//! Port of packages/ai/src/providers/azure-openai-responses.ts
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::StreamExt;
use indexmap::IndexMap;
use serde_json::{Map, Value};

use crate::env_api_keys::get_env_api_key;
use crate::models::clamp_thinking_level;
use crate::types::{
	Api, AssistantMessage, AssistantMessageEvent, Context, Model, ProviderResponse, SimpleStreamOptions,
	StreamFunction, StreamOptions, Usage,
};
use crate::utils::event_stream::{create_assistant_message_event_stream, AssistantMessageEventStream};
use crate::utils::stream_failure::{
	format_stream_failure_message, record_stream_failure, stream_failure_from_stop_reason, ThrownStreamError,
};

use super::openai_responses_shared::{
	convert_responses_messages, convert_responses_tools, process_responses_stream, OpenAIResponsesStreamOptions,
	ResponsesEventStream, ResponsesStreamError,
};
use super::simple_options::build_base_options;

const DEFAULT_AZURE_API_VERSION: &str = "v1";

/// Owns the SDK-style error JSON so [`ThrownStreamError::Value`] can borrow it, exactly like
/// the same-named helper in `openai_responses.rs:76-84`.
struct ThrownValue(Value);

impl ThrownValue {
	fn value(&self) -> &Value {
		&self.0
	}
}

/// The Rust counterpart of a `throw` inside the TypeScript stream body.
///
/// `azure-openai-responses.ts:97` `client.responses.create(...).withResponse()` rejects with
/// the SDK `APIError` and the `catch` at `azure-openai-responses.ts:116-127` runs
/// `formatStreamFailureMessage(error)` + `recordStreamFailure`.
enum AzureRunError {
	/// A `throw` raised by the shared stream helpers.
	Shared(ResponsesStreamError),
	/// An SDK-style error object (`APIError`) whose fields `extractStreamFailureParts`
	/// reads (`utils/stream-failure.ts:130-167`).
	Value(ThrownValue),
}

impl AzureRunError {
	/// `throw new Error(...)`.
	fn message(message: impl Into<String>) -> Self {
		AzureRunError::Shared(ResponsesStreamError::Message(message.into()))
	}

	/// The thrown value as seen by the provider's catch block.
	fn thrown(&self) -> ThrownStreamError<'_> {
		match self {
			AzureRunError::Shared(error) => error.to_thrown(),
			AzureRunError::Value(value) => ThrownStreamError::Value(value.value()),
		}
	}
}

impl From<ResponsesStreamError> for AzureRunError {
	fn from(error: ResponsesStreamError) -> Self {
		AzureRunError::Shared(error)
	}
}

/// TS: `AZURE_TOOL_CALL_PROVIDERS`
fn azure_tool_call_providers(provider: &str) -> bool {
	matches!(provider, "openai" | "openai-codex" | "opencode" | "azure-openai-responses")
}

/// TS: `parseDeploymentNameMap(value)`
fn parse_deployment_name_map(value: Option<&str>) -> IndexMap<String, String> {
	let mut map: IndexMap<String, String> = IndexMap::new();
	let Some(value) = value else {
		return map;
	};
	for entry in value.split(',') {
		let trimmed = entry.trim();
		if trimmed.is_empty() {
			continue;
		}
		// `trimmed.split("=", 2)` (`azure-openai-responses.ts:33`) stops after the first
		// separator and DISCARDS the rest, so `"a=b=c"` yields `["a", "b"]` - unlike
		// `splitn(2, '=')`, which keeps `"b=c"` as the deployment name. Split into three
		// pieces and keep the first two to reproduce the JS limit exactly.
		let mut parts = trimmed.splitn(3, '=');
		let model_id = parts.next().unwrap_or("");
		let deployment_name = parts.next().unwrap_or("");
		if model_id.is_empty() || deployment_name.is_empty() {
			continue;
		}
		map.insert(model_id.trim().to_string(), deployment_name.trim().to_string());
	}
	map
}

/// TS: `resolveDeploymentName(model, options?)`
fn resolve_deployment_name(model: &Model, options: Option<&AzureOpenAIResponsesOptions>) -> String {
	if let Some(name) = options.and_then(|options| options.azure_deployment_name.clone()) {
		if !name.is_empty() {
			return name;
		}
	}
	let env_value = std::env::var("AZURE_OPENAI_DEPLOYMENT_NAME_MAP").ok();
	let mapped = parse_deployment_name_map(env_value.as_deref()).get(&model.id).cloned();
	mapped.unwrap_or_else(|| model.id.clone())
}

/// TS: `AzureOpenAIResponsesOptions extends StreamOptions`
#[derive(Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct AzureOpenAIResponsesOptions {
	#[serde(flatten)]
	pub stream: StreamOptions,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub reasoning_effort: Option<String>,
	/// `reasoningSummary?: "auto" | "detailed" | "concise" | null`
	#[serde(skip_serializing_if = "Option::is_none")]
	pub reasoning_summary: Option<Option<String>>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub azure_api_version: Option<String>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub azure_resource_name: Option<String>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub azure_base_url: Option<String>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub azure_deployment_name: Option<String>,
}

impl std::fmt::Debug for AzureOpenAIResponsesOptions {
	fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		formatter
			.debug_struct("AzureOpenAIResponsesOptions")
			.field("stream", &self.stream)
			.field("reasoning_effort", &self.reasoning_effort)
			.field("reasoning_summary", &self.reasoning_summary)
			.field("azure_api_version", &self.azure_api_version)
			.field("azure_resource_name", &self.azure_resource_name)
			.field("azure_base_url", &self.azure_base_url)
			.field("azure_deployment_name", &self.azure_deployment_name)
			.finish()
	}
}

impl AzureOpenAIResponsesOptions {
	/// TS: the caller passes `StreamOptions`; keep the non-serialisable fields.
	pub fn from_base(base: &StreamOptions) -> Self {
		Self {
			stream: base.clone(),
			reasoning_effort: None,
			reasoning_summary: None,
			azure_api_version: None,
			azure_resource_name: None,
			azure_base_url: None,
			azure_deployment_name: None,
		}
	}
}

/// TS: `streamAzureOpenAIResponses`
pub fn stream_azure_openai_responses(
	model: &Model,
	context: &Context,
	options: Option<AzureOpenAIResponsesOptions>,
) -> AssistantMessageEventStream {
	let stream = create_assistant_message_event_stream();
	let out = stream.clone();
	let model = model.clone();
	let context = context.clone();
	let options = options.unwrap_or_default();
	tokio::spawn(async move {
		let deployment_name = resolve_deployment_name(&model, Some(&options));

		let mut output = AssistantMessage::new(
			"azure-openai-responses".to_string(),
			model.provider.clone(),
			model.id.clone(),
			crate::utils::now_ms(),
		);
		output.usage = Usage::zero();
		output.stop_reason = "stop".to_string();

		match run_azure_openai_responses(&model, &context, &options, &deployment_name, &mut output, &out).await {
			Ok(()) => {}
			Err(error) => {
				// `partialJson` is only a streaming scratch buffer; never persist it.
				// (The port keeps it outside the block, so there is nothing to delete.)
				let aborted = options
					.stream
					.signal
					.as_ref()
					.map(|signal| signal.is_cancelled())
					.unwrap_or(false);
				output.stop_reason = if aborted { "aborted".to_string() } else { "error".to_string() };
				let thrown = error.thrown();
				output.error_message = Some(format_stream_failure_message(&thrown));
				record_stream_failure(&model, &mut output, &thrown);
				out.push(AssistantMessageEvent::Error {
					reason: output.stop_reason.clone(),
					error: output.clone(),
				});
				out.end(None);
			}
		}
	});
	stream
}

/// The TypeScript async IIFE body of `streamAzureOpenAIResponses`.
async fn run_azure_openai_responses(
	model: &Model,
	context: &Context,
	options: &AzureOpenAIResponsesOptions,
	deployment_name: &str,
	output: &mut AssistantMessage,
	stream: &AssistantMessageEventStream,
) -> Result<(), AzureRunError> {
	let api_key = options
		.stream
		.api_key
		.clone()
		.or_else(|| get_env_api_key(&model.provider))
		.unwrap_or_default();
	let client = create_client(model, &api_key, Some(options)).map_err(AzureRunError::message)?;
	let mut params = build_params(model, context, Some(options), deployment_name);

	if let Some(on_payload) = options.stream.on_payload.clone() {
		let next_params = on_payload(Value::Object(params.clone()), model).await;
		if let Some(next_params) = next_params {
			params = match next_params {
				Value::Object(map) => map,
				_ => Map::new(),
			};
		}
	}

	let response = send_request(&client, &params, options).await?;
	if let Some(on_response) = options.stream.on_response.clone() {
		on_response(
			ProviderResponse {
				status: response.status().as_u16() as i64,
				headers: crate::utils::headers::header_map_to_record(response.headers()),
			},
			model,
		)
		.await;
	}
	let request_id = response
		.headers()
		.get("x-request-id")
		.and_then(|value| value.to_str().ok())
		.map(str::to_string);

	stream.push(AssistantMessageEvent::Start {
		partial: output.clone(),
	});

	// A body-read failure rejects the SDK's async iteration, so `processResponsesStream`
	// never completes and the provider catch (azure-openai-responses.ts:116-127) reports the
	// stream as `error`. The scan state is not reachable once the stream is built, so the
	// failure is published through this shared slot as well as `buffer.error`.
	let body_error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
	let body_error_sink = body_error.clone();
	let events: ResponsesEventStream = Box::pin(
		response
			.bytes_stream()
			.scan(super::openai_responses::SseBuffer::default(), move |buffer, chunk| {
				let ready = match chunk {
					Ok(bytes) => buffer.push(&bytes),
					Err(error) => {
						let message = error.to_string();
						buffer.error = Some(message.clone());
						if let Ok(mut slot) = body_error_sink.lock() {
							*slot = Some(message);
						}
						Vec::new()
					}
				};
				futures::future::ready(Some(ready))
			})
			.flat_map(futures::stream::iter),
	);
	let observation_options = options.stream.clone();
	let events = Box::pin(events.inspect(move |event| super::responses_transport::observe_event(&observation_options, event)));
	process_responses_stream(
		events,
		output,
		stream,
		model,
		Some(&OpenAIResponsesStreamOptions {
			on_usage_observation: options.stream.on_usage_observation.clone(),
			..Default::default()
		}),
	)
	.await?;

	// A truncated response must never be reported as a complete one.
	if let Some(message) = body_error.lock().ok().and_then(|slot| slot.clone()) {
		return Err(AzureRunError::message(message));
	}

	if options
		.stream
		.signal
		.as_ref()
		.map(|signal| signal.is_cancelled())
		.unwrap_or(false)
	{
		return Err(AzureRunError::message("Request was aborted"));
	}

	if output.stop_reason == "aborted" || output.stop_reason == "error" {
		return Err(AzureRunError::Shared(ResponsesStreamError::StreamFailure(
			stream_failure_from_stop_reason(output.stop_reason_raw.as_deref(), request_id.as_deref()),
		)));
	}

	stream.push(AssistantMessageEvent::Done {
		reason: output.stop_reason.clone(),
		message: output.clone(),
	});
	stream.end(None);
	Ok(())
}

/// TS: `streamSimpleAzureOpenAIResponses`
pub fn stream_simple_azure_openai_responses(
	model: &Model,
	context: &Context,
	options: Option<SimpleStreamOptions>,
) -> AssistantMessageEventStream {
	let stream = create_assistant_message_event_stream();
	let out = stream.clone();
	let model_owned = model.clone();
	let context_owned = context.clone();
	let options = options.unwrap_or_default();
	tokio::spawn(async move {
		let api_key = options
			.stream
			.api_key
			.clone()
			.or_else(|| get_env_api_key(&model_owned.provider));
		let Some(api_key) = api_key else {
			let mut output = AssistantMessage::new(
				"azure-openai-responses".to_string(),
				model_owned.provider.clone(),
				model_owned.id.clone(),
				crate::utils::now_ms(),
			);
			output.usage = Usage::zero();
			output.stop_reason = "error".to_string();
			output.error_message = Some(format!("No API key for provider: {}", model_owned.provider));
			out.push(AssistantMessageEvent::Error {
				reason: "error".to_string(),
				error: output.clone(),
			});
			out.end(None);
			return;
		};

		let base = build_base_options(&model_owned, Some(&options), Some(&api_key));
		let clamped_reasoning = options
			.reasoning
			.as_ref()
			.map(|level| clamp_thinking_level(&model_owned, level));
		let reasoning_effort = match clamped_reasoning.as_deref() {
			Some("off") | None => None,
			Some(level) => Some(level.to_string()),
		};

		let typed = AzureOpenAIResponsesOptions {
			reasoning_effort,
			..AzureOpenAIResponsesOptions::from_base(&base)
		};
		let inner = stream_azure_openai_responses(&model_owned, &context_owned, Some(typed));
		while let Some(event) = inner.next().await {
			out.push(event);
		}
		out.end(None);
	});
	stream
}

/// TS: `normalizeAzureBaseUrl(baseUrl)`
fn normalize_azure_base_url(base_url: &str) -> Result<String, String> {
	let trimmed = base_url.trim().trim_end_matches('/');
	let mut url = match url::Url::parse(trimmed) {
		Ok(url) => url,
		Err(_) => return Err(format!("Invalid Azure OpenAI base URL: {base_url}")),
	};

	let host = url.host_str().unwrap_or_default().to_string();
	let is_azure_host = host.ends_with(".openai.azure.com") || host.ends_with(".cognitiveservices.azure.com");
	let normalized_path = url.path().trim_end_matches('/').to_string();

	// Ensure Azure hosts have /openai/v1 as base path so the AzureOpenAI SDK
	// can append /deployments/<model>/... and ?api-version=v1 correctly.
	if is_azure_host && (normalized_path.is_empty() || normalized_path == "/" || normalized_path == "/openai") {
		url.set_path("/openai/v1");
		url.set_query(None);
	}

	Ok(url.to_string().trim_end_matches('/').to_string())
}

/// TS: `buildDefaultBaseUrl(resourceName)`
fn build_default_base_url(resource_name: &str) -> String {
	format!("https://{resource_name}.openai.azure.com/openai/v1")
}

/// TS: `resolveAzureConfig(model, options?)`
fn resolve_azure_config(
	model: &Model,
	options: Option<&AzureOpenAIResponsesOptions>,
) -> Result<(String, String), String> {
	// `azure-openai-responses.ts:184`
	// `options?.azureApiVersion || process.env.AZURE_OPENAI_API_VERSION || DEFAULT_AZURE_API_VERSION`:
	// `||` treats an EMPTY string as falsy, so an empty env var must fall through to "v1".
	let api_version = options
		.and_then(|options| options.azure_api_version.clone())
		.filter(|value| !value.is_empty())
		.or_else(|| {
			std::env::var("AZURE_OPENAI_API_VERSION")
				.ok()
				.filter(|value| !value.is_empty())
		})
		.unwrap_or_else(|| DEFAULT_AZURE_API_VERSION.to_string());

	let base_url = options
		.and_then(|options| options.azure_base_url.clone())
		.map(|value| value.trim().to_string())
		.filter(|value| !value.is_empty())
		.or_else(|| {
			std::env::var("AZURE_OPENAI_BASE_URL")
				.ok()
				.map(|value| value.trim().to_string())
				.filter(|value| !value.is_empty())
		});
	let resource_name = options
		.and_then(|options| options.azure_resource_name.clone())
		.filter(|value| !value.is_empty())
		// `azure-openai-responses.ts:187` `options?.azureResourceName || process.env...`:
		// an empty env var is falsy, so it must not build `https://.openai.azure.com/openai/v1`.
		.or_else(|| {
			std::env::var("AZURE_OPENAI_RESOURCE_NAME")
				.ok()
				.filter(|value| !value.is_empty())
		});

	let mut resolved_base_url = base_url;

	if resolved_base_url.is_none() {
		if let Some(resource_name) = resource_name.as_deref() {
			resolved_base_url = Some(build_default_base_url(resource_name));
		}
	}

	if resolved_base_url.is_none() && !model.base_url.is_empty() {
		resolved_base_url = Some(model.base_url.clone());
	}

	let Some(resolved_base_url) = resolved_base_url else {
		return Err(
			"Azure OpenAI base URL is required. Set AZURE_OPENAI_BASE_URL or AZURE_OPENAI_RESOURCE_NAME, or pass azureBaseUrl, azureResourceName, or model.baseUrl.".to_string(),
		);
	};

	Ok((normalize_azure_base_url(&resolved_base_url)?, api_version))
}

/// TS: the `AzureOpenAI` client fields the provider uses.
#[derive(Clone, Debug)]
pub struct AzureClient {
	pub api_key: String,
	pub base_url: String,
	pub api_version: String,
	pub default_headers: IndexMap<String, String>,
}

/// TS: `createClient(model, apiKey, options?)`
fn create_client(
	model: &Model,
	api_key: &str,
	options: Option<&AzureOpenAIResponsesOptions>,
) -> Result<AzureClient, String> {
	let mut api_key = api_key.to_string();
	if api_key.is_empty() {
		match std::env::var("AZURE_OPENAI_API_KEY") {
			Ok(value) if !value.is_empty() => api_key = value,
			_ => {
				return Err(
					"Azure OpenAI API key is required. Set AZURE_OPENAI_API_KEY environment variable or pass it as an argument.".to_string(),
				)
			}
		}
	}

	let mut headers = model.headers.clone().unwrap_or_default();

	if let Some(options) = options {
		if let Some(option_headers) = &options.stream.headers {
			for (key, value) in option_headers {
				headers.insert(key.clone(), value.clone());
			}
		}
	}

	let (base_url, api_version) = resolve_azure_config(model, options)?;

	Ok(AzureClient {
		api_key,
		base_url,
		api_version,
		default_headers: headers,
	})
}

/// TS: `buildParams(model, context, options, deploymentName)`
fn build_params(
	model: &Model,
	context: &Context,
	options: Option<&AzureOpenAIResponsesOptions>,
	deployment_name: &str,
) -> Map<String, Value> {
	let messages = convert_responses_messages(model, context, &azure_tool_call_providers, None)
		.unwrap_or_else(|_| Vec::new());

	let mut params: Map<String, Value> = Map::new();
	params.insert("model".to_string(), Value::String(deployment_name.to_string()));
	params.insert("input".to_string(), Value::Array(messages));
	params.insert("stream".to_string(), Value::Bool(true));
	params.insert("store".to_string(), Value::Bool(false));
	if options.and_then(|options| options.stream.cache_retention.as_deref()) != Some("none") {
		if let Some(session_id) = options.and_then(|options| options.stream.session_id.clone()) {
			params.insert("prompt_cache_key".to_string(), Value::String(session_id));
		}
	}

	if let Some(max_tokens) = options.and_then(|options| options.stream.max_tokens) {
		params.insert("max_output_tokens".to_string(), serde_json::json!(max_tokens));
	}

	if let Some(temperature) = options.and_then(|options| options.stream.temperature) {
		params.insert("temperature".to_string(), serde_json::json!(temperature));
	}

	if let Some(tools) = &context.tools {
		if !tools.is_empty() {
			params.insert(
				"tools".to_string(),
				Value::Array(convert_responses_tools(tools, None)),
			);
		}
	}

	if model.reasoning {
		let reasoning_effort = options.and_then(|options| options.reasoning_effort.clone());
		let reasoning_summary = options.and_then(|options| options.reasoning_summary.clone());
		let has_summary = matches!(reasoning_summary, Some(Some(_)));
		if reasoning_effort.is_some() || has_summary {
			// `azure-openai-responses.ts:269`
			// `model.thinkingLevelMap?.[options.reasoningEffort] ?? options.reasoningEffort`:
			// `??` only replaces a nullish value, so a MISSING key and an explicit `null`
			// entry both fall back to the REQUESTED effort - never to "none". The same
			// expression is already ported at `openai_responses.rs:619-625`.
			let effort = match reasoning_effort {
				Some(effort) => model.thinking_level_map_get(&effort).flatten().unwrap_or(effort),
				None => "medium".to_string(),
			};
			let summary = match reasoning_summary {
				Some(Some(summary)) => summary,
				_ => "auto".to_string(),
			};
			let mut reasoning = Map::new();
			reasoning.insert("effort".to_string(), Value::String(effort));
			reasoning.insert("summary".to_string(), Value::String(summary));
			params.insert("reasoning".to_string(), Value::Object(reasoning));
			params.insert(
				"include".to_string(),
				Value::Array(vec![Value::String("reasoning.encrypted_content".to_string())]),
			);
		} else if model.thinking_level_map_get("off") != Some(None) {
			let effort = model
				.thinking_level_map_get("off")
				.flatten()
				.unwrap_or_else(|| "none".to_string());
			let mut reasoning = Map::new();
			reasoning.insert("effort".to_string(), Value::String(effort));
			params.insert("reasoning".to_string(), Value::Object(reasoning));
		}
	}

	params
}

/// The TypeScript calls `client.responses.create(params, { signal, timeout }).withResponse()`.
/// The Rust port builds the Azure request itself: the AzureOpenAI SDK prefixes `/deployments/<model>`
/// for the deployment endpoints and appends `?api-version=<version>`, and authenticates with
/// the `api-key` header.
async fn send_request(
	client: &AzureClient,
	params: &Map<String, Value>,
	options: &AzureOpenAIResponsesOptions,
) -> Result<reqwest::Response, AzureRunError> {
	let base_url = client.base_url.trim_end_matches('/');
	let model = params.get("model").and_then(Value::as_str).unwrap_or_default();
	// `_deployments_endpoints` in the OpenAI Azure SDK does not list `/responses`, so the SDK
	// does not insert the deployment path for this route.
	let url = format!("{base_url}/responses?api-version={}", client.api_version);

	let mut headers = reqwest::header::HeaderMap::new();
	for (key, value) in client.default_headers.iter() {
		if let (Ok(name), Ok(header_value)) = (
			reqwest::header::HeaderName::from_bytes(key.as_bytes()),
			reqwest::header::HeaderValue::from_str(value),
		) {
			headers.insert(name, header_value);
		}
	}
	if !headers.contains_key("api-key") {
		if let Ok(value) = reqwest::header::HeaderValue::from_str(&client.api_key) {
			headers.insert(reqwest::header::HeaderName::from_static("api-key"), value);
		}
	}
	let _ = model;

	let mut request = crate::providers::shared_http::shared_client(crate::providers::shared_http::ClientPolicy::Default)
		.post(&url)
		.headers(headers)
		.json(&Value::Object(params.clone()));
	if let Some(timeout_ms) = options.stream.timeout_ms {
		request = request.timeout(Duration::from_millis(timeout_ms.max(0.0) as u64));
	}

	let send = request.send();
	let response = match options.stream.signal.as_ref() {
		Some(signal) => tokio::select! {
			_ = signal.cancelled() => return Err(AzureRunError::message("Request was aborted")),
			result = send => result,
		},
		None => send.await,
	};
	let response = response.map_err(|error| AzureRunError::message(error.to_string()))?;
	if !response.status().is_success() {
		return Err(AzureRunError::Value(azure_api_error(response).await));
	}
	Ok(response)
}

/// The OpenAI SDK `APIError.generate(status, error, message, headers)` failure value.
///
/// TS: `client.responses.create(...)` rejects with the SDK `APIError`
/// (`azure-openai-responses.ts:97`, thrown into the `catch` at
/// `azure-openai-responses.ts:116-127`), whose `status`, `headers`, `error` body and
/// `message` are what `extractStreamFailureParts` reads (`utils/stream-failure.ts:142-167`).
/// This mirrors `api_error_from_response` in `openai_responses.rs:731-779`; the two providers
/// share one SDK error class, so the shape must not be re-invented.
async fn azure_api_error(response: reqwest::Response) -> ThrownValue {
	let status = response.status().as_u16() as i64;
	let headers = crate::utils::headers::header_map_to_record(response.headers());
	let text = response.text().await.unwrap_or_default();
	let body: Option<Value> = serde_json::from_str(&text).ok();
	// `APIError.makeMessage(status, error, message)`: the parsed body's `error.message`
	// when present, otherwise the raw text, otherwise "<status> status code (no body)".
	let error_message = body
		.as_ref()
		.and_then(|body| body.get("error"))
		.and_then(|error| error.get("message"))
		.and_then(Value::as_str)
		.map(str::to_string);
	let message = match error_message {
		Some(message) if !message.is_empty() => format!("{status} {message}"),
		_ if !text.is_empty() => format!("{status} {text}"),
		_ => format!("{status} status code (no body)"),
	};
	let mut object = Map::new();
	object.insert("name".to_string(), Value::String("APIError".to_string()));
	object.insert("message".to_string(), Value::String(message));
	object.insert("status".to_string(), Value::Number(status.into()));
	object.insert(
		"headers".to_string(),
		Value::Object(
			headers
				.iter()
				.map(|(key, value)| (key.clone(), Value::String(value.clone())))
				.collect(),
		),
	);
	// The SDK sets `error` to the body's `error` object when it is one, and otherwise to the
	// whole response body (`errorFromResponse(errorResponse) ?? errorResponse`), which is what
	// `extractStreamFailureParts` then reads for the provider type and message.
	let body_error = body.and_then(|body| match body.get("error") {
		Some(Value::Object(_)) => body.get("error").cloned(),
		_ => Some(body.clone()),
	});
	if let Some(error) = body_error {
		object.insert("error".to_string(), error);
	}
	ThrownValue(Value::Object(object))
}

/// Re-exported alias so `register_builtins.rs` can name the api without a type import.
pub type AzureOpenAIResponsesStreamFunction = StreamFunction;

/// Kept so the module references the same `Api` alias the TypeScript module does.
pub fn _api_marker(_api: &Api) {}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::types::{InputModality, Message, Tool, UserContent, UserMessage};

	fn model(provider: &str, id: &str, base_url: &str) -> Model {
		Model {
			id: id.to_string(),
			provider: provider.to_string(),
			api: "azure-openai-responses".to_string(),
			base_url: base_url.to_string(),
			input: vec![InputModality::Text],
			..Default::default()
		}
	}

	fn context() -> Context {
		Context {
			system_prompt: None,
			messages: vec![Message::user(UserMessage::new(UserContent::Text("hi".to_string()), 0))],
			tools: None,
		}
	}

	#[test]
	fn parse_deployment_name_map_matches_typescript() {
		let map = parse_deployment_name_map(Some(" gpt-4=deploy-a , broken , =x , gpt-5 = deploy-b "));
		assert_eq!(map.get("gpt-4").map(String::as_str), Some("deploy-a"));
		assert_eq!(map.get("gpt-5").map(String::as_str), Some("deploy-b"));
		assert_eq!(map.len(), 2);
		assert!(parse_deployment_name_map(None).is_empty());
		assert!(parse_deployment_name_map(Some("")).is_empty());
	}

	#[test]
	fn resolve_deployment_name_prefers_option_then_env_then_model() {
		let mut env = crate::test_env::ScopedEnv::new();
		env.set("AZURE_OPENAI_DEPLOYMENT_NAME_MAP", "model-1=deploy-env");
		let target = model("azure-openai-responses", "model-1", "https://example.openai.azure.com");
		assert_eq!(resolve_deployment_name(&target, None), "deploy-env");

		let options = AzureOpenAIResponsesOptions {
			azure_deployment_name: Some("explicit".to_string()),
			..Default::default()
		};
		assert_eq!(resolve_deployment_name(&target, Some(&options)), "explicit");

		let other = model("azure-openai-responses", "unknown-model", "https://example.openai.azure.com");
		assert_eq!(resolve_deployment_name(&other, None), "unknown-model");
		env.remove("AZURE_OPENAI_DEPLOYMENT_NAME_MAP");
	}

	#[test]
	fn normalize_azure_base_url_rewrites_azure_hosts() {
		assert_eq!(
			normalize_azure_base_url("https://res.openai.azure.com").unwrap(),
			"https://res.openai.azure.com/openai/v1"
		);
		assert_eq!(
			normalize_azure_base_url("https://res.openai.azure.com/").unwrap(),
			"https://res.openai.azure.com/openai/v1"
		);
		assert_eq!(
			normalize_azure_base_url("https://res.openai.azure.com/openai").unwrap(),
			"https://res.openai.azure.com/openai/v1"
		);
		assert_eq!(
			normalize_azure_base_url("https://res.cognitiveservices.azure.com/openai/v1/").unwrap(),
			"https://res.cognitiveservices.azure.com/openai/v1"
		);
	}

	#[test]
	fn normalize_azure_base_url_keeps_custom_paths_and_rejects_invalid() {
		assert_eq!(
			normalize_azure_base_url("https://gateway.example.com/azure-openai/v1").unwrap(),
			"https://gateway.example.com/azure-openai/v1"
		);
		assert_eq!(
			normalize_azure_base_url("https://res.openai.azure.com/custom").unwrap(),
			"https://res.openai.azure.com/custom"
		);
		assert_eq!(
			normalize_azure_base_url("not a url").unwrap_err(),
			"Invalid Azure OpenAI base URL: not a url"
		);
	}

	#[test]
	fn build_default_base_url_matches_typescript() {
		assert_eq!(
			build_default_base_url("my-resource"),
			"https://my-resource.openai.azure.com/openai/v1"
		);
	}

	#[test]
	fn resolve_azure_config_precedence() {
		let mut env = crate::test_env::ScopedEnv::new();
		env.remove("AZURE_OPENAI_BASE_URL");
		env.remove("AZURE_OPENAI_RESOURCE_NAME");
		env.remove("AZURE_OPENAI_API_VERSION");

		let model = model("azure-openai-responses", "m", "https://model.openai.azure.com");
		let (base_url, api_version) = resolve_azure_config(&model, None).unwrap();
		assert_eq!(base_url, "https://model.openai.azure.com/openai/v1");
		assert_eq!(api_version, DEFAULT_AZURE_API_VERSION);

		let options = AzureOpenAIResponsesOptions {
			azure_resource_name: Some("res".to_string()),
			azure_api_version: Some("2024-10-21".to_string()),
			..Default::default()
		};
		let (base_url, api_version) = resolve_azure_config(&model, Some(&options)).unwrap();
		assert_eq!(base_url, "https://res.openai.azure.com/openai/v1");
		assert_eq!(api_version, "2024-10-21");

		let options = AzureOpenAIResponsesOptions {
			azure_base_url: Some("  https://explicit.example.com/v1/  ".to_string()),
			..Default::default()
		};
		let (base_url, _) = resolve_azure_config(&model, Some(&options)).unwrap();
		assert_eq!(base_url, "https://explicit.example.com/v1");
	}

	#[test]
	fn resolve_azure_config_treats_empty_env_values_as_absent() {
		// Held for the whole body: these values are read through the process-global
		// environment, so a parallel test must not write them in between.
		let mut env = crate::test_env::ScopedEnv::new();
		// `azure-openai-responses.ts:184` `||` and `:187` `||` treat an EMPTY env var as
		// falsy, so an empty AZURE_OPENAI_API_VERSION falls back to "v1" and an empty
		// AZURE_OPENAI_RESOURCE_NAME must not produce `https://.openai.azure.com/openai/v1`.
		env.set("AZURE_OPENAI_API_VERSION", "");
		env.set("AZURE_OPENAI_RESOURCE_NAME", "");

		let configured = model("azure-openai-responses", "m", "https://model.openai.azure.com");
		let (base_url, api_version) = resolve_azure_config(&configured, None).unwrap();
		assert_eq!(api_version, DEFAULT_AZURE_API_VERSION);
		assert_eq!(base_url, "https://model.openai.azure.com/openai/v1");

		env.remove("AZURE_OPENAI_BASE_URL");
		let no_base = model("azure-openai-responses", "m", "");
		assert!(
			resolve_azure_config(&no_base, None).is_err(),
			"an empty resource name must not be treated as a configured resource"
		);

	}

	#[test]
	fn resolve_azure_config_errors_without_any_base_url() {
		let mut env = crate::test_env::ScopedEnv::new();
		env.remove("AZURE_OPENAI_BASE_URL");
		env.remove("AZURE_OPENAI_RESOURCE_NAME");
		let model = model("azure-openai-responses", "m", "");
		let error = resolve_azure_config(&model, None).unwrap_err();
		assert_eq!(
			error,
			"Azure OpenAI base URL is required. Set AZURE_OPENAI_BASE_URL or AZURE_OPENAI_RESOURCE_NAME, or pass azureBaseUrl, azureResourceName, or model.baseUrl."
		);
	}

	#[test]
	fn create_client_requires_api_key() {
		let mut env = crate::test_env::ScopedEnv::new();
		env.remove("AZURE_OPENAI_API_KEY");
		let model = model("azure-openai-responses", "m", "https://res.openai.azure.com");
		let error = create_client(&model, "", None).unwrap_err();
		assert_eq!(
			error,
			"Azure OpenAI API key is required. Set AZURE_OPENAI_API_KEY environment variable or pass it as an argument."
		);
	}

	#[test]
	fn create_client_merges_model_and_option_headers() {
		// `azure-openai-responses.ts:184` reads AZURE_OPENAI_API_VERSION, so this test is
		// deterministic only when the ambient value is cleared (same assertions either way).
		// Held for the whole body, so a parallel test cannot set it in between.
		let mut env = crate::test_env::ScopedEnv::new();
		env.remove("AZURE_OPENAI_API_VERSION");
		let mut model = model("azure-openai-responses", "m", "https://res.openai.azure.com");
		let mut model_headers = IndexMap::new();
		model_headers.insert("X-Model".to_string(), "1".to_string());
		model.headers = Some(model_headers);
		let mut option_headers = IndexMap::new();
		option_headers.insert("X-Option".to_string(), "2".to_string());
		let options = AzureOpenAIResponsesOptions {
			stream: StreamOptions {
				headers: Some(option_headers),
				..Default::default()
			},
			..Default::default()
		};
		let client = create_client(&model, "key", Some(&options)).unwrap();
		assert_eq!(client.api_key, "key");
		assert_eq!(client.base_url, "https://res.openai.azure.com/openai/v1");
		assert_eq!(client.api_version, "v1");
		assert_eq!(client.default_headers.get("X-Model").map(String::as_str), Some("1"));
		assert_eq!(client.default_headers.get("X-Option").map(String::as_str), Some("2"));
	}

	#[test]
	fn build_params_matches_typescript_defaults() {
		let model = model("azure-openai-responses", "m", "https://res.openai.azure.com");
		let params = build_params(&model, &context(), None, "deployment-x");
		assert_eq!(params.get("model").and_then(Value::as_str), Some("deployment-x"));
		assert_eq!(params.get("stream").and_then(Value::as_bool), Some(true));
		assert_eq!(params.get("store").and_then(Value::as_bool), Some(false));
		assert!(params.get("input").and_then(Value::as_array).is_some());
		// `prompt_cache_key`, `max_output_tokens` and `temperature` are omitted when unset.
		assert!(!params.contains_key("prompt_cache_key"));
		assert!(!params.contains_key("max_output_tokens"));
		assert!(!params.contains_key("temperature"));
		assert!(!params.contains_key("tools"));
		assert!(!params.contains_key("reasoning"));
	}

	#[test]
	fn build_params_includes_session_tokens_temperature_and_tools() {
		let model = model("azure-openai-responses", "m", "https://res.openai.azure.com");
		let mut context = context();
		context.tools = Some(vec![Tool {
			name: "read".to_string(),
			description: "read".to_string(),
			parameters: serde_json::json!({"type": "object"}),
		}]);
		let options = AzureOpenAIResponsesOptions {
			stream: StreamOptions {
				session_id: Some("session-1".to_string()),
				max_tokens: Some(512.0),
				temperature: Some(0.25),
				..Default::default()
			},
			..Default::default()
		};
		let params = build_params(&model, &context, Some(&options), "m");
		assert_eq!(params.get("prompt_cache_key").and_then(Value::as_str), Some("session-1"));
		assert_eq!(params.get("max_output_tokens").and_then(Value::as_f64), Some(512.0));
		assert_eq!(params.get("temperature").and_then(Value::as_f64), Some(0.25));
		let tools = params.get("tools").and_then(Value::as_array).expect("tools");
		assert_eq!(tools.len(), 1);
		assert_eq!(tools[0].get("type").and_then(Value::as_str), Some("function"));
		assert_eq!(tools[0].get("name").and_then(Value::as_str), Some("read"));
		assert_eq!(tools[0].get("strict").and_then(Value::as_bool), Some(false));
	}

	#[test]
	fn disabled_cache_omits_the_session_cache_key() {
		let model = model("azure-openai-responses", "m", "https://res.openai.azure.com");
		for retention in [None, Some("short"), Some("long"), Some("none")] {
			let options = AzureOpenAIResponsesOptions {
				stream: StreamOptions {
					session_id: Some("session-1".to_string()),
					cache_retention: retention.map(str::to_string),
					..Default::default()
				},
				..Default::default()
			};
			let params = build_params(&model, &context(), Some(&options), "m");
			assert_eq!(params["store"], false);
			assert_eq!(params.contains_key("prompt_cache_key"), retention != Some("none"));
		}
	}

	#[test]
	fn build_params_reasoning_branches() {
		let mut model = model("azure-openai-responses", "m", "https://res.openai.azure.com");
		model.reasoning = true;
		let params = build_params(&model, &context(), None, "m");
		let reasoning = params.get("reasoning").expect("reasoning");
		assert_eq!(reasoning.get("effort").and_then(Value::as_str), Some("none"));
		assert!(!params.contains_key("include"));

		let options = AzureOpenAIResponsesOptions {
			reasoning_effort: Some("high".to_string()),
			reasoning_summary: Some(Some("concise".to_string())),
			..Default::default()
		};
		let params = build_params(&model, &context(), Some(&options), "m");
		let reasoning = params.get("reasoning").expect("reasoning");
		assert_eq!(reasoning.get("effort").and_then(Value::as_str), Some("high"));
		assert_eq!(reasoning.get("summary").and_then(Value::as_str), Some("concise"));
		assert_eq!(
			params.get("include").and_then(Value::as_array).map(|values| values.len()),
			Some(1)
		);
	}

	#[test]
	fn build_params_uses_thinking_level_map_for_effort() {
		let mut model = model("azure-openai-responses", "m", "https://res.openai.azure.com");
		model.reasoning = true;
		model.thinking_level_map = Some(
			[("high".to_string(), Some("high-mapped".to_string()))]
				.into_iter()
				.collect(),
		);
		let options = AzureOpenAIResponsesOptions {
			reasoning_effort: Some("high".to_string()),
			..Default::default()
		};
		let params = build_params(&model, &context(), Some(&options), "m");
		let reasoning = params.get("reasoning").expect("reasoning");
		assert_eq!(reasoning.get("effort").and_then(Value::as_str), Some("high-mapped"));
		// No summary requested -> the TS default "auto".
		assert_eq!(reasoning.get("summary").and_then(Value::as_str), Some("auto"));
	}

	#[test]
	fn build_params_keeps_the_requested_effort_when_the_map_misses() {
		// `azure-openai-responses.ts:269` uses `??`, which only substitutes for a nullish
		// value: a missing key AND an explicit `null` entry both keep the requested effort.
		let mut model = model("azure-openai-responses", "m", "https://res.openai.azure.com");
		model.reasoning = true;
		model.thinking_level_map = Some(
			[("low".to_string(), None), ("high".to_string(), Some("high-mapped".to_string()))]
				.into_iter()
				.collect(),
		);

		let options = AzureOpenAIResponsesOptions {
			reasoning_effort: Some("low".to_string()),
			..Default::default()
		};
		let params = build_params(&model, &context(), Some(&options), "m");
		assert_eq!(
			params["reasoning"]["effort"], serde_json::json!("low"),
			"an explicit null entry falls back to the requested effort, not \"none\""
		);

		let options = AzureOpenAIResponsesOptions {
			reasoning_effort: Some("medium".to_string()),
			..Default::default()
		};
		let params = build_params(&model, &context(), Some(&options), "m");
		assert_eq!(
			params["reasoning"]["effort"], serde_json::json!("medium"),
			"a missing key falls back to the requested effort"
		);
	}

	#[test]
	fn from_base_keeps_non_serialisable_fields() {
		let base = StreamOptions {
			temperature: Some(0.5),
			signal: Some(tokio_util::sync::CancellationToken::new()),
			..Default::default()
		};
		let options = AzureOpenAIResponsesOptions::from_base(&base);
		assert_eq!(options.stream.temperature, Some(0.5));
		assert!(options.stream.signal.is_some());
		assert!(options.azure_base_url.is_none());
	}

	#[test]
	fn parse_deployment_name_map_discards_everything_after_the_first_separator() {
		// `trimmed.split("=", 2)` (`azure-openai-responses.ts:33`) keeps at most two pieces,
		// so `"a=b=c"` yields `["a", "b"]` and `"c"` is dropped by the JS limit.
		// Verified against node: `"a=b=c".split("=", 2)` -> `["a","b"]`.
		let map = parse_deployment_name_map(Some("a=b=c, only-one-part, =noleft, noRight= "));
		assert_eq!(map.get("a").map(String::as_str), Some("b"));
		assert_eq!(map.len(), 1, "entries without both halves are skipped: {map:?}");
	}

	/// A one-response fixture server, the same shape `openai_completions.rs:2464-2501` uses.
	async fn serve_http(
		status: &'static str,
		extra_headers: &'static str,
		body: &'static str,
	) -> (std::net::SocketAddr, tokio::task::JoinHandle<()>) {
		use tokio::io::{AsyncReadExt, AsyncWriteExt};
		let listener = tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).await.unwrap();
		let address = listener.local_addr().unwrap();
		let server = tokio::spawn(async move {
			let (mut socket, _) = listener.accept().await.unwrap();
			let mut request = Vec::new();
			loop {
				let mut buffer = [0u8; 4096];
				let count = socket.read(&mut buffer).await.unwrap();
				if count == 0 {
					return;
				}
				request.extend_from_slice(&buffer[..count]);
				if let Some(end) = request.windows(4).position(|window| window == b"\r\n\r\n") {
					let headers = std::str::from_utf8(&request[..end]).unwrap();
					let length = headers
						.lines()
						.filter_map(|line| line.split_once(':'))
						.find(|(key, _)| key.eq_ignore_ascii_case("content-length"))
						.map(|(_, value)| value.trim().parse::<usize>().unwrap_or(0))
						.unwrap_or(0);
					if request.len() >= end + 4 + length {
						break;
					}
				}
			}
			let response = format!(
				"HTTP/1.1 {status}\r\nContent-Type: application/json\r\n{extra_headers}Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
				body.len()
			);
			socket.write_all(response.as_bytes()).await.unwrap();
			socket.flush().await.unwrap();
		});
		(address, server)
	}

	/// A fixture server that advertises a longer body than it writes, then closes: the
	/// `bytes_stream()` read fails mid-body exactly like a dropped connection.
	async fn serve_truncated_http(partial: &'static str) -> (std::net::SocketAddr, tokio::task::JoinHandle<()>) {
		use tokio::io::{AsyncReadExt, AsyncWriteExt};
		let listener = tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).await.unwrap();
		let address = listener.local_addr().unwrap();
		let server = tokio::spawn(async move {
			let (mut socket, _) = listener.accept().await.unwrap();
			let mut request = Vec::new();
			loop {
				let mut buffer = [0u8; 4096];
				let count = socket.read(&mut buffer).await.unwrap();
				if count == 0 {
					break;
				}
				request.extend_from_slice(&buffer[..count]);
				if request.windows(4).any(|window| window == b"\r\n\r\n") {
					break;
				}
			}
			let response = format!(
				"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: 5000\r\nConnection: close\r\n\r\n{partial}"
			);
			socket.write_all(response.as_bytes()).await.unwrap();
			socket.flush().await.unwrap();
			let _ = socket.shutdown().await;
		});
		(address, server)
	}

	#[tokio::test]
	async fn truncated_body_is_reported_as_an_error_not_a_complete_stream() {
		// The SDK's async iteration rejects when the body read fails, so the TS catch
		// (azure-openai-responses.ts:116-127) reports `stopReason: "error"`. A truncated
		// response must therefore never emit `done`.
		let (address, server) = serve_truncated_http(
			"data: {\"type\":\"response.output_text.delta\",\"delta\":\"partial\"}\n\n",
		)
		.await;
		let mut model = model("azure-openai-responses", "m", &format!("http://{address}"));
		model.max_tokens = 1000.0;
		let options = AzureOpenAIResponsesOptions {
			stream: StreamOptions {
				api_key: Some("fixture-key".to_string()),
				..Default::default()
			},
			..Default::default()
		};
		let mut stream = stream_azure_openai_responses(&model, &context(), Some(options));
		let mut events = Vec::new();
		while let Some(event) = tokio::time::timeout(std::time::Duration::from_secs(5), stream.next())
			.await
			.expect("stream event timeout")
		{
			events.push(event);
		}
		let error = events
			.iter()
			.find_map(|event| match event {
				AssistantMessageEvent::Error { reason, error } => Some((reason.clone(), error.clone())),
				_ => None,
			})
			.expect("truncated stream must produce an error event");
		assert_eq!(error.0, "error");
		assert_eq!(error.1.stop_reason, "error");
		assert!(
			events.iter().all(|event| !matches!(event, AssistantMessageEvent::Done { .. })),
			"a truncated stream must not report completion"
		);
		tokio::time::timeout(std::time::Duration::from_secs(5), server).await.unwrap().unwrap();
	}

	#[tokio::test]
	async fn azure_error_status_throws_the_sdk_api_error_shape() {
		// azure-openai-responses.ts:97 rejects with the SDK `APIError` and the catch at
		// azure-openai-responses.ts:116-127 classifies it, so the Rust throw must carry
		// `status` / `headers` / `error` exactly like `APIError.generate`
		// (`utils/stream-failure.ts:142-167` reads those fields).
		let body = "{\"error\":{\"message\":\"Access denied due to invalid subscription key\",\"type\":\"invalid_request_error\"}}";
		let (address, server) = serve_http(
			"401 Unauthorized",
			"x-request-id: req_azure\r\nRetry-After: 2\r\n",
			body,
		)
		.await;
		let response = reqwest::Client::new()
			.get(format!("http://{address}/probe"))
			.send()
			.await
			.unwrap();
		let value = azure_api_error(response).await;
		let value = value.value();
		assert_eq!(value["name"], serde_json::json!("APIError"));
		assert_eq!(value["status"], serde_json::json!(401));
		assert_eq!(
			value["message"],
			serde_json::json!("401 Access denied due to invalid subscription key"),
			"`APIError.makeMessage` prefixes the status"
		);
		assert_eq!(value["error"]["type"], serde_json::json!("invalid_request_error"));
		assert_eq!(value["headers"]["x-request-id"], serde_json::json!("req_azure"));

		// Downstream classification: the whole point of the shape.
		let thrown = ThrownStreamError::Value(value);
		let parts = crate::utils::stream_failure::extract_stream_failure_parts(&thrown);
		// classifyStreamFailure (stream-failure.ts:70-97): `status === 401` wins over the
		// body type, so the verdict is "auth" - not "invalid_request".
		assert_eq!(parts.info.kind, "auth");
		assert_eq!(parts.info.status, Some(401));
		assert_eq!(parts.info.provider_error_type.as_deref(), Some("invalid_request_error"));
		assert_eq!(parts.info.request_id.as_deref(), Some("req_azure"));
		assert_eq!(parts.info.retry_after_ms, Some(2000.0));
		assert_eq!(
			format_stream_failure_message(&thrown),
			"Provider authentication failed (invalid_request_error, 401): Access denied due to invalid subscription key [request_id: req_azure]"
		);
		tokio::time::timeout(std::time::Duration::from_secs(5), server).await.unwrap().unwrap();
	}

	#[test]
	fn azure_tool_call_providers_matches_typescript_set() {
		for provider in ["openai", "openai-codex", "opencode", "azure-openai-responses"] {
			assert!(azure_tool_call_providers(provider), "{provider}");
		}
		assert!(!azure_tool_call_providers("anthropic"));
	}
}


#[cfg(test)]
mod t15_controls_tests {
	//! T15 owner 'controls': azure-openai-managed/gpt-6-astra route fixture
	//! (installed profile models.json, apiKey redacted; base URL is inert).

	use super::*;
	use crate::types::{InputModality, Message, UserContent, UserMessage};
	use indexmap::IndexMap;

	fn route_model(id: &str) -> Model {
		let mut level_map: IndexMap<String, Option<String>> = IndexMap::new();
		for level in ["off", "low", "medium", "high", "xhigh", "max"] {
			level_map.insert(level.to_string(), Some(level.to_string()));
		}
		Model {
			id: id.to_string(),
			name: id.to_string(),
			api: "azure-openai-responses".to_string(),
			provider: "azure-openai-managed".to_string(),
			base_url: "http://127.0.0.1:43119/azure-openai/v1".to_string(),
			reasoning: true,
			input: vec![InputModality::Text, InputModality::Image],
			thinking_level_map: Some(level_map),
			..Default::default()
		}
	}

	fn context() -> Context {
		Context::new(
			None,
			vec![Message::user(UserMessage::new(UserContent::Text("hi".to_string()), 0))],
			None,
		)
	}

	/// Identity effort map (low/medium/high/xhigh/max) with maxTokens 128000.
	#[test]
	fn t15_route_fixture_azure_openai_managed_gpt_6_astra_payload_contract() {
		let model = route_model("gpt-6-astra");
		let mut options = AzureOpenAIResponsesOptions::default();
		options.stream.max_tokens = Some(128_000.0);
		options.stream.api_key = Some("test-key".to_string());
		options.reasoning_effort = Some("xhigh".to_string());
		let params = build_params(&model, &context(), Some(&options), "gpt-6-astra");
		assert_eq!(params.get("model").and_then(Value::as_str), Some("gpt-6-astra"));
		assert_eq!(params.get("max_output_tokens").and_then(Value::as_f64), Some(128_000.0));
		let reasoning = params.get("reasoning").expect("reasoning");
		assert_eq!(reasoning.get("effort").and_then(Value::as_str), Some("xhigh"));
		assert_eq!(reasoning.get("summary").and_then(Value::as_str), Some("auto"));
		// The full effort ladder of the route passes through unchanged.
		for level in ["low", "medium", "high", "max"] {
			let mut options = AzureOpenAIResponsesOptions::default();
			options.reasoning_effort = Some(level.to_string());
			let params = build_params(&model, &context(), Some(&options), "gpt-6-astra");
			let reasoning = params.get("reasoning").expect("reasoning");
			assert_eq!(reasoning.get("effort").and_then(Value::as_str), Some(level), "level {level}");
		}
	}
}
