//! Port of packages/ai/src/providers/amazon-bedrock-responses.ts
//!
//! The TypeScript talks to the OpenAI SDK pointed at the Bedrock Responses endpoint; the Rust
//! port builds the same JSON request, signs it with SigV4 (see `bedrock_responses_client.rs`),
//! decodes the SSE stream locally and hands the events to `process_responses_stream`, exactly
//! like the TypeScript hands them to `processResponsesStream`.

use std::sync::Arc;

use serde_json::{json, Map, Value};

use crate::models::clamp_thinking_level;
use crate::types::{AssistantMessage, AssistantMessageEvent, Context, Model, SimpleStreamOptions, StreamOptions, Usage};
use crate::utils::event_stream::AssistantMessageEventStream;
use crate::utils::headers::header_map_to_record;
use crate::utils::now_ms;
use crate::utils::stream_failure::{
	format_stream_failure_message, record_stream_failure, stream_failure_from_stop_reason,
};

use super::bedrock_responses_client::{
	create_bedrock_responses_client, responses_event_stream, BedrockResponsesAuthOptions,
};
/// The thrown-value type is owned by the transport module that can raise every variant; it is
/// re-exported here so this module's call sites keep their name.
pub use super::bedrock_responses_client::ResponsesRunError;
use super::openai_responses_shared::{
	convert_responses_messages, convert_responses_tools, process_responses_stream, OpenAIResponsesStreamOptions,
};
use super::simple_options::build_base_options;

/// TS: `const BEDROCK_TOOL_CALL_PROVIDERS = new Set(["amazon-bedrock"])`.
pub const BEDROCK_TOOL_CALL_PROVIDERS: [&str; 1] = ["amazon-bedrock"];

/// TS: `BEDROCK_TOOL_CALL_PROVIDERS.has(provider)`.
pub fn is_bedrock_tool_call_provider(provider: &str) -> bool {
	BEDROCK_TOOL_CALL_PROVIDERS.contains(&provider)
}

/// TS: `interface BedrockResponsesOptions extends BedrockResponsesAuthOptions`.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct BedrockResponsesOptions {
	#[serde(flatten)]
	pub stream: StreamOptions,
	/// `BedrockResponsesAuthOptions.region`
	#[serde(skip_serializing_if = "Option::is_none")]
	pub region: Option<String>,
	/// `BedrockResponsesAuthOptions.profile`
	#[serde(skip_serializing_if = "Option::is_none")]
	pub profile: Option<String>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub reasoning_effort: Option<String>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub reasoning_summary: Option<String>,
	/// TS: `credentialProvider?: ReturnType<typeof defaultProvider>`.
	#[serde(skip)]
	pub credentials: Option<super::amazon_bedrock::AwsCredentials>,
	#[serde(skip)]
	pub has_credential_provider: bool,
}

impl BedrockResponsesOptions {
	/// TS: the caller passes `StreamOptions & Record<string, unknown>`; this keeps the
	/// non-serializable fields (signal, on_payload, on_response, on_usage_observation).
	pub fn from_base(base: &StreamOptions) -> Self {
		Self {
			stream: base.clone(),
			..Default::default()
		}
	}

	/// The `BedrockResponsesAuthOptions` view the client factory takes.
	pub fn auth_options(&self) -> BedrockResponsesAuthOptions {
		BedrockResponsesAuthOptions {
			stream: self.stream.clone(),
			region: self.region.clone(),
			profile: self.profile.clone(),
			credentials: self.credentials.clone(),
			has_credential_provider: self.has_credential_provider,
		}
	}
}

/// TS: `streamBedrockResponses(model, context, options?)` registered as a builtin.
pub fn stream_bedrock_responses(
	model: &Model,
	context: &Context,
	options: Option<&StreamOptions>,
) -> AssistantMessageEventStream {
	stream_bedrock_responses_with_options(
		model,
		context,
		options.map(BedrockResponsesOptions::from_base),
	)
}

/// The typed entry point, mirroring the TypeScript signature.
pub fn stream_bedrock_responses_with_options(
	model: &Model,
	context: &Context,
	options: Option<BedrockResponsesOptions>,
) -> AssistantMessageEventStream {
	let stream = AssistantMessageEventStream::new_owned();

	let model = model.clone();
	let context = context.clone();
	let options = options.unwrap_or_default();
	let producer = stream.producer_handle();
	let out = stream.producer_handle();

	// TS: `(async () => { try { ... } catch (error) { ... } })();`
	producer.spawn(async move {
		let mut output = AssistantMessage {
			role: "assistant".to_string(),
			content: Vec::new(),
			api: "bedrock-responses".to_string(),
			provider: model.provider.clone(),
			model: model.id.clone(),
			usage: Usage::zero(),
			stop_reason: "stop".to_string(),
			timestamp: now_ms(),
			..Default::default()
		};

		let result = run_bedrock_responses_stream(&model, &context, &options, &mut output, &out).await;

		match result {
			Ok(()) => {
				out.push(AssistantMessageEvent::Done {
					reason: output.stop_reason.clone(),
					message: output,
				});
				out.end(None);
			}
			Err(error) => {
				let aborted = options
					.stream
					.signal
					.as_ref()
					.map(|signal| signal.is_cancelled())
					.unwrap_or(false);
				output.stop_reason = if aborted { "aborted" } else { "error" }.to_string();
				// TS: `output.errorMessage = formatStreamFailureMessage(error);`
				let thrown = error.as_thrown();
				output.error_message = Some(format_stream_failure_message(&thrown));
				record_stream_failure(&model, &mut output, &thrown);
				out.push(AssistantMessageEvent::Error {
					reason: output.stop_reason.clone(),
					error: output,
				});
				out.end(None);
			}
		}
	});

	stream
}

/// TS: `streamSimpleBedrockResponses(model, context, options?)`.
pub fn stream_simple_bedrock_responses(
	model: &Model,
	context: &Context,
	options: Option<&StreamOptions>,
) -> AssistantMessageEventStream {
	stream_simple_bedrock_responses_with_options(model, context, options.map(|base| SimpleStreamOptions {
		stream: base.clone(),
		reasoning: None,
		thinking_budgets: None,
	}))
}

/// The options object `streamSimpleBedrockResponses` hands to `streamBedrockResponses`.
///
/// TS (`amazon-bedrock-responses.ts:101-110`):
/// ```text
/// const base = buildBaseOptions(model, options);
/// const reasoningEffort = clampThinkingLevel(model, options?.reasoning ?? "medium");
/// return streamBedrockResponses(model, context, {
///     ...base,
///     onUsageObservation: options?.onUsageObservation,
///     reasoningEffort,
/// });
/// ```
///
/// Unlike the original TypeScript base builder, the shared Rust builder now preserves
/// monitoring callbacks. This provider's explicit forwarding retains the same callback.
pub fn build_simple_stream_options(
	model: &Model,
	options: Option<&SimpleStreamOptions>,
) -> BedrockResponsesOptions {
	// TS: `const base = buildBaseOptions(model, options);`
	let base = build_base_options(model, options, None);
	// TS: `clampThinkingLevel(model, options?.reasoning ?? "medium")`
	let reasoning_effort = clamp_thinking_level(
		model,
		options
			.and_then(|options| options.reasoning.as_deref())
			.unwrap_or("medium"),
	);
	// TS: `{ ...base, onUsageObservation: options?.onUsageObservation, reasoningEffort }`
	let mut stream_options = base;
	stream_options.on_usage_observation =
		options.and_then(|options| options.stream.on_usage_observation.clone());
	BedrockResponsesOptions {
		stream: stream_options,
		reasoning_effort: Some(reasoning_effort),
		..Default::default()
	}
}

/// The typed entry point, mirroring the TypeScript `SimpleStreamOptions` signature.
pub fn stream_simple_bedrock_responses_with_options(
	model: &Model,
	context: &Context,
	options: Option<SimpleStreamOptions>,
) -> AssistantMessageEventStream {
	stream_bedrock_responses_with_options(
		model,
		context,
		Some(build_simple_stream_options(model, options.as_ref())),
	)
}

/// The OpenAI SDK `APIError.generate(status, error, message, headers)` failure value for a
/// non-2xx Bedrock proxy response.
///
/// TS: `client.responses.create(params, requestOptions).withResponse()`
/// (`amazon-bedrock-responses.ts:59`) rejects with the SDK `APIError` when the response status
/// is not ok (`openai@6.x` `makeStatusError`), and the `catch` at
/// `amazon-bedrock-responses.ts:79-90` passes it to `formatStreamFailureMessage` /
/// `recordStreamFailure`, which read `error.status`, `error.headers`, `error.error` and
/// `error.message` (`utils/stream-failure.ts:142-167`).
///
/// The OpenAI and Azure Responses ports share that one SDK error class, so this is the same
/// shape as `api_error_from_response` (`openai_responses.rs:737-779`) and `azure_api_error`
/// (`azure_openai_responses.rs:642-667`). The shape must not be re-invented.
async fn bedrock_api_error(response: reqwest::Response) -> Value {
	let status = response.status().as_u16() as i64;
	let headers = header_map_to_record(response.headers());
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
	Value::Object(object)
}

/// TS: `buildParams(model, context, options?)`.
pub fn build_params(
	model: &Model,
	context: &Context,
	options: Option<&BedrockResponsesOptions>,
) -> Result<Value, String> {
	if let Some(Some(service_tier)) = options.and_then(|options| options.stream.service_tier.as_ref()) {
		if service_tier != "default" && service_tier != "auto" {
			return Err("Bedrock Astra supports only the Standard service tier.".to_string());
		}
	}

	let mut params: Map<String, Value> = Map::new();
	params.insert("model".to_string(), Value::String(model.id.clone()));
	params.insert(
		"input".to_string(),
		Value::Array(convert_responses_messages(
			model,
			context,
			&is_bedrock_tool_call_provider,
			None,
		)?),
	);
	params.insert("stream".to_string(), Value::Bool(true));
	params.insert("store".to_string(), Value::Bool(false));
	params.insert("service_tier".to_string(), Value::String("default".to_string()));
	// TS: `max_output_tokens: options?.maxTokens` - `undefined` drops the key on the wire.
	if let Some(max_tokens) = options.and_then(|options| options.stream.max_tokens) {
		params.insert("max_output_tokens".to_string(), json!(max_tokens));
	}
	if let Some(tools) = context.tools.as_ref() {
		if !tools.is_empty() {
			params.insert(
				"tools".to_string(),
				Value::Array(convert_responses_tools(tools, None)),
			);
		}
	}
	if model.reasoning {
		let effort = options
			.and_then(|options| options.reasoning_effort.clone())
			.unwrap_or_else(|| "medium".to_string());
		if !["low", "medium", "high", "xhigh", "max"].contains(&effort.as_str()) {
			return Err("Bedrock Astra reasoning effort must be low, medium, high, xhigh, or max.".to_string());
		}
		params.insert(
			"reasoning".to_string(),
			json!({
				"effort": effort,
				"summary": options
					.and_then(|options| options.reasoning_summary.clone())
					.unwrap_or_else(|| "auto".to_string()),
			}),
		);
		params.insert("include".to_string(), json!(["reasoning.encrypted_content"]));
	}
	Ok(Value::Object(params))
}

/// The whole `try { ... }` body of the TypeScript async IIFE.
async fn run_bedrock_responses_stream(
	model: &Model,
	context: &Context,
	options: &BedrockResponsesOptions,
	output: &mut AssistantMessage,
	stream: &AssistantMessageEventStream,
) -> Result<(), ResponsesRunError> {
	// TS: `const client = createBedrockResponsesClient(model, options);`
	let client = create_bedrock_responses_client(model, Some(&options.auth_options()))
		.map_err(ResponsesRunError::message)?;
	// TS: `let params = buildParams(model, context, options);`
	let mut params = build_params(model, context, Some(options)).map_err(ResponsesRunError::message)?;
	// TS: `const nextParams = await options?.onPayload?.(params, model);`
	if let Some(on_payload) = options.stream.on_payload.clone() {
		if let Some(next) = on_payload(params.clone(), model).await {
			params = next;
		}
	}

	// TS: `const { data: openaiStream, response } = await client.responses
	//      .create(params, requestOptions).withResponse();`
	let response = client
		.send_responses(
			&match params {
				Value::Object(object) => object,
				_ => serde_json::Map::new(),
			},
			Some(&options.auth_options()),
		)
		.await
		.map_err(ResponsesRunError::message)?;

	// TS: `client.responses.create(...).withResponse()` (amazon-bedrock-responses.ts:59) - the
	// OpenAI SDK rejects the promise with an SDK `APIError` for a non-2xx status, so the
	// `catch` at amazon-bedrock-responses.ts:79-90 runs `formatStreamFailureMessage` +
	// `recordStreamFailure`. Without this check the HTTP error body (JSON, no SSE frames)
	// would decode to zero events and be reported as a completed empty answer.
	// Same check as the Azure adapter, which shares the SDK error class
	// (`azure_openai_responses.rs:611-613`).
	if !response.status().is_success() {
		return Err(ResponsesRunError::ApiError(bedrock_api_error(response).await));
	}

	let status = response.status().as_u16();
	let response_headers = response.headers().clone();
	// TS: `await options?.onResponse?.({ status, headers: headersToRecord(response.headers) }, model);`
	{
		let record = header_map_to_record(&response_headers);
		if let Some(on_response) = options.stream.on_response.clone() {
			on_response(
				crate::types::ProviderResponse {
					status: status as i64,
					headers: record,
				},
				model,
			)
			.await;
		}
	}
	// TS: `const requestId = response.headers.get("x-request-id") ?? undefined;`
	let request_id = response_headers
		.get("x-request-id")
		.and_then(|value| value.to_str().ok())
		.map(str::to_string);

	stream.push(AssistantMessageEvent::Start {
		partial: output.clone(),
	});

	// TS: `{ onUsageObservation: options?.onUsageObservation,
	//      applyServiceTierPricing: applyBedrockAstraContextPricing }`
	let stream_options = OpenAIResponsesStreamOptions {
		on_output_item_done: None,
		service_tier: None,
		resolve_service_tier: None,
		apply_service_tier_pricing: Some(Arc::new(|usage: &mut Usage, _service_tier: Option<&str>| {
			apply_bedrock_astra_context_pricing(usage)
		})),
		on_usage_observation: options.stream.on_usage_observation.clone(),
        idle_timeout: Some(crate::utils::sse_frames::resolve_sse_idle_timeout()),
	};

	// The SSE reader `throw`s out of the `for await` loop for a body read error, an
	// unparseable frame or a `data.error` frame; a Rust stream cannot, so the thrown value is
	// parked in this slot and re-raised below - the same `catch` boundary and the same message
	// (`amazon-bedrock-responses.ts:79-90`). Same pattern as `map_codex_events`
	// (`openai_codex_responses.rs:1058-1084`).
	let sse_error: Arc<std::sync::Mutex<Option<ResponsesRunError>>> =
		Arc::new(std::sync::Mutex::new(None));
	let events = responses_event_stream(response, sse_error.clone());
	let parsed = process_responses_stream(events, output, stream, model, Some(&stream_options)).await;

	// SSE read, JSON parse and data.error failures explain the early EOF. Surface
	// their original details before the shared parser's missing-terminal fallback.
	if let Some(error) = sse_error.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).take() {
		return Err(error);
	}
	parsed.map_err(|error| match error {
		super::openai_responses_shared::ResponsesStreamError::StreamFailure(failure) => {
			ResponsesRunError::Failure(Box::new(failure))
		}
		super::openai_responses_shared::ResponsesStreamError::Message(message) => {
			ResponsesRunError::Message(message)
		}
	})?;

	if options
		.stream
		.signal
		.as_ref()
		.map(|signal| signal.is_cancelled())
		.unwrap_or(false)
	{
		return Err(ResponsesRunError::message("Request was aborted"));
	}

	if output.stop_reason == "aborted" || output.stop_reason == "error" {
		let failure = stream_failure_from_stop_reason(output.stop_reason_raw.as_deref(), request_id.as_deref());
		return Err(ResponsesRunError::Failure(Box::new(failure)));
	}

	Ok(())
}

/// TS: `function applyBedrockAstraContextPricing(usage: Usage): void`.
pub fn apply_bedrock_astra_context_pricing(usage: &mut Usage) {
	if usage.input + usage.cache_read + usage.cache_write <= 272_000.0 {
		return;
	}
	usage.cost.input *= 2.0;
	usage.cost.cache_read *= 2.0;
	usage.cost.cache_write *= 2.0;
	usage.cost.output *= 1.5;
	usage.cost.total = usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write;
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::types::{Message, ModelCost, Tool, UserContent, UserMessage};
	use crate::utils::stream_failure::ThrownStreamError;

	fn model(id: &str) -> Model {
		let mut built = Model::new(
			id,
			"Astra",
			"bedrock-responses",
			"amazon-bedrock",
			"https://bedrock-mantle.us-west-2.api.aws/openai/v1",
		);
		built.reasoning = true;
		built.cost = ModelCost::default();
		built
	}

	fn context(text: &str) -> Context {
		Context::new(
			None,
			vec![Message::user(UserMessage::new(UserContent::Text(text.to_string()), 0))],
			None,
		)
	}

	#[test]
	fn tool_call_providers_match_the_typescript_set() {
		assert!(is_bedrock_tool_call_provider("amazon-bedrock"));
		assert!(!is_bedrock_tool_call_provider("openai"));
	}

	#[test]
	fn astra_context_pricing_applies_above_the_threshold() {
		let mut usage = Usage {
			input: 100_000.0,
			cache_read: 100_000.0,
			cache_write: 72_000.0,
			output: 1_000.0,
			total_tokens: 0.0,
			cost: crate::types::UsageCost {
				input: 1.0,
				output: 1.0,
				cache_read: 1.0,
				cache_write: 1.0,
				total: 4.0,
			},
		};
		// 272_000 is the boundary: `<= 272_000` returns early.
		apply_bedrock_astra_context_pricing(&mut usage);
		assert_eq!(usage.cost.input, 1.0);

		usage.input = 100_001.0;
		apply_bedrock_astra_context_pricing(&mut usage);
		assert_eq!(usage.cost.input, 2.0);
		assert_eq!(usage.cost.cache_read, 2.0);
		assert_eq!(usage.cost.cache_write, 2.0);
		assert_eq!(usage.cost.output, 1.5);
		assert_eq!(usage.cost.total, 7.5);
	}

	#[test]
	fn build_params_matches_the_typescript_shape() {
		let params = build_params(&model("openai.gpt-6-astra"), &context("hello"), None).unwrap();
		let keys: Vec<&String> = params.as_object().unwrap().keys().collect();
		assert_eq!(
			keys,
			vec!["model", "input", "stream", "store", "service_tier", "reasoning", "include"]
		);
		assert_eq!(params["model"], json!("openai.gpt-6-astra"));
		assert_eq!(params["stream"], json!(true));
		assert_eq!(params["store"], json!(false));
		assert_eq!(params["service_tier"], json!("default"));
		assert_eq!(params["reasoning"], json!({ "effort": "medium", "summary": "auto" }));
		assert_eq!(params["include"], json!(["reasoning.encrypted_content"]));
		// `max_output_tokens: undefined` drops the key.
		assert!(params.get("max_output_tokens").is_none());
		assert!(params.get("tools").is_none());
	}

	#[test]
	fn build_params_carries_max_tokens_tools_and_reasoning_options() {
		let mut ctx = context("hello");
		ctx.tools = Some(vec![Tool {
			name: "read".to_string(),
			description: "Read a file".to_string(),
			parameters: json!({ "type": "object" }),
		}]);
		let options = BedrockResponsesOptions {
			stream: StreamOptions {
				max_tokens: Some(4096.0),
				..Default::default()
			},
			reasoning_effort: Some("high".to_string()),
			reasoning_summary: Some("concise".to_string()),
			..Default::default()
		};
		let params = build_params(&model("openai.gpt-6-astra"), &ctx, Some(&options)).unwrap();
		assert_eq!(params["max_output_tokens"], json!(4096.0));
		assert_eq!(params["tools"][0]["type"], json!("function"));
		assert_eq!(params["tools"][0]["name"], json!("read"));
		assert_eq!(params["tools"][0]["strict"], json!(false));
		assert_eq!(params["reasoning"], json!({ "effort": "high", "summary": "concise" }));

		// Empty tool lists do not add the key.
		let mut ctx = context("hello");
		ctx.tools = Some(Vec::new());
		let params = build_params(&model("openai.gpt-6-astra"), &ctx, None).unwrap();
		assert!(params.get("tools").is_none());
	}

	#[test]
	fn build_params_rejects_unsupported_service_tiers_and_efforts() {
		let options = BedrockResponsesOptions {
			stream: StreamOptions {
				service_tier: Some(Some("flex".to_string())),
				..Default::default()
			},
			..Default::default()
		};
		assert_eq!(
			build_params(&model("openai.gpt-6-astra"), &context("hi"), Some(&options)).unwrap_err(),
			"Bedrock Astra supports only the Standard service tier."
		);

		for allowed in ["default", "auto"] {
			let options = BedrockResponsesOptions {
				stream: StreamOptions {
					service_tier: Some(Some(allowed.to_string())),
					..Default::default()
				},
				..Default::default()
			};
			assert!(build_params(&model("openai.gpt-6-astra"), &context("hi"), Some(&options)).is_ok());
		}

		let options = BedrockResponsesOptions {
			reasoning_effort: Some("minimal".to_string()),
			..Default::default()
		};
		assert_eq!(
			build_params(&model("openai.gpt-6-astra"), &context("hi"), Some(&options)).unwrap_err(),
			"Bedrock Astra reasoning effort must be low, medium, high, xhigh, or max."
		);

		for allowed in ["low", "medium", "high", "xhigh", "max"] {
			let options = BedrockResponsesOptions {
				reasoning_effort: Some(allowed.to_string()),
				..Default::default()
			};
			let params = build_params(&model("openai.gpt-6-astra"), &context("hi"), Some(&options)).unwrap();
			assert_eq!(params["reasoning"]["effort"], json!(allowed));
		}
	}

	#[test]
	fn build_params_skips_reasoning_for_non_reasoning_models() {
		let mut non_reasoning_model = model("openai.gpt-6-astra");
		non_reasoning_model.reasoning = false;
		let params = build_params(&non_reasoning_model, &context("hi"), None).unwrap();
		assert!(params.get("reasoning").is_none());
		assert!(params.get("include").is_none());
	}

	#[test]
	fn options_round_trip_through_serde() {
		let options = BedrockResponsesOptions {
			stream: StreamOptions {
				max_tokens: Some(10.0),
				..Default::default()
			},
			region: Some("us-west-2".to_string()),
			profile: Some("dev".to_string()),
			reasoning_effort: Some("high".to_string()),
			reasoning_summary: Some("auto".to_string()),
			..Default::default()
		};
		let value = serde_json::to_value(&options).unwrap();
		assert_eq!(value["maxTokens"], json!(10.0));
		assert_eq!(value["region"], json!("us-west-2"));
		assert_eq!(value["profile"], json!("dev"));
		assert_eq!(value["reasoningEffort"], json!("high"));
		assert_eq!(value["reasoningSummary"], json!("auto"));

		let parsed: BedrockResponsesOptions = serde_json::from_value(value).unwrap();
		assert_eq!(parsed.region, options.region);
		assert_eq!(parsed.stream.max_tokens, Some(10.0));
		assert!(parsed.credentials.is_none());

		let base = StreamOptions {
			temperature: Some(0.2),
			..Default::default()
		};
		let typed = BedrockResponsesOptions::from_base(&base);
		assert_eq!(typed.stream.temperature, Some(0.2));
		assert!(typed.reasoning_effort.is_none());
		let auth = typed.auth_options();
		assert_eq!(auth.stream.temperature, Some(0.2));
		assert!(auth.region.is_none());
	}

	/// A throwaway HTTP/1.1 server that answers one request with a fixed status/body and reads
	/// the request to completion, so a body read cannot race the response write.
	async fn serve_http(
		status: &str,
		extra_headers: &str,
		body: &'static str,
	) -> (std::net::SocketAddr, tokio::task::JoinHandle<()>) {
		use tokio::io::{AsyncReadExt, AsyncWriteExt};
		let listener = tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).await.unwrap();
		let address = listener.local_addr().unwrap();
		let response = format!(
			"HTTP/1.1 {status}\r\nContent-Type: application/json\r\n{extra_headers}Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
			body.len()
		);
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
			socket.write_all(response.as_bytes()).await.unwrap();
			socket.flush().await.unwrap();
		});
		(address, server)
	}

	#[tokio::test]
	async fn http_error_status_throws_the_sdk_api_error_shape() {
		// `amazon-bedrock-responses.ts:59` (`client.responses.create(...).withResponse()`)
		// rejects with the SDK `APIError` for a non-2xx status (the OpenAI SDK calls
		// `makeStatusError`), and the `catch` at `amazon-bedrock-responses.ts:79-90`
		// classifies it through `utils/stream-failure.ts:142-167`, which reads
		// `error.status`, `error.headers`, `error.error` and `error.message`.
		//
		// Before the status check this body decoded to zero SSE events, so the failure was
		// reported to the user as a SUCCESSFUL EMPTY ANSWER (`stopReason: "stop"`).
		let body = "{\"error\":{\"message\":\"The security token included in the request is invalid\",\"type\":\"invalid_request_error\"}}";
		let (address, server) =
			serve_http("401 Unauthorized", "x-request-id: req_bedrock\r\nRetry-After: 2\r\n", body).await;
		let response = reqwest::Client::new()
			.get(format!("http://{address}/probe"))
			.send()
			.await
			.unwrap();
		let value = bedrock_api_error(response).await;
		assert_eq!(value["name"], json!("APIError"));
		assert_eq!(value["status"], json!(401));
		assert_eq!(
			value["message"],
			json!("401 The security token included in the request is invalid")
		);
		assert_eq!(value["error"]["type"], json!("invalid_request_error"));

		// Downstream classification: the whole point of the shape.
		let thrown = ThrownStreamError::Value(&value);
		let parts = crate::utils::stream_failure::extract_stream_failure_parts(&thrown);
		// `classifyStreamFailure` (`utils/stream-failure.ts:70-97`): `status === 401` wins over
		// the body type, so the verdict is "auth" - not "invalid_request".
		assert_eq!(parts.info.kind, "auth");
		assert_eq!(parts.info.status, Some(401));
		assert_eq!(parts.info.provider_error_type.as_deref(), Some("invalid_request_error"));
		assert_eq!(parts.info.request_id.as_deref(), Some("req_bedrock"));
		assert_eq!(
			format_stream_failure_message(&thrown),
			"Provider authentication failed (invalid_request_error, 401): The security token included in the request is invalid [request_id: req_bedrock]"
		);
		tokio::time::timeout(std::time::Duration::from_secs(5), server).await.unwrap().unwrap();
	}

	#[test]
	fn http_error_status_ends_the_run_as_an_error_not_an_empty_answer() {
		// Held for the whole test, so it must not be an async test: a std lock guard live
		// across an `.await` is exactly what the client module's env guards avoid too.
		// This is the crate-wide `AWS_*` lock, so a client test that writes
		// `AWS_BEDROCK_BASE_URL` cannot land between this test's request and its read.
		let _env = crate::providers::bedrock_responses_client::aws_env_test_lock()
			.lock()
			.unwrap_or_else(|poisoned| poisoned.into_inner());
		let runtime = tokio::runtime::Builder::new_current_thread()
			.enable_all()
			.build()
			.expect("current-thread runtime");
		runtime.block_on(http_error_status_ends_the_run_as_an_error_not_an_empty_answer_body());
	}

	async fn http_error_status_ends_the_run_as_an_error_not_an_empty_answer_body() {
		// End-to-end through the real entry point: a 500 with a JSON error body must produce an
		// `error` event carrying the thrown message - never `done` with `stopReason: "stop"`,
		// zero usage and no `errorMessage`.
		//
		// Hermetic: the model base URL points at the throwaway server, so no `AWS_*` process
		// variable is written. The caller holds the shared lock because a concurrent client
		// test sets `AWS_BEDROCK_BASE_URL`, which would otherwise outrank this base URL
		// (`bedrock-responses-client.ts:20`).
		let body = "{\"error\":{\"message\":\"Bedrock is having a bad day\",\"type\":\"server_error\"}}";
		let (address, server) = serve_http("500 Internal Server Error", "", body).await;
		let mut streaming_model = Model::new(
			"openai.gpt-6-astra",
			"Astra",
			"bedrock-responses",
			"amazon-bedrock",
			&format!("http://{address}/openai/v1"),
		);
		streaming_model.reasoning = false;
		streaming_model.cost = ModelCost::default();

		// No `.into_stream()`: `AssistantMessageEventStream::next()` is the port's own
		// async-iterator (`event_stream.rs:225`), and its `into_stream()` wrapper is `!Unpin`.
		let stream = stream_bedrock_responses_with_options(
			&streaming_model,
			&context("hello"),
			Some(BedrockResponsesOptions {
				stream: StreamOptions {
					api_key: Some("token".to_string()),
					..Default::default()
				},
				region: Some("us-west-2".to_string()),
				..Default::default()
			}),
		);

		let mut saw_error = false;
		while let Some(event) = stream.next().await {
			match event {
				AssistantMessageEvent::Error { reason, error } => {
					assert_eq!(reason, "error");
					assert_eq!(error.stop_reason, "error");
					assert!(
						error
							.error_message
							.as_deref()
							.unwrap_or_default()
							.contains("Bedrock is having a bad day"),
						"errorMessage must carry the provider error: {:?}",
						error.error_message
					);
					saw_error = true;
				}
				AssistantMessageEvent::Done { message, .. } => panic!(
					"an HTTP error must not be reported as a completed answer: stopReason={} errorMessage={:?}",
					message.stop_reason, message.error_message
				),
				_ => {}
			}
		}
		assert!(saw_error, "the failed request must surface an error event");
		tokio::time::timeout(std::time::Duration::from_secs(5), server).await.unwrap().unwrap();
	}


	#[test]
	fn responses_run_error_preserves_the_thrown_shape() {
		let error = ResponsesRunError::message("Request was aborted");
		assert_eq!(error.to_string(), "Request was aborted");
		match error.as_thrown() {
			ThrownStreamError::Message(message) => assert_eq!(message, "Request was aborted"),
			ThrownStreamError::Failure(failure) => panic!("unexpected stream failure: {}", failure.message),
			ThrownStreamError::Error(error) => panic!("unexpected thrown error: {error}"),
			ThrownStreamError::Value(value) => panic!("unexpected thrown value: {value}"),
		}

		let failure = stream_failure_from_stop_reason(Some("malformed_model_output"), Some("req-1"));
		let error = ResponsesRunError::Failure(Box::new(failure));
		match error.as_thrown() {
			ThrownStreamError::Failure(failure) => {
				assert_eq!(failure.info.kind, "malformed_response");
				assert_eq!(failure.info.request_id.as_deref(), Some("req-1"));
			}
			ThrownStreamError::Error(error) => panic!("unexpected thrown error: {error}"),
			ThrownStreamError::Value(value) => panic!("unexpected thrown value: {value}"),
			ThrownStreamError::Message(message) => panic!("unexpected thrown message: {message}"),
		}
	}

	#[test]
	fn usage_observation_is_forwarded_from_simple_options() {
		// The shared base builder now preserves monitoring callbacks. The provider's
		// explicit forwarding must retain the same observer, not drop or replace it.
		let simple = SimpleStreamOptions {
			stream: StreamOptions {
				on_usage_observation: Some(Arc::new(
					|_observation: crate::types::ProviderUsageObservation, _model: &Model| {
						Box::pin(async {}) as crate::types::BoxFuture<()>
					},
				)),
				..Default::default()
			},
			reasoning: Some("high".to_string()),
			thinking_budgets: None,
		};
		let mut reasoning_model = model("openai.gpt-6-astra");
		reasoning_model.max_tokens = 1000.0;
		let base = build_base_options(&reasoning_model, Some(&simple), None);
		let expected = simple.stream.on_usage_observation.as_ref().unwrap();
		assert!(Arc::ptr_eq(base.on_usage_observation.as_ref().unwrap(), expected));
		let provider_options = build_simple_stream_options(&reasoning_model, Some(&simple));
		assert!(Arc::ptr_eq(provider_options.stream.on_usage_observation.as_ref().unwrap(), expected));
		assert_eq!(provider_options.reasoning_effort.as_deref(), Some("high"));
		// No observer requested: nothing to forward, and the field stays unset.
		assert!(build_base_options(&reasoning_model, None, None).on_usage_observation.is_none());
		let provider_options = build_simple_stream_options(&reasoning_model, None);
		assert!(provider_options.stream.on_usage_observation.is_none());

		// `clampThinkingLevel(model, options?.reasoning ?? "medium")` on a model without a
		// thinking-level map keeps the requested level.
		reasoning_model.reasoning = true;
		assert_eq!(clamp_thinking_level(&reasoning_model, "high"), "high");
		assert_eq!(clamp_thinking_level(&reasoning_model, "medium"), "medium");

		// A `null` level in the map is unsupported, so the clamp walks to another level.
		let mut mapped = reasoning_model.clone();
		let mut map = crate::types::ThinkingLevelMap::new();
		map.insert("high".to_string(), None);
		map.insert("xhigh".to_string(), None);
		map.insert("max".to_string(), None);
		mapped.thinking_level_map = Some(map);
		assert_eq!(clamp_thinking_level(&mapped, "high"), "medium");
	}
}
