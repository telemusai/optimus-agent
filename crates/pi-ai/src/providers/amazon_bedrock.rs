//! Port of packages/ai/src/providers/amazon-bedrock.ts
//!
//! The TypeScript drives `@aws-sdk/client-bedrock-runtime` (ConverseStream) and lets the SDK do
//! the SigV4 signing and the AWS event-stream decoding. The Rust port reproduces the same wire
//! behaviour with `reqwest`: the ConverseStream JSON request body is built exactly like the SDK
//! serialises it, the request is signed with SigV4 (implemented locally over `sha2`, because the
//! `hmac` crate is not in the workspace dependency list) and the `vnd.amazon.eventstream`
//! response is decoded locally.
//!
//! Ported details that the SDK used to hide (see the `// NOTE:` comments for the gaps):
//! - endpoint/region resolution from `model.baseUrl`, `options.region`, `AWS_REGION`,
//!   `AWS_DEFAULT_REGION` and `AWS_PROFILE`
//! - credentials: the same environment variables and the same precedence as
//!   `@aws-sdk/credential-provider-node`'s `defaultProvider` (env static keys, then the shared
//!   config/credentials profile). No STS/SSO/IMDS network calls are made.
//! - the AWS event-stream frame decoder (`:event-type`, `:message-type` headers)

use base64::Engine as _;
use futures::StreamExt;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use crate::models::{calculate_cost, clamp_thinking_level};
use crate::types::{
	AssistantMessage, AssistantMessageEvent, ContentBlock, Context, ImageOrTextContent, Message, Model,
	SimpleStreamOptions, StreamOptions, TextContent, ThinkingBudgets, ThinkingContent, Tool, ToolCall, Usage,
	UserContent,
};
use crate::utils::event_stream::{create_assistant_message_event_stream, AssistantMessageEventStream};
use crate::utils::json_parse::parse_streaming_json;
use crate::utils::now_ms;
use crate::utils::sanitize_unicode::sanitize_surrogates;
use crate::utils::stream_failure::{
	record_stream_failure, stream_failure_from_stop_reason, StreamFailureError, ThrownStreamError,
};

use super::simple_options::{adjust_max_tokens_for_thinking, build_base_options, clamp_reasoning};
use super::transform_messages::try_transform_messages;

/// TS: `export type BedrockThinkingDisplay = "summarized" | "omitted"`.
pub type BedrockThinkingDisplay = String;

/// TS: `BedrockOptions["toolChoice"]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum BedrockToolChoice {
	/// `"auto" | "any" | "none"`
	Literal(String),
	/// `{ type: "tool"; name: string }`
	Tool {
		#[serde(rename = "type")]
		type_: String,
		name: String,
	},
}

impl Default for BedrockToolChoice {
	fn default() -> Self {
		BedrockToolChoice::Literal("auto".to_string())
	}
}

/// TS: `interface BedrockOptions extends StreamOptions`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct BedrockOptions {
	#[serde(flatten)]
	pub stream: StreamOptions,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub region: Option<String>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub profile: Option<String>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub tool_choice: Option<BedrockToolChoice>,
	/// See https://docs.aws.amazon.com/bedrock/latest/userguide/inference-reasoning.html
	#[serde(skip_serializing_if = "Option::is_none")]
	pub reasoning: Option<String>,
	/// Custom token budgets per thinking level. Overrides default budgets.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub thinking_budgets: Option<ThinkingBudgets>,
	/// Only supported by Claude 4.x models.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub interleaved_thinking: Option<bool>,
	/// Controls how Claude's thinking content is returned in responses.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub thinking_display: Option<BedrockThinkingDisplay>,
	/// Key-value pairs attached to the inference request for cost allocation tagging.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub request_metadata: Option<IndexMap<String, String>>,
	/// Bearer token for Bedrock API key authentication.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub bearer_token: Option<String>,
}

impl BedrockOptions {
	/// TS: the caller passes `StreamOptions & Record<string, unknown>`; this keeps the
	/// non-serializable fields (signal, on_payload, on_response, on_usage_observation).
	pub fn from_base(base: &StreamOptions) -> Self {
		Self {
			stream: base.clone(),
			..Default::default()
		}
	}
}

/// The TypeScript `StreamFunction<"bedrock-converse-stream", BedrockOptions>` accepts either the
/// typed options object or a bare `StreamOptions` (the registered builtin is called with base
/// options). This trait keeps both call shapes compiling.
pub trait IntoBedrockOptions {
	fn into_bedrock_options(self) -> BedrockOptions;
}

impl IntoBedrockOptions for BedrockOptions {
	fn into_bedrock_options(self) -> BedrockOptions {
		self
	}
}

impl IntoBedrockOptions for Option<BedrockOptions> {
	fn into_bedrock_options(self) -> BedrockOptions {
		self.unwrap_or_default()
	}
}

impl IntoBedrockOptions for Option<&StreamOptions> {
	fn into_bedrock_options(self) -> BedrockOptions {
		match self {
			Some(base) => BedrockOptions::from_base(base),
			None => BedrockOptions::default(),
		}
	}
}

/// `streamSimpleBedrock` takes `SimpleStreamOptions`.
pub trait IntoSimpleStreamOptions {
	fn into_simple_stream_options(self) -> Option<SimpleStreamOptions>;
}

impl IntoSimpleStreamOptions for Option<SimpleStreamOptions> {
	fn into_simple_stream_options(self) -> Option<SimpleStreamOptions> {
		self
	}
}

impl IntoSimpleStreamOptions for Option<&SimpleStreamOptions> {
	fn into_simple_stream_options(self) -> Option<SimpleStreamOptions> {
		self.cloned()
	}
}

impl IntoSimpleStreamOptions for Option<&StreamOptions> {
	fn into_simple_stream_options(self) -> Option<SimpleStreamOptions> {
		self.map(|base| SimpleStreamOptions {
			stream: base.clone(),
			reasoning: None,
			thinking_budgets: None,
		})
	}
}

/// The Rust counterpart of a thrown TypeScript value in this provider.
#[derive(Debug, Clone)]
pub enum ProviderError {
	/// An AWS SDK service exception (`BedrockRuntimeServiceException` subclass): carries `name`
	/// and `message`, exactly like the SDK error object.
	Service { name: String, message: String },
	/// A plain `Error`.
	Message(String),
	/// `streamFailureFromStopReason(...)`.
	Failure(StreamFailureError),
}

impl ProviderError {
	pub fn message(message: impl Into<String>) -> Self {
		ProviderError::Message(message.into())
	}

	pub fn service(name: impl Into<String>, message: impl Into<String>) -> Self {
		ProviderError::Service {
			name: name.into(),
			message: message.into(),
		}
	}

	fn message_text(&self) -> String {
		match self {
			ProviderError::Service { message, .. } => message.clone(),
			ProviderError::Message(message) => message.clone(),
			ProviderError::Failure(failure) => failure.message.clone(),
		}
	}
}

impl std::fmt::Display for ProviderError {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "{}", self.message_text())
	}
}

impl std::error::Error for ProviderError {}

impl From<String> for ProviderError {
	fn from(message: String) -> Self {
		ProviderError::Message(message)
	}
}

/// Human-readable prefixes for Bedrock SDK exception names.
/// The downstream retry logic in agent-session matches patterns like
/// `server.?error` and `service.?unavailable`, so we preserve the legacy
/// prefix format rather than using the raw SDK exception name.
pub const BEDROCK_ERROR_PREFIXES: [(&str, &str); 5] = [
	("InternalServerException", "Internal server error"),
	("ModelStreamErrorException", "Model stream error"),
	("ValidationException", "Validation error"),
	("ThrottlingException", "Throttling error"),
	("ServiceUnavailableException", "Service unavailable"),
];

/// Format a Bedrock error with a human-readable prefix.
/// AWS SDK exceptions (both from `client.send()` and from stream event items)
/// extend BedrockRuntimeServiceException. We map the `.name` to a stable
/// human-readable prefix so downstream consumers (retry logic, context-overflow
/// detection) can distinguish error categories via simple string matching.
pub fn format_bedrock_error(error: &ProviderError) -> String {
	match error {
		ProviderError::Service { name, message } => {
			let prefix = BEDROCK_ERROR_PREFIXES
				.iter()
				.find(|(exception, _)| exception == name)
				.map(|(_, prefix)| *prefix)
				.unwrap_or(name.as_str());
			format!("{}: {}", prefix, message)
		}
		ProviderError::Message(message) => message.clone(),
		ProviderError::Failure(failure) => failure.message.clone(),
	}
}

/// Streaming scratch state that the TypeScript keeps on the block objects (`index`,
/// `partialJson`). The Rust port keeps it in a parallel vector so `output.content` never
/// carries the scratch fields.
#[derive(Debug, Clone, Default)]
struct BlockMeta {
	index: Option<usize>,
	partial_json: Option<String>,
}

fn block_position(blocks: &[BlockMeta], content_block_index: usize) -> Option<usize> {
	blocks
		.iter()
		.position(|meta| meta.index == Some(content_block_index))
}

/// TS: `streamBedrock(model, context, options)`.
pub fn stream_bedrock<O: IntoBedrockOptions>(
	model: &Model,
	context: &Context,
	options: O,
) -> AssistantMessageEventStream {
	stream_bedrock_with_options(model, context, Some(options.into_bedrock_options()))
}

/// The typed entry point (`Option<BedrockOptions>`), mirroring the TypeScript signature.
pub fn stream_bedrock_with_options(
	model: &Model,
	context: &Context,
	options: Option<BedrockOptions>,
) -> AssistantMessageEventStream {
	let stream = create_assistant_message_event_stream();

	let model = model.clone();
	let context = context.clone();
	let options = options.unwrap_or_default();
	let producer = stream.clone();
	let out = stream.clone();

	// TS: `(async () => { try { ... } catch (error) { ... } })();`
	producer.spawn(async move {
		let mut output = AssistantMessage {
			role: "assistant".to_string(),
			content: Vec::new(),
			api: "bedrock-converse-stream".to_string(),
			provider: model.provider.clone(),
			model: model.id.clone(),
			usage: Usage::zero(),
			stop_reason: "stop".to_string(),
			timestamp: now_ms(),
			..Default::default()
		};

		let mut blocks: Vec<BlockMeta> = Vec::new();

		let result = run_bedrock_stream(&model, &context, &options, &mut output, &mut blocks, &out).await;

		match result {
			Ok(()) => {
				out.push(AssistantMessageEvent::Done {
					reason: output.stop_reason.clone(),
					message: output,
				});
				out.end(None);
			}
			Err(error) => {
				// TS: deletes `index` / `partialJson` from every block before persisting.
				// The port keeps those fields out of `output.content` entirely.
				blocks.clear();

				let aborted = options
					.stream
					.signal
					.as_ref()
					.map(|signal| signal.is_cancelled())
					.unwrap_or(false);
				output.stop_reason = if aborted { "aborted" } else { "error" }.to_string();
				output.error_message = Some(format_bedrock_error(&error));
				{
					let thrown_value: Value;
					let thrown = match &error {
						ProviderError::Failure(failure) => ThrownStreamError::Failure(failure),
						ProviderError::Service { name, message } => {
							thrown_value = json!({ "name": name, "message": message });
							ThrownStreamError::Value(&thrown_value)
						}
						ProviderError::Message(message) => ThrownStreamError::Message(message),
					};
					record_stream_failure(&model, &mut output, &thrown);
				}
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

/// TS: `streamSimpleBedrock(model, context, options)`.
pub fn stream_simple_bedrock<O: IntoSimpleStreamOptions>(
	model: &Model,
	context: &Context,
	options: O,
) -> AssistantMessageEventStream {
	let options = options.into_simple_stream_options();
	let base = build_base_options(model, options.as_ref(), None);

	let Some(options) = options else {
		return stream_bedrock_with_options(
			model,
			context,
			Some(BedrockOptions {
				stream: base,
				..Default::default()
			}),
		);
	};

	if options.reasoning.is_none() || options.reasoning.as_deref() == Some("off") {
		return stream_bedrock_with_options(
			model,
			context,
			Some(BedrockOptions {
				stream: base,
				reasoning: None,
				..Default::default()
			}),
		);
	}

	let reasoning = options.reasoning.clone().unwrap_or_default();

	if is_anthropic_claude_model(model) {
		if supports_adaptive_thinking(&model.id, Some(&model.name)) {
			return stream_bedrock_with_options(
				model,
				context,
				Some(BedrockOptions {
					stream: base,
					reasoning: Some(reasoning),
					thinking_budgets: options.thinking_budgets.clone(),
					..Default::default()
				}),
			);
		}

		let adjusted = adjust_max_tokens_for_thinking(
			base.max_tokens.unwrap_or(0.0),
			model.max_tokens,
			&reasoning,
			options.thinking_budgets.as_ref(),
		);

		let mut budgets = options.thinking_budgets.clone().unwrap_or_default();
		if let Some(level) = clamp_reasoning(Some(&reasoning)) {
			set_thinking_budget(&mut budgets, &level, adjusted.thinking_budget);
		}

		return stream_bedrock_with_options(
			model,
			context,
			Some(BedrockOptions {
				stream: base,
				reasoning: Some(reasoning),
				thinking_budgets: Some(budgets),
				..Default::default()
			}),
		);
	}

	stream_bedrock_with_options(
		model,
		context,
		Some(BedrockOptions {
			stream: base,
			reasoning: Some(reasoning),
			thinking_budgets: options.thinking_budgets.clone(),
			..Default::default()
		}),
	)
}

/// `{ ...(options.thinkingBudgets || {}), [level]: budget }`.
fn set_thinking_budget(budgets: &mut ThinkingBudgets, level: &str, budget: f64) {
	match level {
		"minimal" => budgets.minimal = Some(budget),
		"low" => budgets.low = Some(budget),
		"medium" => budgets.medium = Some(budget),
		"high" => budgets.high = Some(budget),
		_ => {}
	}
}

// ---------------------------------------------------------------------------
// Endpoint / region / credentials resolution (was the AWS SDK config chain)
// ---------------------------------------------------------------------------

/// TS: `getConfiguredBedrockRegion(options)`.
pub fn get_configured_bedrock_region(options: &BedrockOptions) -> Option<String> {
	if let Some(region) = options.region.clone() {
		return Some(region);
	}
	std::env::var("AWS_REGION")
		.ok()
		.filter(|value| !value.is_empty())
		.or_else(|| std::env::var("AWS_DEFAULT_REGION").ok().filter(|value| !value.is_empty()))
}

/// TS: `hasConfiguredBedrockProfile()`.
pub fn has_configured_bedrock_profile() -> bool {
	std::env::var("AWS_PROFILE")
		.map(|value| !value.is_empty())
		.unwrap_or(false)
}

/// TS: `getStandardBedrockEndpointRegion(baseUrl)`.
pub fn get_standard_bedrock_endpoint_region(base_url: Option<&str>) -> Option<String> {
	let base_url = base_url?;
	let url = url::Url::parse(base_url).ok()?;
	let hostname = url.host_str()?.to_lowercase();
	let rest = hostname.strip_prefix("bedrock-runtime")?;
	// `(?:-fips)?` then `.region.amazonaws.com(.cn)?`
	let rest = rest.strip_prefix("-fips").unwrap_or(rest);
	let rest = rest.strip_prefix('.')?;
	let region = rest.split('.').next()?;
	if !region.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-') || region.is_empty() {
		return None;
	}
	let suffix = &rest[region.len()..];
	if suffix != ".amazonaws.com" && suffix != ".amazonaws.com.cn" {
		return None;
	}
	Some(region.to_string())
}

/// TS: `shouldUseExplicitBedrockEndpoint(baseUrl, configuredRegion, hasConfiguredProfile)`.
pub fn should_use_explicit_bedrock_endpoint(
	base_url: &str,
	configured_region: Option<&str>,
	has_configured_profile: bool,
) -> bool {
	let Some(_endpoint_region) = get_standard_bedrock_endpoint_region(Some(base_url)) else {
		return true;
	};
	configured_region.is_none() && !has_configured_profile
}

/// TS: `isGovCloudBedrockTarget(model, options)`.
pub fn is_gov_cloud_bedrock_target(model: &Model, options: &BedrockOptions) -> bool {
	if let Some(region) = get_configured_bedrock_region(options) {
		if region.to_lowercase().starts_with("us-gov-") {
			return true;
		}
	}

	let model_id = model.id.to_lowercase();
	model_id.starts_with("us-gov.") || model_id.starts_with("arn:aws-us-gov:")
}

/// The resolved `BedrockRuntimeClientConfig` the TypeScript hands to the SDK.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BedrockClientConfig {
	pub region: Option<String>,
	pub endpoint: Option<String>,
	pub profile: Option<String>,
	pub max_attempts: i64,
	/// `config.token` / `config.authSchemePreference` (bearer auth).
	pub bearer_token: Option<String>,
	/// `config.credentials` (dummy keys under `AWS_BEDROCK_SKIP_AUTH=1`, otherwise resolved).
	pub credentials: Option<AwsCredentials>,
	/// `config.requestHandler` was installed because a proxy env var is set.
	pub use_proxy_env: bool,
	/// `config.requestHandler = new NodeHttpHandler()` under `AWS_BEDROCK_FORCE_HTTP1=1`
	/// (`amazon-bedrock.ts:176`). Consumed by `build_http_client`, which then builds the reqwest
	/// client with `http1_only()`.
	pub force_http1: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AwsCredentials {
	pub access_key_id: String,
	pub secret_access_key: String,
	pub session_token: Option<String>,
}

/// TS: the `config` object plus `useBearerToken`, in the same order.
pub fn resolve_bedrock_client_config(model: &Model, options: &BedrockOptions) -> BedrockClientConfig {
	let mut config = BedrockClientConfig {
		profile: options.profile.clone(),
		max_attempts: 1,
		..Default::default()
	};
	let configured_region = get_configured_bedrock_region(options);
	let has_configured_profile = has_configured_bedrock_profile();
	let endpoint_region = get_standard_bedrock_endpoint_region(Some(&model.base_url));
	let use_explicit_endpoint = should_use_explicit_bedrock_endpoint(
		&model.base_url,
		configured_region.as_deref(),
		has_configured_profile,
	);

	// Preserve custom endpoints and AWS region/profile configuration.
	if use_explicit_endpoint {
		config.endpoint = Some(model.base_url.clone());
	}

	let bearer_token = options
		.bearer_token
		.clone()
		.or_else(|| std::env::var("AWS_BEARER_TOKEN_BEDROCK").ok().filter(|v| !v.is_empty()));
	let use_bearer_token = bearer_token.is_some() && std::env::var("AWS_BEDROCK_SKIP_AUTH").ok().as_deref() != Some("1");

	// Region resolution: explicit option > env vars > SDK default chain.
	// When AWS_PROFILE is set, we leave region undefined so the SDK can
	// resolve it from aws profile configs. Otherwise fall back to us-east-1.
	config.region = configured_region.clone();
	if config.region.is_none() && endpoint_region.is_some() && use_explicit_endpoint {
		config.region = endpoint_region.clone();
	}
	if config.region.is_none() && !has_configured_profile {
		config.region = Some("us-east-1".to_string());
	}
	// NOTE: when AWS_PROFILE is set the SDK resolves the region from the aws profile config
	// files; the port reads the same files (`AWS_CONFIG_FILE` / `~/.aws/config`).
	if config.region.is_none() {
		config.region = region_from_profile_files(options.profile.as_deref());
	}

	if std::env::var("AWS_BEDROCK_SKIP_AUTH").ok().as_deref() == Some("1") {
		config.credentials = Some(AwsCredentials {
			access_key_id: "dummy-access-key".to_string(),
			secret_access_key: "dummy-secret-key".to_string(),
			session_token: None,
		});
	}

	let proxy_env = [
		"HTTP_PROXY",
		"HTTPS_PROXY",
		"NO_PROXY",
		"http_proxy",
		"https_proxy",
		"no_proxy",
	]
	.iter()
	.any(|name| std::env::var(name).map(|value| !value.is_empty()).unwrap_or(false));
	if proxy_env {
		config.use_proxy_env = true;
	} else if std::env::var("AWS_BEDROCK_FORCE_HTTP1").ok().as_deref() == Some("1") {
		// TS `amazon-bedrock.ts:176-178`: the HTTP/1.1 handler is only installed in the `else if`
		// branch, i.e. only when no proxy env var selected the proxy handler first.
		config.force_http1 = true;
	}

	if use_bearer_token {
		config.bearer_token = bearer_token;
	}

	config
}

/// `~/.aws/config` / `~/.aws/credentials` static profile lookup.
///
/// NOTE: the SDK default chain also covers SSO, `credential_process`, web identity and IMDS;
/// those need network/process calls and are intentionally not implemented. Static profile keys
/// and the region from the profile config are.
fn region_from_profile_files(profile: Option<&str>) -> Option<String> {
	let profile_name = profile
		.map(str::to_string)
		.or_else(|| std::env::var("AWS_PROFILE").ok())
		.unwrap_or_else(|| "default".to_string());
	let path = std::env::var("AWS_CONFIG_FILE")
		.ok()
		.filter(|value| !value.is_empty())
		.map(std::path::PathBuf::from)
		.or_else(|| home_dir().map(|home| home.join(".aws").join("config")))?;
	let contents = std::fs::read_to_string(path).ok()?;
	let section = if profile_name == "default" {
		"default".to_string()
	} else {
		format!("profile {}", profile_name)
	};
	read_ini_value(&contents, &section, "region")
}

fn home_dir() -> Option<std::path::PathBuf> {
	std::env::var("HOME")
		.ok()
		.filter(|value| !value.is_empty())
		.or_else(|| std::env::var("USERPROFILE").ok().filter(|value| !value.is_empty()))
		.map(std::path::PathBuf::from)
}

fn read_ini_value(contents: &str, section: &str, key: &str) -> Option<String> {
	let mut current: Option<String> = None;
	for raw_line in contents.lines() {
		let line = raw_line.trim();
		if line.is_empty() || line.starts_with('#') || line.starts_with(';') {
			continue;
		}
		if line.starts_with('[') && line.ends_with(']') {
			current = Some(line[1..line.len() - 1].trim().to_string());
			continue;
		}
		if current.as_deref() != Some(section) {
			continue;
		}
		if let Some((name, value)) = line.split_once('=') {
			if name.trim() == key {
				return Some(value.trim().to_string());
			}
		}
	}
	None
}

/// `@aws-sdk/credential-provider-node` `defaultProvider` order, restricted to the file/env
/// sources the port implements (env static keys, then the shared config/credentials profile).
pub fn resolve_aws_credentials(options: &BedrockOptions) -> Result<AwsCredentials, ProviderError> {
	let profile = options
		.profile
		.clone()
		.or_else(|| std::env::var("AWS_PROFILE").ok().filter(|value| !value.is_empty()));

	// TS: `if (profile) throw CredentialsProviderError("AWS_PROFILE is set, skipping fromEnv provider.")`
	if profile.is_none() {
		let access_key_id = std::env::var("AWS_ACCESS_KEY_ID").ok().filter(|v| !v.is_empty());
		let secret_access_key = std::env::var("AWS_SECRET_ACCESS_KEY").ok().filter(|v| !v.is_empty());
		if let (Some(access_key_id), Some(secret_access_key)) = (access_key_id, secret_access_key) {
			return Ok(AwsCredentials {
				access_key_id,
				secret_access_key,
				session_token: std::env::var("AWS_SESSION_TOKEN").ok().filter(|v| !v.is_empty()),
			});
		}
	}

	let profile_name = profile.unwrap_or_else(|| "default".to_string());
	let section = if profile_name == "default" {
		"default".to_string()
	} else {
		profile_name.clone()
	};

	let mut candidates: Vec<std::path::PathBuf> = Vec::new();
	if let Ok(path) = std::env::var("AWS_SHARED_CREDENTIALS_FILE") {
		if !path.is_empty() {
			candidates.push(std::path::PathBuf::from(path));
		}
	}
	if let Some(home) = home_dir() {
		candidates.push(home.join(".aws").join("credentials"));
	}
	if let Ok(path) = std::env::var("AWS_CONFIG_FILE") {
		if !path.is_empty() {
			candidates.push(std::path::PathBuf::from(path));
		}
	}
	if let Some(home) = home_dir() {
		candidates.push(home.join(".aws").join("config"));
	}

	for path in candidates {
		let Ok(contents) = std::fs::read_to_string(&path) else {
			continue;
		};
		let config_section = format!("profile {}", section);
		let section_name = if path.ends_with("config") {
			if section == "default" {
				"default".to_string()
			} else {
				config_section
			}
		} else {
			section.clone()
		};
		let access_key_id = read_ini_value(&contents, &section_name, "aws_access_key_id");
		let secret_access_key = read_ini_value(&contents, &section_name, "aws_secret_access_key");
		if let (Some(access_key_id), Some(secret_access_key)) = (access_key_id, secret_access_key) {
			return Ok(AwsCredentials {
				access_key_id,
				secret_access_key,
				session_token: read_ini_value(&contents, &section_name, "aws_session_token"),
			});
		}
	}

	Err(ProviderError::message(
		"Could not load credentials from any providers",
	))
}

// ---------------------------------------------------------------------------
// Request conversion (TS: buildSystemPrompt / convertMessages / convertToolConfig)
// ---------------------------------------------------------------------------

/// TS: `resolveCacheRetention(cacheRetention)`.
pub fn resolve_cache_retention(cache_retention: Option<&str>) -> String {
	if let Some(cache_retention) = cache_retention {
		return cache_retention.to_string();
	}
	if std::env::var("PI_CACHE_RETENTION").ok().as_deref() == Some("long") {
		return "long".to_string();
	}
	"short".to_string()
}

/// TS: `getModelMatchCandidates(modelId, modelName?)`.
pub fn get_model_match_candidates(model_id: &str, model_name: Option<&str>) -> Vec<String> {
	let values: Vec<&str> = match model_name {
		Some(name) => vec![model_id, name],
		None => vec![model_id],
	};
	values
		.into_iter()
		.flat_map(|value| {
			let lower = value.to_lowercase();
			let collapsed = collapse_model_separators(&lower);
			[lower, collapsed]
		})
		.collect()
}

/// `lower.replace(/[\s_.:]+/g, "-")`.
fn collapse_model_separators(value: &str) -> String {
	let mut result = String::with_capacity(value.len());
	let mut in_run = false;
	for ch in value.chars() {
		if ch.is_whitespace() || ch == '_' || ch == '.' || ch == ':' {
			if !in_run {
				result.push('-');
				in_run = true;
			}
			continue;
		}
		in_run = false;
		result.push(ch);
	}
	result
}

/// Check if the model supports adaptive thinking (Opus 4.6+, Sonnet 4.6).
/// Checks both model ID and model name to support application inference profiles
/// whose ARNs don't contain the model name.
pub fn supports_adaptive_thinking(model_id: &str, model_name: Option<&str>) -> bool {
	let candidates = get_model_match_candidates(model_id, model_name);
	candidates.iter().any(|s| {
		s.contains("opus-4-6")
			|| s.contains("opus-4-7")
			|| s.contains("opus-4-8")
			|| s.contains("opus-5")
			|| s.contains("sonnet-4-6")
			|| s.contains("sonnet-5")
			|| s.contains("fable-5")
			|| s.contains("mythos-5")
			|| s.contains("mythos-preview")
	})
}

/// Fable/Mythos models think every turn and reject sampling params with a 400.
pub fn supports_always_on_adaptive_thinking(model_id: &str, model_name: Option<&str>) -> bool {
	let candidates = get_model_match_candidates(model_id, model_name);
	candidates
		.iter()
		.any(|s| s.contains("fable-5") || s.contains("mythos-5") || s.contains("mythos-preview"))
}

/// TS: `mapThinkingLevelToEffort(model, level)`.
pub fn map_thinking_level_to_effort(model: &Model, level: Option<&str>) -> String {
	// Clamp to what the model actually supports so callers that bypass
	// clampThinkingLevel (e.g. passing reasoning: "xhigh" directly) can't send an
	// effort the model lacks - xhigh on a max-only model resolves to max, not xhigh.
	let effective = level.map(|level| clamp_thinking_level(model, level));
	let mapped = effective
		.as_ref()
		.and_then(|effective| model.thinking_level_map_get(effective));
	if let Some(Some(mapped)) = mapped {
		return mapped;
	}

	match effective.as_deref() {
		Some("minimal") | Some("low") => "low".to_string(),
		Some("medium") => "medium".to_string(),
		Some("high") => "high".to_string(),
		Some("xhigh") => "xhigh".to_string(),
		Some("max") => "max".to_string(),
		_ => "high".to_string(),
	}
}

/// Check if the model is an Anthropic Claude model on Bedrock.
/// Checks both model ID and model name to support application inference profiles
/// whose ARNs don't contain the model name.
pub fn is_anthropic_claude_model(model: &Model) -> bool {
	let id = model.id.to_lowercase();
	let name = model.name.to_lowercase();
	id.contains("anthropic.claude")
		|| id.contains("anthropic/claude")
		|| name.contains("anthropic.claude")
		|| name.contains("anthropic/claude")
		|| name.contains("claude")
}

/// Check if the model supports prompt caching.
/// Supported: Claude 3.5 Haiku, Claude 3.7 Sonnet, Claude 4.x models
pub fn supports_prompt_caching(model: &Model) -> bool {
	let candidates = get_model_match_candidates(&model.id, Some(&model.name));

	let has_claude_ref = candidates.iter().any(|s| s.contains("claude"));
	if !has_claude_ref {
		// Application inference profiles don't contain the model name in the ARN.
		// Allow users to force cache points via environment variable.
		if std::env::var("AWS_BEDROCK_FORCE_CACHE").ok().as_deref() == Some("1") {
			return true;
		}
		return false;
	}
	if candidates.iter().any(|s| s.contains("-4-")) {
		return true;
	}
	if candidates.iter().any(|s| s.contains("claude-3-7-sonnet")) {
		return true;
	}
	if candidates.iter().any(|s| s.contains("claude-3-5-haiku")) {
		return true;
	}
	false
}

/// Check if the model supports thinking signatures in reasoningContent.
/// Only Anthropic Claude models support the signature field.
pub fn supports_thinking_signature(model: &Model) -> bool {
	is_anthropic_claude_model(model)
}

/// `CachePointType.DEFAULT`.
const CACHE_POINT_TYPE_DEFAULT: &str = "default";
/// `CacheTTL.ONE_HOUR`.
const CACHE_TTL_ONE_HOUR: &str = "1h";

fn cache_point(cache_retention: &str) -> Value {
	let mut point = Map::new();
	point.insert("type".to_string(), Value::String(CACHE_POINT_TYPE_DEFAULT.to_string()));
	if cache_retention == "long" {
		point.insert("ttl".to_string(), Value::String(CACHE_TTL_ONE_HOUR.to_string()));
	}
	json!({ "cachePoint": Value::Object(point) })
}

/// TS: `buildSystemPrompt(systemPrompt, model, cacheRetention)`.
pub fn build_system_prompt(
	system_prompt: Option<&str>,
	model: &Model,
	cache_retention: &str,
) -> Option<Vec<Value>> {
	let system_prompt = system_prompt?;
	if system_prompt.is_empty() {
		return None;
	}

	let mut blocks: Vec<Value> = vec![json!({ "text": sanitize_surrogates(system_prompt) })];

	if cache_retention != "none" && supports_prompt_caching(model) {
		blocks.push(cache_point(cache_retention));
	}

	Some(blocks)
}

/// TS: `normalizeToolCallId(id)`.
pub fn normalize_tool_call_id(id: &str) -> String {
	let sanitized: String = id
		.chars()
		.map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
		.collect();
	if sanitized.chars().count() > 64 {
		sanitized.chars().take(64).collect()
	} else {
		sanitized
	}
}

/// TS: `createImageBlock(mimeType, data)`.
pub fn create_image_block(mime_type: &str, data: &str) -> Result<Value, ProviderError> {
	let format = match mime_type {
		"image/jpeg" | "image/jpg" => "jpeg",
		"image/png" => "png",
		"image/gif" => "gif",
		"image/webp" => "webp",
		other => return Err(ProviderError::message(format!("Unknown image type: {}", other))),
	};

	// TS: `atob(data)` then the SDK base64-encodes the bytes again on the wire.
	let bytes = base64::engine::general_purpose::STANDARD
		.decode(data)
		.map_err(|_| ProviderError::message("The string to be decoded is not correctly encoded."))?;
	let encoded = base64::engine::general_purpose::STANDARD.encode(bytes);

	Ok(json!({ "source": { "bytes": encoded }, "format": format }))
}

/// TS: `convertMessages(context, model, cacheRetention)`.
///
/// NOTE: the TypeScript `default: throw new Error("Unknown user content type")`,
/// `"Unknown assistant content type"` and `"Unknown message role"` branches are unreachable in
/// Rust because the message/content unions are closed enums.
pub fn convert_messages(
	context: &Context,
	model: &Model,
	cache_retention: &str,
) -> Result<Vec<Value>, ProviderError> {
	let mut result: Vec<Value> = Vec::new();
	let normalize = |id: &str, _model: &Model, _source: &AssistantMessage| normalize_tool_call_id(id);
	let transformed_messages = try_transform_messages(context.messages.clone(), model, Some(&normalize))
		.map_err(ProviderError::message)?;

	let mut i = 0usize;
	while i < transformed_messages.len() {
		let m = &transformed_messages[i];

		match m {
			Message::User(user) => {
				let content: Vec<Value> = match &user.content {
					UserContent::Text(text) => vec![json!({ "text": sanitize_surrogates(text) })],
					UserContent::Blocks(blocks) => {
						let mut converted = Vec::with_capacity(blocks.len());
						for c in blocks {
							match c {
								ImageOrTextContent::Text(text) => {
									converted.push(json!({ "text": sanitize_surrogates(&text.text) }))
								}
								ImageOrTextContent::Image(image) => converted
									.push(json!({ "image": create_image_block(&image.mime_type, &image.data)? })),
							}
						}
						converted
					}
				};
				result.push(json!({ "role": "user", "content": content }));
			}
			Message::Assistant(assistant) => {
				// Skip assistant messages with empty content (e.g., from aborted requests)
				// Bedrock rejects messages with empty content arrays
				if assistant.content.is_empty() {
					i += 1;
					continue;
				}
				let mut content_blocks: Vec<Value> = Vec::new();
				for c in &assistant.content {
					match c {
						ContentBlock::Text(text) => {
							if text.text.trim().is_empty() {
								continue;
							}
							content_blocks.push(json!({ "text": sanitize_surrogates(&text.text) }));
						}
						ContentBlock::ToolCall(tool_call) => {
							content_blocks.push(json!({
								"toolUse": {
									"toolUseId": tool_call.id,
									"name": tool_call.name,
									"input": Value::Object(tool_call.arguments.clone()),
								}
							}));
						}
						ContentBlock::Thinking(thinking) => {
							if thinking.thinking.trim().is_empty() {
								continue;
							}
							// Only Anthropic models support the signature field in reasoningText.
							// For other models, we omit the signature to avoid errors like:
							// "This model doesn't support the reasoningContent.reasoningText.signature field"
							if supports_thinking_signature(model) {
								// Signatures arrive after thinking deltas. If a partial or externally
								// persisted message lacks a signature, Bedrock rejects the replayed
								// reasoning block. Fall back to plain text, matching Anthropic.
								let signature = thinking.thinking_signature.clone().unwrap_or_default();
								if signature.trim().is_empty() {
									content_blocks.push(json!({ "text": sanitize_surrogates(&thinking.thinking) }));
								} else {
									content_blocks.push(json!({
										"reasoningContent": {
											"reasoningText": {
												"text": sanitize_surrogates(&thinking.thinking),
												"signature": signature,
											}
										}
									}));
								}
							} else {
								content_blocks.push(json!({
									"reasoningContent": {
										"reasoningText": { "text": sanitize_surrogates(&thinking.thinking) }
									}
								}));
							}
						}
					}
				}
				if content_blocks.is_empty() {
					i += 1;
					continue;
				}
				result.push(json!({ "role": "assistant", "content": content_blocks }));
			}
			Message::ToolResult(_) => {
				// Collect all consecutive toolResult messages into a single user message
				// Bedrock requires all tool results to be in one message
				let mut tool_results: Vec<Value> = Vec::new();
				let mut j = i;
				while j < transformed_messages.len() {
					let Message::ToolResult(next_msg) = &transformed_messages[j] else {
						break;
					};
					let mut content: Vec<Value> = Vec::with_capacity(next_msg.content.len());
					for c in &next_msg.content {
						match c {
							ImageOrTextContent::Image(image) => content
								.push(json!({ "image": create_image_block(&image.mime_type, &image.data)? })),
							ImageOrTextContent::Text(text) => {
								content.push(json!({ "text": sanitize_surrogates(&text.text) }))
							}
						}
					}
					tool_results.push(json!({
						"toolResult": {
							"toolUseId": next_msg.tool_call_id,
							"content": content,
							"status": if next_msg.is_error { "error" } else { "success" },
						}
					}));
					j += 1;
				}

				i = j;

				result.push(json!({ "role": "user", "content": tool_results }));
				continue;
			}
		}

		i += 1;
	}

	// Add cache point to the last user message for supported Claude models when caching is enabled
	if cache_retention != "none" && supports_prompt_caching(model) && !result.is_empty() {
		let last_index = result.len() - 1;
		let is_user = result[last_index]
			.get("role")
			.and_then(Value::as_str)
			.map(|role| role == "user")
			.unwrap_or(false);
		let has_content = result[last_index].get("content").is_some();
		if is_user && has_content {
			if let Some(Value::Array(content)) = result[last_index].get_mut("content") {
				content.push(cache_point(cache_retention));
			}
		}
	}

	Ok(result)
}

/// TS: `convertToolConfig(tools, toolChoice)`.
pub fn convert_tool_config(tools: Option<&Vec<Tool>>, tool_choice: Option<&BedrockToolChoice>) -> Option<Value> {
	let tools = tools?;
	if tools.is_empty() {
		return None;
	}
	let literal = match tool_choice {
		Some(BedrockToolChoice::Literal(literal)) => Some(literal.as_str()),
		_ => None,
	};
	if literal == Some("none") {
		return None;
	}

	let bedrock_tools: Vec<Value> = tools
		.iter()
		.map(|tool| {
			json!({
				"toolSpec": {
					"name": tool.name,
					"description": tool.description,
					"inputSchema": { "json": tool.parameters },
				}
			})
		})
		.collect();

	let mut bedrock_tool_choice: Option<Value> = None;
	match literal {
		Some("auto") => bedrock_tool_choice = Some(json!({ "auto": {} })),
		Some("any") => bedrock_tool_choice = Some(json!({ "any": {} })),
		_ => {
			if let Some(BedrockToolChoice::Tool { type_, name }) = tool_choice {
				if type_ == "tool" {
					bedrock_tool_choice = Some(json!({ "tool": { "name": name } }));
				}
			}
		}
	}

	Some(json!({ "tools": bedrock_tools, "toolChoice": bedrock_tool_choice }))
}

/// TS: `mapStopReason(reason)`.
pub fn map_stop_reason(reason: Option<&str>) -> String {
	match reason {
		Some("end_turn") | Some("stop_sequence") => "stop".to_string(),
		Some("max_tokens") | Some("model_context_window_exceeded") => "length".to_string(),
		Some("tool_use") => "toolUse".to_string(),
		_ => "error".to_string(),
	}
}

/// TS: `buildAdditionalModelRequestFields(model, options)`.
pub fn build_additional_model_request_fields(model: &Model, options: &BedrockOptions) -> Option<Value> {
	let reasoning = options.reasoning.clone()?;
	if !model.reasoning {
		return None;
	}

	if is_anthropic_claude_model(model) {
		// GovCloud Bedrock currently rejects the Claude thinking.display field.
		// Omit it there until the GovCloud Converse schema catches up.
		let display = if is_gov_cloud_bedrock_target(model, options) {
			None
		} else {
			Some(
				options
					.thinking_display
					.clone()
					.unwrap_or_else(|| "summarized".to_string()),
			)
		};

		let adaptive = supports_adaptive_thinking(&model.id, Some(&model.name));
		let mut result: Map<String, Value> = Map::new();
		if adaptive {
			let mut thinking = Map::new();
			thinking.insert("type".to_string(), Value::String("adaptive".to_string()));
			if let Some(display) = display.clone() {
				thinking.insert("display".to_string(), Value::String(display));
			}
			result.insert("thinking".to_string(), Value::Object(thinking));
			result.insert(
				"output_config".to_string(),
				json!({ "effort": map_thinking_level_to_effort(model, Some(&reasoning)) }),
			);
		} else {
			let default_budgets: [(&str, f64); 6] = [
				("minimal", 1024.0),
				("low", 2048.0),
				("medium", 8192.0),
				("high", 16384.0),
				// Budget-based Claude has no xhigh tier, clamp to high
				("xhigh", 16384.0),
				// Budget-based Claude has no max tier, clamp to high
				("max", 16384.0),
			];

			let level = if reasoning == "xhigh" || reasoning == "max" {
				"high".to_string()
			} else {
				reasoning.clone()
			};
			let budget = options
				.thinking_budgets
				.as_ref()
				.and_then(|budgets| thinking_budget_for(budgets, &level))
				.or_else(|| {
					default_budgets
						.iter()
						.find(|(name, _)| *name == reasoning)
						.map(|(_, budget)| *budget)
				})
				.unwrap_or(0.0);

			let mut thinking = Map::new();
			thinking.insert("type".to_string(), Value::String("enabled".to_string()));
			thinking.insert("budget_tokens".to_string(), json!(budget));
			if let Some(display) = display {
				thinking.insert("display".to_string(), Value::String(display));
			}
			result.insert("thinking".to_string(), Value::Object(thinking));
		}

		if !adaptive && options.interleaved_thinking.unwrap_or(true) {
			result.insert(
				"anthropic_beta".to_string(),
				json!(["interleaved-thinking-2025-05-14"]),
			);
		}

		return Some(Value::Object(result));
	}

	None
}

/// `options.thinkingBudgets?.[level]` for the budget levels the type carries.
fn thinking_budget_for(budgets: &ThinkingBudgets, level: &str) -> Option<f64> {
	match level {
		"minimal" => budgets.minimal,
		"low" => budgets.low,
		"medium" => budgets.medium,
		"high" => budgets.high,
		_ => None,
	}
}

// ---------------------------------------------------------------------------
// SigV4 (the SDK signed the request; the port signs it here)
// ---------------------------------------------------------------------------

/// `SignatureV4` is constructed with `service` = "bedrock" for the ConverseStream API.
const BEDROCK_SIGNING_SERVICE: &str = "bedrock";

/// `x-amz-content-sha256` (the SDK sets it because `applyChecksum` defaults to true).
const SHA256_HEADER: &str = "x-amz-content-sha256";
const AMZ_DATE_HEADER: &str = "x-amz-date";
const TOKEN_HEADER: &str = "x-amz-security-token";
const HOST_HEADER: &str = "host";
const ALGORITHM_IDENTIFIER: &str = "AWS4-HMAC-SHA256";
const KEY_TYPE_IDENTIFIER: &str = "aws4_request";
const EMPTY_PAYLOAD_SHA256: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

/// `ALWAYS_UNSIGNABLE_HEADERS` from `@smithy/signature-v4`.
const ALWAYS_UNSIGNABLE_HEADERS: [&str; 15] = [
	"authorization",
	"cache-control",
	"connection",
	"expect",
	"from",
	"keep-alive",
	"max-forwards",
	"pragma",
	"referer",
	"te",
	"trailer",
	"transfer-encoding",
	"upgrade",
	"user-agent",
	"x-amzn-trace-id",
];

/// NOTE: the SDK also treats `^proxy-` and `^sec-` headers as unsignable; the port keeps the
/// literal list plus those two prefixes.
fn is_unsignable_header(name: &str) -> bool {
	ALWAYS_UNSIGNABLE_HEADERS.contains(&name) || name.starts_with("proxy-") || name.starts_with("sec-")
}

pub fn hex_encode(bytes: &[u8]) -> String {
	const HEX: &[u8; 16] = b"0123456789abcdef";
	let mut out = String::with_capacity(bytes.len() * 2);
	for byte in bytes {
		out.push(HEX[(byte >> 4) as usize] as char);
		out.push(HEX[(byte & 0x0f) as usize] as char);
	}
	out
}

pub fn sha256_hex(data: &[u8]) -> String {
	use sha2::{Digest, Sha256};
	let mut hasher = Sha256::new();
	hasher.update(data);
	hex_encode(&hasher.finalize())
}

/// HMAC-SHA256 (RFC 2104) implemented locally over `sha2` because the `hmac` crate is not in
/// the workspace dependency list.
pub fn hmac_sha256(key: &[u8], data: &[u8]) -> Vec<u8> {
	use sha2::{Digest, Sha256};
	const BLOCK_SIZE: usize = 64;

	let mut normalized = key.to_vec();
	if normalized.len() > BLOCK_SIZE {
		let mut hasher = Sha256::new();
		hasher.update(&normalized);
		normalized = hasher.finalize().to_vec();
	}
	normalized.resize(BLOCK_SIZE, 0);

	let mut inner_pad = Vec::with_capacity(BLOCK_SIZE);
	let mut outer_pad = Vec::with_capacity(BLOCK_SIZE);
	for byte in &normalized {
		inner_pad.push(byte ^ 0x36);
		outer_pad.push(byte ^ 0x5c);
	}

	let mut inner = Sha256::new();
	inner.update(&inner_pad);
	inner.update(data);
	let inner_digest = inner.finalize();

	let mut outer = Sha256::new();
	outer.update(&outer_pad);
	outer.update(inner_digest);
	outer.finalize().to_vec()
}

/// `escapeUri` from `@smithy/core`: `encodeURIComponent` plus `!'()*`.
fn escape_uri(uri: &str) -> String {
	let mut out = String::with_capacity(uri.len());
	for byte in uri.bytes() {
		let unreserved = byte.is_ascii_alphanumeric()
			|| byte == b'-'
			|| byte == b'_'
			|| byte == b'.'
			|| byte == b'~';
		if unreserved {
			out.push(byte as char);
		} else {
			out.push_str(&format!("%{:02X}", byte));
		}
	}
	out
}

/// `getCanonicalPath`: normalises `..`/`.` segments then URI-escapes each segment.
fn canonical_path(path: &str) -> String {
	let mut segments: Vec<&str> = Vec::new();
	for segment in path.split('/') {
		if segment.is_empty() || segment == "." {
			continue;
		}
		if segment == ".." {
			segments.pop();
		} else {
			segments.push(segment);
		}
	}
	let normalized = format!(
		"{}{}{}",
		if path.starts_with('/') { "/" } else { "" },
		segments.join("/"),
		if !segments.is_empty() && path.ends_with('/') { "/" } else { "" }
	);
	escape_uri(&normalized).replace("%2F", "/")
}

/// `getCanonicalQuery`.
fn canonical_query(query: &[(String, String)]) -> String {
	let mut entries: Vec<(String, String)> = Vec::new();
	for (key, value) in query {
		let encoded_key = escape_uri(key);
		entries.push((encoded_key.clone(), format!("{}={}", encoded_key, escape_uri(value))));
	}
	entries.sort();
	entries
		.into_iter()
		.map(|(_, serialized)| serialized)
		.collect::<Vec<_>>()
		.join("&")
}

/// `iso8601(now).replace(/[-:]/g, "")` -> `{ longDate, shortDate }`.
fn format_signing_date(timestamp_ms: i64) -> (String, String) {
	let seconds = timestamp_ms.div_euclid(1000);
	let datetime = chrono::DateTime::from_timestamp(seconds, 0).unwrap_or_else(|| {
		chrono::DateTime::from_timestamp(0, 0).expect("epoch is a valid timestamp")
	});
	let iso = datetime.format("%Y-%m-%dT%H:%M:%SZ").to_string();
	let long_date = iso.replace('-', "").replace(':', "");
	let short_date = long_date.chars().take(8).collect();
	(long_date, short_date)
}

/// `getSigningKey`: `HMAC(HMAC(HMAC(HMAC("AWS4"+secret, date), region), service), "aws4_request")`.
fn derive_signing_key(secret_access_key: &str, short_date: &str, region: &str, service: &str) -> Vec<u8> {
	let mut key = hmac_sha256(format!("AWS4{}", secret_access_key).as_bytes(), short_date.as_bytes());
	key = hmac_sha256(&key, region.as_bytes());
	key = hmac_sha256(&key, service.as_bytes());
	hmac_sha256(&key, KEY_TYPE_IDENTIFIER.as_bytes())
}

/// The `SignatureV4.sign({...})` result: the headers the SDK adds to the request.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SignedRequestHeaders {
	pub headers: IndexMap<String, String>,
}

/// Sign a request exactly like `SignatureV4.signRequest` does.
/// Sign a Bedrock request (the provider's `SignatureV4({ service: "bedrock" })`).
#[allow(clippy::too_many_arguments)]
pub fn sign_request(
	method: &str,
	path: &str,
	query: &[(String, String)],
	headers: &IndexMap<String, String>,
	body: &str,
	region: &str,
	credentials: &AwsCredentials,
	signing_date_ms: i64,
) -> SignedRequestHeaders {
	sign_request_for_service(
		method,
		path,
		query,
		headers,
		body,
		region,
		BEDROCK_SIGNING_SERVICE,
		credentials,
		signing_date_ms,
	)
}

/// The generic `SignatureV4.signRequest` port (`service` is the signer's service name).
#[allow(clippy::too_many_arguments)]
pub fn sign_request_for_service(
	method: &str,
	path: &str,
	query: &[(String, String)],
	headers: &IndexMap<String, String>,
	body: &str,
	region: &str,
	service: &str,
	credentials: &AwsCredentials,
	signing_date_ms: i64,
) -> SignedRequestHeaders {
	let (long_date, short_date) = format_signing_date(signing_date_ms);
	let scope = format!("{}/{}/{}/{}", short_date, region, service, KEY_TYPE_IDENTIFIER);

	let mut request_headers: IndexMap<String, String> = headers.clone();
	request_headers.insert(AMZ_DATE_HEADER.to_string(), long_date.clone());
	if let Some(token) = &credentials.session_token {
		if !token.is_empty() {
			request_headers.insert(TOKEN_HEADER.to_string(), token.clone());
		}
	}

	let payload_hash = if request_headers
		.keys()
		.any(|name| name.to_lowercase() == SHA256_HEADER)
	{
		request_headers
			.iter()
			.find(|(name, _)| name.to_lowercase() == SHA256_HEADER)
			.map(|(_, value)| value.clone())
			.unwrap_or_else(|| EMPTY_PAYLOAD_SHA256.to_string())
	} else if body.is_empty() {
		EMPTY_PAYLOAD_SHA256.to_string()
	} else {
		sha256_hex(body.as_bytes())
	};
	if !request_headers
		.keys()
		.any(|name| name.to_lowercase() == SHA256_HEADER)
	{
		request_headers.insert(SHA256_HEADER.to_string(), payload_hash.clone());
	}

	// getCanonicalHeaders: sorted by name, trimmed and whitespace-collapsed values.
	let mut canonical: Vec<(String, String)> = Vec::new();
	let mut sorted_names: Vec<String> = request_headers.keys().cloned().collect();
	sorted_names.sort();
	for name in sorted_names {
		let Some(value) = request_headers.get(&name) else {
			continue;
		};
		let canonical_name = name.to_lowercase();
		if is_unsignable_header(&canonical_name) {
			continue;
		}
		canonical.push((canonical_name, collapse_whitespace(value.trim())));
	}
	let canonical_headers = canonical
		.iter()
		.map(|(name, value)| format!("{}:{}", name, value))
		.collect::<Vec<_>>()
		.join("\n");
	let signed_headers = canonical
		.iter()
		.map(|(name, _)| name.clone())
		.collect::<Vec<_>>()
		.join(";");

	let canonical_request = [
		method.to_string(),
		canonical_path(path),
		canonical_query(query),
		canonical_headers,
		String::new(),
		signed_headers.clone(),
		payload_hash,
	]
	.join("\n");

	let string_to_sign = [
		ALGORITHM_IDENTIFIER.to_string(),
		long_date,
		scope.clone(),
		sha256_hex(canonical_request.as_bytes()),
	]
	.join("\n");

	let signing_key = derive_signing_key(&credentials.secret_access_key, &short_date, region, service);
	let signature = hex_encode(&hmac_sha256(&signing_key, string_to_sign.as_bytes()));

	let authorization = format!(
		"{} Credential={}/{}, SignedHeaders={}, Signature={}",
		ALGORITHM_IDENTIFIER, credentials.access_key_id, scope, signed_headers, signature
	);

	let mut signed: IndexMap<String, String> = IndexMap::new();
	signed.insert(AMZ_DATE_HEADER.to_string(), request_headers[AMZ_DATE_HEADER].clone());
	if let Some(token) = request_headers.get(TOKEN_HEADER) {
		signed.insert(TOKEN_HEADER.to_string(), token.clone());
	}
	if let Some(hash) = request_headers.get(SHA256_HEADER) {
		signed.insert(SHA256_HEADER.to_string(), hash.clone());
	}
	signed.insert("authorization".to_string(), authorization);
	SignedRequestHeaders { headers: signed }
}

/// `value.trim().replace(/\s+/g, " ")`.
fn collapse_whitespace(value: &str) -> String {
	let mut out = String::with_capacity(value.len());
	let mut in_run = false;
	for ch in value.chars() {
		if ch.is_whitespace() {
			if !in_run {
				out.push(' ');
				in_run = true;
			}
			continue;
		}
		in_run = false;
		out.push(ch);
	}
	out
}

/// The ConverseStream endpoint the SDK derives: `POST /model/{modelId}/converse-stream`.
pub fn converse_stream_path(model_id: &str) -> String {
	// modelId is a non-greedy HTTP label, so an ARN's slash belongs inside
	// the escaped label rather than introducing another path segment.
	format!("/model/{}/converse-stream", escape_uri(model_id))
}

/// The host the SDK derives when no explicit endpoint is configured.
pub fn bedrock_runtime_host(region: &str) -> String {
	format!("bedrock-runtime.{}.amazonaws.com", region)
}

// ---------------------------------------------------------------------------
// AWS event-stream decoding (was done by the SDK's EventStreamCodec)
// ---------------------------------------------------------------------------

/// One decoded `vnd.amazon.eventstream` message, mapped onto the SDK's
/// `ConverseStreamOutput` union members (`item.messageStart`, `item.contentBlockDelta`, ...).
#[derive(Debug, Clone, PartialEq)]
pub enum BedrockStreamEvent {
	MessageStart { role: String },
	ContentBlockStart { content_block_index: usize, start: Value },
	ContentBlockDelta { content_block_index: usize, delta: Value },
	ContentBlockStop { content_block_index: usize },
	MessageStop { stop_reason: Option<String> },
	Metadata { usage: Value },
	/// `item.internalServerException` / `item.modelStreamErrorException` / ... - thrown by the TS.
	Exception { name: String, message: String },
}

/// A decoded event-stream frame: its headers plus the JSON payload.
#[derive(Debug, Clone, PartialEq)]
pub struct EventStreamFrame {
	pub headers: IndexMap<String, String>,
	pub payload: Vec<u8>,
}

/// Decode every complete frame in `buffer`, leaving the incomplete tail in place.
///
/// Frame layout: total length (u32 BE) | headers length (u32 BE) | prelude CRC (u32) |
/// headers | payload | message CRC (u32).
pub fn decode_event_stream_frames(buffer: &mut Vec<u8>) -> Result<Vec<EventStreamFrame>, ProviderError> {
	let mut frames = Vec::new();
	loop {
		if buffer.len() < 4 {
			break;
		}
		let total_length = u32::from_be_bytes([buffer[0], buffer[1], buffer[2], buffer[3]]) as usize;
		if !(16..=16 * 1024 * 1024).contains(&total_length) {
			return Err(ProviderError::message("Invalid event stream frame length"));
		}
		if buffer.len() < total_length {
			break;
		}
		let frame: Vec<u8> = buffer.drain(..total_length).collect();
		frames.push(decode_event_stream_frame(&frame)?);
	}
	Ok(frames)
}

fn decode_event_stream_frame(frame: &[u8]) -> Result<EventStreamFrame, ProviderError> {
	if frame.len() < 16 {
		return Err(ProviderError::message("Truncated event stream frame"));
	}
	let headers_length = u32::from_be_bytes([frame[4], frame[5], frame[6], frame[7]]) as usize;
	if headers_length > frame.len() - 16 {
		return Err(ProviderError::message("Truncated event stream frame"));
	}
	let headers_end = 12 + headers_length;
	let header_bytes = &frame[..headers_end];

	let mut headers: IndexMap<String, String> = IndexMap::new();
	let mut cursor = 12usize;
	while cursor < headers_end {
		let name_length = frame[cursor] as usize;
		cursor += 1;
		if cursor + name_length >= headers_end {
			return Err(ProviderError::message("Malformed event stream headers"));
		}
		let name = String::from_utf8_lossy(&frame[cursor..cursor + name_length]).to_string();
		cursor += name_length;
		let header_type = frame[cursor];
		cursor += 1;
		match header_type {
			// 7 = string
			7 => {
				if cursor + 2 > headers_end {
					return Err(ProviderError::message("Malformed event stream headers"));
				}
				let value_length =
					u16::from_be_bytes([header_bytes[cursor], header_bytes[cursor + 1]]) as usize;
				cursor += 2;
				if cursor + value_length > headers_end {
					return Err(ProviderError::message("Malformed event stream headers"));
				}
				let value = String::from_utf8_lossy(&frame[cursor..cursor + value_length]).to_string();
				cursor += value_length;
				headers.insert(name, value);
			}
			// 0 = bool true, 1 = bool false
			0 | 1 => {
				headers.insert(name, if header_type == 0 { "true" } else { "false" }.to_string());
			}
			// 2 = byte, 3 = short, 4 = integer
			2 => cursor += 1,
			3 => cursor += 2,
			4 => cursor += 4,
			// 5 = long, 8 = timestamp (8 bytes), 6 = byte array, 9 = uuid (16 bytes)
			5 | 8 => cursor += 8,
			6 => {
				if cursor + 2 > headers_end {
					return Err(ProviderError::message("Malformed event stream headers"));
				}
				let value_length =
					u16::from_be_bytes([header_bytes[cursor], header_bytes[cursor + 1]]) as usize;
				cursor += 2 + value_length;
			}
			9 => cursor += 16,
			_ => return Err(ProviderError::message("Unknown event stream header type")),
		}
		if cursor > headers_end {
			return Err(ProviderError::message("Malformed event stream headers"));
		}
	}

	let payload = frame[headers_end..frame.len() - 4].to_vec();
	Ok(EventStreamFrame { headers, payload })
}

/// The JSON payload of one ConverseStream frame in the consumer's own
/// shape: the hot members land typed (no payload-wide DOM, no subtree
/// clones), the rare members (`start`, `delta`, `usage`) keep their `Value`
/// subtree. Field tolerances mirror the previous Value reads: wrong-typed
/// members read as absent, `null` is absent.
#[derive(serde::Deserialize, Default)]
struct BedrockPayload {
	#[serde(rename = "contentBlockIndex", default, deserialize_with = "crate::utils::typed_json::optional_u64")]
	content_block_index: Option<u64>,
	#[serde(default, deserialize_with = "crate::utils::typed_json::optional_string")]
	role: Option<String>,
	#[serde(default, deserialize_with = "crate::utils::typed_json::optional_value")]
	start: Option<Value>,
	#[serde(default, deserialize_with = "crate::utils::typed_json::optional_value")]
	delta: Option<Value>,
	#[serde(rename = "stopReason", default, deserialize_with = "crate::utils::typed_json::optional_string")]
	stop_reason: Option<String>,
	#[serde(default, deserialize_with = "crate::utils::typed_json::optional_value")]
	usage: Option<Value>,
}

/// The Value fallback: read each member off the DOM exactly the way the
/// pre-typed code did, so a payload the typed struct rejects (wrong-typed
/// members, duplicate keys) keeps its exact behavior.
fn bedrock_payload_from_value(value: &Value) -> BedrockPayload {
	BedrockPayload {
		content_block_index: value.get("contentBlockIndex").and_then(Value::as_u64),
		role: value.get("role").and_then(Value::as_str).map(str::to_string),
		start: value.get("start").cloned(),
		delta: value.get("delta").cloned(),
		stop_reason: value.get("stopReason").and_then(Value::as_str).map(str::to_string),
		usage: value.get("usage").cloned(),
	}
}

/// Map a decoded frame onto the `ConverseStreamOutput` member the TypeScript switches on.
pub fn parse_bedrock_stream_event(frame: &EventStreamFrame) -> Result<BedrockStreamEvent, ProviderError> {
	let message_type = frame.headers.get(":message-type").map(String::as_str).unwrap_or("event");
	if message_type == "exception" || message_type == "error" {
		let name = frame
			.headers
			.get(":exception-type")
			.or_else(|| frame.headers.get(":error-code"))
			.cloned()
			.unwrap_or_else(|| "Unknown".to_string());
		let payload: Value = if frame.payload.is_empty() {
			Value::Object(Map::new())
		} else {
			serde_json::from_slice(&frame.payload).unwrap_or(Value::Null)
		};
		let message = payload
			.get("message")
			.and_then(Value::as_str)
			.map(str::to_string)
			.unwrap_or_else(|| name.clone());
		return Ok(BedrockStreamEvent::Exception { name, message });
	}

	let event_type = frame.headers.get(":event-type").cloned().unwrap_or_default();
	// One typed parse of the payload: the hot members land directly in the
	// event without building the full payload DOM and cloning subtrees; the
	// rare members (`start`, `delta`, `usage`) deserialize as their own
	// `Value` subtree. Tolerances mirror the previous Value reads exactly.
	let payload: BedrockPayload = if frame.payload.is_empty() {
		BedrockPayload::default()
	} else {
		// A non-object payload (serde would map a sequence onto the struct
		// by position) keeps the old Value behavior: every field read absent.
		let looks_like_object = frame
			.payload
			.iter()
			.find(|byte| !byte.is_ascii_whitespace())
			.map(|byte| *byte == b'{')
			.unwrap_or(false);
		if !looks_like_object {
			// Malformed JSON keeps the old error; anything else parses fine
			// with all field reads absent.
			if let Err(error) = serde_json::from_slice::<Value>(&frame.payload) {
				return Err(ProviderError::message(error.to_string()));
			}
			BedrockPayload::default()
		} else {
			match serde_json::from_slice::<BedrockPayload>(&frame.payload) {
				Ok(payload) => payload,
				// A shape the typed struct cannot cover (a wrong-typed
				// member such as a string `contentBlockIndex`, duplicate
				// keys) keeps the unchanged Value consumer semantics: parse
				// the DOM and read each member the way the pre-typed code
				// did. Malformed JSON keeps the old error.
				Err(_) => {
					let value: Value = serde_json::from_slice(&frame.payload)
						.map_err(|error| ProviderError::message(error.to_string()))?;
					bedrock_payload_from_value(&value)
				}
			}
		}
	};

	let content_block_index = payload.content_block_index.unwrap_or(0) as usize;

	match event_type.as_str() {
		"messageStart" => Ok(BedrockStreamEvent::MessageStart {
			role: payload.role.unwrap_or_else(|| "user".to_string()),
		}),
		"contentBlockStart" => Ok(BedrockStreamEvent::ContentBlockStart {
			content_block_index,
			start: payload.start.unwrap_or(Value::Null),
		}),
		"contentBlockDelta" => Ok(BedrockStreamEvent::ContentBlockDelta {
			content_block_index,
			delta: payload.delta.unwrap_or(Value::Null),
		}),
		"contentBlockStop" => Ok(BedrockStreamEvent::ContentBlockStop { content_block_index }),
		"messageStop" => Ok(BedrockStreamEvent::MessageStop {
			stop_reason: payload.stop_reason,
		}),
		"metadata" => Ok(BedrockStreamEvent::Metadata {
			usage: payload.usage.unwrap_or(Value::Null),
		}),
		other => Err(ProviderError::message(format!("Unknown event type: {}", other))),
	}
}

// ---------------------------------------------------------------------------
// HTTP request (was `client.send(command, { abortSignal })`)
// ---------------------------------------------------------------------------

/// The ConverseStream command input, in the SDK's serialization order.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ConverseStreamCommandInput {
	pub model_id: String,
	pub messages: Vec<Value>,
	pub system: Option<Vec<Value>>,
	pub inference_config: Map<String, Value>,
	pub tool_config: Option<Value>,
	pub additional_model_request_fields: Option<Value>,
	pub request_metadata: Option<IndexMap<String, String>>,
}

impl ConverseStreamCommandInput {
	/// The command input as the TypeScript hands it to `onPayload` (and as the SDK receives it):
	/// every present member, `modelId` included.
	pub fn to_json(&self) -> Value {
		let mut body = Map::new();
		body.insert("modelId".to_string(), Value::String(self.model_id.clone()));
		body.insert("messages".to_string(), Value::Array(self.messages.clone()));
		if let Some(system) = &self.system {
			body.insert("system".to_string(), Value::Array(system.clone()));
		}
		body.insert(
			"inferenceConfig".to_string(),
			Value::Object(self.inference_config.clone()),
		);
		if let Some(tool_config) = &self.tool_config {
			body.insert("toolConfig".to_string(), tool_config.clone());
		}
		if let Some(fields) = &self.additional_model_request_fields {
			body.insert("additionalModelRequestFields".to_string(), fields.clone());
		}
		if let Some(metadata) = &self.request_metadata {
			let mut map = Map::new();
			for (key, value) in metadata {
				map.insert(key.clone(), Value::String(value.clone()));
			}
			body.insert("requestMetadata".to_string(), Value::Object(map));
		}
		Value::Object(body)
	}

	/// The JSON body the SDK actually sends.
	///
	/// `modelId` is an `httpLabel` member of `ConverseStreamRequest`, so the SDK serializes it
	/// into the URI (`/model/{modelId}/converse-stream`) and leaves it out of the body.
	pub fn to_wire_json(&self) -> Value {
		let mut body = match self.to_json() {
			Value::Object(body) => body,
			other => return other,
		};
		body.shift_remove("modelId");
		Value::Object(body)
	}
}

/// TS: the `commandInput` object literal in `streamBedrock`.
///
/// NOTE: the TypeScript lets `@aws-sdk/client-bedrock-runtime` apply the ConverseStream input
/// defaults. The port keeps the fields the TS object literal carries and only the SDK defaults
/// this provider can observe through its own options.
pub fn build_command_input(
	context: &Context,
	model: &Model,
	options: &BedrockOptions,
	cache_retention: &str,
) -> Result<ConverseStreamCommandInput, ProviderError> {
	let mut inference_config = Map::new();
	if let Some(max_tokens) = options.stream.max_tokens {
		inference_config.insert("maxTokens".to_string(), json!(max_tokens));
	}
	if let Some(temperature) = options.stream.temperature {
		if !supports_always_on_adaptive_thinking(&model.id, Some(&model.name)) {
			inference_config.insert("temperature".to_string(), json!(temperature));
		}
	}

	Ok(ConverseStreamCommandInput {
		model_id: model.id.clone(),
		messages: convert_messages(context, model, cache_retention)?,
		system: build_system_prompt(context.system_prompt.as_deref(), model, cache_retention),
		inference_config,
		tool_config: convert_tool_config(context.tools.as_ref(), options.tool_choice.as_ref()),
		additional_model_request_fields: build_additional_model_request_fields(model, options),
		request_metadata: options.request_metadata.clone(),
	})
}

/// The endpoint the SDK would use: explicit `config.endpoint` when set, else the regional host.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedEndpoint {
	pub scheme: String,
	pub host: String,
	pub port: Option<u16>,
	pub origin: String,
}

pub fn resolve_endpoint(config: &BedrockClientConfig, region: &str) -> ResolvedEndpoint {
	match &config.endpoint {
		Some(endpoint) => {
			let parsed = url::Url::parse(endpoint).ok();
			let scheme = parsed
				.as_ref()
				.map(|url| url.scheme().to_string())
				.unwrap_or_else(|| "https".to_string());
			let host = parsed
				.as_ref()
				.and_then(|url| url.host_str().map(str::to_string))
				.unwrap_or_else(|| endpoint.clone());
			let port = parsed.as_ref().and_then(url::Url::port);
			let origin = match &port {
				Some(port) => format!("{}://{}:{}", scheme, host, port),
				None => format!("{}://{}", scheme, host),
			};
			ResolvedEndpoint {
				scheme,
				host,
				port,
				origin,
			}
		}
		None => {
			let host = bedrock_runtime_host(region);
			ResolvedEndpoint {
				scheme: "https".to_string(),
				host: host.clone(),
				port: None,
				origin: format!("https://{}", host),
			}
		}
	}
}

// ---------------------------------------------------------------------------
// Stream handling (TS: handleContentBlockStart / Delta / Stop / Metadata)
// ---------------------------------------------------------------------------

/// TS: `handleContentBlockStart(event, blocks, output, stream)`.
pub fn handle_content_block_start(
	content_block_index: usize,
	start: &Value,
	blocks: &mut Vec<BlockMeta>,
	output: &mut AssistantMessage,
	stream: &AssistantMessageEventStream,
) {
	if let Some(tool_use) = start.get("toolUse") {
		let block = ContentBlock::ToolCall(ToolCall {
			type_: "toolCall".to_string(),
			id: tool_use
				.get("toolUseId")
				.and_then(Value::as_str)
				.unwrap_or("")
				.to_string(),
			name: tool_use
				.get("name")
				.and_then(Value::as_str)
				.unwrap_or("")
				.to_string(),
			arguments: Map::new(),
			thought_signature: None,
		});
		output.content.push(block);
		blocks.push(BlockMeta {
			index: Some(content_block_index),
			partial_json: Some(String::new()),
		});
		stream.push(AssistantMessageEvent::ToolCallStart {
			content_index: output.content.len() - 1,
			partial: output.clone(),
		});
	}
}

/// TS: `handleContentBlockDelta(event, blocks, output, stream)`.
pub fn handle_content_block_delta(
	content_block_index: usize,
	delta: &Value,
	blocks: &mut Vec<BlockMeta>,
	output: &mut AssistantMessage,
	stream: &AssistantMessageEventStream,
) {
	let mut index = block_position(blocks.as_slice(), content_block_index);
	let mut block_exists = index.is_some();

	if let Some(text) = delta.get("text").and_then(Value::as_str) {
		// If no text block exists yet, create one, as `handleContentBlockStart` is not sent for text blocks
		if !block_exists {
			output
				.content
				.push(ContentBlock::Text(TextContent::new(String::new())));
			blocks.push(BlockMeta {
				index: Some(content_block_index),
				partial_json: None,
			});
			// TS: `block = blocks[index]` - the new block is the one used below.
			index = Some(output.content.len() - 1);
			stream.push(AssistantMessageEvent::TextStart {
				content_index: index.expect("created text block"),
				partial: output.clone(),
			});
		}
		let index = index.expect("text block");
		if let Some(ContentBlock::Text(text_content)) = output.content.get_mut(index) {
			text_content.text.push_str(text);
			stream.push(AssistantMessageEvent::TextDelta {
				content_index: index,
				delta: text.to_string(),
				partial: output.clone(),
			});
		}
	} else if let Some(tool_use) = delta.get("toolUse") {
		let is_tool_call = index
			.map(|index| matches!(output.content.get(index), Some(ContentBlock::ToolCall(_))))
			.unwrap_or(false);
		if is_tool_call {
			let index = index.expect("tool call block");
			let input = tool_use
				.get("input")
				.and_then(Value::as_str)
				.unwrap_or("")
				.to_string();
			let partial_json = {
				let meta = &mut blocks[index];
				let partial = meta.partial_json.get_or_insert_with(String::new);
				partial.push_str(&input);
				partial.clone()
			};
			let arguments = parse_streaming_json(Some(&partial_json));
			if let Some(ContentBlock::ToolCall(tool_call)) = output.content.get_mut(index) {
				tool_call.arguments = match arguments {
					Value::Object(map) => map,
					_ => Map::new(),
				};
			}
			stream.push(AssistantMessageEvent::ToolCallDelta {
				content_index: index,
				delta: input,
				partial: output.clone(),
			});
		}
	} else if let Some(reasoning_content) = delta.get("reasoningContent") {
		let mut thinking_index = index;

		if thinking_index.is_none() {
			let mut thinking = ThinkingContent::new(String::new());
			thinking.thinking_signature = Some(String::new());
			output.content.push(ContentBlock::Thinking(thinking));
			blocks.push(BlockMeta {
				index: Some(content_block_index),
				partial_json: None,
			});
			thinking_index = Some(output.content.len() - 1);
			stream.push(AssistantMessageEvent::ThinkingStart {
				content_index: thinking_index.expect("created thinking block"),
				partial: output.clone(),
			});
		}

		let thinking_index = thinking_index.expect("thinking block");
		let is_thinking = matches!(output.content.get(thinking_index), Some(ContentBlock::Thinking(_)));
		if is_thinking {
			if let Some(text) = reasoning_content.get("text").and_then(Value::as_str) {
				if !text.is_empty() {
					if let Some(ContentBlock::Thinking(thinking)) = output.content.get_mut(thinking_index) {
						thinking.thinking.push_str(text);
					}
					stream.push(AssistantMessageEvent::ThinkingDelta {
						content_index: thinking_index,
						delta: text.to_string(),
						partial: output.clone(),
					});
				}
			}
			if let Some(signature) = reasoning_content.get("signature").and_then(Value::as_str) {
				if let Some(ContentBlock::Thinking(thinking)) = output.content.get_mut(thinking_index) {
					let current = thinking.thinking_signature.get_or_insert_with(String::new);
					current.push_str(signature);
				}
			}
		}
	}
}

/// TS: `handleMetadata(event, model, output)`.
pub fn handle_metadata(usage: &Value, model: &Model, output: &mut AssistantMessage) {
	if usage.is_null() {
		return;
	}
	output.usage.input = usage.get("inputTokens").and_then(Value::as_f64).unwrap_or(0.0);
	output.usage.output = usage.get("outputTokens").and_then(Value::as_f64).unwrap_or(0.0);
	output.usage.cache_read = usage
		.get("cacheReadInputTokens")
		.and_then(Value::as_f64)
		.unwrap_or(0.0);
	output.usage.cache_write = usage
		.get("cacheWriteInputTokens")
		.and_then(Value::as_f64)
		.unwrap_or(0.0);
	output.usage.total_tokens = usage
		.get("totalTokens")
		.and_then(Value::as_f64)
		.unwrap_or(output.usage.input + output.usage.output);
	calculate_cost(model, &mut output.usage, None);
}

/// TS: `handleContentBlockStop(event, blocks, output, stream)`.
pub fn handle_content_block_stop(
	content_block_index: usize,
	blocks: &mut Vec<BlockMeta>,
	output: &mut AssistantMessage,
	stream: &AssistantMessageEventStream,
) {
	let Some(index) = block_position(blocks.as_slice(), content_block_index) else {
		return;
	};
	blocks[index].index = None;

	match output.content.get(index) {
		Some(ContentBlock::Text(text)) => {
			stream.push(AssistantMessageEvent::TextEnd {
				content_index: index,
				content: text.text.clone(),
				partial: output.clone(),
			});
		}
		Some(ContentBlock::Thinking(thinking)) => {
			stream.push(AssistantMessageEvent::ThinkingEnd {
				content_index: index,
				content: thinking.thinking.clone(),
				partial: output.clone(),
			});
		}
		Some(ContentBlock::ToolCall(_)) => {
			let partial_json = blocks[index].partial_json.clone();
			let arguments = parse_streaming_json(partial_json.as_deref());
			blocks[index].partial_json = None;
			let tool_call = match output.content.get_mut(index) {
				Some(ContentBlock::ToolCall(tool_call)) => {
					tool_call.arguments = match arguments {
						Value::Object(map) => map,
						_ => Map::new(),
					};
					tool_call.clone()
				}
				_ => return,
			};
			stream.push(AssistantMessageEvent::ToolCallEnd {
				content_index: index,
				tool_call,
				partial: output.clone(),
			});
		}
		None => {}
	}
}

// ---------------------------------------------------------------------------
// The request body (was the SDK's `client.send(command, { abortSignal })`)
// ---------------------------------------------------------------------------

/// The `config.requestHandler` half of the client config: proxy handling plus the optional
/// HTTP/1.1 pinning. Split out so tests can observe the builder itself.
///
/// TS `packages/ai/src/providers/amazon-bedrock.ts:172-178` installs a `NodeHttpHandler`; when
/// `AWS_BEDROCK_FORCE_HTTP1 === "1"` (`amazon-bedrock.ts:176`) that handler is what keeps the
/// request on HTTP/1.1 instead of the SDK's default `NodeHttp2Handler`. reqwest has no handler
/// object, so the same intent is expressed on the client with `http1_only()` (ALPN restricted to
/// `http/1.1`; `reqwest-0.12.28/src/async_impl/client.rs`, `http1_only`, and the ALPN list at its
/// line 825). Without the flag the client keeps reqwest's default negotiation, like the SDK.
fn bedrock_proxy_env_present() -> bool {
	[
		"HTTP_PROXY",
		"HTTPS_PROXY",
		"NO_PROXY",
		"http_proxy",
		"https_proxy",
		"no_proxy",
	]
	.iter()
	.any(|name| std::env::var(name).map(|value| !value.is_empty()).unwrap_or(false))
}

fn bedrock_client_policy(config: &BedrockClientConfig) -> crate::providers::shared_http::ClientPolicy {
	crate::providers::shared_http::ClientPolicy::Bedrock {
		force_http1: config.force_http1,
		no_proxy: !bedrock_proxy_env_present(),
	}
}

/// Kept for the client-builder option tests; `build_http_client` shares the
/// pooled client instead of constructing one.
#[cfg(test)]
fn bedrock_http_client_builder(config: &BedrockClientConfig) -> reqwest::ClientBuilder {
	bedrock_client_policy(config).builder()
}

fn build_http_client(config: &BedrockClientConfig) -> Result<reqwest::Client, ProviderError> {
	crate::providers::shared_http::try_shared_client(bedrock_client_policy(config))
		.map_err(ProviderError::message)
}

/// The `$metadata.requestId` the SDK reads from `x-amzn-requestid`.
fn request_id_from_headers(headers: &reqwest::header::HeaderMap) -> Option<String> {
	headers
		.get("x-amzn-requestid")
		.and_then(|value| value.to_str().ok())
		.map(str::to_string)
}

/// The exception name the SDK derives from `x-amzn-errortype` or the JSON `__type`.
fn exception_name_from_body(status: u16, headers: &reqwest::header::HeaderMap, body: &Value) -> String {
	let from_header = headers
		.get("x-amzn-errortype")
		.and_then(|value| value.to_str().ok())
		.map(|value| value.split(':').next().unwrap_or(value).to_string());
	if let Some(name) = from_header {
		return name;
	}
	let from_body = body
		.get("__type")
		.and_then(Value::as_str)
		.map(|value| value.rsplit('#').next().unwrap_or(value).to_string());
	if let Some(name) = from_body {
		return name;
	}
	// NOTE: the SDK maps the HTTP status onto a modelled exception name; the port keeps the
	// status-derived default for the shapes the ConverseStream API documents.
	match status {
		400 => "ValidationException".to_string(),
		403 => "AccessDeniedException".to_string(),
		429 => "ThrottlingException".to_string(),
		500 => "InternalServerException".to_string(),
		503 => "ServiceUnavailableException".to_string(),
		_ => "BedrockRuntimeServiceException".to_string(),
	}
}

/// The whole `try { ... }` body of the TypeScript async IIFE.
async fn run_bedrock_stream(
	model: &Model,
	context: &Context,
	options: &BedrockOptions,
	output: &mut AssistantMessage,
	blocks: &mut Vec<BlockMeta>,
	stream: &AssistantMessageEventStream,
) -> Result<(), ProviderError> {
	let config = resolve_bedrock_client_config(model, options);
	let cache_retention = resolve_cache_retention(options.stream.cache_retention.as_deref());

	let mut command_input = build_command_input(context, model, options, &cache_retention)?;

	// TS: `const nextCommandInput = await options?.onPayload?.(commandInput, model);`
	if let Some(on_payload) = options.stream.on_payload.clone() {
		let payload = command_input.to_json();
		if let Some(next) = on_payload(payload, model).await {
			command_input = command_input_from_json(next);
		}
	}

	let region = config.region.clone().unwrap_or_else(|| "us-east-1".to_string());
	let endpoint = resolve_endpoint(&config, &region);
	let path = converse_stream_path(&command_input.model_id);
	let body = command_input.to_wire_json().to_string();

	let mut headers: IndexMap<String, String> = IndexMap::new();
	headers.insert("content-type".to_string(), "application/json".to_string());
	headers.insert(HOST_HEADER.to_string(), host_header(&endpoint));

	if let Some(bearer_token) = config.bearer_token.clone() {
		headers.insert("authorization".to_string(), format!("Bearer {}", bearer_token));
	} else {
		let credentials = match config.credentials.clone() {
			Some(credentials) => credentials,
			None => resolve_aws_credentials(options)?,
		};
		let signed = sign_request(
			"POST",
			&path,
			&[],
			&headers,
			&body,
			&region,
			&credentials,
			now_ms(),
		);
		for (name, value) in signed.headers {
			headers.insert(name, value);
		}
	}

	// TS: `client.send(command, { abortSignal: options.signal })`.
	if options
		.stream
		.signal
		.as_ref()
		.map(|signal| signal.is_cancelled())
		.unwrap_or(false)
	{
		return Err(ProviderError::message("Request was aborted"));
	}

	// TS: the `config.requestHandler` installed by `resolve_bedrock_client_config`.
	let client = build_http_client(&config)?;
	let mut request = client
		.post(format!("{}{}", endpoint.origin, path))
		.body(body.clone());
	for (name, value) in &headers {
		request = request.header(name.as_str(), value.as_str());
	}

	let response = match &options.stream.signal {
		Some(signal) => {
			let token = signal.clone();
			tokio::select! {
				result = request.send() => result,
				_ = token.cancelled() => {
					return Err(ProviderError::message("Request was aborted"));
				}
			}
		}
		None => request.send().await,
	}
	.map_err(|error| ProviderError::message(error.to_string()))?;

	let status = response.status().as_u16();
	let response_headers = response.headers().clone();
	let request_id = request_id_from_headers(&response_headers);

	if status < 200 || status >= 300 {
		let text = response.text().await.unwrap_or_default();
		let parsed: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
		let name = exception_name_from_body(status, &response_headers, &parsed);
		let message = parsed
			.get("message")
			.and_then(Value::as_str)
			.map(str::to_string)
			.unwrap_or_else(|| {
				if text.is_empty() {
					format!("{}: {}", name, status)
				} else {
					text.clone()
				}
			});
		return Err(ProviderError::Service { name, message });
	}

	// TS: `if (response.$metadata.httpStatusCode !== undefined) { ... await options?.onResponse?.(...) }`
	{
		let mut response_headers_record = IndexMap::new();
		if let Some(request_id) = request_id.clone() {
			response_headers_record.insert("x-amzn-requestid".to_string(), request_id);
		}
			if let Some(on_response) = options.stream.on_response.clone() {
			on_response(
				crate::types::ProviderResponse {
					status: status as i64,
					headers: response_headers_record,
				},
				model,
			)
			.await;
		}
	}

	let mut buffer: Vec<u8> = Vec::new();
	let mut byte_stream = response.bytes_stream();

	loop {
		let chunk = match &options.stream.signal {
			Some(signal) => {
				let token = signal.clone();
				tokio::select! {
					next = byte_stream.next() => next,
					_ = token.cancelled() => {
						return Err(ProviderError::message("Request was aborted"));
					}
				}
			}
			None => byte_stream.next().await,
		};

		match chunk {
			None => break,
			Some(Err(error)) => return Err(ProviderError::message(error.to_string())),
			Some(Ok(bytes)) => buffer.extend_from_slice(&bytes),
		}

		for frame in decode_event_stream_frames(&mut buffer)? {
			let event = parse_bedrock_stream_event(&frame)?;
			match event {
				BedrockStreamEvent::MessageStart { role } => {
					if role != "assistant" {
						return Err(ProviderError::message(
							"Unexpected assistant message start but got user message start instead",
						));
					}
					stream.push(AssistantMessageEvent::Start {
						partial: output.clone(),
					});
				}
				BedrockStreamEvent::ContentBlockStart {
					content_block_index,
					start,
				} => handle_content_block_start(content_block_index, &start, blocks, output, stream),
				BedrockStreamEvent::ContentBlockDelta {
					content_block_index,
					delta,
				} => handle_content_block_delta(content_block_index, &delta, blocks, output, stream),
				BedrockStreamEvent::ContentBlockStop { content_block_index } => {
					handle_content_block_stop(content_block_index, blocks, output, stream)
				}
				BedrockStreamEvent::MessageStop { stop_reason } => {
					output.stop_reason = map_stop_reason(stop_reason.as_deref());
					if output.stop_reason == "error" {
						output.stop_reason_raw = stop_reason;
					}
				}
				BedrockStreamEvent::Metadata { usage } => handle_metadata(&usage, model, output),
				BedrockStreamEvent::Exception { name, message } => {
					return Err(ProviderError::Service { name, message });
				}
			}
		}
	}

	if options
		.stream
		.signal
		.as_ref()
		.map(|signal| signal.is_cancelled())
		.unwrap_or(false)
	{
		return Err(ProviderError::message("Request was aborted"));
	}

	if output.stop_reason == "error" || output.stop_reason == "aborted" {
		return Err(ProviderError::Failure(stream_failure_from_stop_reason(
			output.stop_reason_raw.as_deref(),
			request_id.as_deref(),
		)));
	}

	Ok(())
}

/// `headers.host` - includes the port when the endpoint carries a non-default one.
fn host_header(endpoint: &ResolvedEndpoint) -> String {
	match endpoint.port {
		Some(port) => format!("{}:{}", endpoint.host, port),
		None => endpoint.host.clone(),
	}
}

/// Rebuild the command input from an `onPayload` return value.
fn command_input_from_json(value: Value) -> ConverseStreamCommandInput {
	let mut input = ConverseStreamCommandInput::default();
	let Some(object) = value.as_object() else {
		return input;
	};
	input.model_id = object
		.get("modelId")
		.and_then(Value::as_str)
		.unwrap_or_default()
		.to_string();
	input.messages = object
		.get("messages")
		.and_then(Value::as_array)
		.cloned()
		.unwrap_or_default();
	input.system = object.get("system").and_then(Value::as_array).cloned();
	input.inference_config = object
		.get("inferenceConfig")
		.and_then(Value::as_object)
		.cloned()
		.unwrap_or_default();
	input.tool_config = object.get("toolConfig").cloned();
	input.additional_model_request_fields = object.get("additionalModelRequestFields").cloned();
	input.request_metadata = object
		.get("requestMetadata")
		.and_then(Value::as_object)
		.map(|map| {
			map.iter()
				.filter_map(|(key, value)| value.as_str().map(|value| (key.clone(), value.to_string())))
				.collect()
		});
	input
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {

	#[test]
	fn typed_payload_reads_match_the_value_reads() {
		// (event type, payload) pairs, all through the production
		// decode + parse path; the expected events use the pre-typed Value
		// reads so the typed shape must agree field for field.
		let cases: Vec<(&str, &[u8])> = vec![
			("contentBlockDelta", br#"{"contentBlockIndex":2,"delta":{"text":"hi"}}"#),
			("contentBlockDelta", br#"{"contentBlockIndex":0,"delta":{"toolUse":{"input":"{\"a\":1}"}}}"#),
			("contentBlockStart", br#"{"start":{"toolUse":{"toolUseId":"t1","name":"read"}},"contentBlockIndex":1}"#),
			("messageStart", br#"{"role":"assistant"}"#),
			("messageStop", br#"{"stopReason":"end_turn"}"#),
			("metadata", br#"{"usage":{"inputTokens":5,"outputTokens":2}}"#),
			// wrong-typed members read as absent (no stream abort)
			("contentBlockDelta", br#"{"contentBlockIndex":"2","role":7,"stopReason":42}"#),
			("contentBlockDelta", br#"{"contentBlockIndex":1e3,"delta":{"text":"f"}}"#),
			// null members read as absent
			("contentBlockDelta", br#"{"contentBlockIndex":null,"role":null,"stopReason":null,"delta":null}"#),
			// unknown members ignored
			("contentBlockDelta", br#"{"extra":1,"contentBlockIndex":3}"#),
			// non-object payloads: every read absent, no error
			("contentBlockDelta", b"[1, 2]"),
			("messageStart", b"\"text\""),
		];
		for (event_type, payload) in cases {
			let mut buffer = frame(
				&[(":message-type", "event"), (":event-type", event_type)],
				payload,
			);
			let frames = decode_event_stream_frames(&mut buffer).unwrap();
			assert_eq!(frames.len(), 1);
			let event = parse_bedrock_stream_event(&frames[0])
				.unwrap_or_else(|error| panic!("{event_type} {payload:?}: {error}"));
			let value: Value = serde_json::from_slice(payload).unwrap();
			let content_block_index = value
				.get("contentBlockIndex")
				.and_then(Value::as_u64)
				.unwrap_or(0) as usize;
			let expected = match event_type {
				"messageStart" => BedrockStreamEvent::MessageStart {
					role: value
						.get("role")
						.and_then(Value::as_str)
						.unwrap_or("user")
						.to_string(),
				},
				"contentBlockStart" => BedrockStreamEvent::ContentBlockStart {
					content_block_index,
					start: value.get("start").cloned().unwrap_or(Value::Null),
				},
				"contentBlockDelta" => BedrockStreamEvent::ContentBlockDelta {
					content_block_index,
					delta: value.get("delta").cloned().unwrap_or(Value::Null),
				},
				"messageStop" => BedrockStreamEvent::MessageStop {
					stop_reason: value
						.get("stopReason")
						.and_then(Value::as_str)
						.map(str::to_string),
				},
				"metadata" => BedrockStreamEvent::Metadata {
					usage: value.get("usage").cloned().unwrap_or(Value::Null),
				},
				other => panic!("unknown event type {other}"),
			};
			assert_eq!(event, expected, "mismatch on {event_type} {payload:?}");
		}
	}

	/// Wrong-typed `contentBlockIndex` must not abort the stream: the
	/// pre-typed reads treated it as absent (index 0).
	#[test]
	fn wrong_typed_content_block_index_keeps_the_value_reads() {
		for payload in [
			&br#"{"contentBlockIndex":"2","delta":{"text":"hi"}}"#[..],
			&br#"{"contentBlockIndex":1e3,"delta":{"text":"hi"}}"#[..],
			&br#"{"contentBlockIndex":18446744073709551616,"delta":{"text":"hi"}}"#[..],
		] {
			let mut buffer = frame(
				&[(":message-type", "event"), (":event-type", "contentBlockDelta")],
				payload,
			);
			let frames = decode_event_stream_frames(&mut buffer).unwrap();
			match parse_bedrock_stream_event(&frames[0]) {
				Ok(BedrockStreamEvent::ContentBlockDelta { content_block_index, delta }) => {
					assert_eq!(content_block_index, 0, "wrong-typed index must read as 0");
					assert_eq!(
						delta.get("text").and_then(Value::as_str),
						Some("hi"),
						"delta must still parse on {payload:?}"
					);
				}
				other => panic!("expected ContentBlockDelta, got {other:?} on {payload:?}"),
			}
		}
	}
use super::*;
	use crate::types::{InputModality, ModelCost, ThinkingLevelMap, ToolResultMessage};

	fn model(id: &str, name: &str) -> Model {
		Model {
			id: id.to_string(),
			name: name.to_string(),
			api: "bedrock-converse-stream".to_string(),
			provider: "amazon-bedrock".to_string(),
			base_url: "https://bedrock-runtime.us-east-1.amazonaws.com".to_string(),
			reasoning: true,
			input: vec![InputModality::Text],
			cost: ModelCost::default(),
			context_window: 200_000.0,
			max_tokens: 8192.0,
			..Default::default()
		}
	}

	fn user_context(text: &str) -> Context {
		Context::new(
			None,
			vec![Message::user(crate::types::UserMessage::new(
				UserContent::Text(text.to_string()),
				0,
			))],
			None,
		)
	}

	#[test]
	fn error_prefixes_match_the_typescript_table() {
		let error = ProviderError::service("ThrottlingException", "Too many requests");
		assert_eq!(format_bedrock_error(&error), "Throttling error: Too many requests");

		let error = ProviderError::service("ServiceUnavailableException", "try later");
		assert_eq!(format_bedrock_error(&error), "Service unavailable: try later");

		let error = ProviderError::service("AccessDeniedException", "nope");
		assert_eq!(format_bedrock_error(&error), "AccessDeniedException: nope");

		let error = ProviderError::message("plain failure");
		assert_eq!(format_bedrock_error(&error), "plain failure");
	}

	#[test]
	fn stop_reason_mapping_matches_bedrock_enum() {
		assert_eq!(map_stop_reason(Some("end_turn")), "stop");
		assert_eq!(map_stop_reason(Some("stop_sequence")), "stop");
		assert_eq!(map_stop_reason(Some("max_tokens")), "length");
		assert_eq!(map_stop_reason(Some("model_context_window_exceeded")), "length");
		assert_eq!(map_stop_reason(Some("tool_use")), "toolUse");
		assert_eq!(map_stop_reason(Some("content_filtered")), "error");
		assert_eq!(map_stop_reason(None), "error");
	}

	#[test]
	fn endpoint_region_is_derived_from_standard_hosts() {
		assert_eq!(
			get_standard_bedrock_endpoint_region(Some("https://bedrock-runtime.eu-central-1.amazonaws.com")),
			Some("eu-central-1".to_string())
		);
		assert_eq!(
			get_standard_bedrock_endpoint_region(Some("https://bedrock-runtime-fips.us-west-2.amazonaws.com")),
			Some("us-west-2".to_string())
		);
		assert_eq!(
			get_standard_bedrock_endpoint_region(Some("https://bedrock-runtime.cn-north-1.amazonaws.com.cn")),
			Some("cn-north-1".to_string())
		);
		assert_eq!(get_standard_bedrock_endpoint_region(Some("https://bedrock-vpc.example.com")), None);
		assert_eq!(get_standard_bedrock_endpoint_region(None), None);
		assert_eq!(get_standard_bedrock_endpoint_region(Some("not a url")), None);
	}

	#[test]
	fn explicit_endpoint_is_only_pinned_for_standard_hosts() {
		assert!(should_use_explicit_bedrock_endpoint(
			"https://bedrock-runtime.eu-central-1.amazonaws.com",
			None,
			false
		));
		assert!(!should_use_explicit_bedrock_endpoint(
			"https://bedrock-runtime.eu-central-1.amazonaws.com",
			Some("us-east-2"),
			false
		));
		assert!(!should_use_explicit_bedrock_endpoint(
			"https://bedrock-runtime.eu-central-1.amazonaws.com",
			None,
			true
		));
		assert!(should_use_explicit_bedrock_endpoint("https://bedrock-vpc.example.com", None, false));
	}

	#[test]
	fn govcloud_targets_are_detected_from_region_and_model_id() {
		let mut options = BedrockOptions::default();
		options.region = Some("us-gov-west-1".to_string());
		assert!(is_gov_cloud_bedrock_target(&model("global.anthropic.claude-opus-4-7", ""), &options));

		let options = BedrockOptions::default();
		assert!(is_gov_cloud_bedrock_target(
			&model("us-gov.anthropic.claude-sonnet-4-5-20250929-v1:0", ""),
			&options
		));
		assert!(is_gov_cloud_bedrock_target(
			&model("arn:aws-us-gov:bedrock:us-gov-west-1:1:inference-profile/x", ""),
			&options
		));
		assert!(!is_gov_cloud_bedrock_target(
			&model("global.anthropic.claude-opus-4-7", ""),
			&options
		));
	}

	#[test]
	fn model_match_candidates_collapse_separators() {
		let candidates = get_model_match_candidates("global.anthropic.claude-opus-4-7", Some("Claude Opus 4.7"));
		assert_eq!(candidates[0], "global.anthropic.claude-opus-4-7");
		assert_eq!(candidates[1], "global-anthropic-claude-opus-4-7");
		assert_eq!(candidates[2], "claude opus 4.7");
		assert_eq!(candidates[3], "claude-opus-4-7");
		assert_eq!(get_model_match_candidates("id", None).len(), 2);
	}

	#[test]
	fn adaptive_thinking_detection_covers_profiles_and_names() {
		assert!(supports_adaptive_thinking("global.anthropic.claude-opus-4-6-v1", None));
		assert!(supports_adaptive_thinking("global.anthropic.claude-opus-4-7", None));
		assert!(supports_adaptive_thinking("global.anthropic.claude-opus-4-8", None));
		assert!(supports_adaptive_thinking("global.anthropic.claude-opus-5", None));
		assert!(supports_adaptive_thinking("global.anthropic.claude-sonnet-4-6", None));
		assert!(supports_adaptive_thinking("global.anthropic.claude-sonnet-5", None));
		assert!(supports_adaptive_thinking("global.anthropic.claude-fable-5", None));
		assert!(supports_adaptive_thinking("global.anthropic.claude-mythos-5", None));
		assert!(supports_adaptive_thinking("global.anthropic.claude-mythos-preview", None));
		assert!(supports_adaptive_thinking(
			"arn:aws:bedrock:us-east-1:1:application-inference-profile/my-profile",
			Some("Claude Opus 4.6")
		));
		assert!(!supports_adaptive_thinking(
			"us.anthropic.claude-sonnet-4-5-20250929-v1:0",
			None
		));
	}

	#[test]
	fn always_on_adaptive_thinking_only_covers_fable_and_mythos() {
		assert!(supports_always_on_adaptive_thinking("global.anthropic.claude-fable-5", None));
		assert!(supports_always_on_adaptive_thinking("global.anthropic.claude-mythos-5", None));
		assert!(supports_always_on_adaptive_thinking("global.anthropic.claude-mythos-preview", None));
		assert!(!supports_always_on_adaptive_thinking("global.anthropic.claude-opus-4-7", None));
	}

	#[test]
	fn prompt_caching_support_matches_claude_rules() {
		assert!(supports_prompt_caching(&model("global.anthropic.claude-opus-4-6-v1", "")));
		assert!(supports_prompt_caching(&model("us.anthropic.claude-3-7-sonnet-20250219-v1:0", "")));
		assert!(supports_prompt_caching(&model("us.anthropic.claude-3-5-haiku-20241022-v1:0", "")));
		assert!(!supports_prompt_caching(&model("amazon.nova-2-lite-v1:0", "Nova 2 Lite")));
		assert!(supports_prompt_caching(&model(
			"arn:aws:bedrock:us-east-1:1:application-inference-profile/my-profile",
			"Claude Sonnet 4.6"
		)));
	}

	#[test]
	fn claude_detection_covers_id_and_name() {
		assert!(is_anthropic_claude_model(&model("anthropic.claude-3", "")));
		assert!(is_anthropic_claude_model(&model("anthropic/claude-3", "")));
		assert!(is_anthropic_claude_model(&model("x", "Anthropic.Claude 3")));
		assert!(is_anthropic_claude_model(&model("x", "anthropic/claude")));
		assert!(is_anthropic_claude_model(&model("x", "Claude Opus")));
		assert!(!is_anthropic_claude_model(&model("amazon.nova-lite", "Nova")));
	}

	#[test]
	fn thinking_effort_clamps_to_supported_levels() {
		let mut base = model("global.anthropic.claude-opus-4-7", "");
		// Extended levels require an explicit model map, even for adaptive models.
		assert_eq!(map_thinking_level_to_effort(&base, Some("xhigh")), "high");
		assert_eq!(map_thinking_level_to_effort(&base, Some("max")), "high");
		base.thinking_level_map = Some(ThinkingLevelMap::from([
			("xhigh".to_string(), Some("xhigh".to_string())),
			("max".to_string(), Some("max".to_string())),
		]));
		assert_eq!(map_thinking_level_to_effort(&base, Some("low")), "low");
		assert_eq!(map_thinking_level_to_effort(&base, Some("medium")), "medium");
		assert_eq!(map_thinking_level_to_effort(&base, Some("high")), "high");
		assert_eq!(map_thinking_level_to_effort(&base, Some("xhigh")), "xhigh");
		assert_eq!(map_thinking_level_to_effort(&base, Some("max")), "max");
		assert_eq!(map_thinking_level_to_effort(&base, None), "high");

		// xhigh/max unsupported -> clampThinkingLevel walks up to the supported level.
		let mut clamped = model("global.anthropic.claude-opus-4-6-v1", "");
		let mut map = ThinkingLevelMap::new();
		map.insert("xhigh".to_string(), None);
		map.insert("max".to_string(), Some("max".to_string()));
		clamped.thinking_level_map = Some(map);
		assert_eq!(map_thinking_level_to_effort(&clamped, Some("xhigh")), "max");
		assert_eq!(map_thinking_level_to_effort(&clamped, Some("max")), "max");

		let mut mapped = model("global.anthropic.claude-opus-4-7", "");
		let mut map = ThinkingLevelMap::new();
		map.insert("high".to_string(), Some("custom-effort".to_string()));
		mapped.thinking_level_map = Some(map);
		assert_eq!(map_thinking_level_to_effort(&mapped, Some("high")), "custom-effort");
	}

	#[test]
	fn thinking_payload_matches_the_typescript_shapes() {
		// Adaptive: thinking + output_config, no anthropic_beta.
		let mut adaptive = model("global.anthropic.claude-opus-4-7", "Claude Opus 4.7");
		adaptive.thinking_level_map = Some(ThinkingLevelMap::from([
			("xhigh".to_string(), Some("xhigh".to_string())),
		]));
		let options = BedrockOptions {
			reasoning: Some("xhigh".to_string()),
			..Default::default()
		};
		let fields = build_additional_model_request_fields(&adaptive, &options).unwrap();
		assert_eq!(
			fields["thinking"],
			json!({ "type": "adaptive", "display": "summarized" })
		);
		assert_eq!(fields["output_config"], json!({ "effort": "xhigh" }));
		assert!(fields.get("anthropic_beta").is_none());

		// GovCloud: display is omitted.
		let gov_options = BedrockOptions {
			reasoning: Some("high".to_string()),
			region: Some("us-gov-west-1".to_string()),
			..Default::default()
		};
		let fields = build_additional_model_request_fields(&adaptive, &gov_options).unwrap();
		assert_eq!(fields["thinking"], json!({ "type": "adaptive" }));
		assert_eq!(fields["output_config"], json!({ "effort": "high" }));

		// Fixed budget: enabled + budget_tokens + anthropic_beta.
		let budget_model = model("us.anthropic.claude-sonnet-4-5-20250929-v1:0", "Claude Sonnet 4.5");
		let budget_options = BedrockOptions {
			reasoning: Some("high".to_string()),
			..Default::default()
		};
		let fields = build_additional_model_request_fields(&budget_model, &budget_options).unwrap();
		assert_eq!(fields["thinking"], json!({ "type": "enabled", "budget_tokens": 16384.0, "display": "summarized" }));
		assert_eq!(fields["anthropic_beta"], json!(["interleaved-thinking-2025-05-14"]));

		// xhigh/max clamp to the high budget.
		let max_options = BedrockOptions {
			reasoning: Some("xhigh".to_string()),
			..Default::default()
		};
		let fields = build_additional_model_request_fields(&budget_model, &max_options).unwrap();
		assert_eq!(fields["thinking"]["budget_tokens"].as_f64(), Some(16384.0));

		// Custom budgets win for the clamped level.
		let custom_options = BedrockOptions {
			reasoning: Some("low".to_string()),
			thinking_budgets: Some(ThinkingBudgets {
				low: Some(99.0),
				..Default::default()
			}),
			..Default::default()
		};
		let fields = build_additional_model_request_fields(&budget_model, &custom_options).unwrap();
		assert_eq!(fields["thinking"]["budget_tokens"], json!(99.0));

		// interleavedThinking: false removes the beta flag.
		let no_interleave = BedrockOptions {
			reasoning: Some("high".to_string()),
			interleaved_thinking: Some(false),
			..Default::default()
		};
		let fields = build_additional_model_request_fields(&budget_model, &no_interleave).unwrap();
		assert!(fields.get("anthropic_beta").is_none());

		// thinkingDisplay: "omitted" travels through.
		let omitted = BedrockOptions {
			reasoning: Some("high".to_string()),
			thinking_display: Some("omitted".to_string()),
			..Default::default()
		};
		let fields = build_additional_model_request_fields(&adaptive, &omitted).unwrap();
		assert_eq!(fields["thinking"]["display"], json!("omitted"));

		// No reasoning, or a non-reasoning model -> undefined.
		assert!(build_additional_model_request_fields(&adaptive, &BedrockOptions::default()).is_none());
		let mut non_reasoning = adaptive.clone();
		non_reasoning.reasoning = false;
		assert!(build_additional_model_request_fields(&non_reasoning, &options).is_none());

		// Non-Claude models get no additional fields.
		let nova = model("amazon.nova-2-lite-v1:0", "Nova 2 Lite");
		assert!(build_additional_model_request_fields(&nova, &options).is_none());
	}

	#[test]
	fn system_prompt_builds_cache_points() {
		let claude = model("global.anthropic.claude-opus-4-6-v1", "");
		assert!(build_system_prompt(None, &claude, "short").is_none());
		assert!(build_system_prompt(Some(""), &claude, "short").is_none());

		let blocks = build_system_prompt(Some("You are helpful."), &claude, "short").unwrap();
		assert_eq!(blocks.len(), 2);
		assert_eq!(blocks[0], json!({ "text": "You are helpful." }));
		assert_eq!(blocks[1], json!({ "cachePoint": { "type": "default" } }));

		let blocks = build_system_prompt(Some("hi"), &claude, "long").unwrap();
		assert_eq!(blocks[1], json!({ "cachePoint": { "type": "default", "ttl": "1h" } }));

		let blocks = build_system_prompt(Some("hi"), &claude, "none").unwrap();
		assert_eq!(blocks.len(), 1);

		let nova = model("amazon.nova-2-lite-v1:0", "Nova 2 Lite");
		let blocks = build_system_prompt(Some("hi"), &nova, "short").unwrap();
		assert_eq!(blocks.len(), 1);
	}

	#[test]
	fn tool_call_ids_are_sanitized_and_truncated() {
		assert_eq!(normalize_tool_call_id("call|item"), "call_item");
		assert_eq!(normalize_tool_call_id("a-b_c9"), "a-b_c9");
		let long = "x".repeat(80);
		assert_eq!(normalize_tool_call_id(&long).chars().count(), 64);
	}

	#[test]
	fn image_blocks_use_bedrock_formats() {
		let block = create_image_block("image/jpeg", "AAEC").unwrap();
		assert_eq!(block["format"], json!("jpeg"));
		assert_eq!(block["source"]["bytes"], json!("AAEC"));

		assert_eq!(create_image_block("image/jpg", "AAEC").unwrap()["format"], json!("jpeg"));
		assert_eq!(create_image_block("image/png", "AAEC").unwrap()["format"], json!("png"));
		assert_eq!(create_image_block("image/gif", "AAEC").unwrap()["format"], json!("gif"));
		assert_eq!(create_image_block("image/webp", "AAEC").unwrap()["format"], json!("webp"));

		let error = create_image_block("image/bmp", "AAEC").unwrap_err();
		assert_eq!(format_bedrock_error(&error), "Unknown image type: image/bmp");
	}

	#[test]
	fn messages_convert_to_converse_shapes() {
		let claude = model("global.anthropic.claude-opus-4-6-v1", "");

		let converted = convert_messages(&user_context("hello"), &claude, "none").unwrap();
		assert_eq!(converted.len(), 1);
		assert_eq!(converted[0]["role"], json!("user"));
		assert_eq!(converted[0]["content"], json!([{ "text": "hello" }]));

		// Cache point lands on the last user message when caching is on.
		let converted = convert_messages(&user_context("hello"), &claude, "short").unwrap();
		assert_eq!(converted[0]["content"].as_array().unwrap().len(), 2);
		assert_eq!(converted[0]["content"][1], json!({ "cachePoint": { "type": "default" } }));

		// Consecutive tool results collapse into one user message.
		let mut assistant = AssistantMessage::new("bedrock-converse-stream", "amazon-bedrock", "m", 0);
		assistant.content = vec![ContentBlock::ToolCall(ToolCall::new(
			"tool-1",
			"read",
			Map::new(),
		))];
		assistant.content.push(ContentBlock::ToolCall(ToolCall::new(
			"tool-2", "read", Map::new(),
		)));
		let context = Context::new(
			None,
			vec![
				Message::user(crate::types::UserMessage::new(UserContent::Text("hi".into()), 0)),
				Message::assistant(assistant),
				Message::tool_result(ToolResultMessage::new(
					"tool-1",
					"read",
					vec![ImageOrTextContent::Text(TextContent::new("a"))],
					false,
					0,
				)),
				Message::tool_result(ToolResultMessage::new(
					"tool-2",
					"read",
					vec![ImageOrTextContent::Text(TextContent::new("b"))],
					true,
					0,
				)),
			],
			None,
		);
		let converted = convert_messages(&context, &claude, "none").unwrap();
		assert_eq!(converted.len(), 3);
		assert_eq!(converted[1]["role"], json!("assistant"));
		assert_eq!(converted[1]["content"][0]["toolUse"]["toolUseId"], json!("tool-1"));
		assert_eq!(converted[2]["role"], json!("user"));
		let tool_results = converted[2]["content"].as_array().unwrap();
		assert_eq!(tool_results.len(), 2);
		assert_eq!(tool_results[0]["toolResult"]["status"], json!("success"));
		assert_eq!(tool_results[1]["toolResult"]["status"], json!("error"));

		// Empty assistant content is skipped entirely.
		let empty_assistant = AssistantMessage::new("bedrock-converse-stream", "amazon-bedrock", "m", 0);
		let context = Context::new(
			None,
			vec![
				Message::user(crate::types::UserMessage::new(UserContent::Text("hi".into()), 0)),
				Message::assistant(empty_assistant),
			],
			None,
		);
		let converted = convert_messages(&context, &claude, "none").unwrap();
		assert_eq!(converted.len(), 1);
	}

	#[test]
	fn thinking_replay_uses_signatures_only_for_claude() {
		let claude = model("global.anthropic.claude-opus-4-6-v1", "");
		let nova = model("amazon.nova-2-lite-v1:0", "Nova 2 Lite");

		let mut assistant = AssistantMessage::new("bedrock-converse-stream", "amazon-bedrock", &claude.id, 0);
		let mut signed = ThinkingContent::new("reasoning");
		signed.thinking_signature = Some("sig".to_string());
		let mut unsigned = ThinkingContent::new("reasoning without signature");
		unsigned.thinking_signature = Some("  ".to_string());
		assistant.content = vec![
			ContentBlock::Thinking(signed),
			ContentBlock::Thinking(unsigned),
			ContentBlock::Text(TextContent::new("   ")),
		];
		let nova_assistant = AssistantMessage { model: nova.id.clone(), ..assistant.clone() };
		let context = Context::new(None, vec![Message::assistant(assistant)], None);

		let converted = convert_messages(&context, &claude, "none").unwrap();
		let content = converted[0]["content"].as_array().unwrap();
		assert_eq!(content.len(), 2);
		assert_eq!(content[0]["reasoningContent"]["reasoningText"]["signature"], json!("sig"));
		assert_eq!(content[1]["text"], json!("reasoning without signature"));

		let converted = convert_messages(&context, &nova, "none").unwrap();
		assert_eq!(converted[0]["content"][0], json!({ "text": "reasoning" }));

		let nova_context = Context::new(None, vec![Message::assistant(nova_assistant)], None);
		let converted = convert_messages(&nova_context, &nova, "none").unwrap();
		let content = converted[0]["content"].as_array().unwrap();
		assert_eq!(content.len(), 2);
		assert_eq!(content[0]["reasoningContent"]["reasoningText"]["text"], json!("reasoning"));
		assert!(content[0]["reasoningContent"]["reasoningText"].get("signature").is_none());
	}

	#[test]
	fn tool_config_matches_tool_choice() {
		let tools = vec![Tool {
			name: "read".to_string(),
			description: "Read a file".to_string(),
			parameters: json!({ "type": "object", "properties": {} }),
		}];

		assert!(convert_tool_config(None, None).is_none());
		assert!(convert_tool_config(Some(&Vec::new()), None).is_none());
		assert!(convert_tool_config(
			Some(&tools),
			Some(&BedrockToolChoice::Literal("none".to_string()))
		)
		.is_none());

		let config = convert_tool_config(Some(&tools), None).unwrap();
		assert_eq!(config["tools"][0]["toolSpec"]["name"], json!("read"));
		assert_eq!(
			config["tools"][0]["toolSpec"]["inputSchema"],
			json!({ "json": { "type": "object", "properties": {} } })
		);
		assert!(config["toolChoice"].is_null());

		let config = convert_tool_config(Some(&tools), Some(&BedrockToolChoice::Literal("auto".to_string()))).unwrap();
		assert_eq!(config["toolChoice"], json!({ "auto": {} }));

		let config = convert_tool_config(Some(&tools), Some(&BedrockToolChoice::Literal("any".to_string()))).unwrap();
		assert_eq!(config["toolChoice"], json!({ "any": {} }));

		let config = convert_tool_config(
			Some(&tools),
			Some(&BedrockToolChoice::Tool {
				type_: "tool".to_string(),
				name: "read".to_string(),
			}),
		)
		.unwrap();
		assert_eq!(config["toolChoice"], json!({ "tool": { "name": "read" } }));

		// A tool choice object with another type is ignored.
		let config = convert_tool_config(
			Some(&tools),
			Some(&BedrockToolChoice::Tool {
				type_: "other".to_string(),
				name: "read".to_string(),
			}),
		)
		.unwrap();
		assert!(config["toolChoice"].is_null());
	}

	#[test]
	fn command_input_serializes_in_sdk_order() {
		let claude = model("global.anthropic.claude-opus-4-6-v1", "");
		let options = BedrockOptions {
			stream: StreamOptions {
				max_tokens: Some(2048.0),
				temperature: Some(0.5),
				..Default::default()
			},
			reasoning: Some("high".to_string()),
			request_metadata: Some(IndexMap::from([("team".to_string(), "core".to_string())])),
			..Default::default()
		};
		let context = Context::new(Some("You are helpful.".to_string()), user_context("hello").messages, None);
		let input = build_command_input(&context, &claude, &options, "short").unwrap();
		let body = input.to_json();
		let keys: Vec<&String> = body.as_object().unwrap().keys().collect();
		assert_eq!(
			keys,
			vec![
				"modelId",
				"messages",
				"system",
				"inferenceConfig",
				"additionalModelRequestFields",
				"requestMetadata"
			]
		);
		assert_eq!(body["modelId"], json!("global.anthropic.claude-opus-4-6-v1"));
		assert_eq!(body["inferenceConfig"]["maxTokens"], json!(2048.0));
		assert_eq!(body["inferenceConfig"]["temperature"], json!(0.5));
		assert_eq!(body["requestMetadata"], json!({ "team": "core" }));
		assert_eq!(body["system"].as_array().unwrap().len(), 2);
		assert!(body.get("toolConfig").is_none());

		// The wire body drops `modelId` (it is an httpLabel member of the request).
		let wire = input.to_wire_json();
		let wire_keys: Vec<&String> = wire.as_object().unwrap().keys().collect();
		assert_eq!(
			wire_keys,
			vec![
				"messages",
				"system",
				"inferenceConfig",
				"additionalModelRequestFields",
				"requestMetadata"
			]
		);

		// Fable 5 drops temperature.
		let fable = model("global.anthropic.claude-fable-5", "Claude Fable 5");
		let input = build_command_input(&context, &fable, &options, "short").unwrap();
		let body = input.to_json();
		assert!(body["inferenceConfig"].get("temperature").is_none());
		assert_eq!(body["inferenceConfig"]["maxTokens"], json!(2048.0));
	}

	#[test]
	fn sigv4_helpers_match_published_vectors() {
		assert_eq!(hmac_sha256(&[0x0b; 20], b"Hi There").len(), 32);
		assert_eq!(
			hex_encode(&hmac_sha256(&[0x0b; 20], b"Hi There")),
			"b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"
		);
		assert_eq!(
			sha256_hex(b""),
			"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
		);

		// IAM ListUsers example with Smithy's default payload-checksum header.
		// Its published 5d672d79... signature omits x-amz-content-sha256;
		// including that signed header yields the independently computed value below.
		let credentials = AwsCredentials {
			access_key_id: "AKIDEXAMPLE".to_string(),
			secret_access_key: "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY".to_string(),
			session_token: None,
		};
		let mut headers = IndexMap::new();
		headers.insert("content-type".to_string(), "application/x-www-form-urlencoded; charset=utf-8".to_string());
		headers.insert("host".to_string(), "iam.amazonaws.com".to_string());
		let query = vec![
			("Action".to_string(), "ListUsers".to_string()),
			("Version".to_string(), "2010-05-08".to_string()),
		];
		let signing_date = chrono::DateTime::parse_from_rfc3339("2015-08-30T12:36:00Z")
			.unwrap()
			.timestamp_millis();
		let signed = sign_request_for_service(
			"GET",
			"/",
			&query,
			&headers,
			"",
			"us-east-1",
			"iam",
			&credentials,
			signing_date,
		);
		assert_eq!(signed.headers["x-amz-date"], "20150830T123600Z");
		assert_eq!(
			signed.headers["authorization"],
			concat!(
				"AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/iam/aws4_request, ",
				"SignedHeaders=content-type;host;x-amz-content-sha256;x-amz-date, ",
				"Signature=dd479fa8a80364edf2119ec24bebde66712ee9c9cb2b0d92eb3ab9ccdc0c3947"
			)
		);
		assert_eq!(
			signed.headers["x-amz-content-sha256"],
			"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
		);
	}

	#[test]
	fn signing_adds_session_token_and_payload_hash() {
		let credentials = AwsCredentials {
			access_key_id: "AKID".to_string(),
			secret_access_key: "secret".to_string(),
			session_token: Some("token".to_string()),
		};
		let mut headers = IndexMap::new();
		headers.insert("host".to_string(), "bedrock-runtime.us-east-1.amazonaws.com".to_string());
		headers.insert("content-type".to_string(), "application/json".to_string());
		let body = "{\"modelId\":\"m\"}";
		let signed = sign_request(
			"POST",
			"/model/m/converse-stream",
			&[],
			&headers,
			body,
			"us-east-1",
			&credentials,
			0,
		);
		assert_eq!(signed.headers["x-amz-security-token"], "token");
		assert_eq!(signed.headers["x-amz-content-sha256"], sha256_hex(body.as_bytes()));
		assert!(signed.headers["authorization"].contains("Credential=AKID/19700101/us-east-1/bedrock/aws4_request"));
		assert!(signed.headers["authorization"].contains("SignedHeaders=content-type;host;x-amz-content-sha256;x-amz-date;x-amz-security-token"));
	}

	#[test]
	fn converse_path_escapes_model_ids() {
		assert_eq!(converse_stream_path("model-id"), "/model/model-id/converse-stream");
		assert_eq!(
			converse_stream_path("arn:aws:bedrock:us-east-1:1:inference-profile/x"),
			"/model/arn%3Aaws%3Abedrock%3Aus-east-1%3A1%3Ainference-profile%2Fx/converse-stream"
		);
		assert_eq!(
			converse_stream_path("global.anthropic.claude-opus-4-7"),
			"/model/global.anthropic.claude-opus-4-7/converse-stream"
		);
	}

	#[test]
	fn options_round_trip_through_serde_keeps_camel_case_keys() {
		let options = BedrockOptions {
			stream: StreamOptions {
				max_tokens: Some(10.0),
				..Default::default()
			},
			region: Some("us-west-2".to_string()),
			tool_choice: Some(BedrockToolChoice::Literal("any".to_string())),
			thinking_display: Some("omitted".to_string()),
			request_metadata: Some(IndexMap::from([("a".to_string(), "b".to_string())])),
			bearer_token: Some("token".to_string()),
			..Default::default()
		};
		let value = serde_json::to_value(&options).unwrap();
		assert_eq!(value["maxTokens"], json!(10.0));
		assert_eq!(value["region"], json!("us-west-2"));
		assert_eq!(value["toolChoice"], json!("any"));
		assert_eq!(value["thinkingDisplay"], json!("omitted"));
		assert_eq!(value["requestMetadata"], json!({ "a": "b" }));
		assert_eq!(value["bearerToken"], json!("token"));
		assert!(value.get("reasoning").is_none());

		let parsed: BedrockOptions = serde_json::from_value(value).unwrap();
		assert_eq!(parsed.region, options.region);
		assert_eq!(parsed.stream.max_tokens, Some(10.0));

		let tool_choice: BedrockToolChoice =
			serde_json::from_value(json!({ "type": "tool", "name": "read" })).unwrap();
		assert_eq!(
			tool_choice,
			BedrockToolChoice::Tool {
				type_: "tool".to_string(),
				name: "read".to_string()
			}
		);

		// from_base keeps the non-serialisable fields.
		let mut base = StreamOptions::default();
		let token = tokio_util::sync::CancellationToken::new();
		base.signal = Some(token.clone());
		let typed = BedrockOptions::from_base(&base);
		assert!(typed.stream.signal.is_some());
		token.cancel();
		assert!(typed.stream.signal.unwrap().is_cancelled());
	}

	fn frame(headers: &[(&str, &str)], payload: &[u8]) -> Vec<u8> {
		let mut header_bytes: Vec<u8> = Vec::new();
		for (name, value) in headers {
			header_bytes.push(name.len() as u8);
			header_bytes.extend_from_slice(name.as_bytes());
			header_bytes.push(7);
			header_bytes.extend_from_slice(&(value.len() as u16).to_be_bytes());
			header_bytes.extend_from_slice(value.as_bytes());
		}
		let total_length = 16 + header_bytes.len() + payload.len();
		let mut out = Vec::with_capacity(total_length);
		out.extend_from_slice(&(total_length as u32).to_be_bytes());
		out.extend_from_slice(&(header_bytes.len() as u32).to_be_bytes());
		out.extend_from_slice(&[0, 0, 0, 0]);
		out.extend_from_slice(&header_bytes);
		out.extend_from_slice(payload);
		out.extend_from_slice(&[0, 0, 0, 0]);
		out
	}

	#[test]
	fn event_stream_frames_decode_incrementally() {
		let first = frame(
			&[(":message-type", "event"), (":event-type", "messageStart")],
			br#"{"role":"assistant"}"#,
		);
		let second = frame(
			&[(":message-type", "event"), (":event-type", "contentBlockDelta")],
			br#"{"contentBlockIndex":0,"delta":{"text":"hi"}}"#,
		);

		let mut buffer = Vec::new();
		buffer.extend_from_slice(&first);
		buffer.extend_from_slice(&second[..second.len() - 3]);
		let frames = decode_event_stream_frames(&mut buffer).unwrap();
		assert_eq!(frames.len(), 1);
		assert_eq!(frames[0].headers[":event-type"], "messageStart");
		assert_eq!(
			parse_bedrock_stream_event(&frames[0]).unwrap(),
			BedrockStreamEvent::MessageStart {
				role: "assistant".to_string()
			}
		);

		// The incomplete tail stays buffered until the rest arrives.
		buffer.extend_from_slice(&second[second.len() - 3..]);
		let frames = decode_event_stream_frames(&mut buffer).unwrap();
		assert_eq!(frames.len(), 1);
		assert_eq!(
			parse_bedrock_stream_event(&frames[0]).unwrap(),
			BedrockStreamEvent::ContentBlockDelta {
				content_block_index: 0,
				delta: json!({ "text": "hi" })
			}
		);
		assert!(buffer.is_empty());
	}

	#[test]
	fn event_stream_rejects_invalid_lengths_and_truncated_headers() {
		for total in [0u32, 1, 4, 15, 16 * 1024 * 1024 + 1, u32::MAX] {
			assert!(decode_event_stream_frames(&mut total.to_be_bytes().to_vec()).is_err());
		}
		let mut oversized_headers = frame(&[], b"{}");
		oversized_headers[4..8].copy_from_slice(&u32::MAX.to_be_bytes());
		assert!(decode_event_stream_frames(&mut oversized_headers).is_err());

		let malformed: &[&[u8]] = &[
			&[1, b'x'], // missing type
			&[2, b'x'], // incomplete name
			&[1, b'x', 7], // missing string length
			&[1, b'x', 7, 0],
			&[1, b'x', 7, 0, 4, b'y'], // value extends into payload/CRC
			&[1, b'x', 6], // missing byte-array length
			&[1, b'x', 6, 0, 4, b'y'],
			&[1, b'x', 2], &[1, b'x', 3, 0],
			&[1, b'x', 4, 0], &[1, b'x', 5, 0],
			&[1, b'x', 8, 0], &[1, b'x', 9, 0],
		];
		for headers in malformed {
			let mut bytes = frame(&[], b"payload large enough to hide header overflow");
			bytes.splice(12..12, headers.iter().copied());
			let total = bytes.len() as u32;
			bytes[..4].copy_from_slice(&total.to_be_bytes());
			bytes[4..8].copy_from_slice(&(headers.len() as u32).to_be_bytes());
			assert!(decode_event_stream_frames(&mut bytes).is_err(), "headers: {headers:?}");
		}
	}

	#[test]
	fn event_stream_preserves_frames_split_at_every_byte_boundary() {
		let bytes = frame(&[(":event-type", "messageStart")], br#"{"role":"assistant"}"#);
		for split in 0..bytes.len() {
			let mut buffer = bytes[..split].to_vec();
			assert!(decode_event_stream_frames(&mut buffer).unwrap().is_empty());
			assert_eq!(buffer, bytes[..split]);
			buffer.extend_from_slice(&bytes[split..]);
			assert_eq!(decode_event_stream_frames(&mut buffer).unwrap().len(), 1);
			assert!(buffer.is_empty());
		}
	}

	#[test]
	fn event_stream_parses_every_converse_stream_member() {
		let cases: Vec<(Vec<u8>, BedrockStreamEvent)> = vec![
			(
				frame(&[(":event-type", "contentBlockStart")], br#"{"contentBlockIndex":1,"start":{"toolUse":{"toolUseId":"t","name":"read"}}}"#),
				BedrockStreamEvent::ContentBlockStart {
					content_block_index: 1,
					start: json!({ "toolUse": { "toolUseId": "t", "name": "read" } }),
				},
			),
			(
				frame(&[(":event-type", "contentBlockStop")], br#"{"contentBlockIndex":2}"#),
				BedrockStreamEvent::ContentBlockStop { content_block_index: 2 },
			),
			(
				frame(&[(":event-type", "messageStop")], br#"{"stopReason":"tool_use"}"#),
				BedrockStreamEvent::MessageStop {
					stop_reason: Some("tool_use".to_string()),
				},
			),
			(
				frame(&[(":event-type", "metadata")], br#"{"usage":{"inputTokens":7}}"#),
				BedrockStreamEvent::Metadata {
					usage: json!({ "inputTokens": 7 }),
				},
			),
		];

		for (bytes, expected) in cases {
			let mut buffer = bytes;
			let frames = decode_event_stream_frames(&mut buffer).unwrap();
			assert_eq!(parse_bedrock_stream_event(&frames[0]).unwrap(), expected);
		}
	}

	#[test]
	fn event_stream_exceptions_carry_the_exception_type() {
		let mut buffer = frame(
			&[
				(":message-type", "exception"),
				(":exception-type", "ThrottlingException"),
			],
			br#"{"message":"slow down"}"#,
		);
		let frames = decode_event_stream_frames(&mut buffer).unwrap();
		assert_eq!(
			parse_bedrock_stream_event(&frames[0]).unwrap(),
			BedrockStreamEvent::Exception {
				name: "ThrottlingException".to_string(),
				message: "slow down".to_string()
			}
		);
	}

	#[tokio::test]
	async fn block_start_and_stop_emit_the_same_event_order() {
		let stream = create_assistant_message_event_stream();
		let mut output = AssistantMessage::default();
		let mut blocks: Vec<BlockMeta> = Vec::new();

		handle_content_block_start(
			0,
			&json!({ "toolUse": { "toolUseId": "tool-1", "name": "read" } }),
			&mut blocks,
			&mut output,
			&stream,
		);
		handle_content_block_delta(
			0,
			&json!({ "toolUse": { "input": "{\"path\":" } }),
			&mut blocks,
			&mut output,
			&stream,
		);
		handle_content_block_delta(0, &json!({ "toolUse": { "input": "\"a.txt\"}" } }), &mut blocks, &mut output, &stream);
		handle_content_block_stop(0, &mut blocks, &mut output, &stream);
		// The response producer owns stream completion; a block stop does not end it.
		stream.end(None);

		let mut events: Vec<String> = Vec::new();
		while let Some(event) = stream.next().await {
			events.push(event.event_type().to_string());
		}
		assert_eq!(events, vec!["toolcall_start", "toolcall_delta", "toolcall_delta", "toolcall_end"]);

		match output.content.first() {
			Some(ContentBlock::ToolCall(tool_call)) => {
				assert_eq!(tool_call.id, "tool-1");
				assert_eq!(tool_call.name, "read");
				assert_eq!(tool_call.arguments.get("path"), Some(&json!("a.txt")));
			}
			other => panic!("unexpected block {:?}", other),
		}
	}

	#[tokio::test]
	async fn text_and_thinking_deltas_create_blocks_lazily() {
		let stream = create_assistant_message_event_stream();
		let mut output = AssistantMessage::default();
		let mut blocks: Vec<BlockMeta> = Vec::new();

		handle_content_block_delta(0, &json!({ "text": "he" }), &mut blocks, &mut output, &stream);
		handle_content_block_delta(0, &json!({ "text": "llo" }), &mut blocks, &mut output, &stream);
		handle_content_block_stop(0, &mut blocks, &mut output, &stream);

		handle_content_block_delta(
			1,
			&json!({ "reasoningContent": { "text": "think", "signature": "sig" } }),
			&mut blocks,
			&mut output,
			&stream,
		);
		handle_content_block_stop(1, &mut blocks, &mut output, &stream);
		stream.end(None);

		let mut events: Vec<String> = Vec::new();
		while let Some(event) = stream.next().await {
			events.push(event.event_type().to_string());
		}
		assert_eq!(
			events,
			vec![
				"text_start",
				"text_delta",
				"text_delta",
				"text_end",
				"thinking_start",
				"thinking_delta",
				"thinking_end"
			]
		);

		assert_eq!(output.content.len(), 2);
		match &output.content[0] {
			ContentBlock::Text(text) => assert_eq!(text.text, "hello"),
			other => panic!("unexpected block {:?}", other),
		}
		match &output.content[1] {
			ContentBlock::Thinking(thinking) => {
				assert_eq!(thinking.thinking, "think");
				assert_eq!(thinking.thinking_signature.as_deref(), Some("sig"));
			}
			other => panic!("unexpected block {:?}", other),
		}
	}

	#[test]
	fn metadata_updates_usage_and_cost() {
		let mut priced = model("global.anthropic.claude-opus-4-6-v1", "");
		priced.cost = ModelCost {
			input: 3.0,
			output: 15.0,
			cache_read: 0.3,
			cache_write: 3.75,
		};
		let mut output = AssistantMessage::default();

		handle_metadata(
			&json!({
				"inputTokens": 1_000_000.0,
				"outputTokens": 1_000_000.0,
				"cacheReadInputTokens": 1_000_000.0,
				"cacheWriteInputTokens": 1_000_000.0
			}),
			&priced,
			&mut output,
		);
		assert_eq!(output.usage.input, 1_000_000.0);
		assert_eq!(output.usage.output, 1_000_000.0);
		assert_eq!(output.usage.cache_read, 1_000_000.0);
		assert_eq!(output.usage.cache_write, 1_000_000.0);
		assert_eq!(output.usage.total_tokens, 2_000_000.0);
		assert_eq!(output.usage.cost.input, 3.0);
		assert_eq!(output.usage.cost.output, 15.0);
		assert_eq!(output.usage.cost.cache_read, 0.3);
		assert_eq!(output.usage.cost.cache_write, 3.75);
		assert_eq!(output.usage.cost.total, 22.05);

		// Missing counters default to 0 and totalTokens falls back to input + output.
		let mut output = AssistantMessage::default();
		handle_metadata(&json!({ "inputTokens": 5, "outputTokens": 7 }), &priced, &mut output);
		assert_eq!(output.usage.cache_read, 0.0);
		assert_eq!(output.usage.total_tokens, 12.0);

		// `event.usage` absent -> no update at all.
		let mut output = AssistantMessage::default();
		handle_metadata(&Value::Null, &priced, &mut output);
		assert_eq!(output.usage.total_tokens, 0.0);
	}

	#[test]
	fn client_config_pins_standard_endpoints_only_without_region_or_profile() {
		// Pure helpers used by resolve_bedrock_client_config.
		assert!(should_use_explicit_bedrock_endpoint(
			"https://bedrock-runtime.eu-central-1.amazonaws.com",
			None,
			false
		));
		assert!(!should_use_explicit_bedrock_endpoint(
			"https://bedrock-runtime.eu-central-1.amazonaws.com",
			Some("us-east-2"),
			false
		));
		assert!(should_use_explicit_bedrock_endpoint("https://bedrock-vpc.example.com", None, false));

		// The endpoint used for the signed request follows the same rule.
		let config = BedrockClientConfig {
			endpoint: None,
			..Default::default()
		};
		let endpoint = resolve_endpoint(&config, "us-east-1");
		assert_eq!(endpoint.host, "bedrock-runtime.us-east-1.amazonaws.com");
		assert_eq!(endpoint.origin, "https://bedrock-runtime.us-east-1.amazonaws.com");

		let config = BedrockClientConfig {
			endpoint: Some("https://bedrock-vpc.example.com:8443".to_string()),
			..Default::default()
		};
		let endpoint = resolve_endpoint(&config, "us-east-1");
		assert_eq!(endpoint.host, "bedrock-vpc.example.com");
		assert_eq!(endpoint.port, Some(8443));
		assert_eq!(endpoint.origin, "https://bedrock-vpc.example.com:8443");
		assert_eq!(
			host_header(&endpoint),
			"bedrock-vpc.example.com:8443"
		);
	}

	/// Test-only serialization for the process-global `AWS_*` / proxy variables, shared with the
	/// other pi-ai modules that read them.
	///
	/// The lock is the crate-wide one from [`crate::test_env`], so the `AWS_*` tests in
	/// `env_api_keys.rs`, `bedrock_responses_client.rs` and `amazon_bedrock_responses.rs`
	/// cannot interleave with this one.
	struct BedrockClientEnv {
		env: crate::test_env::ScopedEnv,
	}

	const BEDROCK_CLIENT_ENV_NAMES: [&str; 11] = [
		"AWS_REGION",
		"AWS_DEFAULT_REGION",
		"AWS_PROFILE",
		"AWS_BEDROCK_SKIP_AUTH",
		"AWS_BEDROCK_FORCE_HTTP1",
		"HTTP_PROXY",
		"HTTPS_PROXY",
		"NO_PROXY",
		"http_proxy",
		"https_proxy",
		"no_proxy",
	];

	impl BedrockClientEnv {
		fn new() -> Self {
			// ScopedEnv holds the process-wide environment lock for the whole test body and
			// journals each previous value (or its absence) for restore on drop.
			let mut env = crate::test_env::ScopedEnv::new();
			for name in BEDROCK_CLIENT_ENV_NAMES {
				env.remove(name);
			}
			Self { env }
		}

		/// Set `name` while the lock is held, journaled for restore on drop.
		///
		/// The only test that uses this guard writes `AWS_BEDROCK_FORCE_HTTP1` and
		/// `HTTPS_PROXY`, so no `remove` helper is needed here.
		fn set(&mut self, name: &str, value: impl AsRef<std::ffi::OsStr>) {
			self.env.set(name, value);
		}
	}

	#[test]
	fn force_http1_pins_the_client_to_http1() {
		// TS: `config.requestHandler = new NodeHttpHandler()` under `AWS_BEDROCK_FORCE_HTTP1=1`
		// (amazon-bedrock.ts:176-178). reqwest exposes the same choice on the client, so the
		// builder must carry it: `http1_only()` shows up as `http1_only: true` in the Debug form.
		let default_builder = format!(
			"{:?}",
			bedrock_http_client_builder(&BedrockClientConfig::default())
		);
		assert!(
			!default_builder.contains("http1_only"),
			"without AWS_BEDROCK_FORCE_HTTP1 the client must keep the default protocol negotiation: {default_builder}"
		);

		let pinned = format!(
			"{:?}",
			bedrock_http_client_builder(&BedrockClientConfig {
				force_http1: true,
				..Default::default()
			})
		);
		assert!(
			pinned.contains("http1_only: true"),
			"AWS_BEDROCK_FORCE_HTTP1 must pin the client to HTTP/1.1: {pinned}"
		);
	}

	#[test]
	fn aws_bedrock_force_http1_env_reaches_the_client_builder() {
		let mut env = BedrockClientEnv::new();
		let model = model("global.anthropic.claude-opus-4-6-v1", "Claude Opus 4.6");

		// No flag: the resolved config keeps HTTP/2 available.
		let config = resolve_bedrock_client_config(&model, &BedrockOptions::default());
		assert!(!config.force_http1);
		assert!(!format!("{:?}", bedrock_http_client_builder(&config)).contains("http1_only"));

		env.set("AWS_BEDROCK_FORCE_HTTP1", "1");
		let config = resolve_bedrock_client_config(&model, &BedrockOptions::default());
		assert!(
			config.force_http1,
			"AWS_BEDROCK_FORCE_HTTP1=1 must set config.force_http1 (amazon-bedrock.ts:176)"
		);
		assert!(
			format!("{:?}", bedrock_http_client_builder(&config)).contains("http1_only: true"),
			"the parsed flag must reach the client builder; otherwise the setting is parsed and dropped"
		);

		// `else if` in the TypeScript: a proxy handler wins, so the flag is not applied.
		env.set("HTTPS_PROXY", "http://proxy.example.com:3128");
		let config = resolve_bedrock_client_config(&model, &BedrockOptions::default());
		assert!(config.use_proxy_env);
		assert!(!config.force_http1);
	}

	#[test]
	fn ini_lookup_reads_profiles() {
		let contents = "[default]\nregion = us-east-1\n\n[profile eu]\nregion = eu-central-1\n";
		assert_eq!(read_ini_value(contents, "default", "region"), Some("us-east-1".to_string()));
		assert_eq!(
			read_ini_value(contents, "profile eu", "region"),
			Some("eu-central-1".to_string())
		);
		assert_eq!(read_ini_value(contents, "profile missing", "region"), None);
		assert_eq!(read_ini_value(contents, "default", "output"), None);
	}

	#[test]
	fn exception_names_prefer_the_response_header() {
		let mut headers = reqwest::header::HeaderMap::new();
		headers.insert("x-amzn-errortype", "ThrottlingException:http://internal".parse().unwrap());
		assert_eq!(
			exception_name_from_body(400, &headers, &json!({})),
			"ThrottlingException"
		);

		let headers = reqwest::header::HeaderMap::new();
		assert_eq!(
			exception_name_from_body(400, &headers, &json!({ "__type": "com.amazon#ValidationException" })),
			"ValidationException"
		);
		assert_eq!(exception_name_from_body(503, &headers, &json!({})), "ServiceUnavailableException");
		assert_eq!(exception_name_from_body(418, &headers, &json!({})), "BedrockRuntimeServiceException");
	}

	#[test]
	fn request_id_comes_from_the_requestid_header() {
		let mut headers = reqwest::header::HeaderMap::new();
		headers.insert("x-amzn-requestid", "abc-123".parse().unwrap());
		assert_eq!(request_id_from_headers(&headers).as_deref(), Some("abc-123"));

		let headers = reqwest::header::HeaderMap::new();
		assert_eq!(request_id_from_headers(&headers), None);
	}
}
