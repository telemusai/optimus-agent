//! Port of packages/ai/src/providers/register-builtins.ts
//!
//! The TypeScript registers each provider behind a lazy dynamic `import()`. Rust has no dynamic
//! imports, so `load*ProviderModule` returns the statically linked module directly. Everything
//! else (the lazy stream wrappers, the error message when a module cannot be loaded, the bedrock
//! override, the registration order) is preserved.
use std::sync::{Arc, OnceLock, RwLock};

use crate::api_registry::{
	clear_api_providers, get_api_providers, register_api_provider_simple, ApiProviderSimple,
};
use crate::compaction::CompactionOptions;
use crate::types::CompactFunction;
use crate::api_registry::SimpleStreamFunction;
use crate::types::{
	AssistantMessage, Context, Model, SimpleStreamOptions, StreamFunction, StreamOptions, Usage,
};
use crate::utils::event_stream::{create_assistant_message_event_stream, AssistantMessageEventStream};

use super::amazon_bedrock::{stream_bedrock, stream_simple_bedrock, BedrockOptions};
use super::amazon_bedrock_responses::{
	stream_bedrock_responses_with_options, stream_simple_bedrock_responses_with_options, BedrockResponsesOptions,
};
use super::anthropic::{stream_anthropic as stream_anthropic_provider, stream_simple_anthropic, AnthropicOptions};
use super::azure_openai_responses::{
	stream_azure_openai_responses as stream_azure_openai_responses_provider, stream_simple_azure_openai_responses,
	AzureOpenAIResponsesOptions,
};
use super::google::{stream_google as stream_google_provider, stream_simple_google, GoogleOptions};
use super::google_vertex::{
	stream_google_vertex as stream_google_vertex_provider, stream_simple_google_vertex, GoogleVertexOptions,
};
use super::mistral::{stream_mistral as stream_mistral_provider, stream_simple_mistral, MistralOptions};
use super::openai_codex_responses::{
	compact_openai_codex_responses, stream_openai_codex_responses as stream_openai_codex_responses_provider,
	stream_simple_openai_codex_responses, OpenAICodexResponsesOptions,
};
use super::openai_compaction::supports_openai_compaction;
use super::openai_completions::{
	stream_openai_completions as stream_openai_completions_provider, stream_simple_openai_completions,
	OpenAICompletionsOptions,
};
use super::openai_responses::{
	compact_openai_responses, stream_openai_responses as stream_openai_responses_provider,
	stream_simple_openai_responses, OpenAIResponsesOptions,
};

/// TS: `LazyProviderModule<TApi, TOptions, TSimpleOptions>`
#[derive(Clone)]
pub struct LazyProviderModule {
	pub compact: Option<CompactFunction>,
	pub stream: StreamFunction,
	/// TS types this `TSimpleOptions` (register-builtins.ts:33-37), so the caller's
	/// `SimpleStreamOptions` reach the provider unchanged.
	pub stream_simple: SimpleStreamFunction,
}

/// TS: `BedrockResponsesProviderModule`
///
/// TS types `streamSimpleBedrockResponses` as `StreamFunction<Api, SimpleStreamOptions>`
/// (register-builtins.ts:82-85). The Rust field stays base-shaped because the only
/// implementer, `crates/pi-coding-agent/src/bun/register_bedrock.rs:25-29`, is outside this
/// file and still declares `Option<&StreamOptions>`; [`as_simple_stream`] adapts it and the
/// `reasoning`/`thinkingBudgets` of that override path are lost until that file is updated.
#[derive(Clone)]
pub struct BedrockResponsesProviderModule {
	pub stream_bedrock_responses: StreamFunction,
	pub stream_simple_bedrock_responses: StreamFunction,
}

/// TS: `BedrockProviderModule`
/// TS: `BedrockProviderModule` (register-builtins.ts:87-95). See
/// [`BedrockResponsesProviderModule`] for why `stream_simple_bedrock` stays base-shaped.
#[derive(Clone)]
pub struct BedrockProviderModule {
	pub responses: Option<BedrockResponsesProviderModule>,
	pub stream_bedrock: StreamFunction,
	pub stream_simple_bedrock: StreamFunction,
}

fn bedrock_provider_module_override() -> &'static RwLock<Option<LazyProviderModule>> {
	static OVERRIDE: OnceLock<RwLock<Option<LazyProviderModule>>> = OnceLock::new();
	OVERRIDE.get_or_init(|| RwLock::new(None))
}

fn bedrock_responses_module_override() -> &'static RwLock<Option<BedrockResponsesProviderModule>> {
	static OVERRIDE: OnceLock<RwLock<Option<BedrockResponsesProviderModule>>> = OnceLock::new();
	OVERRIDE.get_or_init(|| RwLock::new(None))
}

/// Convert base `StreamOptions` into a provider-specific options struct.
///
/// The TypeScript passes `StreamOptions & Record<string, unknown>` and the provider reads the
/// fields it knows. The Rust port keeps the base options object itself (so `signal`,
/// `on_payload`, `on_response` and `on_usage_observation`, which are not serialisable, survive)
/// and lets the provider add its own defaults.
trait FromBaseOptions: Sized {
	fn from_base_options(base: &StreamOptions) -> Self;
}

impl FromBaseOptions for AnthropicOptions {
	fn from_base_options(base: &StreamOptions) -> Self {
		AnthropicOptions::from_base(base)
	}
}

impl FromBaseOptions for OpenAICompletionsOptions {
	fn from_base_options(base: &StreamOptions) -> Self {
		OpenAICompletionsOptions::from_base(base)
	}
}

impl FromBaseOptions for OpenAIResponsesOptions {
	fn from_base_options(base: &StreamOptions) -> Self {
		OpenAIResponsesOptions::from_base(base)
	}
}

impl FromBaseOptions for AzureOpenAIResponsesOptions {
	fn from_base_options(base: &StreamOptions) -> Self {
		AzureOpenAIResponsesOptions::from_base(base)
	}
}

impl FromBaseOptions for OpenAICodexResponsesOptions {
	fn from_base_options(base: &StreamOptions) -> Self {
		OpenAICodexResponsesOptions::from_base(base)
	}
}

impl FromBaseOptions for BedrockOptions {
	fn from_base_options(base: &StreamOptions) -> Self {
		BedrockOptions::from_base(base)
	}
}

impl FromBaseOptions for BedrockResponsesOptions {
	fn from_base_options(base: &StreamOptions) -> Self {
		BedrockResponsesOptions::from_base(base)
	}
}

impl FromBaseOptions for GoogleOptions {
	fn from_base_options(base: &StreamOptions) -> Self {
		GoogleOptions::from_base(base)
	}
}

impl FromBaseOptions for GoogleVertexOptions {
	fn from_base_options(base: &StreamOptions) -> Self {
		GoogleVertexOptions::from_base(base)
	}
}

impl FromBaseOptions for MistralOptions {
	fn from_base_options(base: &StreamOptions) -> Self {
		MistralOptions::from_base(base)
	}
}

fn typed_options<T>(options: Option<&StreamOptions>) -> Option<T>
where
	T: FromBaseOptions,
{
	options.map(T::from_base_options)
}

/// TS: `streamSimple: (model, context, options?: TSimpleOptions) => ...`
/// (register-builtins.ts:33-37). The TypeScript hands the caller's `SimpleStreamOptions`
/// object straight to the provider, so `reasoning` / `thinkingBudgets` must survive:
/// `wrapStreamSimple` only checks the api (api-registry.ts:63-68).
fn simple_options(options: Option<&SimpleStreamOptions>) -> Option<SimpleStreamOptions> {
	options.cloned()
}

/// Adapts a base-shaped `streamSimple` to the faithful `SimpleStreamOptions` signature.
/// Only the bedrock module override still uses it (see [`BedrockProviderModule`]).
fn as_simple_stream(stream: StreamFunction) -> SimpleStreamFunction {
	Arc::new(move |model: &Model, context: &Context, options: Option<&SimpleStreamOptions>| {
		stream(model, context, options.map(|options| &options.stream))
	})
}

/// TS: `supportsCompaction` guard used by `registerApiProvider` for the responses apis.
fn compaction_api_guard(api: &str, check: fn(&Model) -> bool) -> Arc<dyn Fn(&Model) -> bool + Send + Sync> {
	let api = api.to_string();
	Arc::new(move |model: &Model| model.api == api && check(model))
}

/// TS: the `compact` wrapper that throws `Mismatched compaction api: ${model.api}`.
fn compact_openai_responses_guarded() -> CompactFunction {
	Arc::new(|model: &Model, context: &Context, options: Option<&CompactionOptions>| {
		if model.api != "openai-responses" {
			panic!("Mismatched compaction api: {}", model.api);
		}
		let model = model.clone();
		let context = context.clone();
		let options = options.cloned();
		Box::pin(async move { compact_openai_responses(&model, &context, options.as_ref()).await })
	})
}

fn compact_openai_codex_responses_guarded() -> CompactFunction {
	Arc::new(|model: &Model, context: &Context, options: Option<&CompactionOptions>| {
		if model.api != "openai-codex-responses" {
			panic!("Mismatched compaction api: {}", model.api);
		}
		let model = model.clone();
		let context = context.clone();
		let options = options.cloned();
		Box::pin(async move { compact_openai_codex_responses(&model, &context, options.as_ref()).await })
	})
}

fn load_bedrock_responses_module() -> LazyProviderModule {
	if let Some(module) = bedrock_responses_module_override().read().unwrap().clone() {
		return LazyProviderModule {
			compact: None,
			stream: module.stream_bedrock_responses,
			stream_simple: as_simple_stream(module.stream_simple_bedrock_responses),
		};
	}
	LazyProviderModule {
		compact: None,
		stream: Arc::new(|model: &Model, context: &Context, options: Option<&StreamOptions>| {
			stream_bedrock_responses_with_options(model, context, typed_options::<BedrockResponsesOptions>(options))
		}),
		stream_simple: Arc::new(|model: &Model, context: &Context, options: Option<&SimpleStreamOptions>| {
			stream_simple_bedrock_responses_with_options(model, context, simple_options(options))
		}),
	}
}

/// TS: `streamBedrockResponses`
pub fn stream_bedrock_responses() -> StreamFunction {
	create_lazy_stream(load_bedrock_responses_module)
}

/// TS: `streamSimpleBedrockResponses`
pub fn stream_simple_bedrock_responses_lazy() -> SimpleStreamFunction {
	create_lazy_simple_stream(load_bedrock_responses_module)
}

/// TS: `setBedrockProviderModule(module)`
pub fn set_bedrock_provider_module(module: BedrockProviderModule) {
	let BedrockProviderModule {
		responses,
		stream_bedrock,
		stream_simple_bedrock,
	} = module;
	*bedrock_responses_module_override().write().unwrap() = responses;
	*bedrock_provider_module_override().write().unwrap() = Some(LazyProviderModule {
		compact: None,
		stream: stream_bedrock,
		stream_simple: as_simple_stream(stream_simple_bedrock),
	});
}

/// TS: `forwardStream(target, source)`
fn forward_stream(target: AssistantMessageEventStream, source: AssistantMessageEventStream) {
	let sink = target.clone();
	target.spawn(async move {
		while let Some(event) = source.next().await {
			sink.push(event);
		}
		sink.end(None);
	});
}

/// TS: `createLazyLoadErrorMessage(model, error)`
fn create_lazy_load_error_message(model: &Model, error: &str) -> AssistantMessage {
	AssistantMessage {
		content: Vec::new(),
		api: model.api.clone(),
		provider: model.provider.clone(),
		model: model.id.clone(),
		usage: Usage::zero(),
		stop_reason: "error".to_string(),
		error_message: Some(error.to_string()),
		timestamp: crate::utils::now_ms(),
		..Default::default()
	}
}

/// TS: `createLazyStream(loadModule)`
fn create_lazy_stream(load_module: fn() -> LazyProviderModule) -> StreamFunction {
	Arc::new(move |model: &Model, context: &Context, options: Option<&StreamOptions>| {
		let outer = create_assistant_message_event_stream();
		let module = load_module();
		let inner = (module.stream)(model, context, options);
		forward_stream(outer.clone(), inner);
		outer
	})
}

/// TS: `createLazySimpleStream(loadModule)` (register-builtins.ts:218-239) - the
/// module's `streamSimple` is invoked with the caller's `SimpleStreamOptions`.
fn create_lazy_simple_stream(load_module: fn() -> LazyProviderModule) -> SimpleStreamFunction {
	Arc::new(move |model: &Model, context: &Context, options: Option<&SimpleStreamOptions>| {
		let outer = create_assistant_message_event_stream();
		let module = load_module();
		let inner = (module.stream_simple)(model, context, options);
		forward_stream(outer.clone(), inner);
		outer
	})
}

fn load_anthropic_provider_module() -> LazyProviderModule {
	LazyProviderModule {
		compact: None,
		stream: Arc::new(|model: &Model, context: &Context, options: Option<&StreamOptions>| {
			stream_anthropic_provider(model, context, typed_options::<AnthropicOptions>(options))
		}),
		stream_simple: Arc::new(|model: &Model, context: &Context, options: Option<&SimpleStreamOptions>| {
			stream_simple_anthropic(model, context, simple_options(options))
		}),
	}
}

fn load_azure_openai_responses_provider_module() -> LazyProviderModule {
	LazyProviderModule {
		compact: None,
		stream: Arc::new(|model: &Model, context: &Context, options: Option<&StreamOptions>| {
			stream_azure_openai_responses_provider(model, context, typed_options::<AzureOpenAIResponsesOptions>(options))
		}),
		stream_simple: Arc::new(|model: &Model, context: &Context, options: Option<&SimpleStreamOptions>| {
			stream_simple_azure_openai_responses(model, context, simple_options(options))
		}),
	}
}

fn load_google_provider_module() -> LazyProviderModule {
	LazyProviderModule {
		compact: None,
		stream: Arc::new(|model: &Model, context: &Context, options: Option<&StreamOptions>| {
			stream_google_provider(model, context, typed_options::<GoogleOptions>(options))
		}),
		stream_simple: Arc::new(|model: &Model, context: &Context, options: Option<&SimpleStreamOptions>| {
			stream_simple_google(model, context, simple_options(options))
		}),
	}
}

fn load_google_vertex_provider_module() -> LazyProviderModule {
	LazyProviderModule {
		compact: None,
		stream: Arc::new(|model: &Model, context: &Context, options: Option<&StreamOptions>| {
			stream_google_vertex_provider(model, context, typed_options::<GoogleVertexOptions>(options))
		}),
		stream_simple: Arc::new(|model: &Model, context: &Context, options: Option<&SimpleStreamOptions>| {
			stream_simple_google_vertex(model, context, simple_options(options))
		}),
	}
}

fn load_mistral_provider_module() -> LazyProviderModule {
	LazyProviderModule {
		compact: None,
		stream: Arc::new(|model: &Model, context: &Context, options: Option<&StreamOptions>| {
			stream_mistral_provider(model, context, typed_options::<MistralOptions>(options))
		}),
		stream_simple: Arc::new(|model: &Model, context: &Context, options: Option<&SimpleStreamOptions>| {
			stream_simple_mistral(model, context, simple_options(options))
		}),
	}
}

fn load_openai_codex_responses_provider_module() -> LazyProviderModule {
	LazyProviderModule {
		compact: Some(compact_openai_codex_responses_guarded()),
		stream: Arc::new(|model: &Model, context: &Context, options: Option<&StreamOptions>| {
			stream_openai_codex_responses_provider(model, context, typed_options::<OpenAICodexResponsesOptions>(options))
		}),
		stream_simple: Arc::new(|model: &Model, context: &Context, options: Option<&SimpleStreamOptions>| {
			stream_simple_openai_codex_responses(model, context, simple_options(options))
		}),
	}
}

fn load_openai_completions_provider_module() -> LazyProviderModule {
	LazyProviderModule {
		compact: None,
		stream: Arc::new(|model: &Model, context: &Context, options: Option<&StreamOptions>| {
			stream_openai_completions_provider(model, context, typed_options::<OpenAICompletionsOptions>(options))
		}),
		stream_simple: Arc::new(|model: &Model, context: &Context, options: Option<&SimpleStreamOptions>| {
			stream_simple_openai_completions(model, context, simple_options(options))
		}),
	}
}

fn load_openai_responses_provider_module() -> LazyProviderModule {
	LazyProviderModule {
		compact: Some(compact_openai_responses_guarded()),
		stream: Arc::new(|model: &Model, context: &Context, options: Option<&StreamOptions>| {
			stream_openai_responses_provider(model, context, typed_options::<OpenAIResponsesOptions>(options))
		}),
		stream_simple: Arc::new(|model: &Model, context: &Context, options: Option<&SimpleStreamOptions>| {
			stream_simple_openai_responses(model, context, simple_options(options))
		}),
	}
}

fn load_bedrock_provider_module() -> LazyProviderModule {
	if let Some(module) = bedrock_provider_module_override().read().unwrap().clone() {
		return module;
	}
	LazyProviderModule {
		compact: None,
		stream: Arc::new(|model: &Model, context: &Context, options: Option<&StreamOptions>| {
			stream_bedrock(model, context, typed_options::<BedrockOptions>(options))
		}),
		stream_simple: Arc::new(|model: &Model, context: &Context, options: Option<&SimpleStreamOptions>| {
			stream_simple_bedrock(model, context, simple_options(options))
		}),
	}
}

/// TS: `streamAnthropic`
pub fn stream_anthropic() -> StreamFunction {
	create_lazy_stream(load_anthropic_provider_module)
}

/// TS: `streamSimpleAnthropic`
pub fn stream_simple_anthropic_lazy() -> SimpleStreamFunction {
	create_lazy_simple_stream(load_anthropic_provider_module)
}

/// TS: `streamAzureOpenAIResponses`
pub fn stream_azure_openai_responses() -> StreamFunction {
	create_lazy_stream(load_azure_openai_responses_provider_module)
}

/// TS: `streamSimpleAzureOpenAIResponses`
pub fn stream_simple_azure_openai_responses_lazy() -> SimpleStreamFunction {
	create_lazy_simple_stream(load_azure_openai_responses_provider_module)
}

/// TS: `streamGoogle`
pub fn stream_google() -> StreamFunction {
	create_lazy_stream(load_google_provider_module)
}

/// TS: `streamSimpleGoogle`
pub fn stream_simple_google_lazy() -> SimpleStreamFunction {
	create_lazy_simple_stream(load_google_provider_module)
}

/// TS: `streamGoogleVertex`
pub fn stream_google_vertex() -> StreamFunction {
	create_lazy_stream(load_google_vertex_provider_module)
}

/// TS: `streamSimpleGoogleVertex`
pub fn stream_simple_google_vertex_lazy() -> SimpleStreamFunction {
	create_lazy_simple_stream(load_google_vertex_provider_module)
}

/// TS: `streamMistral`
pub fn stream_mistral() -> StreamFunction {
	create_lazy_stream(load_mistral_provider_module)
}

/// TS: `streamSimpleMistral`
pub fn stream_simple_mistral_lazy() -> SimpleStreamFunction {
	create_lazy_simple_stream(load_mistral_provider_module)
}

/// TS: `streamOpenAICodexResponses`
pub fn stream_openai_codex_responses() -> StreamFunction {
	create_lazy_stream(load_openai_codex_responses_provider_module)
}

/// TS: `streamSimpleOpenAICodexResponses`
pub fn stream_simple_openai_codex_responses_lazy() -> SimpleStreamFunction {
	create_lazy_simple_stream(load_openai_codex_responses_provider_module)
}

/// TS: `streamOpenAICompletions`
pub fn stream_openai_completions() -> StreamFunction {
	create_lazy_stream(load_openai_completions_provider_module)
}

/// TS: `streamSimpleOpenAICompletions`
pub fn stream_simple_openai_completions_lazy() -> SimpleStreamFunction {
	create_lazy_simple_stream(load_openai_completions_provider_module)
}

/// TS: `streamOpenAIResponses`
pub fn stream_openai_responses() -> StreamFunction {
	create_lazy_stream(load_openai_responses_provider_module)
}

/// TS: `streamSimpleOpenAIResponses`
pub fn stream_simple_openai_responses_lazy() -> SimpleStreamFunction {
	create_lazy_simple_stream(load_openai_responses_provider_module)
}

fn stream_bedrock_lazy() -> StreamFunction {
	create_lazy_stream(load_bedrock_provider_module)
}

fn stream_simple_bedrock_lazy() -> SimpleStreamFunction {
	create_lazy_simple_stream(load_bedrock_provider_module)
}

/// TS: `registerBuiltInApiProviders()`
pub fn register_built_in_api_providers() {
	register_api_provider_simple(ApiProviderSimple {
		api: "kiro-api".into(),
		stream: Arc::new(super::kiro::stream_kiro),
		stream_simple: Arc::new(super::kiro::stream_simple_kiro),
		compact: None,
		supports_compaction: None,
	}, None);
	register_api_provider_simple(ApiProviderSimple {
		api: "bedrock-responses".to_string(),
		stream: stream_bedrock_responses(),
		stream_simple: stream_simple_bedrock_responses_lazy(),
		compact: None,
		supports_compaction: None,
	}, None);
	register_api_provider_simple(ApiProviderSimple {
		api: "anthropic-messages".to_string(),
		stream: stream_anthropic(),
		stream_simple: stream_simple_anthropic_lazy(),
		compact: None,
		supports_compaction: None,
	}, None);

	register_api_provider_simple(ApiProviderSimple {
		api: "openai-completions".to_string(),
		stream: stream_openai_completions(),
		stream_simple: stream_simple_openai_completions_lazy(),
		compact: None,
		supports_compaction: None,
	}, None);

	register_api_provider_simple(ApiProviderSimple {
		api: "mistral-conversations".to_string(),
		stream: stream_mistral(),
		stream_simple: stream_simple_mistral_lazy(),
		compact: None,
		supports_compaction: None,
	}, None);

	register_api_provider_simple(ApiProviderSimple {
		api: "openai-responses".to_string(),
		supports_compaction: Some(compaction_api_guard("openai-responses", supports_openai_compaction)),
		stream: stream_openai_responses(),
		stream_simple: stream_simple_openai_responses_lazy(),
		compact: Some(compact_openai_responses_guarded()),
	}, None);

	register_api_provider_simple(ApiProviderSimple {
		api: "azure-openai-responses".to_string(),
		stream: stream_azure_openai_responses(),
		stream_simple: stream_simple_azure_openai_responses_lazy(),
		compact: None,
		supports_compaction: None,
	}, None);

	register_api_provider_simple(ApiProviderSimple {
		api: "openai-codex-responses".to_string(),
		supports_compaction: Some(compaction_api_guard("openai-codex-responses", supports_openai_compaction)),
		stream: stream_openai_codex_responses(),
		stream_simple: stream_simple_openai_codex_responses_lazy(),
		compact: Some(compact_openai_codex_responses_guarded()),
	}, None);

	register_api_provider_simple(ApiProviderSimple {
		api: "google-generative-ai".to_string(),
		stream: stream_google(),
		stream_simple: stream_simple_google_lazy(),
		compact: None,
		supports_compaction: None,
	}, None);

	register_api_provider_simple(ApiProviderSimple {
		api: "google-vertex".to_string(),
		stream: stream_google_vertex(),
		stream_simple: stream_simple_google_vertex_lazy(),
		compact: None,
		supports_compaction: None,
	}, None);

	register_api_provider_simple(ApiProviderSimple {
		api: "bedrock-converse-stream".to_string(),
		stream: stream_bedrock_lazy(),
		stream_simple: stream_simple_bedrock_lazy(),
		compact: None,
		supports_compaction: None,
	}, None);
}

/// TS: `resetApiProviders()`
pub fn reset_api_providers() {
	clear_api_providers();
	register_built_in_api_providers();
}

/// TS: the module-level `registerBuiltInApiProviders();` call.
///
/// Rust has no module side effects on import, so callers use this instead. A later registration
/// replaces the earlier entry, exactly like the TypeScript `Map.set`.
pub fn ensure_built_in_api_providers_registered() {
	static REGISTERED: OnceLock<()> = OnceLock::new();
	REGISTERED.get_or_init(|| {
		register_built_in_api_providers();
	});
}

/// TS: the registered api ids, in registration order.
pub fn registered_api_ids() -> Vec<String> {
	get_api_providers().into_iter().map(|provider| provider.api).collect()
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::api_registry::API_REGISTRY_TEST_LOCK;
	use crate::types::{InputModality, Message, UserContent, UserMessage};

	fn model(api: &str, provider: &str, id: &str) -> Model {
		Model {
			id: id.to_string(),
			provider: provider.to_string(),
			api: api.to_string(),
			base_url: "https://example.invalid/v1".to_string(),
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
	fn registers_every_built_in_api_in_order() {
		let _guard = API_REGISTRY_TEST_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
		clear_api_providers();
		register_built_in_api_providers();
		assert_eq!(
			registered_api_ids(),
			vec![
				"kiro-api",
				"bedrock-responses",
				"anthropic-messages",
				"openai-completions",
				"mistral-conversations",
				"openai-responses",
				"azure-openai-responses",
				"openai-codex-responses",
				"google-generative-ai",
				"google-vertex",
				"bedrock-converse-stream",
			]
		);
		clear_api_providers();
	}

	#[test]
	fn only_responses_apis_expose_compaction() {
		let _guard = API_REGISTRY_TEST_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
		clear_api_providers();
		register_built_in_api_providers();
		let providers = get_api_providers();
		let with_compaction: Vec<&str> = providers
			.iter()
			.filter(|provider| provider.compact.is_some())
			.map(|provider| provider.api.as_str())
			.collect();
		assert_eq!(with_compaction, vec!["openai-responses", "openai-codex-responses"]);
		clear_api_providers();
	}

	#[test]
	fn supports_compaction_is_api_guarded() {
		let _guard = API_REGISTRY_TEST_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
		clear_api_providers();
		register_built_in_api_providers();
		let providers = get_api_providers();
		let responses = providers
			.iter()
			.find(|provider| provider.api == "openai-responses")
			.expect("openai-responses registered");
		let check = responses.supports_compaction.clone().unwrap();
		assert!(check(&model("openai-responses", "openai", "gpt-6-astra")));
		assert!(!check(&model("openai-completions", "openai", "gpt-6-astra")));
		clear_api_providers();
	}

	#[test]
	#[should_panic(expected = "Mismatched compaction api: openai-completions")]
	fn compaction_guard_reports_api_mismatch() {
		let _guard = API_REGISTRY_TEST_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
		clear_api_providers();
		register_built_in_api_providers();
		let mismatched = model("openai-completions", "openai", "gpt-5");
		let guard = crate::api_registry::get_api_provider("openai-responses").unwrap().compact.unwrap();
		let _ = guard(&mismatched, &context(), None);
	}

	#[tokio::test]
	async fn lazy_load_error_message_shape() {
		let message = create_lazy_load_error_message(&model("anthropic-messages", "anthropic", "claude"), "boom");
		assert_eq!(message.api, "anthropic-messages");
		assert_eq!(message.provider, "anthropic");
		assert_eq!(message.model, "claude");
		assert_eq!(message.stop_reason, "error");
		assert_eq!(message.error_message.as_deref(), Some("boom"));
		assert_eq!(message.usage.total_tokens, 0.0);
		assert!(message.content.is_empty());
	}

	#[test]
	fn typed_options_keeps_base_options_including_abort_signal() {
		let mut options = StreamOptions::default();
		options.temperature = Some(0.5);
		options.max_tokens = Some(1234.0);
		options.api_key = Some("key".to_string());
		options.signal = Some(tokio_util::sync::CancellationToken::new());
		let typed = typed_options::<AnthropicOptions>(Some(&options)).expect("typed options");
		assert_eq!(typed.stream.temperature, Some(0.5));
		assert_eq!(typed.stream.max_tokens, Some(1234.0));
		assert_eq!(typed.stream.api_key.as_deref(), Some("key"));
		// `signal` is not serialisable, so it must survive by cloning the base options.
		assert!(typed.stream.signal.is_some());
		assert!(typed_options::<AnthropicOptions>(None).is_none());
		// The simple path passes the caller's `SimpleStreamOptions` through unchanged
		// (api-registry.ts:63-68), so `reasoning` and `thinkingBudgets` survive.
		let mut simple = crate::types::SimpleStreamOptions {
			stream: options.clone(),
			reasoning: Some("high".to_string()),
			thinking_budgets: Some(crate::types::ThinkingBudgets {
				high: Some(4096.0),
				..Default::default()
			}),
		};
		simple.stream.temperature = Some(0.5);
		let passed = simple_options(Some(&simple)).expect("simple options");
		assert_eq!(passed.stream.temperature, Some(0.5));
		assert_eq!(passed.reasoning.as_deref(), Some("high"));
		assert_eq!(
			passed.thinking_budgets.as_ref().and_then(|budgets| budgets.high),
			Some(4096.0)
		);
		assert!(simple_options(None).is_none());
	}

	#[tokio::test]
	async fn forward_stream_pushes_events_and_ends() {
		let source = create_assistant_message_event_stream();
		let target = create_assistant_message_event_stream();
		forward_stream(target.clone(), source.clone());
		let partial = AssistantMessage {
			api: "test".to_string(),
			provider: "test".to_string(),
			model: "test".to_string(),
			usage: Usage::zero(),
			stop_reason: "stop".to_string(),
			..Default::default()
		};
		source.push(crate::types::AssistantMessageEvent::Start { partial: partial.clone() });
		source.push(crate::types::AssistantMessageEvent::Done {
			reason: "stop".to_string(),
			message: partial.clone(),
		});
		source.end(Some(partial.clone()));
		let first = target.next().await.expect("start event");
		assert!(matches!(first, crate::types::AssistantMessageEvent::Start { .. }));
		let second = target.next().await.expect("done event");
		assert!(matches!(second, crate::types::AssistantMessageEvent::Done { .. }));
		assert!(target.next().await.is_none());
		assert_eq!(target.result().await.model, "test");
	}

	#[test]
	fn bedrock_override_replaces_module() {
		let custom: StreamFunction = Arc::new(|_model: &Model, _context: &Context, _options: Option<&StreamOptions>| {
			create_assistant_message_event_stream()
		});
		set_bedrock_provider_module(BedrockProviderModule {
			responses: Some(BedrockResponsesProviderModule {
				stream_bedrock_responses: custom.clone(),
				stream_simple_bedrock_responses: custom.clone(),
			}),
			stream_bedrock: custom.clone(),
			stream_simple_bedrock: custom,
		});
		let module = load_bedrock_provider_module();
		let model = model("bedrock-converse-stream", "amazon-bedrock", "model");
		let stream = (module.stream)(&model, &context(), None);
		assert!(!stream.is_done());
		let responses = load_bedrock_responses_module();
		let responses_stream = (responses.stream)(&model, &context(), None);
		assert!(!responses_stream.is_done());

		let reset: StreamFunction = Arc::new(|_m: &Model, _c: &Context, _o: Option<&StreamOptions>| {
			create_assistant_message_event_stream()
		});
		set_bedrock_provider_module(BedrockProviderModule {
			responses: None,
			stream_bedrock: reset.clone(),
			stream_simple_bedrock: reset,
		});
	}

	#[test]
	fn reset_api_providers_rebuilds_registry() {
		let _guard = API_REGISTRY_TEST_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
		clear_api_providers();
		assert_eq!(get_api_providers().len(), 0);
		reset_api_providers();
		assert_eq!(get_api_providers().len(), 11);
		clear_api_providers();
	}
}
