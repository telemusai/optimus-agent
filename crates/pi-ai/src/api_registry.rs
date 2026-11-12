//! Port of packages/ai/src/api-registry.ts

use indexmap::IndexMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::compaction::{CompactionOptions, ProviderCompactionResult};
use crate::providers::openai_compaction::CompactionRequestError;
use crate::types::{
    Api, BoxFuture, CompactFunction, Context, Model, SimpleStreamOptions, StreamFunction,
    StreamOptions,
};
use crate::utils::event_stream::AssistantMessageEventStream;

pub type ApiStreamFunction =
    Arc<dyn Fn(&Model, &Context, Option<&StreamOptions>) -> AssistantMessageEventStream + Send + Sync>;

pub type ApiStreamSimpleFunction =
    Arc<dyn Fn(&Model, &Context, Option<&SimpleStreamOptions>) -> AssistantMessageEventStream + Send + Sync>;

/// `StreamFunction<TApi, SimpleStreamOptions>` (types.ts): what
/// `ApiProvider.streamSimple` is typed as (api-registry.ts:27). `Model<TApi>` is not generic
/// in this port, so this is the same function type as [`ApiStreamSimpleFunction`].
pub type SimpleStreamFunction = ApiStreamSimpleFunction;

pub(crate) type FallibleCompactFunction = Arc<
    dyn Fn(
            &Model,
            &Context,
            Option<&CompactionOptions>,
        ) -> BoxFuture<Result<Option<ProviderCompactionResult>, CompactionRequestError>>
        + Send
        + Sync,
>;


/// `interface ApiProvider<TApi, TOptions>` for callers whose `streamSimple` is the base
/// [`StreamFunction`] shape (see [`ApiProviderSimple`] for the faithful
/// `SimpleStreamOptions` signature).
pub struct ApiProvider {
    pub api: Api,
    pub stream: StreamFunction,
    /// Base-shaped simple stream. The TypeScript types this
    /// `StreamFunction<TApi, SimpleStreamOptions>` (api-registry.ts:27); [`ApiProviderSimple`]
    /// is the faithful form, and this legacy one receives `options.stream`.
    pub stream_simple: StreamFunction,
    pub compact: Option<CompactFunction>,
    pub supports_compaction: Option<Arc<dyn Fn(&Model) -> bool + Send + Sync>>,
}

/// `interface ApiProvider<TApi, TOptions>` (api-registry.ts:24-30) with the faithful
/// `streamSimple: StreamFunction<TApi, SimpleStreamOptions>`.
///
/// The Rust port split the struct: the base-shaped [`ApiProvider`] keeps older registrations
/// compiling, while this one preserves `reasoning` / `thinkingBudgets` end to end.
pub struct ApiProviderSimple {
    pub api: Api,
    pub stream: StreamFunction,
    pub stream_simple: SimpleStreamFunction,
    pub compact: Option<CompactFunction>,
    pub supports_compaction: Option<Arc<dyn Fn(&Model) -> bool + Send + Sync>>,
}

impl Clone for ApiProviderSimple {
    fn clone(&self) -> Self {
        Self {
            api: self.api.clone(),
            stream: self.stream.clone(),
            stream_simple: self.stream_simple.clone(),
            compact: self.compact.clone(),
            supports_compaction: self.supports_compaction.clone(),
        }
    }
}

impl Clone for ApiProvider {
    fn clone(&self) -> Self {
        Self {
            api: self.api.clone(),
            stream: self.stream.clone(),
            stream_simple: self.stream_simple.clone(),
            compact: self.compact.clone(),
            supports_compaction: self.supports_compaction.clone(),
        }
    }
}

/// `interface ApiProviderInternal`
#[derive(Clone)]
pub struct ApiProviderInternal {
    pub api: Api,
    pub stream: ApiStreamFunction,
    pub stream_simple: ApiStreamSimpleFunction,
    pub compact: Option<CompactFunction>,
    pub supports_compaction: Option<Arc<dyn Fn(&Model) -> bool + Send + Sync>>,
}

#[derive(Clone)]
struct RegisteredApiProvider {
    provider: ApiProviderInternal,
    source_id: Option<String>,
    fallible_compact: Option<FallibleCompactFunction>,
}

fn api_provider_registry() -> &'static Mutex<IndexMap<String, RegisteredApiProvider>> {
    static REGISTRY: OnceLock<Mutex<IndexMap<String, RegisteredApiProvider>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(IndexMap::new()))
}

fn lock_registry() -> std::sync::MutexGuard<'static, IndexMap<String, RegisteredApiProvider>> {
    api_provider_registry()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// `function wrapStream(api, stream)`.
fn wrap_stream(api: Api, stream: StreamFunction) -> ApiStreamFunction {
    Arc::new(move |model: &Model, context: &Context, options: Option<&StreamOptions>| {
        if model.api != api {
            throw_mismatched_api(&model.api, &api, "Mismatched api");
        }
        stream(model, context, options)
    })
}

/// `function wrapStreamSimple(api, streamSimple)` for the base-shaped registrations: the
/// caller's `SimpleStreamOptions` are reduced to their base [`StreamOptions`] part.
fn wrap_stream_simple_base(api: Api, stream_simple: StreamFunction) -> ApiStreamSimpleFunction {
    Arc::new(
        move |model: &Model, context: &Context, options: Option<&SimpleStreamOptions>| {
            if model.api != api {
                throw_mismatched_api(&model.api, &api, "Mismatched api");
            }
            stream_simple(model, context, options.map(|options| &options.stream))
        },
    )
}

/// `function wrapStreamSimple(api, streamSimple)`.
///
/// TS: `return streamSimple(model as Model<TApi>, context, options);` - the caller's
/// `SimpleStreamOptions` (including `reasoning` and `thinkingBudgets`) are passed through
/// unchanged (api-registry.ts:59-69).
fn wrap_stream_simple(api: Api, stream_simple: SimpleStreamFunction) -> ApiStreamSimpleFunction {
    Arc::new(
        move |model: &Model, context: &Context, options: Option<&SimpleStreamOptions>| {
            if model.api != api {
                throw_mismatched_api(&model.api, &api, "Mismatched api");
            }
            stream_simple(model, context, options)
        },
    )
}

/// The TypeScript `throw new Error(...)` inside the wrapper.
fn throw_mismatched_api(model_api: &str, api: &str, prefix: &str) -> ! {
    panic!("{}: {} expected {}", prefix, model_api, api)
}

fn register_api_provider_simple_inner(
    provider: ApiProviderSimple,
    fallible_compact: Option<FallibleCompactFunction>,
    source_id: Option<String>,
) {
    let api = provider.api.clone();
    let internal = ApiProviderInternal {
        api: provider.api.clone(),
        stream: wrap_stream(provider.api.clone(), provider.stream.clone()),
        stream_simple: wrap_stream_simple(provider.api.clone(), provider.stream_simple.clone()),
        supports_compaction: provider.supports_compaction.clone(),
        compact: provider.compact.clone().map(|compact| {
            let api = provider.api.clone();
            let compact: CompactFunction = compact;
            Arc::new(
                move |model: &Model,
                      context: &Context,
                      options: Option<&CompactionOptions>|
                      -> BoxFuture<Option<crate::compaction::ProviderCompactionResult>> {
                    if model.api != api {
                        panic!("Mismatched compaction api: {}", model.api);
                    }
                    compact(model, context, options)
                },
            ) as CompactFunction
        }),
    };

    lock_registry().insert(
        api,
        RegisteredApiProvider {
            provider: internal,
            source_id,
            fallible_compact,
        },
    );
}

/// `registerApiProvider(provider, sourceId?)` (api-registry.ts:71-92), with the faithful
/// `streamSimple` signature preserved.
pub fn register_api_provider_simple(provider: ApiProviderSimple, source_id: Option<String>) {
    register_api_provider_simple_inner(provider, None, source_id);
}

/// Atomically installs both handlers for built-in providers. Custom registrations use the
/// public entry point above and therefore cannot inherit a built-in fallible handler.
pub(crate) fn register_api_provider_simple_with_fallible_compaction(
    provider: ApiProviderSimple,
    compact: FallibleCompactFunction,
    source_id: Option<String>,
) {
    register_api_provider_simple_inner(provider, Some(compact), source_id);
}

/// `registerApiProvider(provider, sourceId?)` for the base-shaped [`ApiProvider`].
pub fn register_api_provider(provider: ApiProvider, source_id: Option<String>) {
    let api = provider.api.clone();
    let internal = ApiProviderInternal {
        api: provider.api.clone(),
        stream: wrap_stream(provider.api.clone(), provider.stream.clone()),
        stream_simple: wrap_stream_simple_base(provider.api.clone(), provider.stream_simple.clone()),
        supports_compaction: provider.supports_compaction.clone(),
        compact: provider.compact.clone().map(|compact| {
            let api = provider.api.clone();
            let compact: CompactFunction = compact;
            Arc::new(
                move |model: &Model,
                      context: &Context,
                      options: Option<&CompactionOptions>|
                      -> BoxFuture<Option<crate::compaction::ProviderCompactionResult>> {
                    if model.api != api {
                        panic!("Mismatched compaction api: {}", model.api);
                    }
                    compact(model, context, options)
                },
            ) as CompactFunction
        }),
    };

    lock_registry().insert(
        api,
        RegisteredApiProvider {
            provider: internal,
            source_id,
            fallible_compact: None,
        },
    );
}


#[derive(Clone)]
pub(crate) struct ApiProviderCompactionSnapshot {
    pub provider: ApiProviderInternal,
    pub fallible_compact: Option<FallibleCompactFunction>,
}

pub(crate) fn get_api_provider_compaction_snapshot(api: &str) -> Option<ApiProviderCompactionSnapshot> {
    lock_registry().get(api).map(|entry| ApiProviderCompactionSnapshot {
        provider: entry.provider.clone(),
        fallible_compact: entry.fallible_compact.clone(),
    })
}

pub fn get_api_provider(api: &str) -> Option<ApiProviderInternal> {
    lock_registry().get(api).map(|entry| entry.provider.clone())
}

pub fn get_api_providers() -> Vec<ApiProviderInternal> {
    lock_registry()
        .values()
        .map(|entry| entry.provider.clone())
        .collect()
}

pub fn unregister_api_providers(source_id: &str) {
    lock_registry().retain(|_, entry| entry.source_id.as_deref() != Some(source_id));
}

pub fn clear_api_providers() {
    lock_registry().clear();
}

#[cfg(test)]
pub(crate) static API_REGISTRY_TEST_LOCK: Mutex<()> = Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{AssistantMessage, STOP_REASON_STOP};

    fn fake_stream(model_api: &str) -> StreamFunction {
        let model_api = model_api.to_string();
        Arc::new(move |model: &Model, _context: &Context, _options: Option<&StreamOptions>| {
            assert_eq!(model.api, model_api);
            let stream = AssistantMessageEventStream::new();
            stream.push(crate::types::AssistantMessageEvent::Done {
                reason: STOP_REASON_STOP.to_string(),
                message: AssistantMessage::default(),
            });
            stream
        })
    }

    fn provider(api: &str) -> ApiProvider {
        ApiProvider {
            api: api.to_string(),
            stream: fake_stream(api),
            stream_simple: fake_stream(api),
            compact: None,
            supports_compaction: None,
        }
    }

    #[test]
    fn register_get_and_clear() {
        let _guard = API_REGISTRY_TEST_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        clear_api_providers();
        register_api_provider(provider("openai-responses"), Some("src-a".to_string()));
        assert!(get_api_provider("openai-responses").is_some());
        assert_eq!(get_api_providers().len(), 1);

        register_api_provider(provider("anthropic-messages"), Some("src-b".to_string()));
        assert_eq!(get_api_providers().len(), 2);

        unregister_api_providers("src-a");
        assert!(get_api_provider("openai-responses").is_none());
        assert!(get_api_provider("anthropic-messages").is_some());

        clear_api_providers();
        assert!(get_api_providers().is_empty());
    }

    #[test]
    fn registry_is_last_write_wins_per_api() {
        let _guard = API_REGISTRY_TEST_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        clear_api_providers();
        register_api_provider(provider("openai-completions"), Some("first".to_string()));
        register_api_provider(provider("openai-completions"), Some("second".to_string()));
        assert_eq!(get_api_providers().len(), 1);
        unregister_api_providers("first");
        assert_eq!(get_api_providers().len(), 1, "second registration must survive");
        clear_api_providers();
    }

    #[test]
    fn wrapped_stream_returns_events_for_matching_api() {
        let _guard = API_REGISTRY_TEST_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        clear_api_providers();
        register_api_provider(provider("google-generative-ai"), None);
        let internal = get_api_provider("google-generative-ai").unwrap();
        let model = Model::new("m", "M", "google-generative-ai", "google", "https://example.test");
        let context = Context::default();
        let stream = (internal.stream)(&model, &context, None);
        assert!(stream.is_done());
        clear_api_providers();
    }

    #[test]
    fn wrap_stream_simple_passes_reasoning_and_thinking_budgets_through() {
        let _guard = API_REGISTRY_TEST_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        clear_api_providers();
        // TS: `wrapStreamSimple` only checks the api and forwards `options` unchanged
        // (api-registry.ts:59-69), so `reasoning` / `thinkingBudgets` must arrive intact.
        let seen: Arc<Mutex<Option<SimpleStreamOptions>>> = Arc::new(Mutex::new(None));
        let record = Arc::clone(&seen);
        let stream_simple: SimpleStreamFunction = Arc::new(
            move |_model: &Model, _context: &Context, options: Option<&SimpleStreamOptions>| {
                *record.lock().unwrap() = options.cloned();
                AssistantMessageEventStream::new()
            },
        );
        register_api_provider_simple(
            ApiProviderSimple {
                api: "anthropic-messages".to_string(),
                stream: fake_stream("anthropic-messages"),
                stream_simple,
                compact: None,
                supports_compaction: None,
            },
            None,
        );
        let model = Model::new("m", "M", "anthropic-messages", "anthropic", "https://example.test");
        let mut options = SimpleStreamOptions::default();
        options.reasoning = Some("high".to_string());
        options.thinking_budgets = Some(crate::types::ThinkingBudgets {
            high: Some(2048.0),
            ..Default::default()
        });
        let internal = get_api_provider("anthropic-messages").unwrap();
        let _ = (internal.stream_simple)(&model, &Context::default(), Some(&options));
        let received = seen.lock().unwrap().clone().expect("options reached the provider");
        assert_eq!(received.reasoning.as_deref(), Some("high"));
        assert_eq!(
            received.thinking_budgets.as_ref().and_then(|budgets| budgets.high),
            Some(2048.0)
        );
        clear_api_providers();
    }

    #[test]
    #[should_panic(expected = "Mismatched api: openai-responses expected anthropic-messages")]
    fn wrapped_stream_panics_on_mismatched_api() {
        let _guard = API_REGISTRY_TEST_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        clear_api_providers();
        register_api_provider(provider("anthropic-messages"), None);
        let internal = get_api_provider("anthropic-messages").unwrap();
        let model = Model::new("m", "M", "openai-responses", "openai", "https://example.test");
        let context = Context::default();
        let _ = (internal.stream)(&model, &context, None);
    }

    #[test]
    fn replacement_and_removal_preserve_registration_order() {
        let _guard = API_REGISTRY_TEST_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        clear_api_providers();
        for api in ["first", "second", "third"] {
            register_api_provider(provider(api), Some(api.to_string()));
        }
        register_api_provider(provider("second"), Some("replacement".to_string()));
        let ids = || get_api_providers().into_iter().map(|provider| provider.api).collect::<Vec<_>>();
        assert_eq!(ids(), ["first", "second", "third"]);
        unregister_api_providers("first");
        assert_eq!(ids(), ["second", "third"]);
        clear_api_providers();
    }
}
