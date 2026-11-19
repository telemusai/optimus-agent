//! Port of packages/ai/src/stream.ts

use std::sync::Arc;

use crate::api_registry::{
    get_api_provider, get_api_provider_compaction_snapshot, get_api_providers, ApiProviderCompactionSnapshot,
    ApiProviderInternal,
};
use crate::compaction::{CompactionOptions, ProviderCompactionResult};
use crate::providers::openai_compaction::CompactionRequestError;
use crate::types::{
    AssistantMessage, BoxFuture, Context, Model, ProviderStreamOptions, SimpleStreamOptions,
};
use crate::utils::event_stream::AssistantMessageEventStream;

pub use crate::env_api_keys::get_env_api_key;

/// `function resolveApiProvider(api)`.
///
/// `packages/ai/src/stream.ts:1` is `import "./providers/register-builtins.js";` - the
/// built-ins are registered by that side effect before any `stream()` call can resolve an
/// api. Rust has no import side effects, so the equivalent runs here, once per process.
/// Without it `stream_simple("openai-completions")` panics with
/// "No API provider registered for api: ..." the first time a real provider is used.
fn ensure_builtin_api_providers() {
    static REGISTERED: std::sync::Once = std::sync::Once::new();
    REGISTERED.call_once(|| {
        // Tests clear and re-register the registry themselves; only the first call is ours.
        if get_api_providers().is_empty() {
            crate::providers::register_builtins::register_built_in_api_providers();
        }
    });
}

fn resolve_api_provider(api: &str) -> ApiProviderInternal {
    ensure_builtin_api_providers();
    match get_api_provider(api) {
        Some(provider) => provider,
        None => panic!("No API provider registered for api: {}", api),
    }
}

fn resolve_compaction_provider(api: &str) -> ApiProviderCompactionSnapshot {
    ensure_builtin_api_providers();
    get_api_provider_compaction_snapshot(api)
        .unwrap_or_else(|| panic!("No API provider registered for api: {}", api))
}

fn provider_supports_compaction(provider: &ApiProviderInternal, model: &Model) -> bool {
    provider.compact.is_some()
        && provider
            .supports_compaction
            .as_ref()
            .map(|supports| supports(model))
            .unwrap_or(true)
}

pub fn supports_compaction(model: &Model) -> bool {
    provider_supports_compaction(&resolve_api_provider(&model.api), model)
}

pub fn compact_simple(
    model: &Model,
    context: &Context,
    options: Option<&CompactionOptions>,
) -> BoxFuture<Option<ProviderCompactionResult>> {
    if !supports_compaction(model) {
        return Box::pin(async { None });
    }
    let compact = resolve_api_provider(&model.api)
        .compact
        .expect("checked by supports_compaction");
    let model = model.clone();
    let context = context.clone();
    let options = options.cloned();
    Box::pin(async move { compact(&model, &context, options.as_ref()).await })
}

/// Fallible native compaction preserves request failures separately from `None`, which means
/// that compaction is unsupported. Custom providers retain their existing [`compact_simple`]
/// contract; built-in providers opt into typed failures through an internal registration slot.
pub fn try_compact_simple(
    model: &Model,
    context: &Context,
    options: Option<&CompactionOptions>,
) -> BoxFuture<Result<Option<ProviderCompactionResult>, CompactionRequestError>> {
    let snapshot = resolve_compaction_provider(&model.api);
    if !provider_supports_compaction(&snapshot.provider, model) {
        return Box::pin(async { Ok(None) });
    }
    let compact = snapshot
        .provider
        .compact
        .expect("checked by provider_supports_compaction");
    let fallible_compact = snapshot.fallible_compact;
    let model = model.clone();
    let context = context.clone();
    let options = options.cloned();
    Box::pin(async move {
        if let Some(compact) = fallible_compact {
            compact(&model, &context, options.as_ref()).await
        } else {
            Ok(compact(&model, &context, options.as_ref()).await)
        }
    })
}

pub fn stream(
    model: &Model,
    context: &Context,
    options: Option<&ProviderStreamOptions>,
) -> AssistantMessageEventStream {
    let provider = resolve_api_provider(&model.api);
    (provider.stream)(model, context, options)
}

pub async fn complete(
    model: &Model,
    context: &Context,
    options: Option<&ProviderStreamOptions>,
) -> AssistantMessage {
    let stream = stream(model, context, options);
    stream.result().await
}

pub fn stream_simple(
    model: &Model,
    context: &Context,
    options: Option<&SimpleStreamOptions>,
) -> AssistantMessageEventStream {
    let provider = resolve_api_provider(&model.api);
    (provider.stream_simple)(model, context, options)
}

pub async fn complete_simple(
    model: &Model,
    context: &Context,
    options: Option<&SimpleStreamOptions>,
) -> AssistantMessage {
    let stream = stream_simple(model, context, options);
    stream.result().await
}

/// Keeps the `import "./providers/register-builtins.js"` side effect of stream.ts
/// explicit: `stream()` requires the built-in providers to be registered.
pub fn ensure_builtin_api_providers_registered() {
    crate::providers::register_builtins::register_built_in_api_providers();
}

/// Silence unused-import warnings for the `Arc` import kept for API symmetry.
#[allow(dead_code)]
fn _assert_arc_used(_: Arc<()>) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api_registry::{clear_api_providers, register_api_provider, ApiProvider, API_REGISTRY_TEST_LOCK};
    use crate::types::{AssistantMessageEvent, STOP_REASON_STOP};

    fn provider(api: &str) -> ApiProvider {
        let stream_fn = {
            let api = api.to_string();
            Arc::new(move |model: &Model, _context: &Context, _options: Option<&crate::types::StreamOptions>| {
                assert_eq!(model.api, api);
                let stream = AssistantMessageEventStream::new();
                stream.push(AssistantMessageEvent::Done {
                    reason: STOP_REASON_STOP.to_string(),
                    message: AssistantMessage::default(),
                });
                stream
            })
        };
        ApiProvider {
            api: api.to_string(),
            stream: stream_fn.clone(),
            stream_simple: stream_fn,
            compact: None,
            supports_compaction: None,
        }
    }

    #[test]
    fn stream_dispatches_to_the_registered_provider() {
        let _guard = API_REGISTRY_TEST_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        clear_api_providers();
        register_api_provider(provider("anthropic-messages"), None);
        let model = Model::new("m", "M", "anthropic-messages", "anthropic", "https://example.test");
        let stream = stream(&model, &Context::default(), None);
        assert!(stream.is_done());
        clear_api_providers();
    }

    #[test]
    fn supports_compaction_is_false_without_a_compact_function() {
        let _guard = API_REGISTRY_TEST_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        clear_api_providers();
        register_api_provider(provider("mistral-conversations"), None);
        let model = Model::new("m", "M", "mistral-conversations", "mistral", "https://example.test");
        assert!(!supports_compaction(&model));
        clear_api_providers();
    }

    #[test]
    fn built_in_custom_built_in_compaction_uses_the_current_atomic_snapshot() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use crate::providers::register_builtins::register_built_in_api_providers;

        let _guard = API_REGISTRY_TEST_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        clear_api_providers();
        let model = Model::new(
            "gpt-5.1-codex",
            "Synthetic supported Codex",
            "openai-codex-responses",
            "openai-codex",
            "https://unused.invalid",
        );
        let mut options = CompactionOptions::default();
        // Account-ID extraction rejects this synthetic JWT before any network request. The
        // built-in typed path returns Err; the legacy Option wrapper would erase it to None.
        options.simple.stream.api_key = Some("synthetic-malformed-jwt".to_string());

        register_built_in_api_providers();
        assert!(resolve_compaction_provider("openai-codex-responses").fallible_compact.is_some());
        let before = futures::executor::block_on(try_compact_simple(
            &model,
            &Context::default(),
            Some(&options),
        ));
        assert!(matches!(before, Err(ref error) if error.name == "CompactionRequestError" && error.message == "Failed to extract accountId from token"),
            "the current typed built-in handler must run before the custom override");

        let custom_calls = Arc::new(AtomicUsize::new(0));
        let observed_calls = custom_calls.clone();
        let mut custom = provider("openai-codex-responses");
        custom.supports_compaction = Some(Arc::new(|_| true));
        custom.compact = Some(Arc::new(move |_, _, _| {
            observed_calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { None })
        }));
        register_api_provider(custom, None);
        assert!(resolve_compaction_provider("openai-codex-responses").fallible_compact.is_none());
        let custom_result = futures::executor::block_on(try_compact_simple(
            &model,
            &Context::default(),
            Some(&options),
        ));
        assert!(matches!(custom_result, Ok(None)), "custom None must remain unsupported");
        assert_eq!(custom_calls.load(Ordering::SeqCst), 1);

        register_built_in_api_providers();
        assert!(resolve_compaction_provider("openai-codex-responses").fallible_compact.is_some());
        let after = futures::executor::block_on(try_compact_simple(
            &model,
            &Context::default(),
            Some(&options),
        ));
        assert!(matches!(after, Err(ref error) if error.name == "CompactionRequestError" && error.message == "Failed to extract accountId from token"),
            "the current typed built-in handler must run again after the built-in reset");
        assert_eq!(custom_calls.load(Ordering::SeqCst), 1, "stale custom handler must not run after built-in reset");
        clear_api_providers();
    }

    #[test]
    #[should_panic(expected = "No API provider registered for api: unknown-api")]
    fn unregistered_api_panics_with_typescript_message() {
        let _guard = API_REGISTRY_TEST_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        clear_api_providers();
        let model = Model::new("m", "M", "unknown-api", "p", "https://example.test");
        let _ = stream(&model, &Context::default(), None);
    }
}
