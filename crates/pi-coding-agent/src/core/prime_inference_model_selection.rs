//! Port of packages/coding-agent/src/core/prime-inference-model-selection.ts

use pi_ai::types::Model;

use crate::core::prime_inference_auth::PRIME_INFERENCE_PROVIDER_ID;

/// `type ProviderLoginResult` - the union used by the login flows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderLoginResult {
    Success {
        provider_id: String,
        kind: Option<ProviderLoginKind>,
    },
    Cancelled,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderLoginKind {
    Provider,
    Service,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct PrimeInferencePostLoginModelAction {
    pub open_model_picker: bool,
    /// `fallbackModel?: Model` - `None` means the key is absent.
    pub fallback_model: Option<Model>,
}

/// `PRIME_INFERENCE_DEFAULT_MODEL_ID` lives in model-resolver.ts.
use crate::core::model_resolver::PRIME_INFERENCE_DEFAULT_MODEL_ID;

/// `resolvePrimeInferencePostLoginModelAction(authResult, currentModel, modelRegistry)`.
///
/// `modelRegistry` is narrowed to `Pick<ModelRegistry, "find">`, so the Rust
/// signature takes the lookup closure instead of the whole registry.
pub fn resolve_prime_inference_post_login_model_action<F>(
    auth_result: &ProviderLoginResult,
    current_model: Option<&Model>,
    find: F,
) -> PrimeInferencePostLoginModelAction
where
    F: FnOnce(&str, &str) -> Option<Model>,
{
    match auth_result {
        ProviderLoginResult::Success { provider_id, kind } => {
            if *kind == Some(ProviderLoginKind::Service) || provider_id != PRIME_INFERENCE_PROVIDER_ID {
                return PrimeInferencePostLoginModelAction {
                    open_model_picker: false,
                    fallback_model: None,
                };
            }
        }
        _ => {
            return PrimeInferencePostLoginModelAction {
                open_model_picker: false,
                fallback_model: None,
            }
        }
    }

    PrimeInferencePostLoginModelAction {
        open_model_picker: true,
        fallback_model: if current_model.is_some() {
            None
        } else {
            find(PRIME_INFERENCE_PROVIDER_ID, PRIME_INFERENCE_DEFAULT_MODEL_ID)
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pi_ai::types::ModelCost;

    fn model() -> Model {
        Model {
            id: PRIME_INFERENCE_DEFAULT_MODEL_ID.to_string(),
            name: "GLM".to_string(),
            api: "openai-completions".to_string(),
            provider: PRIME_INFERENCE_PROVIDER_ID.to_string(),
            base_url: "https://api.pinference.ai/api/v1".to_string(),
            reasoning: true,
            thinking_level_map: None,
            input: vec![pi_ai::types::InputModality::Text],
            cost: ModelCost::default(),
            context_window: 200_000.0,
            max_input_tokens: None,
            max_tokens: 131_072.0,
            max_tokens_explicit: false,
            featured: None,
            native_compaction: None,
            headers: None,
            compat: None,
        }
    }

    #[test]
    fn non_prime_success_does_not_open_the_picker() {
        let result = resolve_prime_inference_post_login_model_action(
            &ProviderLoginResult::Success {
                provider_id: "anthropic".to_string(),
                kind: None,
            },
            None,
            |_, _| Some(model()),
        );
        assert!(!result.open_model_picker);
        assert!(result.fallback_model.is_none());
    }

    #[test]
    fn service_kind_does_not_open_the_picker() {
        let result = resolve_prime_inference_post_login_model_action(
            &ProviderLoginResult::Success {
                provider_id: PRIME_INFERENCE_PROVIDER_ID.to_string(),
                kind: Some(ProviderLoginKind::Service),
            },
            None,
            |_, _| Some(model()),
        );
        assert!(!result.open_model_picker);
    }

    #[test]
    fn prime_success_without_current_model_returns_fallback() {
        let result = resolve_prime_inference_post_login_model_action(
            &ProviderLoginResult::Success {
                provider_id: PRIME_INFERENCE_PROVIDER_ID.to_string(),
                kind: Some(ProviderLoginKind::Provider),
            },
            None,
            |provider, id| {
                assert_eq!(provider, PRIME_INFERENCE_PROVIDER_ID);
                assert_eq!(id, PRIME_INFERENCE_DEFAULT_MODEL_ID);
                Some(model())
            },
        );
        assert!(result.open_model_picker);
        assert_eq!(result.fallback_model.map(|model| model.id), Some("z-ai/glm-5.3".to_string()));
    }

    #[test]
    fn prime_success_with_current_model_has_no_fallback() {
        let result = resolve_prime_inference_post_login_model_action(
            &ProviderLoginResult::Success {
                provider_id: PRIME_INFERENCE_PROVIDER_ID.to_string(),
                kind: None,
            },
            Some(&model()),
            |_, _| panic!("must not look up a fallback"),
        );
        assert!(result.open_model_picker);
        assert!(result.fallback_model.is_none());
    }

    #[test]
    fn cancelled_and_failed_results_do_nothing() {
        for result in [ProviderLoginResult::Cancelled, ProviderLoginResult::Failed] {
            let action =
                resolve_prime_inference_post_login_model_action(&result, None, |_, _| Some(model()));
            assert!(!action.open_model_picker);
            assert!(action.fallback_model.is_none());
        }
    }
}
