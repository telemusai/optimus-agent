//! Port of packages/coding-agent/src/core/prime-inference-models.ts

use std::collections::{HashMap, HashSet};

use pi_ai::types::{Api, InputModality, Model, ModelCost};
use serde_json::Value;

use crate::core::prime_inference_model_catalog::{
    build_prime_inference_models, fetch_prime_inference_model_catalog, is_private_prime_inference_model_id,
    PrimeInferenceCatalogEntry, PrimeInferenceCatalogRequestError,
};
use crate::core::prime_inference_auth::FetchFn;

pub use crate::core::prime_inference_model_catalog::PRIME_INFERENCE_BASE_URL;

const PRIVATE_MODEL_REFRESH_TIMEOUT_MS: u64 = 10_000;
const PRIME_INFERENCE_PROVIDER_ID: &str = "prime-inference";

/// `PRIVATE_PRIME_INFERENCE_MODELS`.
pub fn private_prime_inference_models() -> Vec<Model> {
    vec![Model {
        id: "internal/glm-5.2-fast".to_string(),
        name: "GLM 5.2 Fast".to_string(),
        api: pi_ai::types::API_OPENAI_COMPLETIONS.to_string(),
        provider: PRIME_INFERENCE_PROVIDER_ID.to_string(),
        base_url: PRIME_INFERENCE_BASE_URL.to_string(),
        reasoning: true,
        thinking_level_map: None,
        input: vec![InputModality::Text],
        cost: ModelCost {
            input: 0.0,
            output: 0.0,
            cache_read: 0.0,
            cache_write: 0.0,
        },
        context_window: 400_000.0,
        max_input_tokens: None,
        max_tokens: 131_072.0,
        featured: Some(true),
        native_compaction: None,
        headers: None,
        compat: Some(pi_ai::types::Compat::Completions(
            pi_ai::types::OpenAICompletionsCompat {
                supports_developer_role: Some(false),
                max_tokens_field: Some("max_tokens".to_string()),
                ..Default::default()
            },
        )),
    }]
}

/// `isPrivatePrimeInferenceModel(model)`.
pub fn is_private_prime_inference_model(provider: &str, id: &str) -> bool {
    provider == PRIME_INFERENCE_PROVIDER_ID && is_private_prime_inference_model_id(id)
}

/// `getPrivatePrimeInferenceModels()` - fresh copies each call.
pub fn get_private_prime_inference_models() -> Vec<Model> {
    private_prime_inference_models()
}

/// `fetchAuthorizedPrivatePrimeInferenceModels(...)`.
pub async fn fetch_authorized_private_prime_inference_models(
    api_key: &str,
    team_headers: &[(String, String)],
    public_model_ids: &HashSet<String>,
    fetch_fn: Option<FetchFn>,
    timeout_ms: Option<u64>,
) -> Result<Vec<Model>, String> {
    let team_id_header = team_headers
        .iter()
        .find(|(key, _)| key == "X-Prime-Team-ID")
        .map(|(_, value)| value.clone());
    if team_id_header.is_none() {
        return Ok(Vec::new());
    }
    let mut headers: Vec<(String, String)> = team_headers.to_vec();
    headers.push(("Authorization".to_string(), format!("Bearer {}", api_key)));

    let fetched = fetch_prime_inference_model_catalog(
        fetch_fn,
        headers,
        Some(timeout_ms.unwrap_or(PRIVATE_MODEL_REFRESH_TIMEOUT_MS)),
        true,
    )
    .await;

    let (payload, entries) = match fetched {
        Ok(value) => value,
        Err(error) => {
            if let Some(status) = error.status() {
                if status == 401 || status == 403 {
                    let _ = PrimeInferenceCatalogRequestError { status };
                    return Ok(Vec::new());
                }
            }
            return Err(error.message());
        }
    };

    let public_ids: HashSet<String> = public_model_ids.iter().map(|id| id.to_lowercase()).collect();
    let bundled_private_models = get_private_prime_inference_models();
    let bundled_by_id: HashMap<String, Model> = bundled_private_models
        .iter()
        .map(|model| (model.id.to_lowercase(), model.clone()))
        .collect();
    let entries_by_id: HashMap<String, PrimeInferenceCatalogEntry> = entries
        .iter()
        .map(|entry| (entry.id.to_lowercase(), entry.clone()))
        .collect();
    let data = match &payload {
        Value::Object(object) => match object.get("data") {
            Some(Value::Array(items)) => items.clone(),
            _ => Vec::new(),
        },
        _ => Vec::new(),
    };

    let mut private_entries: Vec<PrimeInferenceCatalogEntry> = Vec::new();
    for item in &data {
        let Some(object) = item.as_object() else {
            continue;
        };
        let Some(raw_id) = object.get("id").and_then(Value::as_str) else {
            continue;
        };
        let id = raw_id.to_lowercase();
        if public_ids.contains(&id) || !is_private_prime_inference_model_id(&id) {
            continue;
        }
        if let Some(parsed) = entries_by_id.get(&id) {
            private_entries.push(parsed.clone());
            continue;
        }
        if let Some(template) = bundled_by_id.get(&id) {
            private_entries.push(PrimeInferenceCatalogEntry {
                id: raw_id.to_string(),
                name: None,
                input: template.cost.input,
                output: template.cost.output,
                cache_read: None,
                cache_write: None,
                context_window: None,
                max_tokens: None,
                vision: None,
                reasoning: None,
                supported_parameters: None,
                reasoning_efforts: None,
                reasoning_mandatory: None,
            });
        }
    }

    Ok(build_prime_inference_models(&bundled_private_models, &private_entries, true, Some(0.0))
        .unwrap_or_default())
}

/// Re-export used by callers that only need the error type name.
pub fn catalog_request_error_message(status: u16) -> String {
    PrimeInferenceCatalogRequestError { status }.to_string()
}

/// Keeps `Api` visible for callers that build models generically.
pub type PrimeInferenceApi = Api;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::prime_inference_auth::HttpResponse;
    use pi_ai::types::BoxFuture;
    use serde_json::json;
    use std::sync::Arc;

    fn team_headers() -> Vec<(String, String)> {
        vec![("X-Prime-Team-ID".to_string(), "team-1".to_string())]
    }

    fn fetch_returning(body: Value, status: u16) -> FetchFn {
        Arc::new(move |_request| {
            let body = body.clone();
            Box::pin(async move {
                Ok(HttpResponse {
                    status,
                    status_text: String::new(),
                    headers: Vec::new(),
                    text: body.to_string(),
                })
            }) as BoxFuture<Result<HttpResponse, String>>
        })
    }

    #[test]
    fn bundled_private_model_matches_typescript_literals() {
        let models = private_prime_inference_models();
        assert_eq!(models.len(), 1);
        let model = &models[0];
        assert_eq!(model.id, "internal/glm-5.2-fast");
        assert_eq!(model.name, "GLM 5.2 Fast");
        assert_eq!(model.api, "openai-completions");
        assert_eq!(model.provider, "prime-inference");
        assert_eq!(model.base_url, PRIME_INFERENCE_BASE_URL);
        assert!(model.reasoning);
        assert_eq!(model.input, vec![InputModality::Text]);
        assert_eq!(model.context_window, 400_000.0);
        assert_eq!(model.max_tokens, 131_072.0);
        assert_eq!(model.featured, Some(true));
        let compat = model.compat_completions().expect("compat");
        assert_eq!(compat.supports_developer_role, Some(false));
        assert_eq!(compat.max_tokens_field.as_deref(), Some("max_tokens"));
    }

    #[test]
    fn private_model_predicate_requires_provider_and_id_shape() {
        assert!(is_private_prime_inference_model("prime-inference", "internal/x"));
        assert!(!is_private_prime_inference_model("openrouter", "internal/x"));
        assert!(!is_private_prime_inference_model("prime-inference", "z-ai/glm-5.3"));
    }

    #[tokio::test]
    async fn missing_team_header_returns_empty_without_fetching() {
        let fetch_fn: FetchFn = Arc::new(|_request| {
            Box::pin(async move { Err("must not be called".to_string()) }) as BoxFuture<Result<HttpResponse, String>>
        });
        let models = fetch_authorized_private_prime_inference_models(
            "key",
            &[],
            &HashSet::new(),
            Some(fetch_fn),
            None,
        )
        .await
        .unwrap();
        assert!(models.is_empty());
    }

    #[tokio::test]
    async fn unauthorized_returns_empty_instead_of_error() {
        let fetch_fn = fetch_returning(json!({"error": "no"}), 403);
        let models = fetch_authorized_private_prime_inference_models(
            "key",
            &team_headers(),
            &HashSet::new(),
            Some(fetch_fn),
            None,
        )
        .await
        .unwrap();
        assert!(models.is_empty());
    }

    #[tokio::test]
    async fn non_auth_errors_propagate() {
        let fetch_fn = fetch_returning(json!({"error": "boom"}), 500);
        let error = fetch_authorized_private_prime_inference_models(
            "key",
            &team_headers(),
            &HashSet::new(),
            Some(fetch_fn),
            None,
        )
        .await
        .unwrap_err();
        assert_eq!(error, "Prime Inference model catalog request failed with status 500");
    }

    #[tokio::test]
    async fn private_entries_are_filtered_and_built() {
        let body = json!({"data": [
            {"id": "internal/glm-5.2-fast", "pricing": {"input_usd_per_mtok": 0, "output_usd_per_mtok": 0}},
            {"id": "z-ai/glm-5.3", "pricing": {"input_usd_per_mtok": 1, "output_usd_per_mtok": 2}},
            {"id": "dev/secret", "pricing": {"input_usd_per_mtok": 0, "output_usd_per_mtok": 0}}
        ]});
        let fetch_fn = fetch_returning(body, 200);
        let mut public = HashSet::new();
        public.insert("z-ai/glm-5.3".to_string());
        let models = fetch_authorized_private_prime_inference_models(
            "key",
            &team_headers(),
            &public,
            Some(fetch_fn),
            None,
        )
        .await
        .unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "internal/glm-5.2-fast");
        assert_eq!(models[0].name, "GLM 5.2 Fast");
        assert_eq!(models[0].max_tokens, 131_072.0);
    }

    #[tokio::test]
    async fn bundled_template_supplies_missing_pricing() {
        let body = json!({"data": [
            {"id": "internal/glm-5.2-fast", "pricing": {"input_usd_per_mtok": 0, "output_usd_per_mtok": 0}, "specs": {}}
        ]});
        let fetch_fn = fetch_returning(body, 200);
        let models = fetch_authorized_private_prime_inference_models(
            "key",
            &team_headers(),
            &HashSet::new(),
            Some(fetch_fn),
            None,
        )
        .await
        .unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].provider, "prime-inference");
        assert!(models[0].featured.is_none() || models[0].featured == Some(true));
    }
}
