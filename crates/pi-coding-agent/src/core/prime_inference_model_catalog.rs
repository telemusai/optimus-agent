//! Port of packages/coding-agent/src/core/prime-inference-model-catalog.ts

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

use futures::FutureExt;

use indexmap::IndexMap;
use pi_ai::types::{
    Compat, InputModality, Model, ModelCost, OpenAICompletionsCompat, ThinkingLevelMap,
};
use serde_json::Value;
use pi_ai::prime_inference_model_catalog::{get_prime_inference_reasoning_controls, parse_string_array};

use crate::core::prime_inference_auth::FetchFn;
use crate::utils::atomic_file::{write_file_atomic_sync, WriteFileAtomicOptions};

pub const PRIME_INFERENCE_BASE_URL: &str = "https://api.pinference.ai/api/v1";
const FETCH_TIMEOUT_MS: u64 = 5_000;
const MAX_RESPONSE_BYTES: usize = 2 * 1024 * 1024;
const MIN_CATALOG_COVERAGE: f64 = 0.5;

/// `const pendingRefreshes = new Map<string, Promise<...>>()`
type PendingRefresh = Arc<futures::future::Shared<pi_ai::types::BoxFuture<Option<Vec<Model>>>>>;

fn pending_refreshes() -> &'static Mutex<HashMap<String, PendingRefresh>> {
    static PENDING: OnceLock<Mutex<HashMap<String, PendingRefresh>>> = OnceLock::new();
    PENDING.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn default_compat() -> OpenAICompletionsCompat {
    OpenAICompletionsCompat {
        supports_store: Some(false),
        supports_developer_role: Some(false),
        // Live supported parameters override this conservative fallback.
        supports_reasoning_effort: Some(false),
        max_tokens_field: Some("max_tokens".to_string()),
        supports_strict_mode: Some(false),
        ..Default::default()
    }
}

/// `PrimeInferenceCatalogEntry`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PrimeInferenceCatalogEntry {
    pub id: String,
    pub name: Option<String>,
    pub input: f64,
    pub output: f64,
    pub cache_read: Option<f64>,
    pub cache_write: Option<f64>,
    pub context_window: Option<f64>,
    pub max_tokens: Option<f64>,
    pub vision: Option<bool>,
    pub reasoning: Option<bool>,
    pub supported_parameters: Option<Vec<String>>,
    pub reasoning_efforts: Option<Vec<String>>,
    pub reasoning_mandatory: Option<bool>,
}

fn is_record(value: &Value) -> Option<&serde_json::Map<String, Value>> {
    value.as_object()
}

fn non_negative_number(value: Option<&Value>) -> Option<f64> {
    let number = value?.as_f64()?;
    if number.is_finite() && number >= 0.0 {
        Some(number)
    } else {
        None
    }
}

fn positive_integer(value: Option<&Value>) -> Option<f64> {
    let number = value?.as_f64()?;
    if number.is_finite() && number.fract() == 0.0 && number > 0.0 {
        Some(number)
    } else {
        None
    }
}

/// `isPrivatePrimeInferenceModelId(modelId)`.
pub fn is_private_prime_inference_model_id(model_id: &str) -> bool {
    let normalized = model_id.to_lowercase();
    normalized.starts_with("internal/") || normalized.starts_with("dev/") || normalized.contains(':')
}

fn strip_control_characters(value: &str) -> String {
    value
        .chars()
        .filter(|ch| {
            let code = *ch as u32;
            !((0x00..=0x1f).contains(&code) || (0x7f..=0x9f).contains(&code))
        })
        .collect()
}

fn has_control_characters(value: &str) -> bool {
    value.chars().any(|ch| {
        let code = ch as u32;
        (0x00..=0x1f).contains(&code) || (0x7f..=0x9f).contains(&code)
    })
}

/// `parsePrimeInferenceModelCatalog(value, options)`.
pub fn parse_prime_inference_model_catalog(
    value: &Value,
    allow_empty: bool,
) -> Result<Vec<PrimeInferenceCatalogEntry>, String> {
    let object = is_record(value).ok_or_else(|| "Invalid Prime Inference model catalog".to_string())?;
    let data = match object.get("data") {
        Some(Value::Array(items)) => items,
        _ => return Err("Invalid Prime Inference model catalog".to_string()),
    };

    let mut models: Vec<PrimeInferenceCatalogEntry> = Vec::new();
    let mut seen: Vec<String> = Vec::new();
    for item in data {
        let item = match is_record(item) {
            Some(item) => item,
            None => continue,
        };
        let id = match item.get("id").and_then(Value::as_str) {
            Some(id) if !id.is_empty() && id.chars().count() <= 1_024 => id.to_string(),
            _ => continue,
        };
        if has_control_characters(&id) {
            continue;
        }
        if seen.contains(&id) {
            return Err(format!("Duplicate Prime Inference model {}", id));
        }
        let pricing = item.get("pricing").and_then(is_record).cloned().unwrap_or_default();
        let input = non_negative_number(pricing.get("input_usd_per_mtok"));
        let output = non_negative_number(pricing.get("output_usd_per_mtok"));
        let (input, output) = match (input, output) {
            (Some(input), Some(output)) => (input, output),
            _ => continue,
        };

        let name = match item.get("display_name").and_then(Value::as_str) {
            Some(value) => strip_control_characters(value).trim().to_string(),
            None => String::new(),
        };
        let specs = item.get("specs").and_then(is_record).cloned().unwrap_or_default();
        let modalities = specs
            .get("modalities")
            .and_then(is_record)
            .cloned()
            .unwrap_or_default();
        let input_modalities = match modalities.get("input") {
            Some(Value::Array(items)) if items.iter().all(Value::is_string) => Some(items.clone()),
            _ => None,
        };
        let output_modalities = match modalities.get("output") {
            Some(Value::Array(items)) if items.iter().all(Value::is_string) => Some(items.clone()),
            _ => None,
        };
        let context_window = positive_integer(specs.get("context_window"));
        let max_tokens = positive_integer(specs.get("max_output_tokens"));
        let reasoning = match specs.get("supports_reasoning") {
            Some(Value::Bool(value)) => Some(*value),
            _ => None,
        };
        let has_specs = context_window.is_some()
            && max_tokens.is_some()
            && reasoning.is_some()
            && input_modalities.is_some()
            && output_modalities.is_some();
        let cache_read = non_negative_number(pricing.get("cache_read_usd_per_mtok"));
        let cache_write = non_negative_number(pricing.get("cache_write_usd_per_mtok"));

        seen.push(id.clone());
        let mut entry = PrimeInferenceCatalogEntry {
            id,
            name: if name.is_empty() { None } else { Some(name) },
            input,
            output,
            cache_read,
            cache_write,
            context_window: None,
            max_tokens: None,
            vision: None,
            reasoning: None,
            supported_parameters: parse_string_array(item.get("supported_parameters")),
            reasoning_efforts: parse_string_array(item.get("reasoning").and_then(|spec| spec.get("supported_efforts"))),
            reasoning_mandatory: item.get("reasoning").and_then(|spec| spec.get("mandatory")).and_then(Value::as_bool),
        };
        if has_specs {
            let context_window = context_window.expect("checked above");
            let max_tokens = max_tokens.expect("checked above");
            entry.context_window = Some(context_window);
            entry.max_tokens = Some(max_tokens.min(context_window));
            entry.vision = Some(
                input_modalities
                    .expect("checked above")
                    .iter()
                    .any(|modality| modality.as_str() == Some("image")),
            );
            entry.reasoning = reasoning;
        }
        models.push(entry);
    }

    if models.is_empty() && !allow_empty {
        return Err("Prime Inference model catalog is empty".to_string());
    }
    Ok(models)
}

fn cache_costs(entry: &PrimeInferenceCatalogEntry, template: Option<&Model>) -> (f64, f64) {
    let anthropic = entry.id.to_lowercase().starts_with("anthropic/");
    let cache_read = entry
        .cache_read
        .or(template.map(|model| model.cost.cache_read))
        .unwrap_or(if anthropic { entry.input * 0.1 } else { 0.0 });
    let cache_write = entry
        .cache_write
        .or(template.map(|model| model.cost.cache_write))
        .unwrap_or(if anthropic { entry.input * 1.25 } else { 0.0 });
    (cache_read, cache_write)
}

/// `buildPrimeInferenceModels(bundledModels, entries, options)`.
pub fn build_prime_inference_models(
    bundled_models: &[Model],
    entries: &[PrimeInferenceCatalogEntry],
    include_private: bool,
    minimum_models: Option<f64>,
) -> Option<Vec<Model>> {
    let bundled: HashMap<String, &Model> = bundled_models
        .iter()
        .map(|model| (model.id.to_lowercase(), model))
        .collect();
    let mut models: Vec<Model> = Vec::new();
    for entry in entries {
        if !include_private && is_private_prime_inference_model_id(&entry.id) {
            continue;
        }
        let template = bundled.get(&entry.id.to_lowercase()).copied();
        if template.is_none()
            && (entry.context_window.is_none() || entry.max_tokens.is_none() || entry.reasoning.is_none())
        {
            continue;
        }
        let context_window = entry
            .context_window
            .or(template.map(|model| model.context_window))
            .unwrap_or(0.0);
        let max_tokens = entry
            .max_tokens
            .or(template.map(|model| model.max_tokens))
            .unwrap_or(0.0)
            .min(context_window);
        let (cache_read, cache_write) = cache_costs(entry, template);
        let input = if entry.vision.unwrap_or_else(|| template
                .map(|model| model.input.contains(&InputModality::Image))
                .unwrap_or(false))
        {
            vec![InputModality::Text, InputModality::Image]
        } else {
            vec![InputModality::Text]
        };
        let mut thinking_level_map: Option<ThinkingLevelMap> = template.and_then(|model| {
            model
                .thinking_level_map
                .as_ref()
                .map(|map| map.clone())
        });
        let mut compat = template
            .and_then(|model| model.compat.as_ref())
            .and_then(Compat::as_completions)
            .cloned()
            .unwrap_or_else(default_compat);
        if let Some(controls) = get_prime_inference_reasoning_controls(
            entry.supported_parameters.as_deref(), entry.reasoning_efforts.as_deref(),
            entry.reasoning_mandatory == Some(true),
        ) {
            compat.supports_reasoning_effort = Some(controls.supports_reasoning_effort);
            compat.thinking_format = controls.thinking_format;
            thinking_level_map = controls.thinking_level_map.or_else(|| {
                controls.supports_reasoning_effort.then_some(thinking_level_map).flatten()
            });
            if controls.supports_reasoning_effort && entry.reasoning_mandatory == Some(true) {
                thinking_level_map.get_or_insert_with(ThinkingLevelMap::new).insert("off".to_string(), None);
            }
        }
        models.push(Model {
            id: entry.id.clone(),
            name: entry
                .name
                .clone()
                .or_else(|| template.map(|model| model.name.clone()))
                .unwrap_or_else(|| entry.id.clone()),
            api: pi_ai::types::API_OPENAI_COMPLETIONS.to_string(),
            provider: crate::core::prime_inference_auth::PRIME_INFERENCE_PROVIDER_ID.to_string(),
            base_url: PRIME_INFERENCE_BASE_URL.to_string(),
            reasoning: entry
                .reasoning
                .or(template.map(|model| model.reasoning))
                .unwrap_or(false),
            thinking_level_map,
            input,
            cost: ModelCost {
                input: entry.input,
                output: entry.output,
                cache_read,
                cache_write,
            },
            context_window,
            max_input_tokens: None,
            max_tokens,
            featured: if template.map(|model| model.featured == Some(true)).unwrap_or(false) {
                Some(true)
            } else {
                None
            },
            native_compaction: None,
            headers: None,
            compat: Some(Compat::Completions(compat)),
        });
    }
    let minimum_models = minimum_models
        .unwrap_or_else(|| (bundled_models.len() as f64 * MIN_CATALOG_COVERAGE).ceil());
    let covered_bundled_models = models
        .iter()
        .filter(|model| bundled.contains_key(&model.id.to_lowercase()))
        .count();
    if covered_bundled_models as f64 >= minimum_models {
        Some(models)
    } else {
        None
    }
}

/// `mergePrimeInferenceModels(bundledModels, livePrimeInferenceModels?)`.
pub fn merge_prime_inference_models(
    bundled_models: &[Model],
    live_prime_inference_models: Option<&[Model]>,
) -> Vec<Model> {
    match live_prime_inference_models {
        None => bundled_models.to_vec(),
        Some(live) => {
            let mut merged: Vec<Model> = bundled_models
                .iter()
                .filter(|model| model.provider != crate::core::prime_inference_auth::PRIME_INFERENCE_PROVIDER_ID)
                .cloned()
                .collect();
            merged.extend(live.iter().cloned());
            merged
        }
    }
}

/// `readCachedPrimeInferenceModels(cachePath, bundledModels)`.
pub fn read_cached_prime_inference_models(cache_path: &str, bundled_models: &[Model]) -> Option<Vec<Model>> {
    if !Path::new(cache_path).exists() {
        return None;
    }
    let content = std::fs::read_to_string(cache_path).ok()?;
    let parsed: Value = serde_json::from_str(&content).ok()?;
    let entries = parse_prime_inference_model_catalog(&parsed, false).ok()?;
    build_prime_inference_models(bundled_models, &entries, false, None)
}

fn write_cache(cache_path: &str, value: &Value) {
    let payload = serde_json::to_string(value).unwrap_or_else(|_| "null".to_string());
    // The bundled catalog remains available when the cache cannot be persisted.
    let _ = write_file_atomic_sync(
        cache_path,
        &payload,
        WriteFileAtomicOptions {
            mode: Some(0o600),
            fsync: false,
            fsync_dir: false,
            before_rename: None,
        },
    );
}

/// `class PrimeInferenceCatalogRequestError extends Error`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrimeInferenceCatalogRequestError {
    pub status: u16,
}

impl std::fmt::Display for PrimeInferenceCatalogRequestError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "Prime Inference model catalog request failed with status {}",
            self.status
        )
    }
}

impl std::error::Error for PrimeInferenceCatalogRequestError {}

/// Error type of the catalog request path: either the typed status error or a
/// plain message (`Error("Response is too large")`, ...).
#[derive(Debug, Clone)]
pub enum CatalogError {
    Status(u16),
    Message(String),
}

impl CatalogError {
    pub fn status(&self) -> Option<u16> {
        match self {
            CatalogError::Status(status) => Some(*status),
            CatalogError::Message(_) => None,
        }
    }

    pub fn message(&self) -> String {
        match self {
            CatalogError::Status(status) => PrimeInferenceCatalogRequestError { status: *status }.to_string(),
            CatalogError::Message(message) => message.clone(),
        }
    }
}

fn read_response(response: &crate::core::prime_inference_auth::HttpResponse) -> Result<Value, CatalogError> {
    if !response.ok() {
        return Err(CatalogError::Status(response.status));
    }
    let content_length = response
        .header("content-length")
        .and_then(|value| value.trim().parse::<f64>().ok());
    if let Some(content_length) = content_length {
        if content_length.is_finite() && content_length > MAX_RESPONSE_BYTES as f64 {
            return Err(CatalogError::Message("Response is too large".to_string()));
        }
    }
    if response.text.is_empty() {
        return Err(CatalogError::Message("Response body is empty".to_string()));
    }
    if response.text.len() > MAX_RESPONSE_BYTES {
        return Err(CatalogError::Message("Response is too large".to_string()));
    }
    serde_json::from_str(&response.text).map_err(|error| CatalogError::Message(error.to_string()))
}

/// `fetchPrimeInferenceModelCatalog(options)`.
pub async fn fetch_prime_inference_model_catalog(
    fetch_fn: Option<FetchFn>,
    headers: Vec<(String, String)>,
    timeout_ms: Option<u64>,
    allow_empty: bool,
) -> Result<(Value, Vec<PrimeInferenceCatalogEntry>), CatalogError> {
    use crate::core::prime_inference_auth::{default_fetch, HttpRequest};
    let fetch_fn = fetch_fn.unwrap_or_else(default_fetch);
    let mut all_headers = vec![("accept".to_string(), "application/json".to_string())];
    all_headers.extend(headers);
    let response = fetch_fn(HttpRequest {
        method: "GET".to_string(),
        url: format!("{}/models", PRIME_INFERENCE_BASE_URL),
        headers: all_headers,
        body: None,
        timeout_ms: timeout_ms.unwrap_or(FETCH_TIMEOUT_MS),
    })
    .await
    .map_err(CatalogError::Message)?;
    let payload = read_response(&response)?;
    let entries = parse_prime_inference_model_catalog(&payload, allow_empty).map_err(CatalogError::Message)?;
    Ok((payload, entries))
}

/// `refreshPrimeInferenceModels(cachePath, bundledModels, options)`.
pub async fn refresh_prime_inference_models(
    cache_path: &str,
    bundled_models: &[Model],
    fetch_fn: Option<FetchFn>,
    offline: bool,
) -> Option<Vec<Model>> {
    let cached = read_cached_prime_inference_models(cache_path, bundled_models);
    if offline {
        return cached;
    }
    // In-flight refreshes are deduped per cache path (a JS `Promise` shared by
    // callers). The guard's scope ends before the await so the future stays `Send`.
    let in_flight = {
        let pending = pending_refreshes();
        let guard = pending.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.get(cache_path).cloned()
    };
    if let Some(existing) = in_flight {
        return (*existing).clone().await;
    }
    // The shared refresh outlives this call, so it owns its inputs instead of
    // borrowing the caller's `cachePath` / `bundledModels`.
    let cache_path_owned = cache_path.to_string();
    let bundled_models_owned = bundled_models.to_vec();
    let result = async move {
        match fetch_prime_inference_model_catalog(fetch_fn, Vec::new(), None, false).await {
            Ok((payload, entries)) => {
                let models =
                    build_prime_inference_models(&bundled_models_owned, &entries, false, None);
                match models {
                    None => cached,
                    Some(models) => {
                        write_cache(&cache_path_owned, &payload);
                        Some(models)
                    }
                }
            }
            Err(_) => cached,
        }
    };
    // `FutureExt::shared` needs a concrete `'static` future, so the async block is
    // boxed at that type before sharing.
    let shared: PendingRefresh = Arc::new(futures_util::FutureExt::shared(
        result.boxed() as futures::future::BoxFuture<'static, Option<Vec<Model>>>,
    ));
    {
        let pending = pending_refreshes();
        let mut guard = pending.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.insert(cache_path.to_string(), shared.clone());
    }
    let value = (*shared).clone().await;
    {
        let pending = pending_refreshes();
        let mut guard = pending.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(existing) = guard.get(cache_path) {
            if Arc::ptr_eq(existing, &shared) {
                guard.remove(cache_path);
            }
        }
    }
    value
}

/// Keeps the catalog entry order stable when building the picker list.
pub fn entries_by_id(entries: &[PrimeInferenceCatalogEntry]) -> IndexMap<String, PrimeInferenceCatalogEntry> {
    let mut map = IndexMap::new();
    for entry in entries {
        map.insert(entry.id.to_lowercase(), entry.clone());
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::prime_inference_auth::HttpResponse;
    use serde_json::json;
    use std::sync::Arc;

    fn catalog_value() -> Value {
        json!({
            "data": [
                {
                    "id": "z-ai/glm-5.3",
                    "display_name": "GLM 5.3",
                    "pricing": {
                        "input_usd_per_mtok": 0.5,
                        "output_usd_per_mtok": 1.5,
                        "cache_read_usd_per_mtok": 0.05,
                        "cache_write_usd_per_mtok": 0.6
                    },
                    "specs": {
                        "context_window": 200000,
                        "max_output_tokens": 131072,
                        "modalities": {"input": ["text", "image"], "output": ["text"]},
                        "supports_reasoning": true
                    }
                },
                {
                    "id": "anthropic/claude-sonnet-4-5",
                    "display_name": "Claude Sonnet 4.5",
                    "pricing": {"input_usd_per_mtok": 3, "output_usd_per_mtok": 15},
                    "specs": {
                        "context_window": 200000,
                        "max_output_tokens": 64000,
                        "modalities": {"input": ["text"], "output": ["text"]},
                        "supports_reasoning": false
                    }
                },
                {
                    "id": "internal/glm-5.2-fast",
                    "display_name": "GLM 5.2 Fast",
                    "pricing": {"input_usd_per_mtok": 0, "output_usd_per_mtok": 0}
                }
            ]
        })
    }

    fn template() -> Model {
        Model {
            id: "z-ai/glm-5.3".to_string(),
            name: "GLM 5.3".to_string(),
            api: pi_ai::types::API_OPENAI_COMPLETIONS.to_string(),
            provider: "prime-inference".to_string(),
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
            context_window: 200_000.0,
            max_input_tokens: None,
            max_tokens: 131_072.0,
            featured: Some(true),
            native_compaction: None,
            headers: None,
            compat: Some(Compat::Completions(default_compat())),
        }
    }

    #[test]
    fn private_model_ids_match_typescript() {
        assert!(is_private_prime_inference_model_id("internal/glm-5.2-fast"));
        assert!(is_private_prime_inference_model_id("dev/thing"));
        assert!(is_private_prime_inference_model_id("zai/glm:exacto"));
        assert!(!is_private_prime_inference_model_id("z-ai/glm-5.3"));
    }

    #[test]
    fn parse_rejects_invalid_payloads() {
        assert_eq!(
            parse_prime_inference_model_catalog(&json!([]), false).unwrap_err(),
            "Invalid Prime Inference model catalog"
        );
        assert_eq!(
            parse_prime_inference_model_catalog(&json!({"data": []}), false).unwrap_err(),
            "Prime Inference model catalog is empty"
        );
        assert!(parse_prime_inference_model_catalog(&json!({"data": []}), true).unwrap().is_empty());
        assert_eq!(
            parse_prime_inference_model_catalog(
                &json!({"data": [{"id": "a", "pricing": {"input_usd_per_mtok": 1, "output_usd_per_mtok": 1}}, {"id": "a", "pricing": {"input_usd_per_mtok": 1, "output_usd_per_mtok": 1}}]}),
                false
            )
            .unwrap_err(),
            "Duplicate Prime Inference model a"
        );
    }

    #[test]
    fn parse_skips_entries_without_pricing() {
        let entries = parse_prime_inference_model_catalog(
            &json!({"data": [
                {"id": "keep", "pricing": {"input_usd_per_mtok": 1, "output_usd_per_mtok": 2}},
                {"id": "drop-no-pricing"},
                {"id": "", "pricing": {"input_usd_per_mtok": 1, "output_usd_per_mtok": 2}},
                {"id": "drop-negative", "pricing": {"input_usd_per_mtok": -1, "output_usd_per_mtok": 2}}
            ]}),
            false,
        )
        .unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].id, "keep");
        assert_eq!(entries[0].input, 1.0);
        assert_eq!(entries[0].output, 2.0);
        assert!(entries[0].context_window.is_none());
    }

    #[test]
    fn parse_derives_specs_and_clamps_max_tokens() {
        let entries = parse_prime_inference_model_catalog(&catalog_value(), false).unwrap();
        let glm = entries.iter().find(|entry| entry.id == "z-ai/glm-5.3").unwrap();
        assert_eq!(glm.name.as_deref(), Some("GLM 5.3"));
        assert_eq!(glm.context_window, Some(200_000.0));
        assert_eq!(glm.max_tokens, Some(131_072.0));
        assert_eq!(glm.vision, Some(true));
        assert_eq!(glm.reasoning, Some(true));
        assert_eq!(glm.cache_read, Some(0.05));
        assert_eq!(glm.cache_write, Some(0.6));

        let clamped = parse_prime_inference_model_catalog(
            &json!({"data": [{
                "id": "x",
                "pricing": {"input_usd_per_mtok": 1, "output_usd_per_mtok": 1},
                "specs": {
                    "context_window": 1000,
                    "max_output_tokens": 5000,
                    "modalities": {"input": ["text"], "output": ["text"]},
                    "supports_reasoning": false
                }
            }]}),
            false,
        )
        .unwrap();
        assert_eq!(clamped[0].max_tokens, Some(1000.0));
    }

    #[test]
    fn build_models_fills_defaults_from_template() {
        let entries = parse_prime_inference_model_catalog(&catalog_value(), false).unwrap();
        let bundled = vec![template()];
        let models = build_prime_inference_models(&bundled, &entries, false, None).unwrap();
        assert_eq!(models.len(), 2, "private entry is excluded by default");
        let glm = models.iter().find(|model| model.id == "z-ai/glm-5.3").unwrap();
        assert_eq!(glm.provider, "prime-inference");
        assert_eq!(glm.api, pi_ai::types::API_OPENAI_COMPLETIONS);
        assert_eq!(glm.base_url, PRIME_INFERENCE_BASE_URL);
        assert_eq!(glm.featured, Some(true));
        assert!(glm.input.contains(&InputModality::Image));

        let sonnet = models
            .iter()
            .find(|model| model.id == "anthropic/claude-sonnet-4-5")
            .unwrap();
        assert_eq!(sonnet.cost.cache_read, 3.0 * 0.1);
        assert_eq!(sonnet.cost.cache_write, 3.75);
        assert_eq!(sonnet.max_tokens, 64000.0);
        assert!(!sonnet.reasoning);
        assert_eq!(sonnet.input, vec![InputModality::Text]);
    }

    #[test]
    fn live_route_reasoning_metadata_overrides_stale_template_controls() {
        let mut bundled = template();
        bundled.thinking_level_map = Some(ThinkingLevelMap::from([("xhigh".into(), Some("xhigh".into()))]));
        bundled.compat = Some(Compat::Completions(OpenAICompletionsCompat {
            supports_reasoning_effort: Some(false), thinking_format: Some("zai".into()), ..Default::default()
        }));
        let mut value = catalog_value();
        value["data"][0]["supported_parameters"] = json!(["reasoning_effort"]);
        value["data"][0]["reasoning"] = json!({"supported_efforts":["low","high"],"mandatory":true});
        let entries = parse_prime_inference_model_catalog(&value, false).unwrap();
        let models = build_prime_inference_models(&[bundled.clone()], &entries, false, None).unwrap();
        let model = models.iter().find(|model| model.id == bundled.id).unwrap();
        let compat = model.compat.as_ref().unwrap().as_completions().unwrap();
        assert_eq!(compat.supports_reasoning_effort, Some(true));
        assert_eq!(compat.thinking_format, None);
        assert_eq!(pi_ai::models::get_supported_thinking_levels(model), ["low", "high"]);

        value["data"][0]["supported_parameters"] = json!(["temperature"]);
        let entries = parse_prime_inference_model_catalog(&value, false).unwrap();
        let models = build_prime_inference_models(&[bundled], &entries, false, None).unwrap();
        let model = &models[0];
        let compat = model.compat.as_ref().unwrap().as_completions().unwrap();
        assert_eq!(compat.supports_reasoning_effort, Some(false));
        assert_eq!(compat.thinking_format, None);
        assert!(model.thinking_level_map.is_none());
    }

    #[test]
    fn missing_live_efforts_preserve_template_bounds_and_mandatory_reasoning() {
        let mut bundled = template();
        bundled.thinking_level_map = Some(ThinkingLevelMap::from([
            ("minimal".into(), None), ("low".into(), None), ("medium".into(), None),
            ("high".into(), Some("high".into())), ("xhigh".into(), None), ("max".into(), None),
        ]));
        let original = bundled.thinking_level_map.clone();
        let entries = parse_prime_inference_model_catalog(&catalog_value(), false).unwrap();
        let models = build_prime_inference_models(&[bundled.clone()], &entries, false, None).unwrap();
        assert_eq!(models[0].thinking_level_map, original);
        let mut value = catalog_value();
        value["data"][0]["supported_parameters"] = json!(["reasoning_effort"]);
        value["data"][0]["reasoning"] = json!({"mandatory":true});
        let entries = parse_prime_inference_model_catalog(&value, false).unwrap();
        let models = build_prime_inference_models(&[bundled], &entries, false, None).unwrap();
        assert_eq!(pi_ai::models::get_supported_thinking_levels(&models[0]), ["high"]);
    }

    #[test]
    fn build_models_respects_minimum_coverage() {
        let entries = parse_prime_inference_model_catalog(&catalog_value(), false).unwrap();
        let bundled = vec![template()];
        // Only one of two bundled models is covered: below the 0.5 * 2 = 1 floor? it equals 1, so passes.
        assert!(build_prime_inference_models(&bundled, &entries, false, None).is_some());
        assert!(build_prime_inference_models(&bundled, &entries, false, Some(2.0)).is_none());
    }

    #[test]
    fn explicit_text_only_specs_override_a_vision_template() {
        let mut bundled = template();
        bundled.input.push(InputModality::Image);
        let mut entries = parse_prime_inference_model_catalog(&catalog_value(), false).unwrap();
        entries.retain(|entry| entry.id == bundled.id);
        entries[0].vision = Some(false);
        let models = build_prime_inference_models(&[bundled.clone()], &entries, false, None).unwrap();
        assert_eq!(models[0].input, vec![InputModality::Text]);
        entries[0].vision = None;
        let models = build_prime_inference_models(&[bundled], &entries, false, None).unwrap();
        assert!(models[0].input.contains(&InputModality::Image));
    }

    #[test]
    fn merge_replaces_only_prime_inference_models() {
        let bundled = vec![
            template(),
            Model {
                provider: "anthropic".to_string(),
                id: "claude-sonnet-4-5".to_string(),
                ..template()
            },
        ];
        let merged = merge_prime_inference_models(&bundled, None);
        assert_eq!(merged.len(), 2);

        let live = vec![Model {
            id: "live".to_string(),
            ..template()
        }];
        let merged = merge_prime_inference_models(&bundled, Some(&live));
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].id, "claude-sonnet-4-5");
        assert_eq!(merged[1].id, "live");
    }

    #[test]
    fn read_cached_models_returns_none_for_missing_or_bad_cache() {
        let bundled = vec![template()];
        assert!(read_cached_prime_inference_models("Z:/definitely/missing.json", &bundled).is_none());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cache.json");
        let path = path.to_string_lossy().to_string();
        std::fs::write(&path, "{not json").unwrap();
        assert!(read_cached_prime_inference_models(&path, &bundled).is_none());
        std::fs::write(&path, catalog_value().to_string()).unwrap();
        let models = read_cached_prime_inference_models(&path, &bundled).unwrap();
        assert_eq!(models.len(), 2);
    }

    #[tokio::test]
    async fn fetch_catalog_reports_status_and_empty_body() {
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let seen_for_fetch = Arc::clone(&seen);
        let fetch_fn: FetchFn = Arc::new(move |request| {
            seen_for_fetch.lock().unwrap().push(request);
            Box::pin(async move {
                Ok(HttpResponse {
                    status: 401,
                    status_text: "Unauthorized".to_string(),
                    headers: Vec::new(),
                    text: "{}".to_string(),
                })
            }) as pi_ai::types::BoxFuture<Result<HttpResponse, String>>
        });
        let error = fetch_prime_inference_model_catalog(Some(fetch_fn), Vec::new(), None, false)
            .await
            .unwrap_err();
        assert_eq!(error.status(), Some(401));
        assert_eq!(
            error.message(),
            "Prime Inference model catalog request failed with status 401"
        );
        let requests = seen.lock().unwrap();
        assert_eq!(requests[0].url, "https://api.pinference.ai/api/v1/models");
        assert_eq!(requests[0].headers[0], ("accept".to_string(), "application/json".to_string()));
    }

    #[tokio::test]
    async fn fetch_catalog_rejects_oversized_response() {
        let fetch_fn: FetchFn = Arc::new(|_request| {
            Box::pin(async move {
                Ok(HttpResponse {
                    status: 200,
                    status_text: String::new(),
                    headers: vec![("content-length".to_string(), "999999999".to_string())],
                    text: "{}".to_string(),
                })
            }) as pi_ai::types::BoxFuture<Result<HttpResponse, String>>
        });
        let error = fetch_prime_inference_model_catalog(Some(fetch_fn), Vec::new(), None, false)
            .await
            .unwrap_err();
        assert_eq!(error.message(), "Response is too large");
    }

    #[tokio::test]
    async fn refresh_writes_cache_and_dedupes() {
        let dir = tempfile::tempdir().unwrap();
        let cache_path = dir.path().join("prime-inference-models-cache.json");
        let cache_path = cache_path.to_string_lossy().to_string();
        let bundled = vec![template()];
        let fetch_fn: FetchFn = Arc::new(|_request| {
            Box::pin(async move {
                Ok(HttpResponse {
                    status: 200,
                    status_text: String::new(),
                    headers: Vec::new(),
                    text: catalog_value().to_string(),
                })
            }) as pi_ai::types::BoxFuture<Result<HttpResponse, String>>
        });
        let models = refresh_prime_inference_models(&cache_path, &bundled, Some(fetch_fn), false)
            .await
            .unwrap();
        assert_eq!(models.len(), 2);
        assert!(std::path::Path::new(&cache_path).exists());

        // Offline reads the cache and never fetches.
        let offline_fetch: FetchFn = Arc::new(|_request| {
            Box::pin(async move { Err("must not be called".to_string()) }) as pi_ai::types::BoxFuture<Result<HttpResponse, String>>
        });
        let cached = refresh_prime_inference_models(&cache_path, &bundled, Some(offline_fetch), true)
            .await
            .unwrap();
        assert_eq!(cached.len(), 2);
    }

    #[tokio::test]
    async fn refresh_keeps_cache_on_fetch_failure() {
        let dir = tempfile::tempdir().unwrap();
        let cache_path = dir.path().join("cache.json");
        let cache_path = cache_path.to_string_lossy().to_string();
        let bundled = vec![template()];
        let failing: FetchFn = Arc::new(|_request| {
            Box::pin(async move { Err("network down".to_string()) }) as pi_ai::types::BoxFuture<Result<HttpResponse, String>>
        });
        assert!(refresh_prime_inference_models(&cache_path, &bundled, Some(failing), false)
            .await
            .is_none());
        assert!(!std::path::Path::new(&cache_path).exists());
    }
}
