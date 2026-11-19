//! Port of packages/ai/src/compaction.ts

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::types::{Api, Model, SimpleStreamOptions, Usage};

/// An opaque provider checkpoint; replay the entire window without rewriting its items.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderCompactionCheckpoint {
    pub version: i64,
    pub provider: String,
    pub api: Api,
    pub model: String,
    pub base_url: String,
    /// Exact request endpoint when recorded by a newer adapter. Legacy checkpoints omit it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
    pub items: Vec<Map<String, Value>>,
    pub estimated_tokens: f64,
}

/// Typed unary provider-compaction failure. `status == 0` means no HTTP status.
/// `Ok(None)` is reserved for a provider-declared unsupported compaction route.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("{message}")]
pub struct CompactionRequestError {
    pub message: String,
    pub name: &'static str,
    pub status: i64,
    pub retry_after_ms: Option<f64>,
}

impl CompactionRequestError {
    pub fn new(message: impl Into<String>, status: i64, retry_after_ms: Option<f64>) -> Self {
        Self { message: message.into(), name: "CompactionRequestError", status, retry_after_ms }
    }
}

impl From<String> for CompactionRequestError {
    fn from(message: String) -> Self {
        Self::new(message, 0, None)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderCompactionResult {
    pub checkpoint: ProviderCompactionCheckpoint,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactionOptions {
    #[serde(flatten)]
    pub simple: SimpleStreamOptions,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub custom_instructions: Option<String>,
}

/// `isCompactionCheckpoint(value: unknown)`.
pub fn is_compaction_checkpoint(value: &Value) -> bool {
    let Some(checkpoint) = value.as_object() else {
        return false;
    };
    let version_ok = checkpoint.get("version").and_then(Value::as_i64) == Some(1);
    let provider_ok = checkpoint.get("provider").map(Value::is_string).unwrap_or(false);
    let api_ok = checkpoint.get("api").map(Value::is_string).unwrap_or(false);
    let model_ok = checkpoint.get("model").map(Value::is_string).unwrap_or(false);
    let base_url_ok = checkpoint.get("baseUrl").map(Value::is_string).unwrap_or(false);
    let endpoint_ok = match checkpoint.get("endpoint") {
        None => true,
        Some(Value::String(endpoint)) => !endpoint.is_empty(),
        Some(_) => false,
    };
    let estimated_ok = match checkpoint.get("estimatedTokens") {
        Some(Value::Number(number)) => number
            .as_f64()
            .map(|n| n.is_finite() && n >= 0.0)
            .unwrap_or(false),
        _ => false,
    };
    let items_ok = match checkpoint.get("items") {
        Some(Value::Array(items)) => {
            !items.is_empty()
                && items
                    .iter()
                    .all(|item| matches!(item, Value::Object(_)))
        }
        _ => false,
    };

    version_ok
        && provider_ok
        && api_ok
        && model_ok
        && base_url_ok
        && endpoint_ok
        && estimated_ok
        && items_ok
}

fn trim_trailing_slashes(value: &str) -> &str {
    value.trim_end_matches('/')
}

/// `compactionMatchesModel(checkpoint, model)`.
pub fn compaction_matches_model(checkpoint: &ProviderCompactionCheckpoint, model: &Model) -> bool {
    if checkpoint.provider != model.provider {
        return false;
    }
    if checkpoint.model != model.id {
        return false;
    }
    if checkpoint.api != model.api {
        return false;
    }
    if trim_trailing_slashes(&checkpoint.base_url) != trim_trailing_slashes(&model.base_url) {
        return false;
    }
    match (&checkpoint.endpoint, &model.native_compaction) {
        (None, None) => true,
        (Some(checkpoint_endpoint), Some(native)) => {
            trim_trailing_slashes(checkpoint_endpoint) == trim_trailing_slashes(&native.endpoint)
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn checkpoint_json() -> Value {
        json!({
            "version": 1,
            "provider": "openai",
            "api": "openai-responses",
            "model": "gpt-5",
            "baseUrl": "https://api.openai.com/v1",
            "items": [{"type": "message"}],
            "estimatedTokens": 10,
        })
    }

    #[test]
    fn accepts_minimal_checkpoint() {
        assert!(is_compaction_checkpoint(&checkpoint_json()));
        let checkpoint: ProviderCompactionCheckpoint = serde_json::from_value(checkpoint_json()).unwrap();
        let serialized = serde_json::to_value(checkpoint).unwrap();
        assert!(is_compaction_checkpoint(&serialized));
        assert_eq!(serialized["baseUrl"], checkpoint_json()["baseUrl"]);
        assert_eq!(serialized["estimatedTokens"].as_f64(), Some(10.0));
    }

    #[test]
    fn rejects_non_objects_and_bad_shapes() {
        assert!(!is_compaction_checkpoint(&json!(null)));
        assert!(!is_compaction_checkpoint(&json!("x")));
        let mut value = checkpoint_json();
        value["version"] = json!(2);
        assert!(!is_compaction_checkpoint(&value));
        let mut value = checkpoint_json();
        value["items"] = json!([]);
        assert!(!is_compaction_checkpoint(&value));
        let mut value = checkpoint_json();
        value["items"] = json!([1]);
        assert!(!is_compaction_checkpoint(&value));
        let mut value = checkpoint_json();
        value["estimatedTokens"] = json!(-1);
        assert!(!is_compaction_checkpoint(&value));
        let mut value = checkpoint_json();
        value["endpoint"] = json!("");
        assert!(!is_compaction_checkpoint(&value));
        value["endpoint"] = Value::Null;
        assert!(!is_compaction_checkpoint(&value));
    }

    #[test]
    fn accepts_endpoint_when_undefined_or_non_empty() {
        let mut value = checkpoint_json();
        value["endpoint"] = json!("https://api.openai.com/v1/responses/compact");
        assert!(is_compaction_checkpoint(&value));
    }

    #[test]
    fn matches_model_ignores_trailing_slashes() {
        let mut model = Model::default();
        model.provider = "openai".to_string();
        model.id = "gpt-5".to_string();
        model.api = "openai-responses".to_string();
        model.base_url = "https://api.openai.com/v1/".to_string();

        let checkpoint = ProviderCompactionCheckpoint {
            version: 1,
            provider: "openai".to_string(),
            api: "openai-responses".to_string(),
            model: "gpt-5".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            endpoint: None,
            items: vec![Map::new()],
            estimated_tokens: 1.0,
        };
        assert!(compaction_matches_model(&checkpoint, &model));

        let mut other = model.clone();
        other.id = "gpt-4".to_string();
        assert!(!compaction_matches_model(&checkpoint, &other));
    }
}
