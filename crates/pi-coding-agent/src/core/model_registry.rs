//! Port of packages/coding-agent/src/core/model-registry.ts
//!
//! Model registry - manages built-in and custom models, provides API key resolution.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

use indexmap::IndexMap;
use pi_ai::api_registry::ApiStreamSimpleFunction;
use pi_ai::types::{
    Api, Compat, Context, Model, ModelCost, NativeCompactionCapability, SimpleStreamOptions, ThinkingLevelMap,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::core::auth_storage::{
    register_oauth_provider, resolve_config_value_or_throw, resolve_config_value_uncached,
    resolve_headers_or_throw, AuthCredential, AuthSourceToken, AuthStatus, AuthStorage,
};
use crate::core::prime_inference_auth::{default_fetch, FetchFn, HttpRequest, PRIME_INFERENCE_PROVIDER_ID};
use crate::core::prime_inference_model_catalog::{
    build_prime_inference_models, merge_prime_inference_models, parse_prime_inference_model_catalog,
    read_cached_prime_inference_models, refresh_prime_inference_models,
};
use crate::core::prime_inference_models::{
    fetch_authorized_private_prime_inference_models, get_private_prime_inference_models,
    is_private_prime_inference_model,
};
use crate::core::provider_display_names::built_in_provider_display_names;

// ---------------------------------------------------------------------------
// pi-ai boundary: models.ts `getProviders()` / `getModels(provider)`
// ---------------------------------------------------------------------------

/// `getProviders()` from packages/ai/src/models.ts.
fn get_providers() -> Vec<String> {
    pi_ai::models_generated::models().keys().cloned().collect()
}

/// `getModels(provider)` from packages/ai/src/models.ts.
fn get_models(provider: &str) -> Vec<Model> {
    pi_ai::models_generated::models_for_provider(provider)
        .map(|provider_models| provider_models.values().cloned().collect())
        .unwrap_or_default()
}

/// `resetApiProviders()` from packages/ai/src/api-registry.ts.
fn reset_api_providers() {
    // `resetApiProviders` (providers/register-builtins.ts:449-452) CLEARS and then
    // re-registers the built-ins. Clearing alone leaves the registry empty, and since
    // the `--print` path can refresh the model registry before it streams, every
    // provider resolution would then panic with
    // "No API provider registered for api: ...".
    pi_ai::api_registry::clear_api_providers();
    pi_ai::providers::register_builtins::register_built_in_api_providers();
}

// ---------------------------------------------------------------------------
// Schema types (typebox -> serde structs)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelDefinition {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_level_map: Option<ThinkingLevelMap>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost: Option<ModelCost>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_window: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_input_tokens: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub native_compaction: Option<NativeCompactionCapability>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<IndexMap<String, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compat: Option<Compat>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelOverride {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_level_map: Option<ThinkingLevelMap>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost: Option<PartialModelCost>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_window: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_input_tokens: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<IndexMap<String, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compat: Option<Compat>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PartialModelCost {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<f64>,
    #[serde(rename = "cacheRead", skip_serializing_if = "Option::is_none")]
    pub cache_read: Option<f64>,
    #[serde(rename = "cacheWrite", skip_serializing_if = "Option::is_none")]
    pub cache_write: Option<f64>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<IndexMap<String, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compat: Option<Compat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auth_header: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub models: Option<Vec<ModelDefinition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_overrides: Option<IndexMap<String, ModelOverride>>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelsConfig {
    pub providers: IndexMap<String, ProviderConfig>,
}

// ---------------------------------------------------------------------------
// models.json schema validation (typebox Compile(ModelsConfigSchema) equivalent)
// ---------------------------------------------------------------------------

fn validation_path(path: &str, key: &str) -> String {
    if path.is_empty() {
        key.to_string()
    } else {
        format!("{}.{}", path, key)
    }
}

fn expect_string(value: &Value, path: &str, errors: &mut Vec<String>, min_length: usize) {
    match value.as_str() {
        Some(text) if text.chars().count() >= min_length => {}
        Some(_) => errors.push(format!(
            "  - {}: Expected string length greater or equal to {}",
            path, min_length
        )),
        None => errors.push(format!("  - {}: Expected string", path)),
    }
}

fn expect_number(value: &Value, path: &str, errors: &mut Vec<String>) {
    match value.as_f64() {
        Some(number) if number.is_finite() => {}
        _ => errors.push(format!("  - {}: Expected number", path)),
    }
}

fn expect_integer(value: &Value, path: &str, minimum: f64, errors: &mut Vec<String>) {
    match value.as_f64() {
        Some(number) if number.is_finite() && number.fract() == 0.0 && number >= minimum => {}
        _ => errors.push(format!("  - {}: Expected integer greater or equal to {}", path, minimum)),
    }
}

fn expect_boolean(value: &Value, path: &str, errors: &mut Vec<String>) {
    if !value.is_boolean() {
        errors.push(format!("  - {}: Expected boolean", path));
    }
}

fn expect_string_record(value: &Value, path: &str, errors: &mut Vec<String>) {
    match value.as_object() {
        Some(object) => {
            for (key, entry) in object {
                if !entry.is_string() {
                    errors.push(format!("  - {}: Expected string", validation_path(path, key)));
                }
            }
        }
        None => errors.push(format!("  - {}: Expected object", path)),
    }
}

fn validate_thinking_level_map(value: &Value, path: &str, errors: &mut Vec<String>) {
    let Some(object) = value.as_object() else {
        errors.push(format!("  - {}: Expected object", path));
        return;
    };
    for (key, entry) in object {
        if !entry.is_null() && !entry.is_string() {
            errors.push(format!(
                "  - {}: Expected union",
                validation_path(path, key)
            ));
        }
    }
}

fn validate_compat(value: &Value, path: &str, errors: &mut Vec<String>) {
    let Some(object) = value.as_object() else {
        errors.push(format!("  - {}: Expected union", path));
        return;
    };
    for key in [
        "supportsStore",
        "supportsDeveloperRole",
        "supportsReasoningEffort",
        "supportsUsageInStreaming",
        "requiresToolResultName",
        "requiresAssistantAfterToolResult",
        "requiresThinkingAsText",
        "requiresReasoningContentOnAssistantMessages",
        "supportsStrictMode",
        "supportsLongCacheRetention",
        "sendSessionIdHeader",
        "supportsEagerToolInputStreaming",
    ] {
        if let Some(entry) = object.get(key) {
            expect_boolean(entry, &validation_path(path, key), errors);
        }
    }
    if let Some(entry) = object.get("maxTokensField") {
        let valid = matches!(
            entry.as_str(),
            Some("max_completion_tokens") | Some("max_tokens")
        );
        if !valid {
            errors.push(format!(
                "  - {}: Expected union",
                validation_path(path, "maxTokensField")
            ));
        }
    }
    if let Some(entry) = object.get("thinkingFormat") {
        let valid = matches!(
            entry.as_str(),
            Some("openai")
                | Some("openrouter")
                | Some("deepseek")
                | Some("zai")
                | Some("qwen")
                | Some("qwen-chat-template")
        );
        if !valid {
            errors.push(format!(
                "  - {}: Expected union",
                validation_path(path, "thinkingFormat")
            ));
        }
    }
    if let Some(entry) = object.get("cacheControlFormat") {
        if entry.as_str() != Some("anthropic") {
            errors.push(format!(
                "  - {}: Expected literal",
                validation_path(path, "cacheControlFormat")
            ));
        }
    }
    if let Some(entry) = object.get("openRouterRouting") {
        match entry.as_object() {
            Some(routing) => {
                for key in ["allow_fallbacks", "require_parameters", "zdr", "enforce_distillable_text"] {
                    if let Some(inner) = routing.get(key) {
                        expect_boolean(inner, &validation_path(&validation_path(path, "openRouterRouting"), key), errors);
                    }
                }
                if let Some(inner) = routing.get("data_collection") {
                    let valid = matches!(inner.as_str(), Some("deny") | Some("allow"));
                    if !valid {
                        errors.push(format!(
                            "  - {}: Expected union",
                            validation_path(&validation_path(path, "openRouterRouting"), "data_collection")
                        ));
                    }
                }
            }
            None => errors.push(format!(
                "  - {}: Expected object",
                validation_path(path, "openRouterRouting")
            )),
        }
    }
    if let Some(entry) = object.get("vercelGatewayRouting") {
        if !entry.is_object() {
            errors.push(format!(
                "  - {}: Expected object",
                validation_path(path, "vercelGatewayRouting")
            ));
        }
    }
}

fn validate_native_compaction_schema(value: &Value, path: &str, errors: &mut Vec<String>) {
    let Some(object) = value.as_object() else {
        errors.push(format!("  - {}: Expected object", path));
        return;
    };
    for key in ["protocol", "provider", "model", "endpoint", "apiVersion"] {
        match object.get(key) {
            None => errors.push(format!("  - {}: Expected required property", validation_path(path, key))),
            Some(entry) => {
                if key == "protocol" && entry.as_str() != Some("openai-responses-compact-v1") {
                    errors.push(format!("  - {}: Expected literal", validation_path(path, key)));
                } else if key == "apiVersion" && entry.as_str() != Some("v1") {
                    errors.push(format!("  - {}: Expected literal", validation_path(path, key)));
                } else if !entry.is_string() {
                    errors.push(format!("  - {}: Expected string", validation_path(path, key)));
                }
            }
        }
    }
    if let Some(entry) = object.get("enabled") {
        expect_boolean(entry, &validation_path(path, "enabled"), errors);
    } else {
        errors.push(format!("  - {}: Expected required property", validation_path(path, "enabled")));
    }
    match object.get("validation") {
        Some(entry) => {
            let valid = matches!(
                entry.as_str(),
                Some("unverified") | Some("documentation-verified") | Some("live-verified")
            );
            if !valid {
                errors.push(format!("  - {}: Expected union", validation_path(path, "validation")));
            }
        }
        None => errors.push(format!(
            "  - {}: Expected required property",
            validation_path(path, "validation")
        )),
    }
}

fn validate_model_definition(value: &Value, path: &str, errors: &mut Vec<String>) {
    let Some(object) = value.as_object() else {
        errors.push(format!("  - {}: Expected object", path));
        return;
    };
    match object.get("id") {
        None => errors.push(format!("  - {}: Expected required property", validation_path(path, "id"))),
        Some(entry) => {
            expect_string(entry, &validation_path(path, "id"), errors, 1);
        }
    }
    for key in ["name", "api", "baseUrl"] {
        if let Some(entry) = object.get(key) {
            expect_string(entry, &validation_path(path, key), errors, 1);
        }
    }
    if let Some(entry) = object.get("reasoning") {
        expect_boolean(entry, &validation_path(path, "reasoning"), errors);
    }
    if let Some(entry) = object.get("thinkingLevelMap") {
        validate_thinking_level_map(entry, &validation_path(path, "thinkingLevelMap"), errors);
    }
    if let Some(entry) = object.get("input") {
        match entry.as_array() {
            Some(items) => {
                for (index, item) in items.iter().enumerate() {
                    let valid = matches!(item.as_str(), Some("text") | Some("image"));
                    if !valid {
                        errors.push(format!(
                            "  - {}: Expected union",
                            validation_path(&validation_path(path, "input"), &index.to_string())
                        ));
                    }
                }
            }
            None => errors.push(format!("  - {}: Expected array", validation_path(path, "input"))),
        }
    }
    if let Some(entry) = object.get("cost") {
        match entry.as_object() {
            Some(cost) => {
                for key in ["input", "output", "cacheRead", "cacheWrite"] {
                    match cost.get(key) {
                        None => errors.push(format!(
                            "  - {}: Expected required property",
                            validation_path(&validation_path(path, "cost"), key)
                        )),
                        Some(inner) => {
                            expect_number(inner, &validation_path(&validation_path(path, "cost"), key), errors);
                        }
                    }
                }
            }
            None => errors.push(format!("  - {}: Expected object", validation_path(path, "cost"))),
        }
    }
    if let Some(entry) = object.get("contextWindow") {
        expect_number(entry, &validation_path(path, "contextWindow"), errors);
    }
    if let Some(entry) = object.get("maxInputTokens") {
        expect_integer(entry, &validation_path(path, "maxInputTokens"), 1.0, errors);
    }
    if let Some(entry) = object.get("maxTokens") {
        expect_number(entry, &validation_path(path, "maxTokens"), errors);
    }
    if let Some(entry) = object.get("nativeCompaction") {
        validate_native_compaction_schema(entry, &validation_path(path, "nativeCompaction"), errors);
    }
    if let Some(entry) = object.get("headers") {
        expect_string_record(entry, &validation_path(path, "headers"), errors);
    }
    if let Some(entry) = object.get("compat") {
        validate_compat(entry, &validation_path(path, "compat"), errors);
    }
}

fn validate_model_override(value: &Value, path: &str, errors: &mut Vec<String>) {
    let Some(object) = value.as_object() else {
        errors.push(format!("  - {}: Expected object", path));
        return;
    };
    if let Some(entry) = object.get("name") {
        expect_string(entry, &validation_path(path, "name"), errors, 1);
    }
    if let Some(entry) = object.get("reasoning") {
        expect_boolean(entry, &validation_path(path, "reasoning"), errors);
    }
    if let Some(entry) = object.get("thinkingLevelMap") {
        validate_thinking_level_map(entry, &validation_path(path, "thinkingLevelMap"), errors);
    }
    if let Some(entry) = object.get("input") {
        match entry.as_array() {
            Some(items) => {
                for (index, item) in items.iter().enumerate() {
                    let valid = matches!(item.as_str(), Some("text") | Some("image"));
                    if !valid {
                        errors.push(format!(
                            "  - {}: Expected union",
                            validation_path(&validation_path(path, "input"), &index.to_string())
                        ));
                    }
                }
            }
            None => errors.push(format!("  - {}: Expected array", validation_path(path, "input"))),
        }
    }
    if let Some(entry) = object.get("cost") {
        match entry.as_object() {
            Some(cost) => {
                for key in ["input", "output", "cacheRead", "cacheWrite"] {
                    if let Some(inner) = cost.get(key) {
                        expect_number(inner, &validation_path(&validation_path(path, "cost"), key), errors);
                    }
                }
            }
            None => errors.push(format!("  - {}: Expected object", validation_path(path, "cost"))),
        }
    }
    if let Some(entry) = object.get("contextWindow") {
        expect_number(entry, &validation_path(path, "contextWindow"), errors);
    }
    if let Some(entry) = object.get("maxInputTokens") {
        expect_integer(entry, &validation_path(path, "maxInputTokens"), 1.0, errors);
    }
    if let Some(entry) = object.get("maxTokens") {
        expect_number(entry, &validation_path(path, "maxTokens"), errors);
    }
    if let Some(entry) = object.get("headers") {
        expect_string_record(entry, &validation_path(path, "headers"), errors);
    }
    if let Some(entry) = object.get("compat") {
        validate_compat(entry, &validation_path(path, "compat"), errors);
    }
}

/// `Compile(ModelsConfigSchema)` equivalent: collect `  - <path>: <message>` lines.
pub fn validate_models_config_schema(value: &Value) -> Vec<String> {
    let mut errors: Vec<String> = Vec::new();
    let Some(root) = value.as_object() else {
        errors.push("  - root: Expected object".to_string());
        return errors;
    };
    let Some(providers) = root.get("providers") else {
        errors.push("  - providers: Expected required property".to_string());
        return errors;
    };
    let Some(providers) = providers.as_object() else {
        errors.push("  - providers: Expected object".to_string());
        return errors;
    };
    for (provider_name, provider) in providers {
        // typebox instancePath is "/providers/<name>" -> "providers.<name>".
        let path = validation_path("providers", provider_name);
        let Some(provider) = provider.as_object() else {
            errors.push(format!("  - {}: Expected object", path));
            continue;
        };
        for key in ["name", "baseUrl", "apiKey", "api"] {
            if let Some(entry) = provider.get(key) {
                expect_string(entry, &validation_path(&path, key), &mut errors, 1);
            }
        }
        if let Some(entry) = provider.get("headers") {
            expect_string_record(entry, &validation_path(&path, "headers"), &mut errors);
        }
        if let Some(entry) = provider.get("compat") {
            validate_compat(entry, &validation_path(&path, "compat"), &mut errors);
        }
        if let Some(entry) = provider.get("authHeader") {
            expect_boolean(entry, &validation_path(&path, "authHeader"), &mut errors);
        }
        if let Some(entry) = provider.get("models") {
            match entry.as_array() {
                Some(items) => {
                    for (index, item) in items.iter().enumerate() {
                        validate_model_definition(
                            item,
                            &validation_path(&validation_path(&path, "models"), &index.to_string()),
                            &mut errors,
                        );
                    }
                }
                None => errors.push(format!("  - {}: Expected array", validation_path(&path, "models"))),
            }
        }
        if let Some(entry) = provider.get("modelOverrides") {
            match entry.as_object() {
                Some(overrides) => {
                    for (model_id, override_value) in overrides {
                        validate_model_override(
                            override_value,
                            &validation_path(&validation_path(&path, "modelOverrides"), model_id),
                            &mut errors,
                        );
                    }
                }
                None => errors.push(format!(
                    "  - {}: Expected object",
                    validation_path(&path, "modelOverrides")
                )),
            }
        }
    }
    errors
}

// ---------------------------------------------------------------------------
// Config helpers
// ---------------------------------------------------------------------------

/// Strip `//` line comments and trailing commas from JSON, leaving string
/// literals untouched.
pub fn strip_json_comments(input: &str) -> String {
    // Pass 1: drop `//` comments that are outside string literals.
    let mut without_comments = String::with_capacity(input.len());
    let chars: Vec<char> = input.chars().collect();
    let mut index = 0usize;
    while index < chars.len() {
        let ch = chars[index];
        if ch == '"' {
            without_comments.push(ch);
            index += 1;
            while index < chars.len() {
                let inner = chars[index];
                without_comments.push(inner);
                if inner == '\\' {
                    index += 1;
                    if index < chars.len() {
                        without_comments.push(chars[index]);
                        index += 1;
                    }
                    continue;
                }
                index += 1;
                if inner == '"' {
                    break;
                }
            }
            continue;
        }
        if ch == '/' && chars.get(index + 1) == Some(&'/') {
            while index < chars.len() && chars[index] != '\n' {
                index += 1;
            }
            continue;
        }
        without_comments.push(ch);
        index += 1;
    }

    // Pass 2: drop a trailing comma before `}` or `]` that is outside a string.
    let chars: Vec<char> = without_comments.chars().collect();
    let mut out = String::with_capacity(chars.len());
    let mut index = 0usize;
    while index < chars.len() {
        let ch = chars[index];
        if ch == '"' {
            out.push(ch);
            index += 1;
            while index < chars.len() {
                let inner = chars[index];
                out.push(inner);
                if inner == '\\' {
                    index += 1;
                    if index < chars.len() {
                        out.push(chars[index]);
                        index += 1;
                    }
                    continue;
                }
                index += 1;
                if inner == '"' {
                    break;
                }
            }
            continue;
        }
        if ch == ',' {
            let mut lookahead = index + 1;
            while lookahead < chars.len() && chars[lookahead].is_whitespace() {
                lookahead += 1;
            }
            if matches!(chars.get(lookahead).copied(), Some('}') | Some(']')) {
                index += 1;
                continue;
            }
        }
        out.push(ch);
        index += 1;
    }
    out
}

fn normalized_http_endpoint(value: &str) -> Option<String> {
    let url = url::Url::parse(value).ok()?;
    if url.scheme() != "http" && url.scheme() != "https" {
        return None;
    }
    if !url.username().is_empty() || url.password().is_some() || url.query().is_some() || url.fragment().is_some() {
        return None;
    }
    let trimmed_path = url.path().trim_end_matches('/').to_string();
    let mut normalized = url.clone();
    normalized.set_path(&trimmed_path);
    let text = normalized.to_string();
    Some(text.strip_suffix('/').unwrap_or(&text).to_string())
}

/// The `modelDef` half of `validateNativeCompactionCapability` (both the
/// models.json `ModelDefinition` and the `registerProvider` model shape).
struct NativeCompactionModelRef<'a> {
    id: &'a str,
    api: Option<&'a str>,
    base_url: Option<&'a str>,
    capability: Option<&'a NativeCompactionCapability>,
}

/// The `providerConfig` half of `validateNativeCompactionCapability`.
struct NativeCompactionProviderRef<'a> {
    api: Option<&'a str>,
    base_url: Option<&'a str>,
}

impl ProviderConfig {
    fn native_compaction_provider_ref(&self) -> NativeCompactionProviderRef<'_> {
        NativeCompactionProviderRef {
            api: self.api.as_deref(),
            base_url: self.base_url.as_deref(),
        }
    }
}

impl ModelDefinition {
    fn native_compaction_model_ref(&self) -> NativeCompactionModelRef<'_> {
        NativeCompactionModelRef {
            id: &self.id,
            api: self.api.as_deref(),
            base_url: self.base_url.as_deref(),
            capability: self.native_compaction.as_ref(),
        }
    }
}

/// `validateNativeCompactionCapability(providerName, modelDef, providerConfig)`.
fn validate_native_compaction_capability(
    provider_name: &str,
    model_def: NativeCompactionModelRef<'_>,
    provider_config: NativeCompactionProviderRef<'_>,
) -> Result<(), String> {
    let Some(capability) = model_def.capability else {
        return Ok(());
    };
    let label = format!(
        "Provider {}, model {}: nativeCompaction",
        provider_name, model_def.id
    );
    if provider_name != "azure-openai-managed" || model_def.id != "gpt-6-astra" {
        return Err(format!(
            "{} is allowlisted only for azure-openai-managed/gpt-6-astra.",
            label
        ));
    }
    let api = model_def.api.or(provider_config.api);
    if api != Some("openai-responses") {
        return Err(format!("{} requires api \"openai-responses\".", label));
    }
    if capability.provider != provider_name || capability.model != model_def.id {
        return Err(format!(
            "{} provider/model ownership does not match its enclosing model.",
            label
        ));
    }
    if capability.enabled && capability.validation != pi_ai::types::NativeCompactionValidation::LiveVerified {
        return Err(format!(
            "{} cannot be enabled until validation is \"live-verified\".",
            label
        ));
    }
    let base_url = normalized_http_endpoint(model_def.base_url.or(provider_config.base_url).unwrap_or(""));
    let endpoint = normalized_http_endpoint(&capability.endpoint);
    let base_url_path_ok = base_url
        .as_deref()
        .and_then(|base| url::Url::parse(base).ok())
        .map(|url| url.path() == "/azure-openai/v1")
        .unwrap_or(false);
    if base_url.is_none() || endpoint.is_none() || !base_url_path_ok {
        return Err(format!(
            "{} requires a safe /azure-openai/v1 gateway base URL.",
            label
        ));
    }
    let expected = normalized_http_endpoint(&format!("{}/responses/compact", base_url.unwrap_or_default()));
    if endpoint != expected {
        return Err(format!(
            "{}.endpoint must equal the model base URL plus /responses/compact.",
            label
        ));
    }
    Ok(())
}

/// `interface ProviderOverride`.
#[derive(Debug, Clone, Default)]
struct ProviderOverride {
    base_url: Option<String>,
    compat: Option<Compat>,
}

#[derive(Debug, Clone, Default)]
struct ProviderRequestConfig {
    api_key: Option<String>,
    headers: Option<IndexMap<String, String>>,
    auth_header: Option<bool>,
}

#[derive(Clone)]
struct ProviderRequestAuthSource {
    source: String,
    configured: bool,
    label: Option<String>,
    identity_fingerprint: String,
    value_fingerprint: Option<String>,
    resolve_value_fingerprint: Option<Arc<dyn Fn() -> Option<String> + Send + Sync>>,
}

impl std::fmt::Debug for ProviderRequestAuthSource {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ProviderRequestAuthSource")
            .field("source", &self.source)
            .field("configured", &self.configured)
            .field("label", &self.label)
            .field("identity_fingerprint", &self.identity_fingerprint)
            .field("value_fingerprint", &self.value_fingerprint)
            .finish()
    }
}

/// `type ResolvedRequestAuth`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ResolvedRequestAuth {
    pub ok: bool,
    pub api_key: Option<String>,
    pub headers: Option<IndexMap<String, String>>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ModelCatalogSnapshot {
    pub models: Vec<Model>,
    pub configured_providers: Vec<String>,
}

#[derive(Debug, Clone, Default)]
struct CustomModelsResult {
    models: Vec<Model>,
    /// Providers with baseUrl/headers/apiKey overrides for built-in models.
    overrides: IndexMap<String, ProviderOverride>,
    /// Per-model overrides: provider -> modelId -> override.
    model_overrides: IndexMap<String, IndexMap<String, ModelOverride>>,
    error: Option<String>,
}

fn empty_custom_models_result(error: Option<String>) -> CustomModelsResult {
    CustomModelsResult {
        models: Vec::new(),
        overrides: IndexMap::new(),
        model_overrides: IndexMap::new(),
        error,
    }
}

/// `Model.compat` normalised to the shape selected by `api`.
///
/// `Model<TApi>.compat` (packages/ai/src/types.ts:494-500) is chosen by the
/// model's own `api`, so the TypeScript always holds the compat object in the
/// shape that `api` selects. The Rust `Compat` enum instead picks its variant by
/// sniffing keys during `Deserialize`, so a compat object that omits the
/// discriminating keys is classified as `OpenAICompletionsCompat` even when the
/// model's `api` is `anthropic-messages` or `openai-responses`. Every consumer
/// then reads the wrong accessor and silently falls back to its defaults.
///
/// This re-shapes the value through `pi_ai::types::parse_compat_for_api` so the
/// Rust matches the TypeScript's type-directed shape selection.
fn compat_for_api(api: &str, compat: Option<&Compat>) -> Option<Compat> {
    let value = serde_json::to_value(compat?).ok()?;
    pi_ai::types::parse_compat_for_api(api, value)
}

/// `mergeCompat(baseCompat, overrideCompat)`.
///
/// The TypeScript does `{ ...base, ...override }` on the (already narrowed)
/// compat objects, then re-merges the two nested routing objects field by field.
pub fn merge_compat(base_compat: Option<&Compat>, override_compat: Option<&Compat>) -> Option<Compat> {
    let Some(override_compat) = override_compat else {
        return base_compat.cloned();
    };

    let base_value = base_compat
        .and_then(|compat| serde_json::to_value(compat).ok())
        .unwrap_or(Value::Object(Map::new()));
    let override_value = serde_json::to_value(override_compat).unwrap_or(Value::Object(Map::new()));

    let mut merged = match (base_value, override_value) {
        (Value::Object(base), Value::Object(over)) => {
            let mut merged = base;
            for (key, value) in over {
                merged.insert(key, value);
            }
            Value::Object(merged)
        }
        (_, over) => over,
    };

    if let Value::Object(object) = &mut merged {
        let base_object = base_compat
            .and_then(|compat| serde_json::to_value(compat).ok())
            .and_then(|value| match value {
                Value::Object(object) => Some(object),
                _ => None,
            })
            .unwrap_or_default();
        let override_object = serde_json::to_value(override_compat)
            .ok()
            .and_then(|value| match value {
                Value::Object(object) => Some(object),
                _ => None,
            })
            .unwrap_or_default();
        for key in ["openRouterRouting", "vercelGatewayRouting"] {
            let base_routing = base_object.get(key).cloned();
            let override_routing = override_object.get(key).cloned();
            if base_routing.is_some() || override_routing.is_some() {
                let mut routing = match base_routing {
                    Some(Value::Object(object)) => object,
                    _ => Map::new(),
                };
                if let Some(Value::Object(over)) = override_routing {
                    for (inner_key, inner_value) in over {
                        routing.insert(inner_key, inner_value);
                    }
                }
                object.insert(key.to_string(), Value::Object(routing));
            }
        }
    }

    serde_json::from_value(merged).ok()
}

/// Deep merge a model override into a model.
/// Handles nested objects (cost, compat) by merging rather than replacing.
pub fn apply_model_override(model: &Model, override_value: &ModelOverride) -> Model {
    let mut result = model.clone();

    if let Some(name) = &override_value.name {
        result.name = name.clone();
    }
    if let Some(reasoning) = override_value.reasoning {
        result.reasoning = reasoning;
    }
    if let Some(thinking_level_map) = &override_value.thinking_level_map {
        let mut merged = model.thinking_level_map.clone().unwrap_or_default();
        for (key, value) in thinking_level_map {
            merged.insert(key.clone(), value.clone());
        }
        result.thinking_level_map = Some(merged);
    }
    if let Some(input) = &override_value.input {
        result.input = input
            .iter()
            .filter_map(|value| match value.as_str() {
                "text" => Some(pi_ai::types::InputModality::Text),
                "image" => Some(pi_ai::types::InputModality::Image),
                _ => None,
            })
            .collect();
    }
    if let Some(context_window) = override_value.context_window {
        result.context_window = context_window;
    }
    if let Some(max_input_tokens) = override_value.max_input_tokens {
        result.max_input_tokens = Some(max_input_tokens);
    }
    if let Some(max_tokens) = override_value.max_tokens {
        result.max_tokens = max_tokens;
    }

    if let Some(cost) = &override_value.cost {
        result.cost = ModelCost {
            input: cost.input.unwrap_or(model.cost.input),
            output: cost.output.unwrap_or(model.cost.output),
            cache_read: cost.cache_read.unwrap_or(model.cost.cache_read),
            cache_write: cost.cache_write.unwrap_or(model.cost.cache_write),
        };
    }

    // TS `applyModelOverride` (model-registry.ts:415) assigns the merge of two
    // `Model<TApi>.compat` values, which are already in the shape `model.api`
    // selects. Re-shape by `api` so the Rust result matches that layering.
    result.compat = compat_for_api(
        &model.api,
        merge_compat(model.compat.as_ref(), override_value.compat.as_ref()).as_ref(),
    );

    result
}

fn read_openai_codex_account_id(token: &str) -> Option<String> {
    use base64::Engine;
    let payload = token.split('.').nth(1)?;
    let decoded = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(payload)
        .ok()?;
    let parsed: Value = serde_json::from_slice(&decoded).ok()?;
    let account_id = parsed
        .as_object()?
        .get("https://api.openai.com/auth")?
        .as_object()?
        .get("chatgpt_account_id")?;
    let account_id = account_id.as_str()?;
    if account_id.is_empty() {
        None
    } else {
        Some(account_id.to_string())
    }
}

/// The Codex backend gates its model catalog on the reported client version: it
/// answers HTTP 200 with a catalog that grows as the version rises, so a low
/// version yields a silently empty or partial list rather than an error.
///
/// Shipping a new Codex model takes two edits, and both are required:
/// 1. Add reviewed metadata to `pi-ai/src/models.subscription.json`.
/// 2. Raise this constant to a Codex CLI release whose catalog includes that
///    model. Sol/Luna are bundled in rust-v0.156.1 (PR #47332); their declared
///    minimum client version is 0.155.0. Account rollout is still provider-owned.
///
/// Catalog behaviour measured 2026-08-13; see #702.
const OPENAI_CODEX_CLIENT_VERSION: &str = "0.156.1";

fn openai_codex_models_url(base_url: &str) -> String {
    let normalized = base_url.trim_end_matches('/');
    let path = if let Some(prefix) = normalized.strip_suffix("/codex/responses") {
        format!("{}/codex/models", prefix)
    } else if normalized.ends_with("/codex") {
        format!("{}/models", normalized)
    } else {
        format!("{}/codex/models", normalized)
    };
    match url::Url::parse(&path) {
        Ok(mut url) => {
            url.query_pairs_mut()
                .append_pair("client_version", OPENAI_CODEX_CLIENT_VERSION);
            url.to_string()
        }
        Err(_) => path,
    }
}

fn read_openai_codex_model_ids(value: &Value) -> Result<HashSet<String>, String> {
    let object = value.as_object().ok_or_else(|| "Invalid OpenAI Codex model catalog".to_string())?;
    let models = object
        .get("models")
        .and_then(Value::as_array)
        .ok_or_else(|| "Invalid OpenAI Codex model catalog".to_string())?;
    let mut ids = HashSet::new();
    for model in models {
        let Some(object) = model.as_object() else {
            continue;
        };
        if let Some(slug) = object.get("slug").and_then(Value::as_str) {
            ids.insert(slug.to_string());
        }
    }
    Ok(ids)
}

const PRIVATE_PRIME_AUTHORIZATION_CACHE_FILE: &str = "prime-inference-private-models.json";
const PRIVATE_PRIME_AUTHORIZATION_CACHE_TTL_MS: i64 = 5 * 60_000;
const PRIVATE_PRIME_BACKGROUND_REFRESH_TIMEOUT_MS: u64 = 3_000;

#[derive(Debug, Clone, Default)]
struct PrivatePrimeAuthorizationCache {
    fingerprint: String,
    models: Vec<Model>,
    refreshed_at: i64,
}

fn private_prime_authorization_fingerprint(api_key: &str, team_id: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(api_key.as_bytes());
    hasher.update(b"\0");
    hasher.update(team_id.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn is_offline_mode_enabled() -> bool {
    match std::env::var("PI_OFFLINE") {
        Err(_) => false,
        Ok(value) => {
            if value.is_empty() {
                return false;
            }
            value == "1" || value.to_lowercase() == "true" || value.to_lowercase() == "yes"
        }
    }
}

fn now_millis() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_millis() as i64)
        .unwrap_or(0)
}

/// Use the shared resolver so the model picker respects an isolated agent directory.
fn get_agent_dir() -> String {
    crate::config::get_agent_dir()
}

/// `interface ProviderConfigInput` for `registerProvider`.
#[derive(Clone, Default)]
pub struct ProviderConfigInput {
    pub name: Option<String>,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub api: Option<Api>,
    /// `streamSimple?: (model, context, options?) => AssistantMessageEventStream`
    pub stream_simple: Option<ApiStreamSimpleFunction>,
    pub headers: Option<IndexMap<String, String>>,
    pub auth_header: Option<bool>,
    /// OAuth provider for /login support (`Omit<OAuthProviderInterface, "id">`).
    pub oauth: Option<ProviderOAuthInput>,
    pub models: Option<Vec<ModelDefinition>>,
}

impl std::fmt::Debug for ProviderConfigInput {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ProviderConfigInput")
            .field("name", &self.name)
            .field("base_url", &self.base_url)
            .field("api", &self.api)
            .field("auth_header", &self.auth_header)
            .finish_non_exhaustive()
    }
}

/// `Omit<OAuthProviderInterface, "id">`.
#[derive(Clone)]
pub struct ProviderOAuthInput {
    pub name: String,
    pub login: Arc<
        dyn Fn(pi_ai::utils::oauth::types::OAuthLoginCallbacks) -> pi_ai::types::BoxFuture<
                Result<pi_ai::utils::oauth::types::OAuthCredentials, String>,
            > + Send
            + Sync,
    >,
    pub uses_callback_server: Option<bool>,
    pub refresh_token: Arc<
        dyn Fn(pi_ai::utils::oauth::types::OAuthCredentials) -> pi_ai::types::BoxFuture<
                Result<pi_ai::utils::oauth::types::OAuthCredentials, String>,
            > + Send
            + Sync,
    >,
    pub get_api_key: Arc<dyn Fn(&pi_ai::utils::oauth::types::OAuthCredentials) -> String + Send + Sync>,
    pub modify_models:
        Option<Arc<dyn Fn(Vec<Model>, &pi_ai::utils::oauth::types::OAuthCredentials) -> Vec<Model> + Send + Sync>>,
}

impl std::fmt::Debug for ProviderOAuthInput {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ProviderOAuthInput")
            .field("name", &self.name)
            .field("uses_callback_server", &self.uses_callback_server)
            .finish_non_exhaustive()
    }
}

impl ProviderOAuthInput {
    fn to_oauth_provider_interface(&self, id: &str) -> pi_ai::utils::oauth::types::OAuthProviderInterface {
        pi_ai::utils::oauth::types::OAuthProviderInterface {
            id: id.to_string(),
            name: self.name.clone(),
            login: self.login.clone(),
            uses_callback_server: self.uses_callback_server,
            refresh_token: self.refresh_token.clone(),
            get_api_key: self.get_api_key.clone(),
            modify_models: self.modify_models.clone(),
        }
    }
}

/// Model registry - loads and manages models, resolves API keys via AuthStorage.
pub struct ModelRegistry {
    auth_storage: AuthStorage,
    models_json_path: Option<String>,
    models: Vec<Model>,
    provider_request_configs: IndexMap<String, ProviderRequestConfig>,
    /// `Map<string, AuthSourceToken[]>` - behind a mutex so the read-only auth
    /// checks can prune stale markers like the TypeScript does.
    stale_provider_request_auth_sources: Arc<std::sync::Mutex<HashMap<String, Vec<AuthSourceToken>>>>,
    last_provider_auth_source_tokens: HashMap<String, AuthSourceToken>,
    model_request_headers: IndexMap<String, IndexMap<String, String>>,
    registered_providers: IndexMap<String, ProviderConfigInput>,
    authorized_private_prime_inference_model_ids: HashSet<String>,
    authorized_private_prime_inference_models: Vec<Model>,
    authorized_private_prime_inference_team_id: Option<String>,
    explicit_private_prime_inference_model_ids: HashSet<String>,
    openai_codex_models_cache: Option<OpenAiCodexModelsCache>,
    background_private_prime_authorization: Option<BackgroundAuthorization>,
    live_prime_inference_models: Option<Vec<Model>>,
    /// Slot the detached catalog refresh writes into; drained by `loadModels`.
    live_prime_inference_models_slot: Arc<std::sync::Mutex<Option<Vec<Model>>>>,
    load_error: Option<String>,
    /// Re-register dynamic OAuth providers (e.g. user MCP servers) after refresh()
    /// resets the registry.
    on_oauth_providers_reset: Option<Arc<dyn Fn() + Send + Sync>>,
    fetch_fn: Option<FetchFn>,
    /// `entitlementRefreshChain` - serializes entitlement refreshes.
    entitlement_refresh_chain: Arc<tokio::sync::Mutex<()>>,
}

#[derive(Debug, Clone, Default)]
struct OpenAiCodexModelsCache {
    auth_fingerprint: String,
    model_ids: HashSet<String>,
    refreshed_at: i64,
}

#[derive(Debug, Clone, Default)]
struct BackgroundAuthorization {
    fingerprint: String,
}


impl ModelRegistry {
    fn new(auth_storage: AuthStorage, models_json_path: Option<String>) -> Self {
        let mut registry = Self {
            auth_storage,
            models_json_path,
            models: Vec::new(),
            provider_request_configs: IndexMap::new(),
            stale_provider_request_auth_sources: Arc::new(std::sync::Mutex::new(HashMap::new())),
            last_provider_auth_source_tokens: HashMap::new(),
            model_request_headers: IndexMap::new(),
            registered_providers: IndexMap::new(),
            authorized_private_prime_inference_model_ids: HashSet::new(),
            authorized_private_prime_inference_models: Vec::new(),
            authorized_private_prime_inference_team_id: None,
            explicit_private_prime_inference_model_ids: HashSet::new(),
            openai_codex_models_cache: None,
            background_private_prime_authorization: None,
            live_prime_inference_models: None,
            live_prime_inference_models_slot: Arc::new(std::sync::Mutex::new(None)),
            load_error: None,
            on_oauth_providers_reset: None,
            fetch_fn: None,
            entitlement_refresh_chain: Arc::new(tokio::sync::Mutex::new(())),
        };
        registry.load_models();
        registry
    }

    pub fn create(auth_storage: AuthStorage, models_json_path: Option<String>) -> Self {
        let path = models_json_path.unwrap_or_else(|| {
            Path::new(&get_agent_dir())
                .join("models.json")
                .to_string_lossy()
                .to_string()
        });
        Self::new(auth_storage, Some(path))
    }

    pub fn in_memory(auth_storage: AuthStorage) -> Self {
        Self::new(auth_storage, None)
    }

    /// The registry's own auth storage (`modelRegistry.authStorage` in
    /// TypeScript — a public field there; a read accessor here so kernel env
    /// provisioning can read credentials without exposing the whole registry).
    pub fn auth_storage(&self) -> &AuthStorage {
        &self.auth_storage
    }

    /// Set a runtime API key override on the registry's own `AuthStorage`.
    ///
    /// `auth-storage.ts:307 setRuntimeApiKey(provider, apiKey)` is what
    /// `main.ts:882` calls on the instance the registry shares
    /// (`model-registry.ts:525 \`readonly authStorage: AuthStorage\`,
    /// `agent-session-services.ts:152 ModelRegistry.create(authStorage, ...)`).
    /// The Rust registry owns its `AuthStorage` by value, so the CLI runtime
    /// override reaches request auth through this passthrough
    /// (`model-registry.ts` resolves request auth via the same instance in
    /// `getApiKeyAndHeaders`).
    pub fn set_runtime_api_key(&mut self, provider: &str, api_key: &str) {
        self.auth_storage.set_runtime_api_key(provider, api_key);
    }

    pub fn set_fetch_fn(&mut self, fetch_fn: Option<FetchFn>) {
        self.fetch_fn = fetch_fn;
    }

    pub fn set_on_oauth_providers_reset(&mut self, hook: Arc<dyn Fn() + Send + Sync>) {
        self.on_oauth_providers_reset = Some(hook);
    }

    /// Reload models from disk (built-in + custom from models.json).
    pub fn refresh(&mut self) {
        self.provider_request_configs.clear();
        self.model_request_headers.clear();
        self.last_provider_auth_source_tokens.clear();
        self.explicit_private_prime_inference_model_ids.clear();
        self.load_error = None;

        // Credentials may have been written by another process (e.g. the UI
        // process saving a login while the session lives in the daemon).
        self.auth_storage.reload();
        let team_id = self
            .auth_storage
            .get_provider_headers(PRIME_INFERENCE_PROVIDER_ID)
            .and_then(|headers| headers.get("X-Prime-Team-ID").cloned());
        // Direct refreshes must preserve same-team stale recovery, but invalidate
        // changed auth immediately.
        let auth_status = self.auth_storage.get_auth_status(PRIME_INFERENCE_PROVIDER_ID);
        if auth_status.source.as_deref() != Some("stale")
            || team_id.is_none()
            || team_id != self.authorized_private_prime_inference_team_id
        {
            self.authorized_private_prime_inference_model_ids.clear();
            self.authorized_private_prime_inference_models = Vec::new();
            self.authorized_private_prime_inference_team_id = None;
        }
        reset_api_providers();
        crate::core::auth_storage::reset_oauth_providers();
        // reset drops everything but model-provider built-ins; re-add MCP integrations
        // (built-in catalog + this session's user-declared servers via the hook).
        crate::core::auth_storage::register_builtin_mcp_oauth_providers();
        if let Some(hook) = &self.on_oauth_providers_reset {
            hook();
        }

        self.reload_models_after_catalog_change();
    }

    fn reload_models_after_catalog_change(&mut self) {
        self.load_models();
        self.reapply_registered_providers();
    }

    fn reapply_registered_providers(&mut self) {
        let providers: Vec<(String, ProviderConfigInput)> = self
            .registered_providers
            .iter()
            .map(|(name, config)| (name.clone(), config.clone()))
            .collect();
        for (provider_name, config) in providers {
            self.apply_provider_config(&provider_name, &config);
        }
    }

    fn prime_inference_catalog_cache_path(&self) -> Option<String> {
        self.models_json_path.as_ref().map(|path| {
            Path::new(path)
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join("prime-inference-models-cache.json")
                .to_string_lossy()
                .to_string()
        })
    }

    fn bundled_prime_inference_models(&self) -> Vec<Model> {
        get_models(PRIME_INFERENCE_PROVIDER_ID)
    }

    /// Get any error from loading models.json (`None` if no error).
    pub fn get_error(&self) -> Option<&str> {
        self.load_error.as_deref()
    }
}

impl ModelRegistry {
    fn load_models(&mut self) {
        let models_json_path = self.models_json_path.clone();
        let custom = match models_json_path.as_deref() {
            Some(path) => self.load_custom_models(path),
            None => empty_custom_models_result(None),
        };
        let CustomModelsResult {
            models: custom_models,
            overrides,
            model_overrides,
            error,
        } = custom;

        if let Some(error) = error {
            self.load_error = Some(error);
        }

        self.explicit_private_prime_inference_model_ids = custom_models
            .iter()
            .filter(|model| is_private_prime_inference_model(&model.provider, &model.id))
            .map(|model| model.id.clone())
            .collect();

        // `this.livePrimeInferenceModels ??= ...` plus the detached refresh's
        // `this.livePrimeInferenceModels = models`.
        {
            let mut slot = self
                .live_prime_inference_models_slot
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if let Some(models) = slot.take() {
                self.live_prime_inference_models = Some(models);
            }
        }

        let cache_path = self.prime_inference_catalog_cache_path();
        if self.live_prime_inference_models.is_none() {
            // `cachePath ? readCachedPrimeInferenceModels(cachePath, ...) : undefined`
            // - the reader itself may also return undefined, so `and_then` (not `map`)
            // keeps the slot at `Option<Vec<Model>>`.
            self.live_prime_inference_models = cache_path
                .as_deref()
                .and_then(|path| read_cached_prime_inference_models(path, &self.bundled_prime_inference_models()));
        }

        let mut private_models: IndexMap<String, Model> = IndexMap::new();
        for model in get_private_prime_inference_models()
            .into_iter()
            .chain(self.authorized_private_prime_inference_models.iter().cloned())
        {
            private_models.insert(model.id.clone(), model);
        }

        let mut built_in_models = self.load_built_in_models(
            &overrides,
            &model_overrides,
            self.live_prime_inference_models.clone(),
        );
        built_in_models.extend(private_models.into_values());

        let mut combined = self.merge_custom_models(built_in_models, custom_models);

        for oauth_provider in self.auth_storage.get_oauth_providers() {
            let credential = self.auth_storage.get(&oauth_provider.id);
            if let (Some(AuthCredential::OAuth { credentials }), Some(modify_models)) =
                (credential.as_ref(), oauth_provider.modify_models.as_ref())
            {
                combined = modify_models(combined, credentials);
            }
        }

        for model in &mut combined {
            super::model_reasoning_policy::apply_azure_reasoning_levels(model);
        }
        self.models = combined;
    }

    /// Load built-in models and apply provider/model overrides.
    fn load_built_in_models(
        &self,
        overrides: &IndexMap<String, ProviderOverride>,
        model_overrides: &IndexMap<String, IndexMap<String, ModelOverride>>,
        live_prime_inference_models: Option<Vec<Model>>,
    ) -> Vec<Model> {
        let bundled_models: Vec<Model> = get_providers()
            .iter()
            .flat_map(|provider| get_models(provider))
            .collect();
        merge_prime_inference_models(&bundled_models, live_prime_inference_models.as_deref())
            .into_iter()
            .map(|model| {
                let provider_override = overrides.get(&model.provider);
                let per_model_overrides = model_overrides.get(&model.provider);
                let mut configured_model = model.clone();

                if let Some(provider_override) = provider_override {
                    if let Some(base_url) = &provider_override.base_url {
                        configured_model.base_url = base_url.clone();
                    }
                    // TS `loadBuiltInModels` (model-registry.ts:665) merges the
                    // provider override into the built-in model's compat, both of
                    // them `Model<TApi>` values shaped by the model's `api`.
                    configured_model.compat = compat_for_api(
                        &configured_model.api,
                        merge_compat(
                            configured_model.compat.as_ref(),
                            provider_override.compat.as_ref(),
                        )
                        .as_ref(),
                    );
                }

                match per_model_overrides.and_then(|overrides| overrides.get(&model.id)) {
                    Some(model_override) => apply_model_override(&configured_model, model_override),
                    None => configured_model,
                }
            })
            .collect()
    }

    /// Merge custom models into built-in list by provider+id (custom wins).
    fn merge_custom_models(&self, built_in_models: Vec<Model>, custom_models: Vec<Model>) -> Vec<Model> {
        let mut merged = built_in_models;
        for custom_model in custom_models {
            match merged
                .iter()
                .position(|model| model.provider == custom_model.provider && model.id == custom_model.id)
            {
                Some(index) => merged[index] = custom_model,
                None => merged.push(custom_model),
            }
        }
        merged
    }

    fn load_custom_models(&mut self, models_json_path: &str) -> CustomModelsResult {
        if !Path::new(models_json_path).exists() {
            return empty_custom_models_result(None);
        }

        let content = match std::fs::read_to_string(models_json_path) {
            Ok(content) => content,
            Err(error) => {
                return empty_custom_models_result(Some(format!(
                    "Failed to load models.json: {}\n\nFile: {}",
                    error, models_json_path
                )))
            }
        };

        let parsed: Value = match serde_json::from_str(&strip_json_comments(&content)) {
            Ok(parsed) => parsed,
            Err(error) => {
                return empty_custom_models_result(Some(format!(
                    "Failed to parse models.json: {}\n\nFile: {}",
                    error, models_json_path
                )))
            }
        };

        // The typebox validator loads lazily in TypeScript (~300ms import). The
        // schema is compiled eagerly here, so the first load validates
        // synchronously and reports schema errors through `loadError`.
        let schema_errors = validate_models_config_schema(&parsed);
        if !schema_errors.is_empty() {
            let errors = schema_errors.join("\n");
            return empty_custom_models_result(Some(format!(
                "Invalid models.json schema:\n{}\n\nFile: {}",
                errors, models_json_path
            )));
        }

        let config: ModelsConfig = match serde_json::from_value(parsed) {
            Ok(config) => config,
            Err(error) => {
                return empty_custom_models_result(Some(format!(
                    "Failed to load models.json: {}\n\nFile: {}",
                    error, models_json_path
                )))
            }
        };

        // `validateConfig` throws inside the TypeScript try block, so its message
        // is reported through the generic "Failed to load models.json" path.
        if let Err(error) = self.validate_config(&config) {
            return empty_custom_models_result(Some(format!(
                "Failed to load models.json: {}\n\nFile: {}",
                error, models_json_path
            )));
        }

        let mut overrides: IndexMap<String, ProviderOverride> = IndexMap::new();
        let mut model_overrides: IndexMap<String, IndexMap<String, ModelOverride>> = IndexMap::new();

        for (provider_name, provider_config) in &config.providers {
            if provider_config.base_url.is_some() || provider_config.compat.is_some() {
                overrides.insert(
                    provider_name.clone(),
                    ProviderOverride {
                        base_url: provider_config.base_url.clone(),
                        compat: provider_config.compat.clone(),
                    },
                );
            }

            self.store_provider_request_config(
                provider_name,
                provider_config.api_key.as_deref(),
                provider_config.headers.as_ref(),
                provider_config.auth_header,
            );

            if let Some(per_model) = &provider_config.model_overrides {
                model_overrides.insert(provider_name.clone(), per_model.clone());
                for (model_id, model_override) in per_model {
                    self.store_model_headers(provider_name, model_id, model_override.headers.as_ref());
                }
            }
        }

        CustomModelsResult {
            models: self.parse_models(&config),
            overrides,
            model_overrides,
            error: None,
        }
    }

    fn validate_config(&self, config: &ModelsConfig) -> Result<(), String> {
        let built_in_providers: HashSet<String> = get_providers().into_iter().collect();

        for (provider_name, provider_config) in &config.providers {
            let is_built_in = built_in_providers.contains(provider_name);
            let has_provider_api = provider_config.api.is_some();
            let models = provider_config.models.clone().unwrap_or_default();
            let has_model_overrides = provider_config
                .model_overrides
                .as_ref()
                .map(|overrides| !overrides.is_empty())
                .unwrap_or(false);

            if models.is_empty() {
                if provider_config.base_url.is_none()
                    && provider_config.headers.is_none()
                    && provider_config.compat.is_none()
                    && !has_model_overrides
                {
                    return Err(format!(
                        "Provider {}: must specify \"baseUrl\", \"headers\", \"compat\", \"modelOverrides\", or \"models\".",
                        provider_name
                    ));
                }
            } else if !is_built_in {
                if provider_config.base_url.is_none() {
                    return Err(format!(
                        "Provider {}: \"baseUrl\" is required when defining custom models.",
                        provider_name
                    ));
                }
                if provider_config.api_key.is_none() {
                    return Err(format!(
                        "Provider {}: \"apiKey\" is required when defining custom models.",
                        provider_name
                    ));
                }
            }
            // inherited from built-in models. Auth comes from env vars / auth storage.

            for model_def in &models {
                let has_model_api = model_def.api.is_some();
                validate_native_compaction_capability(
                    provider_name,
                    model_def.native_compaction_model_ref(),
                    provider_config.native_compaction_provider_ref(),
                )?;

                if !has_provider_api && !has_model_api && !is_built_in {
                    return Err(format!(
                        "Provider {}, model {}: no \"api\" specified. Set at provider or model level.",
                        provider_name, model_def.id
                    ));
                }

                if model_def.id.is_empty() {
                    return Err(format!("Provider {}: model missing \"id\"", provider_name));
                }
                if model_def.context_window.map(|value| value <= 0.0).unwrap_or(false) {
                    return Err(format!(
                        "Provider {}, model {}: invalid contextWindow",
                        provider_name, model_def.id
                    ));
                }
                if model_def.max_tokens.map(|value| value <= 0.0).unwrap_or(false) {
                    return Err(format!(
                        "Provider {}, model {}: invalid maxTokens",
                        provider_name, model_def.id
                    ));
                }
            }
        }
        Ok(())
    }

    fn parse_models(&mut self, config: &ModelsConfig) -> Vec<Model> {
        let mut models: Vec<Model> = Vec::new();
        let built_in_providers: HashSet<String> = get_providers().into_iter().collect();

        let mut built_in_defaults_cache: HashMap<String, (String, String)> = HashMap::new();
        let mut get_built_in_defaults = |provider_name: &str| -> Option<(String, String)> {
            if !built_in_providers.contains(provider_name) {
                return None;
            }
            if let Some(cached) = built_in_defaults_cache.get(provider_name) {
                return Some(cached.clone());
            }
            let built_in = get_models(provider_name);
            let first = built_in.first()?;
            let defaults = (first.api.clone(), first.base_url.clone());
            built_in_defaults_cache.insert(provider_name.to_string(), defaults.clone());
            Some(defaults)
        };

        let providers: Vec<(String, ProviderConfig)> = config
            .providers
            .iter()
            .map(|(name, config)| (name.clone(), config.clone()))
            .collect();

        for (provider_name, provider_config) in providers {
            let model_defs = provider_config.models.clone().unwrap_or_default();
            if model_defs.is_empty() {
                continue; // Override-only, no custom models
            }

            let built_in_defaults = get_built_in_defaults(&provider_name);

            for model_def in model_defs {
                let api = model_def
                    .api
                    .clone()
                    .or_else(|| provider_config.api.clone())
                    .or_else(|| built_in_defaults.as_ref().map(|defaults| defaults.0.clone()));
                let Some(api) = api else { continue };

                let base_url = model_def
                    .base_url
                    .clone()
                    .or_else(|| provider_config.base_url.clone())
                    .or_else(|| built_in_defaults.as_ref().map(|defaults| defaults.1.clone()));
                let Some(base_url) = base_url else { continue };

                // TS `parseModels` (model-registry.ts:829) merges the provider
                // config compat with the model definition compat. Both sides are
                // typed `ProviderCompatSchema` (model-registry.ts:159-163) and the
                // result is assigned to a `Model<Api>`, whose compat shape `Api`
                // selects (packages/ai/src/types.ts:494-500). Deriving the shape
                // from the resolved `api` here is what makes a `models.json` model
                // that omits `compat` still end up with the TS's resolved compat.
                let compat = compat_for_api(
                    &api,
                    merge_compat(provider_config.compat.as_ref(), model_def.compat.as_ref()).as_ref(),
                );
                self.store_model_headers(&provider_name, &model_def.id, model_def.headers.as_ref());

                models.push(Model {
                    id: model_def.id.clone(),
                    name: model_def.name.clone().unwrap_or_else(|| model_def.id.clone()),
                    api,
                    provider: provider_name.clone(),
                    base_url,
                    reasoning: model_def.reasoning.unwrap_or(false),
                    thinking_level_map: model_def.thinking_level_map.clone(),
                    input: model_def
                        .input
                        .clone()
                        .unwrap_or_else(|| vec!["text".to_string()])
                        .iter()
                        .filter_map(|value| match value.as_str() {
                            "text" => Some(pi_ai::types::InputModality::Text),
                            "image" => Some(pi_ai::types::InputModality::Image),
                            _ => None,
                        })
                        .collect(),
                    cost: model_def.cost.clone().unwrap_or_default(),
                    context_window: model_def.context_window.unwrap_or(128_000.0),
                    max_input_tokens: model_def.max_input_tokens,
                    max_tokens: model_def.max_tokens.unwrap_or(16_384.0),
                    featured: None,
                    native_compaction: model_def.native_compaction.clone(),
                    headers: None,
                    compat,
                });
            }
        }

        models
    }

    /// Get all models (built-in + custom).
    /// If models.json had errors, returns only built-in models.
    pub fn get_all(&self) -> Vec<Model> {
        self.models.clone()
    }

    /// Get only models that have auth configured.
    /// This is a fast check that doesn't refresh OAuth tokens.
    pub fn get_available(&self) -> Vec<Model> {
        self.models
            .iter()
            .filter(|model| {
                if is_private_prime_inference_model(&model.provider, &model.id)
                    && !self.is_authorized_private_prime_inference_model(model)
                {
                    return false;
                }
                self.has_configured_auth(model)
            })
            .cloned()
            .collect()
    }

    /// Reload local state and private authorization. Public Prime Inference
    /// models return from the disk/bundled fallback immediately and refresh in
    /// the background.
    pub async fn refresh_available_models(&mut self) -> Vec<Model> {
        // `runSerializedEntitlementRefresh(async () => { ... })`: the TypeScript
        // chains promises so the next task runs after the current one settles.
        // Rust models that with a mutex held for the whole body.
        let chain = self.entitlement_refresh_chain.clone();
        let _guard = chain.lock().await;

        let previous_private_model_ids = self.authorized_private_prime_inference_model_ids.clone();
        let previous_team_id = self.authorized_private_prime_inference_team_id.clone();
        let previous_private_models = self.authorized_private_prime_inference_models.clone();
        self.refresh();

        // `void refreshPrimeInferenceModels(...).then((models) => { ... })`:
        // detached so the public model list returns from the disk/bundled
        // fallback without waiting on the network. The detached task publishes
        // its result through `live_prime_inference_models_slot`, which
        // `loadModels` drains on the next call.
        if let Some(cache_path) = self.prime_inference_catalog_cache_path() {
            let bundled = self.bundled_prime_inference_models();
            let offline = is_offline_mode_enabled();
            let fetch_fn = self.fetch_fn.clone();
            let slot = self.live_prime_inference_models_slot.clone();
            tokio::spawn(async move {
                let models = refresh_prime_inference_models(&cache_path, &bundled, fetch_fn, offline).await;
                if let Some(models) = models {
                    let mut slot = slot.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                    *slot = Some(models);
                }
            });
        }

        self.refresh_private_prime_inference_authorization(
            previous_private_model_ids,
            previous_team_id,
            previous_private_models,
        )
        .await;
        self.get_available()
    }

    async fn refresh_private_prime_inference_authorization(
        &mut self,
        previous_private_model_ids: HashSet<String>,
        previous_team_id: Option<String>,
        previous_private_models: Vec<Model>,
    ) {
        self.refresh_private_prime_inference_authorization_with_offline(
            previous_private_model_ids,
            previous_team_id,
            previous_private_models,
            is_offline_mode_enabled(),
        ).await;
    }

    async fn refresh_private_prime_inference_authorization_with_offline(
        &mut self,
        previous_private_model_ids: HashSet<String>,
        previous_team_id: Option<String>,
        previous_private_models: Vec<Model>,
        offline: bool,
    ) {
        let api_key = self
            .auth_storage
            .get_api_key(PRIME_INFERENCE_PROVIDER_ID, false)
            .await
            .unwrap_or(None);
        let team_headers = self.auth_storage.get_provider_headers(PRIME_INFERENCE_PROVIDER_ID);
        let team_id = team_headers
            .as_ref()
            .and_then(|headers| headers.get("X-Prime-Team-ID").cloned());
        let current_team_id = team_id.clone();
        let (Some(api_key), Some(team_headers), Some(team_id)) = (api_key, team_headers, team_id) else {
            // Stale is not logout: keep fetched entitlements for explicit
            // re-selection (the auth filter still hides the models while stale) -
            // but only for the team they were fetched for; a team switch
            // invalidates them.
            let status = self.auth_storage.get_auth_status(PRIME_INFERENCE_PROVIDER_ID);
            if status.source.as_deref() == Some("stale")
                && current_team_id.is_some()
                && current_team_id == previous_team_id
            {
                self.authorized_private_prime_inference_model_ids = previous_private_model_ids;
                self.authorized_private_prime_inference_models = previous_private_models;
                self.authorized_private_prime_inference_team_id = previous_team_id;
                self.reload_models_after_catalog_change();
                return;
            }
            self.authorized_private_prime_inference_model_ids.clear();
            self.authorized_private_prime_inference_models = Vec::new();
            self.authorized_private_prime_inference_team_id = None;
            self.reload_models_after_catalog_change();
            return;
        };

        let fingerprint = private_prime_authorization_fingerprint(&api_key, &team_id);
        let cached = self.read_private_prime_authorization_cache();
        if let Some(cached) = cached.filter(|cached| cached.fingerprint == fingerprint) {
            // Serve the credential-scoped cache so startup and model lists don't
            // block on the network. Stale entries refresh in the background.
            self.authorized_private_prime_inference_models = cached.models.clone();
            self.authorized_private_prime_inference_model_ids =
                cached.models.iter().map(|model| model.id.clone()).collect();
            self.authorized_private_prime_inference_team_id = Some(team_id.clone());
            self.reload_models_after_catalog_change();
            let cache_is_fresh = now_millis() - cached.refreshed_at < PRIVATE_PRIME_AUTHORIZATION_CACHE_TTL_MS;
            if offline || cache_is_fresh {
                return;
            }
            self.start_background_private_prime_authorization_refresh(
                &api_key,
                &team_headers,
                &team_id,
                &fingerprint,
            )
            .await;
            return;
        }
        if offline {
            self.authorized_private_prime_inference_model_ids.clear();
            self.authorized_private_prime_inference_models = Vec::new();
            self.authorized_private_prime_inference_team_id = None;
            self.reload_models_after_catalog_change();
            return;
        }

        let mut authorized_models: Option<Vec<Model>> = None;
        let public_model_ids: HashSet<String> = self
            .live_prime_inference_models
            .clone()
            .unwrap_or_else(|| self.bundled_prime_inference_models())
            .iter()
            .map(|model| model.id.clone())
            .collect();
        let team_headers_pairs: Vec<(String, String)> = team_headers
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        match fetch_authorized_private_prime_inference_models(
            &api_key,
            &team_headers_pairs,
            &public_model_ids,
            self.fetch_fn.clone(),
            None,
        )
        .await
        {
            Ok(models) => authorized_models = Some(models),
            // Fall back to the previous authorization below.
            Err(_) => {}
        }
        // Leave newer state untouched if the credentials changed while fetching.
        if self.current_private_prime_authorization_fingerprint().await != Some(fingerprint.clone()) {
            return;
        }
        match authorized_models {
            Some(authorized_models) => {
                self.authorized_private_prime_inference_models = authorized_models.clone();
                self.authorized_private_prime_inference_model_ids =
                    authorized_models.iter().map(|model| model.id.clone()).collect();
                self.authorized_private_prime_inference_team_id = Some(team_id.clone());
                self.reload_models_after_catalog_change();
                self.write_private_prime_authorization_cache(&PrivatePrimeAuthorizationCache {
                    fingerprint,
                    models: authorized_models,
                    refreshed_at: now_millis(),
                });
            }
            None if previous_team_id.as_deref() == Some(team_id.as_str()) => {
                self.authorized_private_prime_inference_model_ids = previous_private_model_ids;
                self.authorized_private_prime_inference_models = previous_private_models;
                self.authorized_private_prime_inference_team_id = Some(team_id);
                self.reload_models_after_catalog_change();
            }
            None => {
                self.authorized_private_prime_inference_model_ids.clear();
                self.authorized_private_prime_inference_models = Vec::new();
                self.authorized_private_prime_inference_team_id = None;
                self.reload_models_after_catalog_change();
            }
        }
    }

    /// Stale cache hits refresh in the background; failures keep the cached ids.
    /// Refreshes for the same credentials are deduped, a changed-credentials
    /// refresh is queued after the in-flight one, and a result is only applied
    /// if the credentials it was fetched with are still current.
    ///
    /// Port note: the TypeScript schedules this as a detached promise and never
    /// awaits it. Rust cannot mutate the registry from a detached task, so this
    /// is an awaited step with the same ordering and dedupe rules. The
    /// caller-visible model list after the call is identical; only the timing of
    /// the in-flight window differs.
    async fn start_background_private_prime_authorization_refresh(
        &mut self,
        api_key: &str,
        team_headers: &IndexMap<String, String>,
        team_id: &str,
        fingerprint: &str,
    ) {
        if self
            .background_private_prime_authorization
            .as_ref()
            .map(|pending| pending.fingerprint == fingerprint)
            .unwrap_or(false)
        {
            return;
        }
        self.background_private_prime_authorization = Some(BackgroundAuthorization {
            fingerprint: fingerprint.to_string(),
        });

        let public_model_ids: HashSet<String> = self
            .live_prime_inference_models
            .clone()
            .unwrap_or_else(|| self.bundled_prime_inference_models())
            .iter()
            .map(|model| model.id.clone())
            .collect();
        let team_headers: Vec<(String, String)> = team_headers
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        let fetched = fetch_authorized_private_prime_inference_models(
            api_key,
            &team_headers,
            &public_model_ids,
            self.fetch_fn.clone(),
            Some(PRIVATE_PRIME_BACKGROUND_REFRESH_TIMEOUT_MS),
        )
        .await;

        self.background_private_prime_authorization = None;

        // Keep the cached authorization on failure.
        let Ok(authorized_models) = fetched else { return };
        if self.current_private_prime_authorization_fingerprint().await != Some(fingerprint.to_string()) {
            return;
        }
        self.authorized_private_prime_inference_models = authorized_models.clone();
        self.authorized_private_prime_inference_model_ids =
            authorized_models.iter().map(|model| model.id.clone()).collect();
        self.authorized_private_prime_inference_team_id = Some(team_id.to_string());
        self.reload_models_after_catalog_change();
        self.write_private_prime_authorization_cache(&PrivatePrimeAuthorizationCache {
            fingerprint: fingerprint.to_string(),
            models: authorized_models,
            refreshed_at: now_millis(),
        });
    }

    async fn current_private_prime_authorization_fingerprint(&mut self) -> Option<String> {
        let api_key = self
            .auth_storage
            .get_api_key(PRIME_INFERENCE_PROVIDER_ID, false)
            .await
            .unwrap_or(None);
        let team_id = self
            .auth_storage
            .get_provider_headers(PRIME_INFERENCE_PROVIDER_ID)
            .and_then(|headers| headers.get("X-Prime-Team-ID").cloned());
        match (api_key, team_id) {
            (Some(api_key), Some(team_id)) => {
                Some(private_prime_authorization_fingerprint(&api_key, &team_id))
            }
            _ => None,
        }
    }

    fn private_prime_authorization_cache_path(&self) -> Option<String> {
        self.models_json_path.as_ref().map(|path| {
            Path::new(path)
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join(PRIVATE_PRIME_AUTHORIZATION_CACHE_FILE)
                .to_string_lossy()
                .to_string()
        })
    }

    fn read_private_prime_authorization_cache(&self) -> Option<PrivatePrimeAuthorizationCache> {
        let cache_path = self.private_prime_authorization_cache_path()?;
        let content = std::fs::read_to_string(&cache_path).ok()?;
        let parsed: Value = serde_json::from_str(&content).ok()?;
        let object = parsed.as_object()?;
        let fingerprint = object.get("fingerprint")?.as_str()?.to_string();
        let data = object.get("data")?;
        if !data.is_array() {
            return None;
        }
        let refreshed_at = object.get("refreshedAt")?.as_f64()?;
        let entries = parse_prime_inference_model_catalog(&serde_json::json!({ "data": data }), true).ok()?;
        let entries: Vec<_> = entries
            .into_iter()
            .filter(|entry| is_private_prime_inference_model(PRIME_INFERENCE_PROVIDER_ID, &entry.id))
            .collect();
        let models = build_prime_inference_models(
            &get_private_prime_inference_models(),
            &entries,
            true,
            Some(0.0),
        )
        .unwrap_or_default();
        Some(PrivatePrimeAuthorizationCache {
            fingerprint,
            models,
            refreshed_at: refreshed_at as i64,
        })
    }

    fn write_private_prime_authorization_cache(&self, cache: &PrivatePrimeAuthorizationCache) {
        let Some(cache_path) = self.private_prime_authorization_cache_path() else {
            return;
        };
        let data: Vec<Value> = cache
            .models
            .iter()
            .map(|model| {
                serde_json::json!({
                    "id": model.id,
                    "display_name": model.name,
                    "pricing": {
                        "input_usd_per_mtok": model.cost.input,
                        "output_usd_per_mtok": model.cost.output,
                        "cache_read_usd_per_mtok": model.cost.cache_read,
                        "cache_write_usd_per_mtok": model.cost.cache_write,
                    },
                    "specs": {
                        "context_window": model.context_window,
                        "max_output_tokens": model.max_tokens,
                        "modalities": {
                            "input": model.input.iter().map(|value| value.as_str()).collect::<Vec<_>>(),
                            "output": ["text"],
                        },
                        "supports_reasoning": model.reasoning,
                    },
                })
            })
            .collect();
        let body = serde_json::json!({
            "fingerprint": cache.fingerprint,
            "data": data,
            "refreshedAt": cache.refreshed_at,
        });
        // A failed cache write only requires a later refetch.
        let _ = crate::utils::atomic_file::write_file_atomic_sync(
            &cache_path,
            &body.to_string(),
            crate::utils::atomic_file::WriteFileAtomicOptions {
                mode: Some(0o600),
                fsync: false,
                fsync_dir: false,
                before_rename: None,
            },
        );
    }

    pub async fn refresh_model_catalog(&mut self) -> ModelCatalogSnapshot {
        let available_models = self.refresh_available_models().await;
        let available_private_models: HashSet<String> = available_models
            .iter()
            .filter(|model| is_private_prime_inference_model(&model.provider, &model.id))
            .map(|model| format!("{}/{}", model.provider, model.id))
            .collect();
        let mut configured_providers: Vec<String> = Vec::new();
        for model in &available_models {
            if !configured_providers.contains(&model.provider) {
                configured_providers.push(model.provider.clone());
            }
        }
        ModelCatalogSnapshot {
            models: self
                .models
                .iter()
                .filter(|model| {
                    !is_private_prime_inference_model(&model.provider, &model.id)
                        || available_private_models.contains(&format!("{}/{}", model.provider, model.id))
                })
                .cloned()
                .collect(),
            configured_providers,
        }
    }

    /// `assumeAuthConfigured` validates an explicit stale-provider selection
    /// BEFORE the clear commits.
    pub async fn can_use_model(&mut self, model: &Model, assume_auth_configured: bool) -> bool {
        if assume_auth_configured {
            // Must be side-effect-free: a keyless refresh would drop the cached
            // entitlements it needs.
            return !is_private_prime_inference_model(&model.provider, &model.id)
                || self.is_authorized_private_prime_inference_model(model);
        }
        if !self.has_configured_auth(model) {
            return false;
        }
        if !is_private_prime_inference_model(&model.provider, &model.id) {
            return true;
        }

        let available_models = self.refresh_available_models().await;
        available_models
            .iter()
            .any(|candidate| candidate.provider == model.provider && candidate.id == model.id)
    }

    fn is_authorized_private_prime_inference_model(&self, model: &Model) -> bool {
        self.explicit_private_prime_inference_model_ids.contains(&model.id)
            || self.authorized_private_prime_inference_model_ids.contains(&model.id)
    }

    pub async fn get_executable_models(&mut self) -> Vec<Model> {
        // `runSerializedEntitlementRefresh(() => this.refreshPrivatePrimeInferenceAuthorization())`
        let chain = self.entitlement_refresh_chain.clone();
        let _guard = chain.lock().await;
        let previous_private_model_ids = self.authorized_private_prime_inference_model_ids.clone();
        let previous_team_id = self.authorized_private_prime_inference_team_id.clone();
        let previous_private_models = self.authorized_private_prime_inference_models.clone();
        self.refresh_private_prime_inference_authorization(
            previous_private_model_ids,
            previous_team_id,
            previous_private_models,
        )
        .await;

        let available_models = self.get_available();
        let codex_models: Vec<Model> = available_models
            .iter()
            .filter(|model| model.provider == "openai-codex")
            .cloned()
            .collect();
        if codex_models.is_empty() {
            return available_models;
        }

        let auth = self.get_api_key_and_headers(&codex_models[0]).await;
        let Some(api_key) = auth.api_key.clone().filter(|_| auth.ok) else {
            return available_models
                .into_iter()
                .filter(|model| model.provider != "openai-codex")
                .collect();
        };
        let auth_fingerprint = {
            use sha2::{Digest, Sha256};
            let mut hasher = Sha256::new();
            hasher.update(api_key.as_bytes());
            format!("{:x}", hasher.finalize())
        };
        let cached = self.openai_codex_models_cache.clone();
        if let Some(cached) = cached
            .as_ref()
            .filter(|cached| cached.auth_fingerprint == auth_fingerprint)
            .filter(|cached| now_millis() - cached.refreshed_at < 300_000)
        {
            let model_ids = cached.model_ids.clone();
            return available_models
                .into_iter()
                .filter(|model| model.provider != "openai-codex" || model_ids.contains(&model.id))
                .collect();
        }

        let Some(account_id) = read_openai_codex_account_id(&api_key) else {
            return available_models
                .into_iter()
                .filter(|model| model.provider != "openai-codex")
                .collect();
        };

        let mut headers: Vec<(String, String)> = auth
            .headers
            .clone()
            .unwrap_or_default()
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        headers.push(("Authorization".to_string(), format!("Bearer {}", api_key)));
        headers.push(("chatgpt-account-id".to_string(), account_id));
        headers.push(("originator".to_string(), "pi".to_string()));

        let fetch = self.fetch_fn.clone().unwrap_or_else(default_fetch);
        let response = fetch(HttpRequest {
            method: "GET".to_string(),
            url: openai_codex_models_url(&codex_models[0].base_url),
            headers,
            body: None,
            timeout_ms: 5_000,
        })
        .await;

        let discovery = match response {
            Ok(response) if response.status >= 200 && response.status < 300 => {
                serde_json::from_str::<Value>(&response.text)
                    .ok()
                    .and_then(|value| read_openai_codex_model_ids(&value).ok())
            }
            _ => None,
        };

        match discovery {
            Some(model_ids) => {
                self.openai_codex_models_cache = Some(OpenAiCodexModelsCache {
                    auth_fingerprint,
                    model_ids: model_ids.clone(),
                    refreshed_at: now_millis(),
                });
                available_models
                    .into_iter()
                    .filter(|model| model.provider != "openai-codex" || model_ids.contains(&model.id))
                    .collect()
            }
            None => {
                if let Some(cached) = cached
                    .as_ref()
                    .filter(|cached| cached.auth_fingerprint == auth_fingerprint)
                    .filter(|cached| now_millis() - cached.refreshed_at < 300_000)
                {
                    let model_ids = cached.model_ids.clone();
                    return available_models
                        .into_iter()
                        .filter(|model| model.provider != "openai-codex" || model_ids.contains(&model.id))
                        .collect();
                }
                available_models
                    .into_iter()
                    .filter(|model| model.provider != "openai-codex")
                    .collect()
            }
        }
    }

    /// Find a model by provider and ID.
    pub fn find(&self, provider: &str, model_id: &str) -> Option<Model> {
        self.models
            .iter()
            .find(|model| model.provider == provider && model.id == model_id)
            .cloned()
    }

    /// Get API key for a model.
    pub fn has_configured_auth(&self, model: &Model) -> bool {
        self.auth_storage.has_auth(&model.provider) || self.has_configured_provider_request_auth(&model.provider)
    }

    fn create_provider_request_auth_source(
        &self,
        source: &str,
        identity_material: &str,
        value_material: Option<&str>,
        label: Option<&str>,
        resolve_value_material: Option<Arc<dyn Fn() -> Option<String> + Send + Sync>>,
    ) -> ProviderRequestAuthSource {
        let identity_fingerprint =
            fingerprint_provider_request_auth_source(source, &format!("identity:{}", identity_material));
        let value_fingerprint = value_material.map(|value_material| {
            fingerprint_provider_request_auth_source(
                source,
                &format!("value:{}\0{}", identity_material, value_material),
            )
        });
        let resolve_value_fingerprint = resolve_value_material.map(|resolver| {
            let source = source.to_string();
            let identity_material = identity_material.to_string();
            Arc::new(move || {
                let value_material = resolver()?;
                Some(fingerprint_provider_request_auth_source(
                    &source,
                    &format!("value:{}\0{}", identity_material, value_material),
                ))
            }) as Arc<dyn Fn() -> Option<String> + Send + Sync>
        });
        ProviderRequestAuthSource {
            source: source.to_string(),
            configured: true,
            label: label.map(|value| value.to_string()),
            identity_fingerprint,
            value_fingerprint,
            resolve_value_fingerprint,
        }
    }

    fn get_provider_request_auth_source(
        &self,
        provider: &str,
        resolved_api_key: Option<&str>,
    ) -> Option<ProviderRequestAuthSource> {
        let provider_api_key = self.provider_request_configs.get(provider)?.api_key.clone()?;

        if let Some(command) = provider_api_key.strip_prefix('!') {
            let command = format!("!{}", command);
            let value_material = resolved_api_key.map(|resolved| format!("{}\0{}", command, resolved));
            let resolver_command = command.clone();
            return Some(self.create_provider_request_auth_source(
                "models_json_command",
                &command,
                value_material.as_deref(),
                None,
                Some(Arc::new(move || {
                    let resolved = resolve_config_value_uncached(&resolver_command)?;
                    Some(format!("{}\0{}", resolver_command, resolved))
                })),
            ));
        }

        if let Ok(env_value) = std::env::var(&provider_api_key) {
            if !env_value.is_empty() {
                return Some(self.create_provider_request_auth_source(
                    "environment",
                    &provider_api_key,
                    Some(&format!("{}\0{}", provider_api_key, env_value)),
                    Some(&provider_api_key),
                    None,
                ));
            }
        }

        Some(self.create_provider_request_auth_source(
            "models_json_key",
            provider,
            Some(&provider_api_key),
            None,
            None,
        ))
    }

    fn is_provider_request_auth_stale(&self, provider: &str, source: &ProviderRequestAuthSource) -> bool {
        let matching_stale = self.get_matching_stale_provider_request_auth_sources(provider, source);
        if matching_stale.is_empty() {
            return false;
        }
        let value_fingerprint = source
            .value_fingerprint
            .clone()
            .or_else(|| source.resolve_value_fingerprint.as_ref().and_then(|resolver| resolver()));
        match value_fingerprint {
            Some(value_fingerprint) => matching_stale
                .iter()
                .any(|token| token.value_fingerprint == value_fingerprint),
            None => false,
        }
    }

    fn is_provider_request_auth_stale_for_status(
        &self,
        provider: &str,
        source: &ProviderRequestAuthSource,
    ) -> bool {
        let matching_stale = self.get_matching_stale_provider_request_auth_sources(provider, source);
        if matching_stale.is_empty() {
            return false;
        }
        let Some(value_fingerprint) = source.value_fingerprint.clone() else {
            return true;
        };
        let is_stale = matching_stale
            .iter()
            .any(|token| token.value_fingerprint == value_fingerprint);
        if !is_stale {
            self.clear_stale_provider_request_auth_source(provider, source);
        }
        is_stale
    }

    fn stale_provider_request_auth_sources(
        &self,
    ) -> std::sync::MutexGuard<'_, HashMap<String, Vec<AuthSourceToken>>> {
        self.stale_provider_request_auth_sources
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn get_matching_stale_provider_request_auth_sources(
        &self,
        provider: &str,
        source: &ProviderRequestAuthSource,
    ) -> Vec<AuthSourceToken> {
        let stale = self.stale_provider_request_auth_sources();
        let Some(stale) = stale.get(provider) else {
            return Vec::new();
        };
        stale
            .iter()
            .filter(|token| {
                token.source == source.source && token.identity_fingerprint == source.identity_fingerprint
            })
            .cloned()
            .collect()
    }

    fn clear_stale_provider_request_auth_source(&self, provider: &str, source: &ProviderRequestAuthSource) {
        let mut stale = self.stale_provider_request_auth_sources();
        let Some(entries) = stale.get(provider) else {
            return;
        };
        let next: Vec<AuthSourceToken> = entries
            .iter()
            .filter(|token| {
                token.source != source.source || token.identity_fingerprint != source.identity_fingerprint
            })
            .cloned()
            .collect();
        if next.is_empty() {
            stale.remove(provider);
        } else {
            stale.insert(provider.to_string(), next);
        }
    }

    fn get_provider_request_auth_source_token(
        &self,
        provider: &str,
        source: &ProviderRequestAuthSource,
    ) -> Option<AuthSourceToken> {
        let value_fingerprint = source
            .value_fingerprint
            .clone()
            .or_else(|| source.resolve_value_fingerprint.as_ref().and_then(|resolver| resolver()))?;
        Some(AuthSourceToken {
            provider: provider.to_string(),
            source: source.source.clone(),
            identity_fingerprint: source.identity_fingerprint.clone(),
            value_fingerprint,
        })
    }

    fn set_last_provider_auth_source_token(&mut self, provider: &str, token: Option<AuthSourceToken>) {
        match token {
            Some(token) => {
                self.last_provider_auth_source_tokens
                    .insert(provider.to_string(), token);
            }
            None => {
                self.last_provider_auth_source_tokens.remove(provider);
            }
        }
    }

    fn has_configured_provider_request_auth(&self, provider: &str) -> bool {
        let Some(source) = self.get_provider_request_auth_source(provider, None) else {
            return false;
        };
        !self.is_provider_request_auth_stale_for_status(provider, &source)
    }

    pub fn mark_provider_auth_stale(&mut self, provider: &str) -> bool {
        if self.auth_storage.mark_auth_stale(provider) {
            return true;
        }

        match self.get_current_provider_auth_source_token(provider) {
            Some(token) => self.mark_provider_auth_source_stale(&token),
            None => false,
        }
    }

    /// Forget stale-auth markings; a structured auth failure on the next request
    /// re-marks the provider.
    pub fn clear_provider_auth_stale(&mut self, provider: &str) {
        self.stale_provider_request_auth_sources().remove(provider);
        self.auth_storage.clear_auth_stale(provider);
    }

    pub fn get_current_provider_auth_source_token(&self, provider: &str) -> Option<AuthSourceToken> {
        if let Some(last_request_token) = self.last_provider_auth_source_tokens.get(provider) {
            return Some(last_request_token.clone());
        }

        if let Some(auth_storage_token) = self.auth_storage.get_current_auth_source_token(provider) {
            return Some(auth_storage_token);
        }

        let provider_api_key = self.provider_request_configs.get(provider)?.api_key.clone()?;
        let resolved_api_key = resolve_config_value_uncached(&provider_api_key);
        let source = self.get_provider_request_auth_source(provider, resolved_api_key.as_deref());
        let value_fingerprint = source.as_ref().and_then(|source| {
            source
                .value_fingerprint
                .clone()
                .or_else(|| source.resolve_value_fingerprint.as_ref().and_then(|resolver| resolver()))
        });
        let Some(source) = source else { return None };
        let value_fingerprint = value_fingerprint?;
        if self.is_provider_request_auth_stale(provider, &source) {
            return None;
        }

        Some(AuthSourceToken {
            provider: provider.to_string(),
            source: source.source.clone(),
            identity_fingerprint: source.identity_fingerprint.clone(),
            value_fingerprint,
        })
    }

    pub fn mark_provider_auth_source_stale(&mut self, token: &AuthSourceToken) -> bool {
        let mut marked = false;
        if let Some(provider_request_source) = self.get_provider_request_auth_source(&token.provider, None) {
            if provider_request_source.source == token.source
                && provider_request_source.identity_fingerprint == token.identity_fingerprint
            {
                let mut all_stale = self.stale_provider_request_auth_sources();
                let mut stale = all_stale.get(&token.provider).cloned().unwrap_or_default();
                if !stale.iter().any(|existing| {
                    existing.source == token.source
                        && existing.identity_fingerprint == token.identity_fingerprint
                        && existing.value_fingerprint == token.value_fingerprint
                }) {
                    stale.push(token.clone());
                }
                all_stale.insert(token.provider.clone(), stale);
                marked = true;
            }
        }

        if token.source != "models_json_key" && token.source != "models_json_command" {
            marked = self.auth_storage.mark_auth_source_stale(token) || marked;
        }

        marked
    }

    fn get_model_request_key(provider: &str, model_id: &str) -> String {
        format!("{}:{}", provider, model_id)
    }

    fn store_provider_request_config(
        &mut self,
        provider_name: &str,
        api_key: Option<&str>,
        headers: Option<&IndexMap<String, String>>,
        auth_header: Option<bool>,
    ) {
        if api_key.is_none() && headers.is_none() && !auth_header.unwrap_or(false) {
            return;
        }

        self.provider_request_configs.insert(
            provider_name.to_string(),
            ProviderRequestConfig {
                api_key: api_key.map(|value| value.to_string()),
                headers: headers.cloned(),
                auth_header,
            },
        );
    }

    fn store_model_headers(
        &mut self,
        provider_name: &str,
        model_id: &str,
        headers: Option<&IndexMap<String, String>>,
    ) {
        let key = Self::get_model_request_key(provider_name, model_id);
        match headers {
            Some(headers) if !headers.is_empty() => {
                self.model_request_headers.insert(key, headers.clone());
            }
            _ => {
                self.model_request_headers.remove(&key);
            }
        }
    }

    /// Get API key and request headers for a model.
    pub async fn get_api_key_and_headers(&mut self, model: &Model) -> ResolvedRequestAuth {
        let provider_config = self.provider_request_configs.get(&model.provider).cloned();

        let auth_storage_auth = match self
            .auth_storage
            .get_api_key_with_source_token(&model.provider, false)
            .await
        {
            Ok(result) => result,
            Err(error) => {
                return ResolvedRequestAuth {
                    ok: false,
                    api_key: None,
                    headers: None,
                    error: Some(error),
                }
            }
        };
        let mut api_key = auth_storage_auth.api_key;
        let mut auth_source_token = auth_storage_auth.source_token;

        if api_key.is_none() {
            if let Some(provider_api_key) = provider_config.as_ref().and_then(|config| config.api_key.clone()) {
                let resolved_api_key = match resolve_config_value_or_throw(
                    &provider_api_key,
                    &format!("API key for provider \"{}\"", model.provider),
                ) {
                    Ok(value) => value,
                    Err(error) => {
                        return ResolvedRequestAuth {
                            ok: false,
                            api_key: None,
                            headers: None,
                            error: Some(error),
                        }
                    }
                };
                let provider_request_auth_source =
                    self.get_provider_request_auth_source(&model.provider, Some(&resolved_api_key));
                if let Some(source) = provider_request_auth_source {
                    if !self.is_provider_request_auth_stale(&model.provider, &source) {
                        self.clear_stale_provider_request_auth_source(&model.provider, &source);
                        api_key = Some(resolved_api_key);
                        auth_source_token = self.get_provider_request_auth_source_token(&model.provider, &source);
                    }
                }
            }
        }
        self.set_last_provider_auth_source_token(
            &model.provider,
            if api_key.is_none() { None } else { auth_source_token },
        );

        let provider_headers = match resolve_headers_or_throw(
            provider_config.as_ref().and_then(|config| config.headers.as_ref()),
            &format!("provider \"{}\"", model.provider),
        ) {
            Ok(headers) => headers,
            Err(error) => {
                return ResolvedRequestAuth {
                    ok: false,
                    api_key: None,
                    headers: None,
                    error: Some(error),
                }
            }
        };
        let auth_storage_headers = self.auth_storage.get_provider_headers(&model.provider);
        let model_headers = match resolve_headers_or_throw(
            self.model_request_headers
                .get(&Self::get_model_request_key(&model.provider, &model.id)),
            &format!("model \"{}/{}\"", model.provider, model.id),
        ) {
            Ok(headers) => headers,
            Err(error) => {
                return ResolvedRequestAuth {
                    ok: false,
                    api_key: None,
                    headers: None,
                    error: Some(error),
                }
            }
        };

        let mut headers = if model.headers.is_some()
            || auth_storage_headers.is_some()
            || provider_headers.is_some()
            || model_headers.is_some()
        {
            let mut merged: IndexMap<String, String> = IndexMap::new();
            if let Some(model_headers) = &model.headers {
                for (key, value) in model_headers {
                    merged.insert(key.clone(), value.clone());
                }
            }
            if let Some(auth_storage_headers) = &auth_storage_headers {
                for (key, value) in auth_storage_headers {
                    merged.insert(key.clone(), value.clone());
                }
            }
            if let Some(provider_headers) = &provider_headers {
                for (key, value) in provider_headers {
                    merged.insert(key.clone(), value.clone());
                }
            }
            if let Some(model_headers) = &model_headers {
                for (key, value) in model_headers {
                    merged.insert(key.clone(), value.clone());
                }
            }
            Some(merged)
        } else {
            None
        };

        if provider_config
            .as_ref()
            .and_then(|config| config.auth_header)
            .unwrap_or(false)
        {
            let Some(api_key_value) = api_key.clone() else {
                return ResolvedRequestAuth {
                    ok: false,
                    api_key: None,
                    headers: None,
                    error: Some(format!("No API key found for \"{}\"", model.provider)),
                };
            };
            let mut merged = headers.unwrap_or_default();
            merged.insert("Authorization".to_string(), format!("Bearer {}", api_key_value));
            headers = Some(merged);
        }

        ResolvedRequestAuth {
            ok: true,
            api_key,
            headers: headers.filter(|headers| !headers.is_empty()),
            error: None,
        }
    }

    /// Return auth status for a provider, including request auth configured in
    /// models.json. This intentionally does not execute command-backed config
    /// values.
    pub fn get_provider_auth_status(&self, provider: &str) -> AuthStatus {
        let auth_status = self.auth_storage.get_auth_status(provider);
        if auth_status.source.is_some() && auth_status.source.as_deref() != Some("stale") {
            return auth_status;
        }

        let Some(source) = self.get_provider_request_auth_source(provider, None) else {
            return auth_status;
        };

        if self.is_provider_request_auth_stale_for_status(provider, &source) {
            return AuthStatus {
                configured: false,
                source: Some("stale".to_string()),
                label: Some("expired".to_string()),
            };
        }

        AuthStatus {
            configured: true,
            source: Some(source.source.clone()),
            label: source.label.clone(),
        }
    }

    /// Get display name for a provider.
    pub fn get_provider_display_name(&self, provider: &str) -> String {
        let registered_provider = self.registered_providers.get(provider);
        let oauth_provider = self
            .auth_storage
            .get_oauth_providers()
            .into_iter()
            .find(|candidate| candidate.id == provider);

        registered_provider
            .and_then(|config| config.name.clone())
            .or_else(|| registered_provider.and_then(|config| config.oauth.as_ref().map(|oauth| oauth.name.clone())))
            .or_else(|| oauth_provider.map(|oauth| oauth.name))
            .or_else(|| built_in_provider_display_names().get(provider).cloned())
            .unwrap_or_else(|| provider.to_string())
    }

    /// Get API key for a provider.
    pub async fn get_api_key_for_provider(&mut self, provider: &str) -> Option<String> {
        let auth_storage_auth = self
            .auth_storage
            .get_api_key_with_source_token(provider, false)
            .await
            .ok()?;
        if let Some(api_key) = auth_storage_auth.api_key {
            self.set_last_provider_auth_source_token(provider, auth_storage_auth.source_token);
            return Some(api_key);
        }

        let provider_api_key = self.provider_request_configs.get(provider)?.api_key.clone()?;
        let Some(resolved_api_key) = resolve_config_value_uncached(&provider_api_key) else {
            self.set_last_provider_auth_source_token(provider, None);
            return None;
        };
        let source = self.get_provider_request_auth_source(provider, Some(&resolved_api_key));
        let Some(source) = source else {
            self.set_last_provider_auth_source_token(provider, None);
            return None;
        };
        if self.is_provider_request_auth_stale(provider, &source) {
            self.set_last_provider_auth_source_token(provider, None);
            return None;
        }
        self.clear_stale_provider_request_auth_source(provider, &source);
        let token = self.get_provider_request_auth_source_token(provider, &source);
        self.set_last_provider_auth_source_token(provider, token);

        Some(resolved_api_key)
    }

    /// Check if a model is using OAuth credentials (subscription).
    pub fn is_using_oauth(&self, model: &Model) -> bool {
        matches!(self.auth_storage.get(&model.provider), Some(AuthCredential::OAuth { .. }))
    }

    /// Register a provider dynamically (from extensions).
    ///
    /// If provider has models: replaces all existing models for this provider.
    /// If provider has only baseUrl/headers: overrides existing models' URLs.
    /// If provider has oauth: registers OAuth provider for /login support.
    pub fn register_provider(
        &mut self,
        provider_name: &str,
        config: ProviderConfigInput,
    ) -> Result<(), String> {
        self.validate_provider_config(provider_name, &config)?;
        self.apply_provider_config(provider_name, &config);
        self.upsert_registered_provider(provider_name, config);
        Ok(())
    }

    /// Unregister a previously registered provider.
    ///
    /// Removes the provider from the registry and reloads models from disk so
    /// that built-in models overridden by this provider are restored to their
    /// original state. Also resets dynamic OAuth and API stream registrations
    /// before reapplying remaining dynamic providers. Has no effect if the
    /// provider was never registered.
    pub fn unregister_provider(&mut self, provider_name: &str) {
        if !self.registered_providers.contains_key(provider_name) {
            return;
        }
        self.registered_providers.shift_remove(provider_name);
        self.refresh();
    }

    /// Upsert a provider config into registeredProviders. If the provider is
    /// already registered, defined values in the incoming config override
    /// existing ones; undefined values are preserved from the stored config.
    fn upsert_registered_provider(&mut self, provider_name: &str, config: ProviderConfigInput) {
        if !self.registered_providers.contains_key(provider_name) {
            self.registered_providers
                .insert(provider_name.to_string(), config);
            return;
        }
        {
            let existing = self
                .registered_providers
                .get_mut(provider_name)
                .expect("registered provider checked above");
            {
                if config.name.is_some() {
                    existing.name = config.name;
                }
                if config.base_url.is_some() {
                    existing.base_url = config.base_url;
                }
                if config.api_key.is_some() {
                    existing.api_key = config.api_key;
                }
                if config.api.is_some() {
                    existing.api = config.api;
                }
                if config.stream_simple.is_some() {
                    existing.stream_simple = config.stream_simple;
                }
                if config.headers.is_some() {
                    existing.headers = config.headers;
                }
                if config.auth_header.is_some() {
                    existing.auth_header = config.auth_header;
                }
                if config.oauth.is_some() {
                    existing.oauth = config.oauth;
                }
                if config.models.is_some() {
                    existing.models = config.models;
                }
            }
        }
    }

    fn validate_provider_config(&self, provider_name: &str, config: &ProviderConfigInput) -> Result<(), String> {
        if config.stream_simple.is_some() && config.api.is_none() {
            return Err(format!(
                "Provider {}: \"api\" is required when registering streamSimple.",
                provider_name
            ));
        }

        let models = match &config.models {
            Some(models) if !models.is_empty() => models,
            _ => return Ok(()),
        };

        if config.base_url.is_none() {
            return Err(format!(
                "Provider {}: \"baseUrl\" is required when defining models.",
                provider_name
            ));
        }
        if config.api_key.is_none() && config.oauth.is_none() {
            return Err(format!(
                "Provider {}: \"apiKey\" or \"oauth\" is required when defining models.",
                provider_name
            ));
        }

        for model_def in models {
            validate_native_compaction_capability(
                provider_name,
                model_def.native_compaction_model_ref(),
                NativeCompactionProviderRef {
                    api: config.api.as_deref(),
                    base_url: config.base_url.as_deref(),
                },
            )?;
            let api = model_def.api.as_deref().or(config.api.as_deref());
            if api.is_none() {
                return Err(format!(
                    "Provider {}, model {}: no \"api\" specified.",
                    provider_name, model_def.id
                ));
            }
        }
        Ok(())
    }

    fn apply_provider_config(&mut self, provider_name: &str, config: &ProviderConfigInput) {
        if let Some(oauth) = &config.oauth {
            register_oauth_provider(oauth.to_oauth_provider_interface(provider_name));
        }

        if let Some(stream_simple) = &config.stream_simple {
            let api = config.api.clone().unwrap_or_default();
            let stream_simple = stream_simple.clone();
            let stream: pi_ai::types::StreamFunction = {
                let stream_simple = stream_simple.clone();
                Arc::new(move |model: &Model, context: &Context, options: Option<&pi_ai::types::StreamOptions>| {
                    let simple = options.map(|options| SimpleStreamOptions {
                        stream: options.clone(),
                        reasoning: None,
                        thinking_budgets: None,
                    });
                    stream_simple(model, context, simple.as_ref())
                })
            };
            // `api-registry.ts` types `stream_simple` as
            // `StreamFunction<TApi, SimpleStreamOptions>`; the Rust port declares
            // `ApiProvider.stream_simple` as a plain `StreamFunction` (base
            // `StreamOptions`), so the same adapter serves both fields, exactly as
            // the TypeScript registers one function for `stream` and `streamSimple`.
            // REPAIR CURSOR: shared-contract drift in crates/pi-ai/src/api_registry.rs
            // (`ApiProvider` / `ApiProviderInternal.stream_simple` should be
            // `ApiStreamSimpleFunction`); pi-ai is outside this pack.
            pi_ai::api_registry::register_api_provider(
                pi_ai::api_registry::ApiProvider {
                    api: api.clone(),
                    stream: stream.clone(),
                    stream_simple: stream,
                    compact: None,
                    supports_compaction: None,
                },
                Some(format!("provider:{}", provider_name)),
            );
        }

        self.store_provider_request_config(
            provider_name,
            config.api_key.as_deref(),
            config.headers.as_ref(),
            config.auth_header,
        );

        let models = config.models.clone().unwrap_or_default();
        if !models.is_empty() {
            self.models.retain(|model| model.provider != provider_name);

            for model_def in &models {
                let api = model_def.api.clone().or_else(|| config.api.clone());
                self.store_model_headers(provider_name, &model_def.id, model_def.headers.as_ref());

                self.models.push(Model {
                    id: model_def.id.clone(),
                    name: model_def.name.clone().unwrap_or_else(|| model_def.id.clone()),
                    api: api.unwrap_or_default(),
                    provider: provider_name.to_string(),
                    base_url: model_def
                        .base_url
                        .clone()
                        .or_else(|| config.base_url.clone())
                        .unwrap_or_default(),
                    reasoning: model_def.reasoning.unwrap_or(false),
                    thinking_level_map: model_def.thinking_level_map.clone(),
                    input: model_def
                        .input
                        .clone()
                        .unwrap_or_default()
                        .iter()
                        .filter_map(|value| match value.as_str() {
                            "text" => Some(pi_ai::types::InputModality::Text),
                            "image" => Some(pi_ai::types::InputModality::Image),
                            _ => None,
                        })
                        .collect(),
                    cost: model_def.cost.clone().unwrap_or_default(),
                    context_window: model_def.context_window.unwrap_or(0.0),
                    max_input_tokens: model_def.max_input_tokens,
                    max_tokens: model_def.max_tokens.unwrap_or(0.0),
                    featured: None,
                    native_compaction: model_def.native_compaction.clone(),
                    headers: None,
                    compat: model_def.compat.clone(),
                });
            }
            if let Some(modify_models) = config.oauth.as_ref().and_then(|oauth| oauth.modify_models.as_ref()) {
                if let Some(AuthCredential::OAuth { credentials }) = self.auth_storage.get(provider_name) {
                    self.models = modify_models(self.models.clone(), &credentials);
                }
            }
        } else if config.base_url.is_some() || config.headers.is_some() {
            let base_url = config.base_url.clone();
            self.models = self
                .models
                .iter()
                .map(|model| {
                    if model.provider != provider_name {
                        return model.clone();
                    }
                    let mut updated = model.clone();
                    if let Some(base_url) = &base_url {
                        updated.base_url = base_url.clone();
                    }
                    updated
                })
                .collect();
        }
    }
}

/// `fingerprintProviderRequestAuthSource(source, material)`:
/// `sha256(source + "\0" + material)` hex, prefixed with `source:`.
fn fingerprint_provider_request_auth_source(source: &str, material: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(source.as_bytes());
    hasher.update(b"\0");
    hasher.update(material.as_bytes());
    format!("{}:{:x}", source, hasher.finalize())
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::auth_storage::{AuthStorage, AuthStorageOptions};
    use pi_ai::types::InputModality;
    use serde_json::json;

    fn in_memory_auth() -> AuthStorage {
        AuthStorage::in_memory(
            IndexMap::new(),
            Some(AuthStorageOptions {
                prime_cli_config_path: None,
                use_prime_cli_config: false,
            }),
        )
    }

    fn registry_with_config(dir: &Path, body: &str) -> ModelRegistry {
        let path = dir.join("models.json");
        std::fs::write(&path, body).unwrap();
        ModelRegistry::create(in_memory_auth(), Some(path.to_string_lossy().to_string()))
    }

    #[test]
    fn strip_json_comments_keeps_string_literals() {
        assert_eq!(strip_json_comments("{\"a\": 1} // trailing\n"), "{\"a\": 1} \n");
        assert_eq!(strip_json_comments("{\"a\": \"// not a comment\"}"), "{\"a\": \"// not a comment\"}");
        assert_eq!(strip_json_comments("{\"a\": [1, 2,]}"), "{\"a\": [1, 2]}");
        assert_eq!(strip_json_comments("{\"a\": \"x,\"}"), "{\"a\": \"x,\"}");
    }

    #[test]
    fn schema_validation_reports_required_properties_and_unions() {
        let errors = validate_models_config_schema(&json!({}));
        assert_eq!(errors, vec!["  - providers: Expected required property"]);

        let errors = validate_models_config_schema(&json!({"providers": {"custom": {"models": [{}]}}}));
        assert_eq!(errors, vec!["  - providers.custom.models.0.id: Expected required property"]);

        let errors = validate_models_config_schema(&json!({
            "providers": {"custom": {"models": [{"id": "m", "input": ["audio"]}]}}
        }));
        assert_eq!(errors, vec!["  - providers.custom.models.0.input.0: Expected union"]);

        let errors = validate_models_config_schema(&json!({
            "providers": {"custom": {"modelOverrides": {"m": {"contextWindow": "big"}}}}
        }));
        assert_eq!(
            errors,
            vec!["  - providers.custom.modelOverrides.m.contextWindow: Expected number"]
        );

        assert!(validate_models_config_schema(&json!({
            "providers": {"custom": {"baseUrl": "https://x.test", "apiKey": "K", "api": "openai-completions", "models": [{"id": "m"}]}}
        }))
        .is_empty());
    }

    #[test]
    fn native_compaction_capability_is_allowlisted() {
        let provider = ProviderConfig {
            api: Some("openai-responses".to_string()),
            base_url: Some("https://gateway.test/azure-openai/v1".to_string()),
            ..Default::default()
        };
        let capability = NativeCompactionCapability {
            protocol: "openai-responses-compact-v1".to_string(),
            provider: "azure-openai-managed".to_string(),
            model: "gpt-6-astra".to_string(),
            endpoint: "https://gateway.test/azure-openai/v1/responses/compact".to_string(),
            api_version: "v1".to_string(),
            enabled: false,
            validation: pi_ai::types::NativeCompactionValidation::Unverified,
        };
        let model = ModelDefinition {
            id: "gpt-6-astra".to_string(),
            native_compaction: Some(capability.clone()),
            ..Default::default()
        };
        assert!(validate_native_compaction_capability(
            "azure-openai-managed",
            model.native_compaction_model_ref(),
            provider.native_compaction_provider_ref()
        )
        .is_ok());

        let other = ModelDefinition {
            id: "other".to_string(),
            native_compaction: Some(capability.clone()),
            ..Default::default()
        };
        assert_eq!(
            validate_native_compaction_capability(
                "azure-openai-managed",
                other.native_compaction_model_ref(),
                provider.native_compaction_provider_ref()
            )
            .unwrap_err(),
            "Provider azure-openai-managed, model other: nativeCompaction is allowlisted only for azure-openai-managed/gpt-6-astra."
        );

        let enabled = ModelDefinition {
            id: "gpt-6-astra".to_string(),
            native_compaction: Some(NativeCompactionCapability {
                enabled: true,
                ..capability.clone()
            }),
            ..Default::default()
        };
        assert_eq!(
            validate_native_compaction_capability(
                "azure-openai-managed",
                enabled.native_compaction_model_ref(),
                provider.native_compaction_provider_ref()
            )
            .unwrap_err(),
            "Provider azure-openai-managed, model gpt-6-astra: nativeCompaction cannot be enabled until validation is \"live-verified\"."
        );

        let wrong_endpoint = ModelDefinition {
            id: "gpt-6-astra".to_string(),
            native_compaction: Some(NativeCompactionCapability {
                endpoint: "https://gateway.test/azure-openai/v1/responses/other".to_string(),
                ..capability.clone()
            }),
            ..Default::default()
        };
        assert_eq!(
            validate_native_compaction_capability(
                "azure-openai-managed",
                wrong_endpoint.native_compaction_model_ref(),
                provider.native_compaction_provider_ref()
            )
            .unwrap_err(),
            "Provider azure-openai-managed, model gpt-6-astra: nativeCompaction.endpoint must equal the model base URL plus /responses/compact."
        );
    }

    #[test]
    fn merge_compat_merges_nested_routing_objects() {
        let base: Compat = serde_json::from_value(json!({
            "supportsStore": true,
            "maxTokensField": "max_tokens",
            "openRouterRouting": {"allow_fallbacks": true, "data_collection": "allow"}
        }))
        .unwrap();
        let over: Compat = serde_json::from_value(json!({
            "supportsStore": false,
            "openRouterRouting": {"data_collection": "deny"}
        }))
        .unwrap();
        let merged = merge_compat(Some(&base), Some(&over)).unwrap();
        let value = serde_json::to_value(&merged).unwrap();
        assert_eq!(value.get("supportsStore"), Some(&json!(false)));
        assert_eq!(value.get("maxTokensField"), Some(&json!("max_tokens")));
        assert_eq!(
            value.pointer("/openRouterRouting/allow_fallbacks"),
            Some(&json!(true))
        );
        assert_eq!(
            value.pointer("/openRouterRouting/data_collection"),
            Some(&json!("deny"))
        );
        assert_eq!(merge_compat(None, None), None);
        assert_eq!(merge_compat(Some(&base), None).unwrap(), base);
    }

    #[test]
    fn apply_model_override_deep_merges_cost_and_compat() {
        let model: Model = serde_json::from_value(json!({
            "id": "m",
            "name": "M",
            "api": "openai-completions",
            "provider": "custom",
            "baseUrl": "https://x.test",
            "reasoning": false,
            "input": ["text"],
            "cost": {"input": 1.0, "output": 2.0, "cacheRead": 3.0, "cacheWrite": 4.0},
            "contextWindow": 1000.0,
            "maxTokens": 100.0,
            "compat": {"supportsStore": true, "maxTokensField": "max_tokens"}
        }))
        .unwrap();
        let over: ModelOverride = serde_json::from_value(json!({
            "name": "Renamed",
            "reasoning": true,
            "input": ["text", "image"],
            "cost": {"output": 9.0},
            "maxTokens": 500.0,
            "compat": {"supportsStore": false}
        }))
        .unwrap();
        let updated = apply_model_override(&model, &over);
        assert_eq!(updated.name, "Renamed");
        assert!(updated.reasoning);
        assert_eq!(updated.input, vec![InputModality::Text, InputModality::Image]);
        assert_eq!(updated.cost.input, 1.0);
        assert_eq!(updated.cost.output, 9.0);
        assert_eq!(updated.cost.cache_read, 3.0);
        assert_eq!(updated.max_tokens, 500.0);
        assert_eq!(updated.context_window, 1000.0);
        let compat = serde_json::to_value(updated.compat.unwrap()).unwrap();
        assert_eq!(compat.get("supportsStore"), Some(&json!(false)));
        assert_eq!(compat.get("maxTokensField"), Some(&json!("max_tokens")));
    }

    /// TS: `parseModels` (model-registry.ts:829) `compat: mergeCompat(providerConfig.compat, modelDef.compat)`
    /// assigned into a `Model<Api>` whose compat shape `Api` selects
    /// (packages/ai/src/types.ts:494-500). A `models.json` entry whose `api` is
    /// `anthropic-messages` and whose compat omits `supportsEagerToolInputStreaming`
    /// must still arrive as `AnthropicMessagesCompat`; otherwise
    /// `getAnthropicCompat` (providers/anthropic.ts:178-183) reads
    /// `compat_anthropic()` as `None` and silently returns the wrong defaults.
    #[test]
    fn custom_model_compat_is_shaped_by_api_not_by_key_sniffing() {
        let dir = tempfile::tempdir().unwrap();
        let registry = registry_with_config(
            dir.path(),
            r#"{
                "providers": {
                    "proxy": {
                        "baseUrl": "https://proxy.test",
                        "apiKey": "PROXY_KEY",
                        "compat": {"supportsLongCacheRetention": false},
                        "models": [
                            {"id": "claude-sonnet-4-5", "api": "anthropic-messages"}
                        ]
                    }
                }
            }"#,
        );
        assert!(registry.get_error().is_none(), "{:?}", registry.get_error());

        let model = registry.find("proxy", "claude-sonnet-4-5").unwrap();
        assert_eq!(model.api, "anthropic-messages");
        // The provider-level compat carries no `supportsEagerToolInputStreaming`,
        // so key-sniffing classified it as `OpenAICompletionsCompat` and dropped it.
        // Plain `expect` (not a formatting closure): the message names the compat
        // key that key-sniffing dropped, so a broken shape selection reports the
        // real symptom instead of a bare `None`.
        let compat = model
            .compat_anthropic()
            .expect("compat must be AnthropicMessagesCompat so supportsLongCacheRetention is readable");
        assert_eq!(
            compat.supports_long_cache_retention,
            Some(false),
            "compat.supportsLongCacheRetention must survive the api-directed shape selection"
        );
    }

    /// Same divergence for `openai-responses`: `getCompat`
    /// (providers/openai-responses.ts:80-85) reads
    /// `model.compat?.sendSessionIdHeader`, which is only reachable through
    /// `compat_responses()`. A provider compat object that carries no
    /// `sendSessionIdHeader` key must still land in `OpenAIResponsesCompat`.
    #[test]
    fn custom_model_responses_compat_is_shaped_by_api() {
        let dir = tempfile::tempdir().unwrap();
        let registry = registry_with_config(
            dir.path(),
            r#"{
                "providers": {
                    "proxy": {
                        "baseUrl": "https://proxy.test",
                        "apiKey": "PROXY_KEY",
                        "compat": {"supportsLongCacheRetention": false},
                        "models": [
                            {"id": "gpt-5", "api": "openai-responses"}
                        ]
                    }
                }
            }"#,
        );
        assert!(registry.get_error().is_none(), "{:?}", registry.get_error());

        let model = registry.find("proxy", "gpt-5").unwrap();
        let compat = model
            .compat_responses()
            .expect("compat must be OpenAIResponsesCompat for api openai-responses");
        assert_eq!(
            compat.supports_long_cache_retention,
            Some(false),
            "compat.supportsLongCacheRetention must survive the api-directed shape selection"
        );
    }

    #[test]
    fn custom_models_load_and_override_built_ins() {
        let dir = tempfile::tempdir().unwrap();
        let mut registry = registry_with_config(
            dir.path(),
            r#"{
                // local provider
                "providers": {
                    "anthropic": {
                        "baseUrl": "https://proxy.test",
                        "modelOverrides": {"claude-sonnet-4-5": {"maxTokens": 4096}}
                    },
                    "ollama": {
                        "name": "Ollama",
                        "baseUrl": "http://localhost:11434/v1",
                        "apiKey": "OLLAMA_KEY",
                        "api": "openai-completions",
                        "models": [
                            {"id": "llama3", "name": "Llama 3", "reasoning": false, "input": ["text"],
                             "cost": {"input": 0.0, "output": 0.0, "cacheRead": 0.0, "cacheWrite": 0.0},
                             "contextWindow": 8192, "maxTokens": 2048}
                        ]
                    }
                }
            }"#,
        );
        assert!(registry.get_error().is_none(), "{:?}", registry.get_error());

        let anthropic = registry.find("anthropic", "claude-sonnet-4-5").unwrap();
        assert_eq!(anthropic.base_url, "https://proxy.test");
        assert_eq!(anthropic.max_tokens, 4096.0);

        let ollama = registry.find("ollama", "llama3").unwrap();
        assert_eq!(ollama.name, "Llama 3");
        assert_eq!(ollama.api, "openai-completions");
        assert_eq!(ollama.context_window, 8192.0);
        assert_eq!(ollama.max_tokens, 2048.0);

        // The provider-level request config is stored for auth resolution.
        assert_eq!(
            registry.provider_request_configs.get("ollama").unwrap().api_key.as_deref(),
            Some("OLLAMA_KEY")
        );
    }

    #[test]
    fn invalid_models_json_reports_the_typescript_messages() {
        let dir = tempfile::tempdir().unwrap();
        let registry = registry_with_config(dir.path(), "{ not json }");
        let error = registry.get_error().unwrap();
        assert!(error.starts_with("Failed to parse models.json: "), "{}", error);
        assert!(error.contains("models.json"));

        let registry = registry_with_config(dir.path(), r#"{"providers": {"custom": {}}}"#);
        let error = registry.get_error().unwrap();
        assert_eq!(
            error,
            format!(
                "Failed to load models.json: Provider custom: must specify \"baseUrl\", \"headers\", \"compat\", \"modelOverrides\", or \"models\".\n\nFile: {}",
                dir.path().join("models.json").to_string_lossy()
            )
        );

        let registry = registry_with_config(dir.path(), r#"{"providers": {"custom": {"headers": {"X": "1"}}}}"#);
        assert!(registry.get_error().is_none());

        let registry = registry_with_config(
            dir.path(),
            r#"{"providers": {"custom": {"models": [{"id": "m"}]}}}"#,
        );
        let error = registry.get_error().unwrap();
        assert!(
            error.starts_with("Failed to load models.json: Provider custom: \"baseUrl\" is required when defining custom models."),
            "{}",
            error
        );
    }

    #[test]
    fn provider_validation_matches_typescript_errors() {
        let dir = tempfile::tempdir().unwrap();
        let mut registry = ModelRegistry::in_memory(in_memory_auth());

        let error = registry
            .register_provider(
                "custom",
                ProviderConfigInput {
                    stream_simple: Some(Arc::new(|_model, _context, _options| {
                        pi_ai::utils::event_stream::AssistantMessageEventStream::new()
                    })),
                    ..Default::default()
                },
            )
            .unwrap_err();
        assert_eq!(error, "Provider custom: \"api\" is required when registering streamSimple.");

        let error = registry
            .register_provider(
                "custom",
                ProviderConfigInput {
                    models: Some(vec![ModelDefinition {
                        id: "m".to_string(),
                        ..Default::default()
                    }]),
                    ..Default::default()
                },
            )
            .unwrap_err();
        assert_eq!(error, "Provider custom: \"baseUrl\" is required when defining models.");

        let error = registry
            .register_provider(
                "custom",
                ProviderConfigInput {
                    base_url: Some("https://x.test".to_string()),
                    models: Some(vec![ModelDefinition {
                        id: "m".to_string(),
                        ..Default::default()
                    }]),
                    ..Default::default()
                },
            )
            .unwrap_err();
        assert_eq!(error, "Provider custom: \"apiKey\" or \"oauth\" is required when defining models.");

        let error = registry
            .register_provider(
                "custom",
                ProviderConfigInput {
                    base_url: Some("https://x.test".to_string()),
                    api_key: Some("K".to_string()),
                    models: Some(vec![ModelDefinition {
                        id: "m".to_string(),
                        ..Default::default()
                    }]),
                    ..Default::default()
                },
            )
            .unwrap_err();
        assert_eq!(error, "Provider custom, model m: no \"api\" specified.");

        let _ = dir;
    }

    #[test]
    fn register_provider_adds_and_replaces_models() {
        let mut registry = ModelRegistry::in_memory(in_memory_auth());
        registry
            .register_provider(
                "custom",
                ProviderConfigInput {
                    name: Some("Custom".to_string()),
                    base_url: Some("https://x.test".to_string()),
                    api_key: Some("CUSTOM_KEY".to_string()),
                    api: Some("openai-completions".to_string()),
                    models: Some(vec![ModelDefinition {
                        id: "m".to_string(),
                        name: Some("M".to_string()),
                        cost: Some(ModelCost {
                            input: 1.0,
                            output: 2.0,
                            cache_read: 3.0,
                            cache_write: 4.0,
                        }),
                        context_window: Some(1000.0),
                        max_tokens: Some(100.0),
                        ..Default::default()
                    }]),
                    ..Default::default()
                },
            )
            .unwrap();
        let model = registry.find("custom", "m").unwrap();
        assert_eq!(model.base_url, "https://x.test");
        assert_eq!(model.context_window, 1000.0);
        assert_eq!(registry.get_provider_display_name("custom"), "Custom");
        assert_eq!(registry.get_provider_display_name("anthropic"), "Anthropic (Claude Pro/Max)");
        assert_eq!(registry.get_provider_display_name("unknown-provider"), "unknown-provider");

        // baseUrl-only registration rewrites the provider's existing models.
        registry
            .register_provider(
                "custom",
                ProviderConfigInput {
                    base_url: Some("https://y.test".to_string()),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(registry.find("custom", "m").unwrap().base_url, "https://y.test");
        assert_eq!(registry.get_provider_display_name("custom"), "Custom");

        registry.unregister_provider("custom");
        assert!(registry.find("custom", "m").is_none());
    }

    #[test]
    fn provider_auth_status_prefers_models_json_keys() {
        let dir = tempfile::tempdir().unwrap();
        let registry = registry_with_config(
            dir.path(),
            r#"{"providers": {"ollama": {"apiKey": "OLLAMA_KEY", "baseUrl": "http://localhost:11434/v1"}}}"#,
        );
        std::env::remove_var("OLLAMA_KEY");
        let status = registry.get_provider_auth_status("ollama");
        assert!(status.configured);
        assert_eq!(status.source.as_deref(), Some("models_json_key"));

        std::env::set_var("OLLAMA_KEY", "secret");
        let status = registry.get_provider_auth_status("ollama");
        std::env::remove_var("OLLAMA_KEY");
        assert!(status.configured);
        assert_eq!(status.source.as_deref(), Some("environment"));
        assert_eq!(status.label.as_deref(), Some("OLLAMA_KEY"));
    }

    #[test]
    fn command_backed_keys_are_not_executed_for_status() {
        let dir = tempfile::tempdir().unwrap();
        let registry = registry_with_config(
            dir.path(),
            r#"{"providers": {"custom": {"apiKey": "!echo secret", "baseUrl": "https://x.test"}}}"#,
        );
        let status = registry.get_provider_auth_status("custom");
        assert!(status.configured);
        assert_eq!(status.source.as_deref(), Some("models_json_command"));
    }

    #[test]
    fn stale_provider_request_auth_hides_and_recovers() {
        let dir = tempfile::tempdir().unwrap();
        let mut registry = registry_with_config(
            dir.path(),
            r#"{"providers": {"custom": {"apiKey": "CUSTOM_KEY", "baseUrl": "https://x.test"}}}"#,
        );
        std::env::remove_var("CUSTOM_KEY");
        let token = registry.get_current_provider_auth_source_token("custom").unwrap();
        assert_eq!(token.source, "models_json_key");
        assert!(registry.mark_provider_auth_source_stale(&token));
        assert!(!registry.has_configured_provider_request_auth("custom"));
        let status = registry.get_provider_auth_status("custom");
        assert_eq!(status.source.as_deref(), Some("stale"));
        assert_eq!(status.label.as_deref(), Some("expired"));
        assert!(registry.get_current_provider_auth_source_token("custom").is_none());

        registry.clear_provider_auth_stale("custom");
        assert!(registry.has_configured_provider_request_auth("custom"));
    }

    #[tokio::test]
    async fn get_api_key_and_headers_merges_headers_in_order() {
        let dir = tempfile::tempdir().unwrap();
        let mut registry = registry_with_config(
            dir.path(),
            r#"{"providers": {"custom": {"apiKey": "CUSTOM_KEY", "baseUrl": "https://x.test",
                "headers": {"X-Provider": "p"},
                "modelOverrides": {"m": {"headers": {"X-Model": "mo"}}}}}}"#,
        );
        std::env::set_var("CUSTOM_KEY", "secret");
        let model = Model {
            id: "m".to_string(),
            name: "M".to_string(),
            api: "openai-completions".to_string(),
            provider: "custom".to_string(),
            base_url: "https://x.test".to_string(),
            headers: Some(IndexMap::from([("X-Base".to_string(), "b".to_string())])),
            ..Default::default()
        };
        let auth = registry.get_api_key_and_headers(&model).await;
        std::env::remove_var("CUSTOM_KEY");
        assert!(auth.ok);
        assert_eq!(auth.api_key.as_deref(), Some("secret"));
        let headers = auth.headers.unwrap();
        assert_eq!(headers.get("X-Base").map(String::as_str), Some("b"));
        assert_eq!(headers.get("X-Provider").map(String::as_str), Some("p"));
        assert_eq!(headers.get("X-Model").map(String::as_str), Some("mo"));
        assert!(registry.is_using_oauth(&model) == false);
    }

    #[tokio::test]
    async fn auth_header_adds_the_bearer_token() {
        let dir = tempfile::tempdir().unwrap();
        let mut registry = registry_with_config(
            dir.path(),
            r#"{"providers": {"custom": {"apiKey": "CUSTOM_KEY", "baseUrl": "https://x.test", "authHeader": true}}}"#,
        );
        std::env::set_var("CUSTOM_KEY", "secret");
        let model = Model {
            id: "m".to_string(),
            provider: "custom".to_string(),
            api: "openai-completions".to_string(),
            ..Default::default()
        };
        let auth = registry.get_api_key_and_headers(&model).await;
        std::env::remove_var("CUSTOM_KEY");
        assert_eq!(
            auth.headers.unwrap().get("Authorization").map(String::as_str),
            Some("Bearer secret")
        );

        let mut registry = registry_with_config(
            dir.path(),
            r#"{"providers": {"custom": {"baseUrl": "https://x.test", "authHeader": true}}}"#,
        );
        let auth = registry.get_api_key_and_headers(&model).await;
        assert!(!auth.ok);
        assert_eq!(auth.error.as_deref(), Some("No API key found for \"custom\""));
    }

    #[tokio::test]
    async fn get_api_key_for_provider_prefers_auth_storage() {
        let dir = tempfile::tempdir().unwrap();
        let mut registry = registry_with_config(
            dir.path(),
            r#"{"providers": {"custom": {"apiKey": "CUSTOM_KEY", "baseUrl": "https://x.test"}}}"#,
        );
        std::env::set_var("CUSTOM_KEY", "secret");
        assert_eq!(
            registry.get_api_key_for_provider("custom").await.as_deref(),
            Some("secret")
        );
        assert_eq!(
            registry
                .last_provider_auth_source_tokens
                .get("custom")
                .map(|token| token.source.as_str()),
            Some("environment")
        );
        std::env::remove_var("CUSTOM_KEY");
        assert_eq!(
            registry.get_api_key_for_provider("custom").await.as_deref(),
            Some("CUSTOM_KEY")
        );
        assert!(registry.get_api_key_for_provider("unknown").await.is_none());
    }

    #[test]
    fn read_codex_account_id_decodes_the_jwt_claim() {
        use base64::Engine;
        let payload = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(r#"{"https://api.openai.com/auth":{"chatgpt_account_id":"acct_1"}}"#);
        let token = format!("header.{}.signature", payload);
        assert_eq!(read_openai_codex_account_id(&token).as_deref(), Some("acct_1"));
        assert_eq!(read_openai_codex_account_id("not-a-jwt"), None);
        let empty = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(r#"{"https://api.openai.com/auth":{"chatgpt_account_id":""}}"#);
        assert_eq!(read_openai_codex_account_id(&format!("h.{}.s", empty)), None);
    }

    #[test]
    fn codex_models_url_rewrites_the_path_and_adds_the_client_version() {
        assert_eq!(
            openai_codex_models_url("https://api.openai.com"),
            format!(
                "https://api.openai.com/codex/models?client_version={}",
                OPENAI_CODEX_CLIENT_VERSION
            )
        );
        assert_eq!(
            openai_codex_models_url("https://api.openai.com/codex/responses"),
            format!(
                "https://api.openai.com/codex/models?client_version={}",
                OPENAI_CODEX_CLIENT_VERSION
            )
        );
    }

    #[test]
    fn codex_model_ids_read_the_slug_list() {
        let ids = read_openai_codex_model_ids(&json!({"models": [{"slug": "gpt-5"}, {"slug": "gpt-5-mini"}, {}]}))
            .unwrap();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains("gpt-5"));
        assert!(read_openai_codex_model_ids(&json!({})).is_err());
    }

    #[test]
    fn private_authorization_fingerprint_is_stable_and_credential_scoped() {
        let first = private_prime_authorization_fingerprint("key", "team");
        assert_eq!(first, private_prime_authorization_fingerprint("key", "team"));
        assert_ne!(first, private_prime_authorization_fingerprint("key", "other"));
        assert_ne!(first, private_prime_authorization_fingerprint("other", "team"));
        assert_eq!(first.len(), 64);
    }

    #[test]
    fn private_authorization_cache_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("models.json");
        std::fs::write(&path, "{}").unwrap();
        let mut registry = ModelRegistry::create(in_memory_auth(), Some(path.to_string_lossy().to_string()));
        assert!(registry.read_private_prime_authorization_cache().is_none());

        registry.write_private_prime_authorization_cache(&PrivatePrimeAuthorizationCache {
            fingerprint: "abc".to_string(),
            models: get_private_prime_inference_models(),
            refreshed_at: 1234,
        });
        let cache = registry.read_private_prime_authorization_cache().unwrap();
        assert_eq!(cache.fingerprint, "abc");
        assert_eq!(cache.refreshed_at, 1234);
        assert!(!cache.models.is_empty());

        std::fs::write(
            registry.private_prime_authorization_cache_path().unwrap(),
            "{\"fingerprint\": 1, \"data\": [], \"refreshedAt\": 2}",
        )
        .unwrap();
        assert!(registry.read_private_prime_authorization_cache().is_none());
    }

    #[test]
    fn catalog_cache_path_sits_next_to_models_json() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("models.json");
        std::fs::write(&path, "{}").unwrap();
        let registry = ModelRegistry::create(in_memory_auth(), Some(path.to_string_lossy().to_string()));
        assert_eq!(
            registry.prime_inference_catalog_cache_path().as_deref(),
            Some(
                dir.path()
                    .join("prime-inference-models-cache.json")
                    .to_string_lossy()
                    .as_ref()
            )
        );
        assert_eq!(
            registry.private_prime_authorization_cache_path().as_deref(),
            Some(
                dir.path()
                    .join(PRIVATE_PRIME_AUTHORIZATION_CACHE_FILE)
                    .to_string_lossy()
                    .as_ref()
            )
        );
        let in_memory = ModelRegistry::in_memory(in_memory_auth());
        assert!(in_memory.prime_inference_catalog_cache_path().is_none());
        assert!(in_memory.private_prime_authorization_cache_path().is_none());
    }

    /// Restores the process environment when the test ends, including on panic.
    struct EnvRestore(Vec<(String, Option<String>)>);

    impl Drop for EnvRestore {
        fn drop(&mut self) {
            for (name, value) in self.0.drain(..) {
                match value {
                    Some(value) => std::env::set_var(&name, value),
                    None => std::env::remove_var(&name),
                }
            }
        }
    }

    /// Clear every ambient API-key/credential variable, so "no auth configured"
    /// means what the TypeScript test means by it.
    ///
    /// `hasConfiguredAuth` is `authStorage.hasAuth(provider)`
    /// (model-registry.ts:1220-1222), and `AuthStorage.hasAuth` accepts the
    /// environment candidate (`auth-storage.ts:757-759` ->
    /// `getEnvironmentAuthCandidate`, `auth-storage.ts:449-465` ->
    /// `getEnvApiKey`). The TypeScript test that pins this behaviour,
    /// `model-registry.test.ts:1189-1218`, therefore deletes
    /// `PRIME_API_KEY`/`OPENAI_API_KEY` (lines 1193-1194) before asserting that
    /// `configuredProviders` excludes "openai" - otherwise ambient credentials on
    /// the host legitimately make models available. This test asserts the same TS
    /// behaviour, so it clears the same ambient variables for the same reason.
    fn clear_ambient_auth_env() -> EnvRestore {
        let saved: Vec<(String, Option<String>)> =
            crate::core::auth_storage::ambient_auth_env_var_names()
                .into_iter()
                .map(|name| (name.to_string(), std::env::var(name).ok()))
                .collect();
        for (name, _) in &saved {
            std::env::remove_var(name);
        }
        EnvRestore(saved)
    }

    #[tokio::test]
    async fn refresh_and_available_models_use_built_in_catalog() {
        let _env = clear_ambient_auth_env();
        let mut registry = ModelRegistry::in_memory(in_memory_auth());
        registry.refresh();
        assert!(registry.get_error().is_none());
        let all = registry.get_all();
        assert!(!all.is_empty());
        assert!(all.iter().any(|model| model.provider == "anthropic"));

        let available = registry.refresh_available_models().await;
        // No auth in the in-memory storage and no ambient credentials, so nothing
        // is available (TS model-registry.test.ts:1198-1200).
        assert!(available.is_empty());
        assert!(!registry.has_configured_auth(&all[0]));

        let snapshot = registry.refresh_model_catalog().await;
        assert!(snapshot.configured_providers.is_empty());
        assert!(!snapshot.models.is_empty());
        assert!(!registry.can_use_model(&all[0], false).await);
        assert!(registry.can_use_model(&all[0], true).await);
    }

    #[test]
    fn offline_mode_reads_the_typescript_env_flag() {
        std::env::remove_var("PI_OFFLINE");
        assert!(!is_offline_mode_enabled());
        std::env::set_var("PI_OFFLINE", "1");
        assert!(is_offline_mode_enabled());
        std::env::set_var("PI_OFFLINE", "true");
        assert!(is_offline_mode_enabled());
        std::env::set_var("PI_OFFLINE", "no");
        assert!(!is_offline_mode_enabled());
        std::env::remove_var("PI_OFFLINE");
    }

    #[test]
    fn merged_custom_models_replace_built_ins_by_provider_and_id() {
        let registry = ModelRegistry::in_memory(in_memory_auth());
        let built_in = Model {
            id: "m".to_string(),
            provider: "p".to_string(),
            ..Default::default()
        };
        let custom = Model {
            id: "m".to_string(),
            provider: "p".to_string(),
            name: "custom".to_string(),
            ..Default::default()
        };
        let merged = registry.merge_custom_models(vec![built_in.clone()], vec![custom.clone()]);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].name, "custom");

        let extra = Model {
            id: "n".to_string(),
            provider: "p".to_string(),
            ..Default::default()
        };
        let merged = registry.merge_custom_models(vec![built_in], vec![extra]);
        assert_eq!(merged.len(), 2);
    }

    #[tokio::test]
    async fn executable_models_filter_openai_codex_by_catalog() {
        use base64::Engine;
        let payload = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(r#"{"https://api.openai.com/auth":{"chatgpt_account_id":"acct_1"}}"#);
        let token = format!("header.{}.signature", payload);
        std::env::set_var("OPENAI_CODEX_TOKEN", &token);

        let mut storage = in_memory_auth();
        storage.set_runtime_api_key("openai-codex", &token);
        let mut registry = ModelRegistry::in_memory(storage);
        registry.set_fetch_fn(Some(Arc::new(|request: HttpRequest| {
            assert!(request.url.contains("client_version="));
            Box::pin(async move {
                Ok(crate::core::prime_inference_auth::HttpResponse {
                    status: 200,
                    status_text: "OK".to_string(),
                    headers: Vec::new(),
                    text: serde_json::json!({"models": [{"slug": "gpt-5-codex"}]}).to_string(),
                })
            }) as pi_ai::types::BoxFuture<Result<crate::core::prime_inference_auth::HttpResponse, String>>
        })));

        let models = registry.get_executable_models().await;
        std::env::remove_var("OPENAI_CODEX_TOKEN");
        let codex: Vec<&Model> = models.iter().filter(|model| model.provider == "openai-codex").collect();
        for model in &codex {
            assert_eq!(model.id, "gpt-5-codex");
        }
        // The catalog response is cached for the same credentials.
        assert!(registry.openai_codex_models_cache.is_some());
    }

    #[tokio::test]
    async fn subscription_models_codex_discovery_preserves_exact_ids_and_fails_closed() {
        use base64::Engine;
        use std::sync::atomic::{AtomicUsize, Ordering};
        let payload = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(
            r#"{"https://api.openai.com/auth":{"chatgpt_account_id":"fixture-account"}}"#,
        );
        let token = format!("fixture.{}.signature", payload);
        for (status, body, expected) in [
            (200, json!({"models": [{"slug": "gpt-6-sol"}, {"slug": "gpt-6-luna"}]}),
                vec!["gpt-6-sol", "gpt-6-luna"]),
            (200, json!({"models": [{"slug": "gpt-6-sol"}]}), vec!["gpt-6-sol"]),
            (200, json!({"models": []}), vec![]),
            (200, json!({"unexpected": []}), vec![]),
            (503, json!({"error": "fixture catalog unavailable"}), vec![]),
        ] {
            let mut storage = in_memory_auth();
            storage.set_runtime_api_key("openai-codex", &token);
            storage.set_runtime_api_key("anthropic", "synthetic-other");
            let mut registry = ModelRegistry::in_memory(storage);
            // Catalog reloads discard ad hoc pushes to registry.models. Use a
            // persisted built-in provider as the unrelated-provider control.
            let unrelated_before: Vec<Model> = registry.get_available().into_iter()
                .filter(|model| model.provider == "anthropic").collect();
            assert!(!unrelated_before.is_empty());
            let calls = Arc::new(AtomicUsize::new(0));
            let observed_calls = Arc::clone(&calls);
            let expected_token = token.clone();
            registry.set_fetch_fn(Some(Arc::new(move |request: HttpRequest| {
                observed_calls.fetch_add(1, Ordering::SeqCst);
                assert_eq!(request.method, "GET");
                assert_eq!(request.url,
                    "https://chatgpt.com/backend-api/codex/models?client_version=0.156.1");
                assert!(request.body.is_none());
                assert!(request.headers.contains(&(
                    "Authorization".to_string(), format!("Bearer {}", expected_token),
                )));
                assert!(request.headers.contains(&(
                    "chatgpt-account-id".to_string(), "fixture-account".to_string(),
                )));
                let text = body.to_string();
                Box::pin(async move {
                    Ok(crate::core::prime_inference_auth::HttpResponse {
                        status, status_text: "Fixture".to_string(), headers: Vec::new(), text,
                    })
                }) as pi_ai::types::BoxFuture<Result<crate::core::prime_inference_auth::HttpResponse, String>>
            })));
            let available = registry.get_executable_models().await;
            let actual: Vec<&str> = available.iter()
                .filter(|model| model.provider == "openai-codex")
                .map(|model| model.id.as_str()).collect();
            assert_eq!(actual, expected);
            let unrelated_after: Vec<Model> = available.iter()
                .filter(|model| model.provider == "anthropic").cloned().collect();
            assert_eq!(unrelated_after, unrelated_before);
            assert_eq!(calls.load(Ordering::SeqCst), 1);
            if registry.openai_codex_models_cache.is_some() {
                assert_eq!(registry.get_executable_models().await, available);
                assert_eq!(calls.load(Ordering::SeqCst), 1, "cached discovery must not fetch again");
            }
        }
    }

    #[tokio::test]
    async fn executable_models_drop_codex_without_an_account_id() {
        let mut storage = in_memory_auth();
        storage.set_runtime_api_key("openai-codex", "not-a-jwt");
        let mut registry = ModelRegistry::in_memory(storage);
        let models = registry.get_executable_models().await;
        assert!(models.iter().all(|model| model.provider != "openai-codex"));
    }

    #[tokio::test]
    async fn can_use_model_rejects_private_models_without_authorization() {
        let mut registry = ModelRegistry::in_memory(in_memory_auth());
        let private_model = get_private_prime_inference_models().into_iter().next().unwrap();
        assert!(!registry.can_use_model(&private_model, true).await);

        registry
            .explicit_private_prime_inference_model_ids
            .insert(private_model.id.clone());
        assert!(registry.can_use_model(&private_model, true).await);
        // Without auth the model still fails the full check.
        assert!(!registry.can_use_model(&private_model, false).await);
    }

    #[test]
    fn get_available_hides_unauthorized_private_models() {
        let mut storage = in_memory_auth();
        storage.set_runtime_api_key(PRIME_INFERENCE_PROVIDER_ID, "prime-key");
        let mut registry = ModelRegistry::in_memory(storage);
        let private_model = get_private_prime_inference_models().into_iter().next().unwrap();
        registry.models.push(private_model.clone());
        let available = registry.get_available();
        assert!(!available.iter().any(|model| model.id == private_model.id));

        registry
            .explicit_private_prime_inference_model_ids
            .insert(private_model.id.clone());
        let available = registry.get_available();
        assert!(available.iter().any(|model| model.id == private_model.id));
    }

    #[test]
    fn oauth_provider_registration_is_wired_to_the_shared_registry() {
        let mut registry = ModelRegistry::in_memory(in_memory_auth());
        crate::core::auth_storage::reset_oauth_providers();
        registry
            .register_provider(
                "custom-oauth",
                ProviderConfigInput {
                    name: Some("Custom".to_string()),
                    oauth: Some(ProviderOAuthInput {
                        name: "Custom OAuth".to_string(),
                        login: Arc::new(|_callbacks| {
                            Box::pin(async { Ok(pi_ai::utils::oauth::types::OAuthCredentials::default()) })
                                as pi_ai::types::BoxFuture<
                                    Result<pi_ai::utils::oauth::types::OAuthCredentials, String>,
                                >
                        }),
                        uses_callback_server: Some(false),
                        refresh_token: Arc::new(|credentials| {
                            Box::pin(async move { Ok(credentials) })
                                as pi_ai::types::BoxFuture<
                                    Result<pi_ai::utils::oauth::types::OAuthCredentials, String>,
                                >
                        }),
                        get_api_key: Arc::new(|credentials| credentials.access.clone()),
                        modify_models: None,
                    }),
                    ..Default::default()
                },
            )
            .unwrap();
        let provider = crate::core::auth_storage::get_oauth_provider("custom-oauth").unwrap();
        assert_eq!(provider.name, "Custom OAuth");
        assert_eq!(registry.get_provider_display_name("custom-oauth"), "Custom");

        registry.unregister_provider("custom-oauth");
        assert!(crate::core::auth_storage::get_oauth_provider("custom-oauth").is_none());
    }

    #[tokio::test]
    async fn private_prime_authorization_refresh_uses_injected_fetch() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("models.json");
        std::fs::write(&path, "{}").unwrap();
        let mut storage = in_memory_auth();
        storage.set(
            PRIME_INFERENCE_PROVIDER_ID,
            AuthCredential::ApiKey {
                key: "prime-key".to_string(),
                prime_team: Some(Some(crate::core::auth_storage::PrimeTeamCredential {
                    team_id: "team-1".to_string(),
                    name: "Team".to_string(),
                    slug: None,
                    role: None,
                    created_at: None,
                })),
            },
        );
        let mut registry = ModelRegistry::create(storage, Some(path.to_string_lossy().to_string()));
        registry.set_fetch_fn(Some(Arc::new(|request: HttpRequest| {
            assert_eq!(request.url, format!("{}/models", crate::core::prime_inference_models::PRIME_INFERENCE_BASE_URL));
            assert!(request.headers.iter().any(|(key, value)| key == "X-Prime-Team-ID" && value == "team-1"));
            Box::pin(async move {
                Ok(crate::core::prime_inference_auth::HttpResponse {
                    status: 200,
                    status_text: "OK".to_string(),
                    headers: Vec::new(),
                    text: serde_json::json!({"data": [{"id": "internal/glm-5.2-fast", "pricing": {"input_usd_per_mtok": 1.0, "output_usd_per_mtok": 2.0}}]}).to_string(),
                })
            }) as pi_ai::types::BoxFuture<Result<crate::core::prime_inference_auth::HttpResponse, String>>
        })));

        let previous_ids = registry.authorized_private_prime_inference_model_ids.clone();
        let previous_team = registry.authorized_private_prime_inference_team_id.clone();
        let previous_models = registry.authorized_private_prime_inference_models.clone();
        registry
            .refresh_private_prime_inference_authorization_with_offline(previous_ids, previous_team, previous_models, false)
            .await;
        assert_eq!(
            registry.authorized_private_prime_inference_team_id.as_deref(),
            Some("team-1")
        );
        assert!(registry
            .authorized_private_prime_inference_model_ids
            .contains("internal/glm-5.2-fast"));
        // The fetched authorization is cached for the next start.
        let cache = registry.read_private_prime_authorization_cache().unwrap();
        assert!(cache.models.iter().any(|model| model.id == "internal/glm-5.2-fast"));
    }

    #[tokio::test]
    async fn private_prime_authorization_without_credentials_clears_state() {
        let mut registry = ModelRegistry::in_memory(in_memory_auth());
        registry.authorized_private_prime_inference_model_ids.insert("internal/x".to_string());
        registry.authorized_private_prime_inference_team_id = Some("team".to_string());
        registry
            .refresh_private_prime_inference_authorization(
                HashSet::from(["internal/x".to_string()]),
                Some("team".to_string()),
                Vec::new(),
            )
            .await;
        assert!(registry.authorized_private_prime_inference_model_ids.is_empty());
        assert!(registry.authorized_private_prime_inference_team_id.is_none());
    }

    #[test]
    fn provider_oauth_input_exposes_an_id_bound_interface() {
        let input = ProviderOAuthInput {
            name: "Name".to_string(),
            login: Arc::new(|_callbacks| {
                Box::pin(async { Ok(pi_ai::utils::oauth::types::OAuthCredentials::default()) })
                    as pi_ai::types::BoxFuture<Result<pi_ai::utils::oauth::types::OAuthCredentials, String>>
            }),
            uses_callback_server: None,
            refresh_token: Arc::new(|credentials| {
                Box::pin(async move { Ok(credentials) })
                    as pi_ai::types::BoxFuture<Result<pi_ai::utils::oauth::types::OAuthCredentials, String>>
            }),
            get_api_key: Arc::new(|credentials| credentials.access.clone()),
            modify_models: None,
        };
        let interface = input.to_oauth_provider_interface("custom");
        assert_eq!(interface.id, "custom");
        assert_eq!(interface.name, "Name");
        assert_eq!((interface.get_api_key)(&Default::default()), "");
    }

    #[test]
    fn provider_config_input_debug_does_not_leak_the_api_key() {
        let config = ProviderConfigInput {
            name: Some("Custom".to_string()),
            api_key: Some("super-secret".to_string()),
            ..Default::default()
        };
        let text = format!("{:?}", config);
        assert!(!text.contains("super-secret"), "{}", text);
        assert!(text.contains("Custom"));
    }
}
