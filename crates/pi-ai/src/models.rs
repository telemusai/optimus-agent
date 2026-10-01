//! Port of packages/ai/src/models.ts

use std::sync::OnceLock;

use indexmap::IndexMap;

use crate::models_generated::models;
use crate::types::{Model, Usage};

/// `const modelRegistry: Map<string, Map<string, Model<Api>>>` - built once from MODELS.
fn model_registry() -> &'static IndexMap<String, IndexMap<String, Model>> {
    static REGISTRY: OnceLock<IndexMap<String, IndexMap<String, Model>>> = OnceLock::new();
    REGISTRY.get_or_init(|| models().clone())
}

/// `getModel(provider, modelId)`.
///
/// The TypeScript returns `undefined` when the provider or the model id is
/// unknown; the Rust port keeps that with `Option`.
pub fn get_model(provider: &str, model_id: &str) -> Option<&'static Model> {
    model_registry()
        .get(provider)
        .and_then(|provider_models| provider_models.get(model_id))
}

/// `getProviders()` - declaration order of MODELS.
pub fn get_providers() -> Vec<&'static String> {
    model_registry().keys().collect()
}

/// `getModels(provider)` - `[]` when the provider is unknown.
pub fn get_models(provider: &str) -> Vec<&'static Model> {
    match model_registry().get(provider) {
        Some(provider_models) => provider_models.values().collect(),
        None => Vec::new(),
    }
}

pub const GPT_5_4: &str = "gpt-5.4";
pub const GPT_5_5: &str = "gpt-5.5";
pub const GPT_5_6: &str = "gpt-5.6";
pub const GPT_5_6_PREFIX: &str = "gpt-5.6-";
pub const GPT_6_ASTRA: &str = "gpt-6-astra";
pub const GPT_6_SOL: &str = "gpt-6-sol";
pub const GPT_6_1_SOL: &str = "gpt-6.1-sol";

/// `supportsFastMode(model)`.
pub fn supports_fast_mode(model: &Model) -> bool {
    let eligible_id = model.id == GPT_5_4
        || model.id == GPT_5_5
        || model.id == GPT_5_6
        || model.id.starts_with(GPT_5_6_PREFIX)
        || model.id == GPT_6_ASTRA;
    let codex_sol_id = model.id == GPT_6_SOL || model.id == GPT_6_1_SOL;
    let codex_route = model.provider == "openai-codex" && model.api == "openai-codex-responses";
    let openai_route = model.provider == "openai" && model.api == "openai-responses";
    (codex_route && (eligible_id || codex_sol_id)) || (openai_route && eligible_id)
}

/// Astra reserves part of its context window for output.
pub const ASTRA_INPUT_LIMIT: f64 = 922_000.0;

/// Input limits can be smaller than the total input + output context window.
pub fn get_model_input_limit(model: &Model) -> f64 {
    let astra_limit = if model.provider == "openai"
        && model.api == "openai-responses"
        && model.id == GPT_6_ASTRA
    {
        ASTRA_INPUT_LIMIT
    } else {
        model.context_window
    };
    let configured = model.max_input_tokens;
    let configured = match configured {
        Some(value) if value.is_finite() && value > 0.0 => value,
        _ => model.context_window,
    };
    model.context_window.min(astra_limit).min(configured)
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct CostOverrides {
    pub cache_write: Option<f64>,
}

/// `calculateCost(model, usage, overrides?)` - mutates and returns `usage.cost`.
pub fn calculate_cost(model: &Model, usage: &mut Usage, overrides: Option<&CostOverrides>) -> () {
    usage.cost.input = (model.cost.input / 1_000_000.0) * usage.input;
    usage.cost.output = (model.cost.output / 1_000_000.0) * usage.output;
    usage.cost.cache_read = (model.cost.cache_read / 1_000_000.0) * usage.cache_read;
    let cache_write_cost = overrides
        .and_then(|overrides| overrides.cache_write)
        .unwrap_or(model.cost.cache_write);
    usage.cost.cache_write = (cache_write_cost / 1_000_000.0) * usage.cache_write;
    usage.cost.total =
        usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write;
}

pub const EXTENDED_THINKING_LEVELS: [&str; 7] = ["off", "minimal", "low", "medium", "high", "xhigh", "max"];

/// `getSupportedThinkingLevels(model)`.
pub fn get_supported_thinking_levels(model: &Model) -> Vec<String> {
    if !model.reasoning {
        return vec!["off".to_string()];
    }

    EXTENDED_THINKING_LEVELS
        .iter()
        .filter(|level| {
            let mapped = model.thinking_level_map_get(level);
            if matches!(mapped, Some(None)) {
                return false;
            }
            if **level == "xhigh" || **level == "max" {
                return mapped.is_some();
            }
            true
        })
        .map(|level| (*level).to_string())
        .collect()
}

/// `clampThinkingLevel(model, level)`.
pub fn clamp_thinking_level(model: &Model, level: &str) -> String {
    let available_levels = get_supported_thinking_levels(model);
    if available_levels.iter().any(|available| available == level) {
        return level.to_string();
    }

    let requested_index = EXTENDED_THINKING_LEVELS
        .iter()
        .position(|candidate| *candidate == level);
    let Some(requested_index) = requested_index else {
        return available_levels
            .first()
            .cloned()
            .unwrap_or_else(|| "off".to_string());
    };

    for candidate in EXTENDED_THINKING_LEVELS.iter().skip(requested_index) {
        if available_levels.iter().any(|available| available == candidate) {
            return (*candidate).to_string();
        }
    }
    for candidate in EXTENDED_THINKING_LEVELS.iter().take(requested_index).rev() {
        if available_levels.iter().any(|available| available == candidate) {
            return (*candidate).to_string();
        }
    }
    available_levels
        .first()
        .cloned()
        .unwrap_or_else(|| "off".to_string())
}

/// `modelsAreEqual(a, b)` - `null`/`undefined` operands are never equal.
pub fn models_are_equal(a: Option<&Model>, b: Option<&Model>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => a.id == b.id && a.provider == b.provider,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{InputModality, ModelCost, UsageCost};

    #[test]
    fn registry_matches_generated_catalog() {
        assert_eq!(get_providers().len(), crate::models_generated::models().len());
        assert!(get_model("anthropic", "claude-sonnet-4-5").is_some());
        assert!(get_model("anthropic", "does-not-exist").is_none());
        assert!(get_model("does-not-exist", "claude-sonnet-4-5").is_none());
        assert!(get_models("does-not-exist").is_empty());
        assert_eq!(
            get_models("anthropic").len(),
            crate::models_generated::models_for_provider("anthropic").unwrap().len()
        );
    }

    #[test]
    fn calculate_cost_matches_typescript_formula() {
        let mut model = Model::default();
        model.cost = ModelCost {
            input: 3.0,
            output: 15.0,
            cache_read: 0.3,
            cache_write: 3.75,
        };
        let mut usage = Usage {
            input: 1_000_000.0,
            output: 2_000_000.0,
            cache_read: 1_000_000.0,
            cache_write: 0.0,
            total_tokens: 0.0,
            cost: UsageCost::default(),
        };
        calculate_cost(&model, &mut usage, None);
        assert_eq!(usage.cost.input, 3.0);
        assert_eq!(usage.cost.output, 30.0);
        assert_eq!(usage.cost.cache_read, 0.3);
        assert_eq!(usage.cost.cache_write, 0.0);
        assert_eq!(usage.cost.total, 33.3);
    }

    #[test]
    fn calculate_cost_honours_cache_write_override() {
        let mut model = Model::default();
        model.cost.cache_write = 3.75;
        let mut usage = Usage {
            cache_write: 1_000_000.0,
            ..Default::default()
        };
        calculate_cost(
            &model,
            &mut usage,
            Some(&CostOverrides {
                cache_write: Some(7.5),
            }),
        );
        assert_eq!(usage.cost.cache_write, 7.5);
    }

    #[test]
    fn input_limit_uses_astra_ceiling_and_max_input_tokens() {
        let mut model = Model::default();
        model.context_window = 1_000_000.0;
        model.max_tokens = 100.0;
        assert_eq!(get_model_input_limit(&model), 1_000_000.0);

        model.max_input_tokens = Some(500.0);
        assert_eq!(get_model_input_limit(&model), 500.0);

        model.max_input_tokens = Some(-1.0);
        assert_eq!(get_model_input_limit(&model), 1_000_000.0);

        model.provider = "openai".to_string();
        model.api = "openai-responses".to_string();
        model.id = GPT_6_ASTRA.to_string();
        model.max_input_tokens = None;
        assert_eq!(get_model_input_limit(&model), ASTRA_INPUT_LIMIT);
    }

    #[test]
    fn supported_thinking_levels_follow_map() {
        let mut model = Model::default();
        model.reasoning = false;
        assert_eq!(get_supported_thinking_levels(&model), vec!["off".to_string()]);

        model.reasoning = true;
        assert_eq!(
            get_supported_thinking_levels(&model),
            vec!["off", "minimal", "low", "medium", "high"]
        );

        model.thinking_level_map = Some(crate::types::ThinkingLevelMap::from([
            ("xhigh".to_string(), Some("xhigh".to_string())),
            ("max".to_string(), Some("max".to_string())),
        ]));
        assert_eq!(get_supported_thinking_levels(&model), EXTENDED_THINKING_LEVELS);

        let mut map = crate::types::ThinkingLevelMap::new();
        map.insert("xhigh".to_string(), None);
        map.insert("max".to_string(), None);
        map.insert("high".to_string(), None);
        model.thinking_level_map = Some(map);
        assert_eq!(
            get_supported_thinking_levels(&model),
            vec!["off", "minimal", "low", "medium"]
        );
    }

    #[test]
    fn clamp_thinking_level_walks_up_then_down() {
        let mut model = Model::default();
        model.reasoning = true;
        let mut map = crate::types::ThinkingLevelMap::new();
        map.insert("high".to_string(), None);
        map.insert("xhigh".to_string(), None);
        map.insert("max".to_string(), None);
        model.thinking_level_map = Some(map);

        assert_eq!(clamp_thinking_level(&model, "low"), "low");
        assert_eq!(clamp_thinking_level(&model, "high"), "medium");
        assert_eq!(clamp_thinking_level(&model, "bogus"), "off");
        assert_eq!(clamp_thinking_level(&model, "max"), "medium");

        let mut non_reasoning = Model::default();
        non_reasoning.reasoning = false;
        assert_eq!(clamp_thinking_level(&non_reasoning, "high"), "off");
    }

    #[test]
    fn fast_mode_requires_eligible_id_and_route() {
        for id in [
            GPT_5_4, GPT_5_5, GPT_5_6, "gpt-5.6-astra", "gpt-5.6-anything",
            GPT_6_ASTRA, GPT_6_SOL, GPT_6_1_SOL,
        ] {
            for provider in [
                "openai", "openai-codex", "github-copilot", "openrouter", "custom",
                "OpenAI-Codex", "openai-codex/", "",
            ] {
                for api in [
                    "openai-responses", "openai-codex-responses", "openai-completions",
                    "anthropic-messages", "openai-codex-responses-v2", "OpenAI-Codex-Responses", "",
                ] {
                    let model = Model::new(id, id, api, provider, "https://fixture.invalid");
                    let codex = provider == "openai-codex" && api == "openai-codex-responses";
                    let openai = provider == "openai" && api == "openai-responses"
                        && id != GPT_6_SOL && id != GPT_6_1_SOL;
                    assert_eq!(supports_fast_mode(&model), codex || openai, "{provider}/{id} via {api}");
                }
            }
        }
    }

    #[test]
    fn fast_mode_rejects_unreviewed_model_ids() {
        for id in [
            "gpt-4o", "gpt-5.60", "gpt-6", "gpt-6.1", "gpt-6-luna", "gpt-6.1-luna",
            "gpt-6-terra", "gpt-6.2-sol", "gpt-6.1-astra", "gpt-6-astra-preview",
            "gpt-6-sol-preview", "gpt-6.1-sol-preview", "gpt-6-sol-2026-09-30",
            "gpt-6.1-sol-2026-09-30", "gpt-6-solx", "gpt-6.1-solx", "GPT-6-SOL",
            " gpt-6-sol", "gpt-6-sol ", "openai-codex/gpt-6-sol", "",
        ] {
            for (provider, api) in [
                ("openai", "openai-responses"),
                ("openai-codex", "openai-codex-responses"),
            ] {
                let model = Model::new(id, id, api, provider, "https://fixture.invalid");
                assert!(!supports_fast_mode(&model), "{provider}/{id} via {api}");
            }
        }
    }

    #[test]
    fn models_are_equal_requires_both_operands() {
        let mut a = Model::default();
        a.id = "m".to_string();
        a.provider = "p".to_string();
        let mut b = a.clone();
        assert!(models_are_equal(Some(&a), Some(&b)));
        assert!(!models_are_equal(None, Some(&b)));
        assert!(!models_are_equal(Some(&a), None));
        b.provider = "other".to_string();
        assert!(!models_are_equal(Some(&a), Some(&b)));
    }

    #[test]
    fn modality_serde_names() {
        assert_eq!(
            serde_json::to_value(InputModality::Text).unwrap(),
            serde_json::json!("text")
        );
    }
}
