//! Port of packages/coding-agent/src/core/model-resolver.ts
//!
//! Model resolution, scoping, and initial selection.

use std::collections::HashMap;

use globset::GlobBuilder;
use indexmap::IndexMap;
use pi_ai::types::Model;
use serde_json::json;

use crate::core::model_registry::ModelRegistry;
use crate::core::prime_inference_models::is_private_prime_inference_model;

pub const PRIME_INFERENCE_DEFAULT_MODEL_ID: &str = "z-ai/glm-5.3";

/// `THINKING_LEVELS` from core/thinking-levels.ts (owned by another slice).
pub const THINKING_LEVELS: [&str; 7] = ["off", "minimal", "low", "medium", "high", "xhigh", "max"];
/// `DEFAULT_THINKING_LEVEL` from core/defaults.ts (owned by another slice).
pub const DEFAULT_THINKING_LEVEL: &str = "medium";

/// `isValidThinkingLevel(level)` from cli/args.ts.
pub fn is_valid_thinking_level(level: &str) -> bool {
    THINKING_LEVELS.contains(&level)
}

/// Installed CLI name used in model-selection guidance.
pub const APP_NAME: &str = "optimus-agent";

/// Default model IDs for each known provider, in declaration order.
pub fn default_model_per_provider() -> &'static IndexMap<&'static str, &'static str> {
    static DEFAULTS: once_cell::sync::Lazy<IndexMap<&'static str, &'static str>> =
        once_cell::sync::Lazy::new(|| {
            let mut map = IndexMap::new();
            for (provider, model_id) in [
                ("amazon-bedrock", "us.anthropic.claude-opus-4-6-v1"),
                ("anthropic", "claude-opus-4-7"),
                ("openai", "gpt-5.4"),
                ("azure-openai-responses", "gpt-5.4"),
                ("openai-codex", "gpt-5.5"),
                ("prime-inference", PRIME_INFERENCE_DEFAULT_MODEL_ID),
                ("deepseek", "deepseek-v4-pro"),
                ("google", "gemini-3.1-pro-preview"),
                ("google-vertex", "gemini-3.1-pro-preview"),
                ("github-copilot", "gpt-5.4"),
                ("openrouter", "moonshotai/kimi-k2.6"),
                ("vercel-ai-gateway", "zai/glm-5.1"),
                ("xai", "grok-4.20-0309-reasoning"),
                ("groq", "openai/gpt-oss-120b"),
                ("cerebras", "gpt-oss-120b"),
                ("zai", "glm-5.3"),
                ("mistral", "devstral-medium-latest"),
                ("minimax", "MiniMax-M2.7"),
                ("minimax-cn", "MiniMax-M2.7"),
                ("moonshotai", "kimi-k2.6"),
                ("moonshotai-cn", "kimi-k2.6"),
                ("huggingface", "moonshotai/Kimi-K2.6"),
                ("fireworks", "accounts/fireworks/models/kimi-k2p6"),
                ("opencode", "kimi-k2.6"),
                ("opencode-go", "kimi-k2.6"),
                ("kimi-coding", "kimi-for-coding"),
                ("cloudflare-workers-ai", "@cf/moonshotai/kimi-k2.6"),
                ("cloudflare-ai-gateway", "claude-sonnet-4.5"),
                ("xiaomi", "mimo-v2.5-pro"),
                ("xiaomi-token-plan-cn", "mimo-v2.5-pro"),
                ("xiaomi-token-plan-ams", "mimo-v2.5-pro"),
                ("xiaomi-token-plan-sgp", "mimo-v2.5-pro"),
            ] {
                map.insert(provider, model_id);
            }
            map
        });
    &DEFAULTS
}

/// `modelsAreEqual(a, b)` from pi-ai/models.ts.
pub fn models_are_equal(a: Option<&Model>, b: Option<&Model>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => a.id == b.id && a.provider == b.provider,
        _ => false,
    }
}

/// `interface ScopedModel`.
#[derive(Debug, Clone, PartialEq)]
pub struct ScopedModel {
    pub model: Model,
    /// Thinking level if explicitly specified in pattern (e.g. "model:high"),
    /// `None` otherwise.
    pub thinking_level: Option<String>,
}

/// Helper to check if a model ID looks like an alias (no date suffix).
/// Dates are typically in format: -20241022 or -20250929.
fn is_alias(id: &str) -> bool {
    if id.ends_with("-latest") {
        return true;
    }
    !ends_with_date_suffix(id)
}

fn ends_with_date_suffix(id: &str) -> bool {
    let bytes = id.as_bytes();
    if bytes.len() < 9 {
        return false;
    }
    let tail = &bytes[bytes.len() - 9..];
    tail[0] == b'-' && tail[1..].iter().all(|byte| byte.is_ascii_digit())
}

/// Find an exact model reference match.
/// Supports either a bare model id or a canonical provider/modelId reference.
/// When matching by bare id, ambiguous matches across providers are rejected.
pub fn find_exact_model_reference_match(
    model_reference: &str,
    available_models: &[Model],
) -> Option<Model> {
    let trimmed_reference = model_reference.trim();
    if trimmed_reference.is_empty() {
        return None;
    }

    let normalized_reference = trimmed_reference.to_lowercase();

    let canonical_matches: Vec<&Model> = available_models
        .iter()
        .filter(|model| format!("{}/{}", model.provider, model.id).to_lowercase() == normalized_reference)
        .collect();
    if canonical_matches.len() == 1 {
        return Some(canonical_matches[0].clone());
    }
    if canonical_matches.len() > 1 {
        return None;
    }

    if let Some(slash_index) = trimmed_reference.find('/') {
        let provider = trimmed_reference[..slash_index].trim();
        let model_id = trimmed_reference[slash_index + 1..].trim();
        if !provider.is_empty() && !model_id.is_empty() {
            let provider_matches: Vec<&Model> = available_models
                .iter()
                .filter(|model| {
                    model.provider.to_lowercase() == provider.to_lowercase()
                        && model.id.to_lowercase() == model_id.to_lowercase()
                })
                .collect();
            if provider_matches.len() == 1 {
                return Some(provider_matches[0].clone());
            }
            if provider_matches.len() > 1 {
                return None;
            }
        }
    }

    let id_matches: Vec<&Model> = available_models
        .iter()
        .filter(|model| model.id.to_lowercase() == normalized_reference)
        .collect();
    if id_matches.len() == 1 {
        Some(id_matches[0].clone())
    } else {
        None
    }
}

/// Try to match a pattern to a model from the available models list.
fn try_match_model(model_pattern: &str, available_models: &[Model]) -> Option<Model> {
    if let Some(exact_match) = find_exact_model_reference_match(model_pattern, available_models) {
        return Some(exact_match);
    }
    let pattern = model_pattern.to_lowercase();
    let mut matches: Vec<&Model> = available_models
        .iter()
        .filter(|model| {
            model.id.to_lowercase().contains(&pattern) || model.name.to_lowercase().contains(&pattern)
        })
        .collect();

    if matches.is_empty() {
        return None;
    }
    let mut aliases: Vec<&Model> = matches.iter().copied().filter(|m| is_alias(&m.id)).collect();
    let mut dated_versions: Vec<&Model> = matches.drain(..).filter(|m| !is_alias(&m.id)).collect();

    if !aliases.is_empty() {
        aliases.sort_by(|a, b| b.id.cmp(&a.id));
        Some(aliases[0].clone())
    } else {
        dated_versions.sort_by(|a, b| b.id.cmp(&a.id));
        dated_versions.first().map(|model| (*model).clone())
    }
}

#[derive(Debug, Clone, PartialEq)]
struct ParsedModelResult {
    model: Option<Model>,
    /// Thinking level if explicitly specified in pattern, `None` otherwise.
    thinking_level: Option<String>,
    warning: Option<String>,
}

impl ParsedModelResult {
    fn empty() -> Self {
        Self {
            model: None,
            thinking_level: None,
            warning: None,
        }
    }
}

fn build_fallback_model(provider: &str, model_id: &str, available_models: &[Model]) -> Option<Model> {
    let provider_models: Vec<&Model> = available_models
        .iter()
        .filter(|model| model.provider == provider)
        .collect();
    if provider_models.is_empty() {
        return None;
    }

    let default_id = default_model_per_provider().get(provider).copied();
    let base_model = match default_id {
        Some(default_id) => provider_models
            .iter()
            .find(|model| model.id == default_id)
            .copied()
            .unwrap_or(provider_models[0]),
        None => provider_models[0],
    };

    let mut fallback = base_model.clone();
    fallback.id = model_id.to_string();
    fallback.name = model_id.to_string();
    Some(fallback)
}

fn find_preferred_default_model(available_models: &[Model]) -> Option<Model> {
    if let Some(prime_inference_default) = available_models.iter().find(|model| {
        model.provider == "prime-inference" && model.id == PRIME_INFERENCE_DEFAULT_MODEL_ID
    }) {
        return Some(prime_inference_default.clone());
    }

    for (provider, default_id) in default_model_per_provider() {
        if let Some(found) = available_models
            .iter()
            .find(|model| model.provider == *provider && model.id == *default_id)
        {
            return Some(found.clone());
        }
    }

    None
}

/// Parse a pattern to extract model and thinking level.
/// Handles models with colons in their IDs (e.g., OpenRouter's :exacto suffix).
fn parse_model_pattern(
    pattern: &str,
    available_models: &[Model],
    allow_invalid_thinking_level_fallback: bool,
) -> ParsedModelResult {
    if let Some(exact_match) = try_match_model(pattern, available_models) {
        return ParsedModelResult {
            model: Some(exact_match),
            thinking_level: None,
            warning: None,
        };
    }
    let Some(last_colon_index) = pattern.rfind(':') else {
        return ParsedModelResult::empty();
    };

    let prefix = &pattern[..last_colon_index];
    let suffix = &pattern[last_colon_index + 1..];

    if is_valid_thinking_level(suffix) {
        let result = parse_model_pattern(prefix, available_models, allow_invalid_thinking_level_fallback);
        if result.model.is_some() {
            return ParsedModelResult {
                model: result.model,
                thinking_level: if result.warning.is_some() {
                    None
                } else {
                    Some(suffix.to_string())
                },
                warning: result.warning,
            };
        }
        return result;
    }

    if !allow_invalid_thinking_level_fallback {
        // In strict mode (CLI --model parsing), treat it as part of the model id and fail.
        // This avoids accidentally resolving to a different model.
        return ParsedModelResult::empty();
    }
    let result = parse_model_pattern(prefix, available_models, allow_invalid_thinking_level_fallback);
    if result.model.is_some() {
        return ParsedModelResult {
            model: result.model,
            thinking_level: None,
            warning: Some(format!(
                "Invalid thinking level \"{}\" in pattern \"{}\". Using default instead.",
                suffix, pattern
            )),
        };
    }
    result
}

/// `minimatch(value, pattern, { nocase: true })` from model-resolver.ts:279.
/// minimatch matches path segments, so `*` never crosses `/`; globset's default
/// `literal_separator(false)` would let `*anthropic*` reach into the model id and
/// scope models the TypeScript refuses.
fn glob_matches(pattern: &str, value: &str) -> bool {
    GlobBuilder::new(pattern)
        .case_insensitive(true)
        .literal_separator(true)
        .backslash_escape(false)
        .build()
        .map(|glob| glob.compile_matcher().is_match(value))
        .unwrap_or(false)
}

/// Resolve model patterns to actual Model objects with optional thinking levels.
/// Format: "pattern:level" where :level is optional.
pub fn resolve_model_scope_from_models(patterns: &[String], available_models: &[Model]) -> Vec<ScopedModel> {
    let log = pi_ai::log::get_logger("coding-agent.model-resolver");
    let mut scoped_models: Vec<ScopedModel> = Vec::new();

    for pattern in patterns {
        if pattern.contains('*') || pattern.contains('?') || pattern.contains('[') {
            let colon_index = pattern.rfind(':');
            let mut glob_pattern = pattern.clone();
            let mut thinking_level: Option<String> = None;

            if let Some(colon_index) = colon_index {
                let suffix = &pattern[colon_index + 1..];
                if is_valid_thinking_level(suffix) {
                    thinking_level = Some(suffix.to_string());
                    glob_pattern = pattern[..colon_index].to_string();
                }
            }
            // This allows "*sonnet*" to match without requiring "anthropic/*sonnet*"
            let matching_models: Vec<&Model> = available_models
                .iter()
                .filter(|model| {
                    let full_id = format!("{}/{}", model.provider, model.id);
                    glob_matches(&glob_pattern, &full_id) || glob_matches(&glob_pattern, &model.id)
                })
                .collect();

            if matching_models.is_empty() {
                log.warn(
                    "no models match pattern",
                    Some(serde_json::Map::from_iter([("pattern".to_string(), json!(pattern))])),
                );
                eprintln!("Warning: No models match pattern \"{}\"", pattern);
                continue;
            }

            for model in matching_models {
                if !scoped_models
                    .iter()
                    .any(|scoped| models_are_equal(Some(&scoped.model), Some(model)))
                {
                    scoped_models.push(ScopedModel {
                        model: model.clone(),
                        thinking_level: thinking_level.clone(),
                    });
                }
            }
            continue;
        }

        let parsed = parse_model_pattern(pattern, available_models, true);

        if let Some(warning) = &parsed.warning {
            log.warn(
                warning,
                Some(serde_json::Map::from_iter([("pattern".to_string(), json!(pattern))])),
            );
            eprintln!("Warning: {}", warning);
        }

        let Some(model) = parsed.model else {
            log.warn(
                "no models match pattern",
                Some(serde_json::Map::from_iter([("pattern".to_string(), json!(pattern))])),
            );
            eprintln!("Warning: No models match pattern \"{}\"", pattern);
            continue;
        };
        if !scoped_models
            .iter()
            .any(|scoped| models_are_equal(Some(&scoped.model), Some(&model)))
        {
            scoped_models.push(ScopedModel {
                model,
                thinking_level: parsed.thinking_level,
            });
        }
    }

    scoped_models
}

pub async fn resolve_model_scope(
    patterns: &[String],
    model_registry: &mut ModelRegistry,
) -> Vec<ScopedModel> {
    let available_models = model_registry.refresh_available_models().await;
    resolve_model_scope_from_models(patterns, &available_models)
}

/// `interface ResolveCliModelResult`.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolveCliModelResult {
    pub model: Option<Model>,
    pub thinking_level: Option<String>,
    pub warning: Option<String>,
    /// Error message suitable for CLI display. When set, model is `None`.
    pub error: Option<String>,
}

impl ResolveCliModelResult {
    fn empty() -> Self {
        Self {
            model: None,
            thinking_level: None,
            warning: None,
            error: None,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ResolveCliModelOptions {
    pub cli_provider: Option<String>,
    pub cli_model: Option<String>,
}

/// Resolve a single model from CLI flags.
pub fn resolve_cli_model(
    options: &ResolveCliModelOptions,
    model_registry: &ModelRegistry,
) -> ResolveCliModelResult {
    let cli_provider = options.cli_provider.as_deref();
    let Some(cli_model) = options.cli_model.as_deref() else {
        return ResolveCliModelResult::empty();
    };

    // Important: use *all* models here, not just models with pre-configured auth.
    // This allows "--api-key" to be used for first-time setup.
    let available_models = model_registry.get_all();
    if available_models.is_empty() {
        return ResolveCliModelResult {
            model: None,
            thinking_level: None,
            warning: None,
            error: Some("No models available. Check your installation or add models to models.json.".to_string()),
        };
    }
    let mut provider_map: HashMap<String, String> = HashMap::new();
    for model in &available_models {
        provider_map.insert(model.provider.to_lowercase(), model.provider.clone());
    }

    let mut provider = cli_provider.and_then(|value| provider_map.get(&value.to_lowercase()).cloned());
    if cli_provider.is_some() && provider.is_none() {
        return ResolveCliModelResult {
            model: None,
            thinking_level: None,
            warning: None,
            error: Some(format!(
                "Unknown provider \"{}\". Use \"{} model list\" to see available providers/models.",
                cli_provider.unwrap_or_default(),
                APP_NAME
            )),
        };
    }

    // If no explicit --provider, try to interpret "provider/model" format first.
    let mut pattern = cli_model.to_string();
    let mut inferred_provider = false;

    if provider.is_none() {
        if let Some(slash_index) = cli_model.find('/') {
            let maybe_provider = &cli_model[..slash_index];
            if let Some(canonical) = provider_map.get(&maybe_provider.to_lowercase()) {
                provider = Some(canonical.clone());
                pattern = cli_model[slash_index + 1..].to_string();
                inferred_provider = true;
            }
        }
    }

    if provider.is_none() {
        let lower = cli_model.to_lowercase();
        let exact = available_models.iter().find(|model| {
            model.id.to_lowercase() == lower
                || format!("{}/{}", model.provider, model.id).to_lowercase() == lower
        });
        if let Some(exact) = exact {
            return ResolveCliModelResult {
                model: Some(exact.clone()),
                thinking_level: None,
                warning: None,
                error: None,
            };
        }
    }

    if cli_provider.is_some() && provider.is_some() {
        // If both were provided, tolerate --model <provider>/<pattern> by stripping the provider prefix
        let provider_value = provider.clone().unwrap_or_default();
        let prefix = format!("{}/", provider_value);
        if cli_model.to_lowercase().starts_with(&prefix.to_lowercase()) {
            pattern = cli_model[prefix.len()..].to_string();
        }
    }

    let candidates: Vec<Model> = match &provider {
        Some(provider) => available_models
            .iter()
            .filter(|model| &model.provider == provider)
            .cloned()
            .collect(),
        None => available_models.clone(),
    };
    let parsed = parse_model_pattern(&pattern, &candidates, false);

    if let Some(model) = parsed.model {
        return ResolveCliModelResult {
            model: Some(model),
            thinking_level: parsed.thinking_level,
            warning: parsed.warning,
            error: None,
        };
    }

    // If we inferred a provider from the slash but found no match within that provider,
    // fall back to matching the full input as a raw model id across all models.
    if inferred_provider {
        let lower = cli_model.to_lowercase();
        let exact = available_models.iter().find(|model| {
            model.id.to_lowercase() == lower
                || format!("{}/{}", model.provider, model.id).to_lowercase() == lower
        });
        if let Some(exact) = exact {
            return ResolveCliModelResult {
                model: Some(exact.clone()),
                thinking_level: None,
                warning: None,
                error: None,
            };
        }
        let fallback = parse_model_pattern(cli_model, &available_models, false);
        if let Some(model) = fallback.model {
            return ResolveCliModelResult {
                model: Some(model),
                thinking_level: fallback.thinking_level,
                warning: fallback.warning,
                error: None,
            };
        }
    }

    if let Some(provider) = &provider {
        if let Some(fallback_model) = build_fallback_model(provider, &pattern, &available_models) {
            let fallback_warning = match &parsed.warning {
                Some(warning) => format!(
                    "{} Model \"{}\" not found for provider \"{}\". Using custom model id.",
                    warning, pattern, provider
                ),
                None => format!(
                    "Model \"{}\" not found for provider \"{}\". Using custom model id.",
                    pattern, provider
                ),
            };
            return ResolveCliModelResult {
                model: Some(fallback_model),
                thinking_level: None,
                warning: Some(fallback_warning),
                error: None,
            };
        }
    }

    let display = match &provider {
        Some(provider) => format!("{}/{}", provider, pattern),
        None => cli_model.to_string(),
    };
    ResolveCliModelResult {
        model: None,
        thinking_level: None,
        warning: parsed.warning,
        error: Some(format!(
            "Model \"{}\" not found. Use \"{} model list\" to see available models.",
            display, APP_NAME
        )),
    }
}

/// `interface InitialModelResult`.
#[derive(Debug, Clone, PartialEq)]
pub struct InitialModelResult {
    pub model: Option<Model>,
    pub thinking_level: String,
    pub fallback_message: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct FindInitialModelOptions {
    pub cli_provider: Option<String>,
    pub cli_model: Option<String>,
    pub scoped_models: Vec<ScopedModel>,
    pub is_continuing: bool,
    pub default_provider: Option<String>,
    pub default_model_id: Option<String>,
    pub default_thinking_level: Option<String>,
}

/// Find the initial model to use based on priority.
///
/// The TypeScript `process.exit(1)` paths return `Err(message)`; the caller
/// prints the message to stderr and exits with code 1.
pub async fn find_initial_model(
    options: &FindInitialModelOptions,
    model_registry: &mut ModelRegistry,
) -> Result<InitialModelResult, String> {
    let cli_provider = options.cli_provider.as_deref();
    let cli_model = options.cli_model.as_deref();

    let mut cached_available_models: Option<Vec<Model>> = None;
    let mut model: Option<Model> = None;
    let mut thinking_level = DEFAULT_THINKING_LEVEL.to_string();

    if let (Some(cli_provider), Some(cli_model)) = (cli_provider, cli_model) {
        let resolved = resolve_cli_model(
            &ResolveCliModelOptions {
                cli_provider: Some(cli_provider.to_string()),
                cli_model: Some(cli_model.to_string()),
            },
            model_registry,
        );
        if let Some(error) = resolved.error {
            let log = pi_ai::log::get_logger("coding-agent.model-resolver");
            log.error(
                &error,
                Some(serde_json::Map::from_iter([
                    ("cliProvider".to_string(), json!(cli_provider)),
                    ("cliModel".to_string(), json!(cli_model)),
                ])),
            );
            return Err(error);
        }
        if let Some(resolved_model) = resolved.model {
            if is_private_prime_inference_model(&resolved_model.provider, &resolved_model.id) {
                if cached_available_models.is_none() {
                    cached_available_models = Some(model_registry.refresh_available_models().await);
                }
                let available_model = cached_available_models
                    .as_ref()
                    .and_then(|models| {
                        models
                            .iter()
                            .find(|candidate| models_are_equal(Some(candidate), Some(&resolved_model)))
                    })
                    .cloned();
                let Some(available_model) = available_model else {
                    let error = format!(
                        "Model \"{}/{}\" is not available for the current Prime team.",
                        resolved_model.provider, resolved_model.id
                    );
                    let log = pi_ai::log::get_logger("coding-agent.model-resolver");
                    log.error(
                        &error,
                        Some(serde_json::Map::from_iter([
                            ("cliProvider".to_string(), json!(cli_provider)),
                            ("cliModel".to_string(), json!(cli_model)),
                        ])),
                    );
                    return Err(error);
                };
                return Ok(InitialModelResult {
                    model: Some(available_model),
                    thinking_level: DEFAULT_THINKING_LEVEL.to_string(),
                    fallback_message: None,
                });
            }
            return Ok(InitialModelResult {
                model: Some(resolved_model),
                thinking_level: DEFAULT_THINKING_LEVEL.to_string(),
                fallback_message: None,
            });
        }
    }
    if !options.scoped_models.is_empty() && !options.is_continuing {
        let scoped = &options.scoped_models[0];
        return Ok(InitialModelResult {
            model: Some(scoped.model.clone()),
            thinking_level: scoped
                .thinking_level
                .clone()
                .or_else(|| options.default_thinking_level.clone())
                .unwrap_or_else(|| DEFAULT_THINKING_LEVEL.to_string()),
            fallback_message: None,
        });
    }
    if cached_available_models.is_none() {
        cached_available_models = Some(model_registry.refresh_available_models().await);
    }
    let available_models = cached_available_models.clone().unwrap_or_default();
    if let (Some(default_provider), Some(default_model_id)) =
        (options.default_provider.as_deref(), options.default_model_id.as_deref())
    {
        // Rebuild from the provider template when the saved id is missing from this
        // build's snapshot (e.g. prime-inference catalog churn), so it survives updates.
        let found = available_models
            .iter()
            .find(|candidate| candidate.provider == default_provider && candidate.id == default_model_id)
            .cloned()
            .or_else(|| {
                if is_private_prime_inference_model(default_provider, default_model_id) {
                    None
                } else {
                    build_fallback_model(default_provider, default_model_id, &available_models)
                }
            });
        if let Some(found) = found {
            model = Some(found);
            if let Some(default_thinking_level) = &options.default_thinking_level {
                thinking_level = default_thinking_level.clone();
            }
            return Ok(InitialModelResult {
                model,
                thinking_level,
                fallback_message: None,
            });
        }
    }
    if !available_models.is_empty() {
        if let Some(default_model) = find_preferred_default_model(&available_models) {
            return Ok(InitialModelResult {
                model: Some(default_model),
                thinking_level: DEFAULT_THINKING_LEVEL.to_string(),
                fallback_message: None,
            });
        }
        return Ok(InitialModelResult {
            model: Some(available_models[0].clone()),
            thinking_level: DEFAULT_THINKING_LEVEL.to_string(),
            fallback_message: None,
        });
    }
    Ok(InitialModelResult {
        model: None,
        thinking_level: DEFAULT_THINKING_LEVEL.to_string(),
        fallback_message: None,
    })
}

/// `restoreModelFromSession(savedProvider, savedModelId, currentModel, shouldPrintMessages, modelRegistry)`.
pub async fn restore_model_from_session(
    saved_provider: &str,
    saved_model_id: &str,
    current_model: Option<&Model>,
    should_print_messages: bool,
    model_registry: &mut ModelRegistry,
) -> (Option<Model>, Option<String>) {
    let available_models = model_registry.refresh_available_models().await;
    let restored_model = available_models
        .iter()
        .find(|candidate| candidate.provider == saved_provider && candidate.id == saved_model_id)
        .cloned();

    if let Some(restored_model) = restored_model {
        if should_print_messages {
            println!("Restored model: {}/{}", saved_provider, saved_model_id);
        }
        return (Some(restored_model), None);
    }
    let registered_model = model_registry.find(saved_provider, saved_model_id);
    let reason = if registered_model.is_none() {
        "model no longer exists"
    } else if !registered_model
        .as_ref()
        .map(|model| model_registry.has_configured_auth(model))
        .unwrap_or(false)
    {
        "no auth configured"
    } else {
        "model is not available"
    };
    let log = pi_ai::log::get_logger("coding-agent.model-resolver");
    log.warn(
        "could not restore model",
        Some(serde_json::Map::from_iter([
            ("provider".to_string(), json!(saved_provider)),
            ("model".to_string(), json!(saved_model_id)),
            ("reason".to_string(), json!(reason)),
        ])),
    );

    if should_print_messages {
        eprintln!(
            "Warning: Could not restore model {}/{} ({}).",
            saved_provider, saved_model_id, reason
        );
    }
    let available_current_model = current_model.and_then(|current| {
        available_models
            .iter()
            .find(|candidate| models_are_equal(Some(candidate), Some(current)))
            .cloned()
    });
    let fallback_current_model = match current_model {
        Some(current)
            if !is_private_prime_inference_model(&current.provider, &current.id)
                || available_current_model.is_some() =>
        {
            Some(available_current_model.clone().unwrap_or_else(|| current.clone()))
        }
        _ => None,
    };
    if let Some(fallback_current_model) = fallback_current_model {
        if should_print_messages {
            println!(
                "Falling back to: {}/{}",
                fallback_current_model.provider, fallback_current_model.id
            );
        }
        return (
            Some(fallback_current_model.clone()),
            Some(format!(
                "Could not restore model {}/{} ({}). Using {}/{}.",
                saved_provider,
                saved_model_id,
                reason,
                fallback_current_model.provider,
                fallback_current_model.id
            )),
        );
    }
    if !available_models.is_empty() {
        let fallback_model = find_preferred_default_model(&available_models)
            .unwrap_or_else(|| available_models[0].clone());

        if should_print_messages {
            println!("Falling back to: {}/{}", fallback_model.provider, fallback_model.id);
        }

        return (
            Some(fallback_model.clone()),
            Some(format!(
                "Could not restore model {}/{} ({}). Using {}/{}.",
                saved_provider, saved_model_id, reason, fallback_model.provider, fallback_model.id
            )),
        );
    }
    (None, None)
}

/// `THINKING_LEVELS` re-export so callers of this module can validate a level.
pub fn thinking_levels() -> &'static [&'static str] {
    &THINKING_LEVELS
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(provider: &str, id: &str) -> Model {
        Model::new(
            id,
            id,
            "openai-completions",
            provider,
            "https://example.invalid",
        )
    }

    fn scoped_ids(pattern: &str, models: &[Model]) -> Vec<(String, String)> {
        resolve_model_scope_from_models(&[pattern.to_string()], models)
            .into_iter()
            .map(|scoped| (scoped.model.provider.clone(), scoped.model.id.clone()))
            .collect()
    }

    /// model-resolver.ts:277-279 matches with
    /// `minimatch(value, globPattern, { nocase: true })`, which is
    /// segment-bound: `*` never crosses `/`. `literal_separator(false)` let
    /// `*anthropic*` reach across `provider/id` and scope models the TypeScript
    /// leaves out. `modelsAreEqual` is a `provider/id` equality check
    /// (model-resolver.ts:288-290), so the scoped set is directly observable.
    #[test]
    fn a_glob_star_does_not_cross_the_provider_separator() {
        let models = vec![
            model("anthropic", "claude-opus-4-7"),
            model("openai", "anthropic-compat"),
        ];

        // minimatch("anthropic/claude-opus-4-7", "*anthropic*") is false, and
        // minimatch("claude-opus-4-7", "*anthropic*") is false too, so only the
        // model whose id literally contains "anthropic" stays in scope.
        assert_eq!(
            scoped_ids("*anthropic*", &models),
            vec![("openai".to_string(), "anthropic-compat".to_string())],
            "`*` crossed the provider separator and scoped extra models"
        );

        // A glob that is meant to cross the separator states the separator.
        assert_eq!(
            scoped_ids("anthropic/*", &models),
            vec![("anthropic".to_string(), "claude-opus-4-7".to_string())]
        );
    }
}
