//! Built-in DGX model catalog.
//!
//! Rust-native module with no TypeScript counterpart: the three DGX gateway
//! models ship in the binary so `model list` and the model pickers surface
//! them on a fresh install. They deliberately carry NO endpoint and NO
//! credentials - no baseUrl, apiKey, or headers. A user enables them by
//! importing a DGX Auth File (`optimus-agent dgx-auth import <file>`), which
//! merges `dgx-*` providers into models.json; those entries then override
//! these built-ins by provider+id (`ModelRegistry::merge_custom_models`).
//! Until then, using one fails fast with the import hint instead of a
//! network error (`dgx_credentials_required_message`).

use pi_ai::types::{
    API_OPENAI_COMPLETIONS, Compat, InputModality, MAX_TOKENS_FIELD_MAX_TOKENS, Model,
    OpenAICompletionsCompat, THINKING_FORMAT_OPENAI,
};

/// Suffix appended to the display name of a built-in DGX model whose provider
/// has no imported credentials. It disappears once a models.json entry
/// replaces the built-in model.
pub const DGX_CREDENTIALS_MARKER: &str = "DGX credentials required";

/// Context window every DGX gateway model reports.
const DGX_CONTEXT_WINDOW: f64 = 1_048_576.0;
/// Output ceiling every DGX gateway model reports.
const DGX_MAX_TOKENS: f64 = 262_144.0;

/// The actionable error for using a DGX model before credentials are
/// imported (or after the imported gateway session expired).
pub fn dgx_credentials_required_message(provider: &str) -> String {
    format!(
        "DGX credentials required for \"{}\" - import a DGX Auth File: {} dgx-auth import <file>",
        provider,
        crate::config::APP_COMMAND
    )
}

/// Whether `provider` is a DGX gateway provider. Matches the `dgx-` id
/// prefix the auth-file format reserves (`core::dgx_auth`).
pub fn is_dgx_provider(provider: &str) -> bool {
    provider.starts_with("dgx-")
}

/// Build the bundled DGX models. `has_request_auth` reports whether a
/// provider already holds models.json request credentials (a DGX Auth File
/// import); models without them carry the needs-credentials marker in their
/// display name.
pub fn build_dgx_models(has_request_auth: &dyn Fn(&str) -> bool) -> Vec<Model> {
    [
        (
            "dgx-glm53",
            "glm-5.3",
            "GLM-5.3 - DGX MTP",
            &[InputModality::Text][..],
            Some(glm53_compat()),
        ),
        (
            "dgx-k3",
            "kimi-k3",
            "Kimi K3 - DGX MAX",
            &[InputModality::Text, InputModality::Image][..],
            Some(k3_compat()),
        ),
        (
            "dgx-deepseek41",
            "deepseek-v4.1-flash-1017",
            "DeepSeek V4.1 Flash 1017 - DGX",
            &[InputModality::Text, InputModality::Image][..],
            // The live deployment ships no compat for this provider; the
            // openai-completions URL auto-detection defaults apply.
            None,
        ),
    ]
    .into_iter()
    .map(|(provider, id, name, input, compat)| {
        dgx_model(
            provider,
            id,
            name,
            input,
            compat,
            !has_request_auth(provider),
        )
    })
    .collect()
}

fn dgx_model(
    provider: &str,
    id: &str,
    name: &str,
    input: &[InputModality],
    compat: Option<OpenAICompletionsCompat>,
    needs_credentials: bool,
) -> Model {
    let mut model = Model::new(
        id,
        if needs_credentials {
            format!("{name} ({DGX_CREDENTIALS_MARKER})")
        } else {
            name.to_string()
        },
        API_OPENAI_COMPLETIONS,
        provider,
        // No endpoint ships with the catalog; a models.json provider entry
        // (the auth-file import) supplies baseUrl.
        "",
    );
    model.reasoning = true;
    model.input = input.to_vec();
    model.context_window = DGX_CONTEXT_WINDOW;
    model.max_tokens = DGX_MAX_TOKENS;
    model.max_tokens_explicit = true;
    model.compat = compat.map(Compat::Completions);
    model
}

/// `dgx-glm53` compat, mirroring the gateway deployment's models.json.
fn glm53_compat() -> OpenAICompletionsCompat {
    OpenAICompletionsCompat {
        supports_store: Some(false),
        supports_developer_role: Some(false),
        supports_reasoning_effort: Some(false),
        max_tokens_field: Some(MAX_TOKENS_FIELD_MAX_TOKENS.to_string()),
        supports_strict_mode: Some(false),
        supports_long_cache_retention: Some(false),
        sglang_token_budget: Some(true),
        thinking_format: Some(THINKING_FORMAT_OPENAI.to_string()),
        requires_thinking_as_text: Some(false),
        requires_reasoning_content_on_assistant_messages: Some(true),
        ..Default::default()
    }
}

/// `dgx-k3` compat, mirroring the gateway deployment's models.json.
fn k3_compat() -> OpenAICompletionsCompat {
    OpenAICompletionsCompat {
        supports_store: Some(false),
        supports_developer_role: Some(false),
        supports_reasoning_effort: Some(true),
        max_tokens_field: Some(MAX_TOKENS_FIELD_MAX_TOKENS.to_string()),
        supports_strict_mode: Some(false),
        supports_long_cache_retention: Some(false),
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn providers_and_ids(models: &[Model]) -> Vec<(String, String)> {
        models
            .iter()
            .map(|model| (model.provider.clone(), model.id.clone()))
            .collect()
    }

    #[test]
    fn builds_the_three_gateway_models() {
        let models = build_dgx_models(&|_| false);
        assert_eq!(
            providers_and_ids(&models),
            vec![
                ("dgx-glm53".to_string(), "glm-5.3".to_string()),
                ("dgx-k3".to_string(), "kimi-k3".to_string()),
                (
                    "dgx-deepseek41".to_string(),
                    "deepseek-v4.1-flash-1017".to_string()
                ),
            ]
        );
        let names: Vec<&str> = models.iter().map(|model| model.name.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "GLM-5.3 - DGX MTP (DGX credentials required)",
                "Kimi K3 - DGX MAX (DGX credentials required)",
                "DeepSeek V4.1 Flash 1017 - DGX (DGX credentials required)",
            ]
        );
        for model in &models {
            assert_eq!(model.api, "openai-completions");
            assert!(model.reasoning);
        }
    }

    #[test]
    fn shipped_catalog_carries_no_endpoint_or_credentials() {
        for model in build_dgx_models(&|_| false) {
            assert!(model.base_url.is_empty(), "{} ships an endpoint", model.id);
            assert!(model.headers.is_none(), "{} ships headers", model.id);
            assert_eq!(model.cost, pi_ai::types::ModelCost::zero());
            assert!(model.reasoning);
            assert_eq!(model.context_window, DGX_CONTEXT_WINDOW);
            assert_eq!(model.max_tokens, DGX_MAX_TOKENS);
            assert!(model.max_tokens_explicit);
        }
    }

    #[test]
    fn marker_tracks_request_auth_per_provider() {
        let models = build_dgx_models(&|provider| provider == "dgx-k3");
        let kimi = models.iter().find(|model| model.id == "kimi-k3").unwrap();
        assert!(!kimi.name.contains(DGX_CREDENTIALS_MARKER));
        let glm = models.iter().find(|model| model.id == "glm-5.3").unwrap();
        assert!(glm.name.contains(DGX_CREDENTIALS_MARKER));
    }

    #[test]
    fn compat_mirrors_the_gateway_configuration() {
        let models = build_dgx_models(&|_| false);
        let glm = models.iter().find(|model| model.id == "glm-5.3").unwrap();
        let glm_compat = glm
            .compat
            .as_ref()
            .and_then(Compat::as_completions)
            .unwrap();
        assert_eq!(glm_compat.supports_store, Some(false));
        assert_eq!(glm_compat.supports_reasoning_effort, Some(false));
        assert_eq!(glm_compat.max_tokens_field.as_deref(), Some("max_tokens"));
        assert_eq!(glm_compat.sglang_token_budget, Some(true));
        assert_eq!(glm_compat.thinking_format.as_deref(), Some("openai"));
        assert_eq!(
            glm_compat.requires_reasoning_content_on_assistant_messages,
            Some(true)
        );
        assert_eq!(glm_compat.requires_thinking_as_text, Some(false));

        let kimi = models.iter().find(|model| model.id == "kimi-k3").unwrap();
        let kimi_compat = kimi
            .compat
            .as_ref()
            .and_then(Compat::as_completions)
            .unwrap();
        assert_eq!(kimi_compat.supports_reasoning_effort, Some(true));
        assert_eq!(kimi_compat.max_tokens_field.as_deref(), Some("max_tokens"));
        assert_eq!(kimi_compat.supports_strict_mode, Some(false));

        let deepseek = models
            .iter()
            .find(|model| model.id == "deepseek-v4.1-flash-1017")
            .unwrap();
        assert!(deepseek.compat.is_none());
    }

    #[test]
    fn modalities_match_the_gateway_configuration() {
        let models = build_dgx_models(&|_| false);
        let glm = models.iter().find(|model| model.id == "glm-5.3").unwrap();
        assert_eq!(glm.input, vec![InputModality::Text]);
        let kimi = models.iter().find(|model| model.id == "kimi-k3").unwrap();
        assert_eq!(kimi.input, vec![InputModality::Text, InputModality::Image]);
        let deepseek = models
            .iter()
            .find(|model| model.id == "deepseek-v4.1-flash-1017")
            .unwrap();
        assert_eq!(
            deepseek.input,
            vec![InputModality::Text, InputModality::Image]
        );
    }

    #[test]
    fn is_dgx_provider_matches_the_auth_file_prefix() {
        assert!(is_dgx_provider("dgx-glm53"));
        assert!(is_dgx_provider("dgx-k3"));
        assert!(is_dgx_provider("dgx-deepseek41"));
        assert!(!is_dgx_provider("deepseek"));
        assert!(!is_dgx_provider(""));
    }

    #[test]
    fn credentials_message_points_at_the_import_command() {
        let message = dgx_credentials_required_message("dgx-glm53");
        assert!(message.contains("\"dgx-glm53\""), "{message}");
        assert!(message.contains("DGX credentials required"), "{message}");
        assert!(
            message.contains(&format!("{} dgx-auth import", crate::config::APP_COMMAND)),
            "{message}"
        );
    }
}
