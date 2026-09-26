//! CMP-002: offline model-aware auto-compaction policy regressions.

use pi_ai::models::get_model_input_limit;
use pi_ai::types::Model;
use pi_coding_agent::core::compaction::compaction::{
    default_compaction_settings, should_compact, should_compact_for_model, CompactionSettings,
    MAX_COMPACTION_CONTEXT_TOKENS,
};

const AZURE_PROVIDERS: [&str; 3] = [
    "azure-openai-managed",
    "azure-foundry-managed",
    "azure-openai-responses",
];

fn model(provider: &str, id: &str) -> Model {
    Model {
        id: id.into(),
        name: "Offline threshold fixture".into(),
        provider: provider.into(),
        api: "openai-responses".into(),
        base_url: "https://fixture.invalid/v1".into(),
        context_window: 1_050_000.0,
        max_tokens: 32_000.0,
        ..Default::default()
    }
}

fn assert_boundary(model: &Model, settings: &CompactionSettings, boundary: f64) {
    let original = model.clone();
    for (tokens, expected) in [
        (boundary - 1.0, false),
        (boundary, true),
        (boundary + 1.0, true),
    ] {
        assert_eq!(
            should_compact_for_model(tokens, model, settings),
            expected,
            "{}/{} at {tokens}, expected boundary {boundary}",
            model.provider,
            model.id,
        );
    }
    assert_eq!(
        model, &original,
        "threshold checks must not change the model"
    );
}

#[test]
fn azure_gpt_routes_use_an_inclusive_400k_cap() {
    let settings = default_compaction_settings();
    for provider in AZURE_PROVIDERS {
        for id in ["gpt-5.6-sol", "gpt-6-astra", "GPT-6-ASTRA", "gPt-4o"] {
            let model = model(provider, id);
            assert_boundary(&model, &settings, 400_000.0);
            for tokens in [250_000.0, 250_905.0, 399_999.0] {
                assert!(!should_compact_for_model(tokens, &model, &settings));
            }
            assert_eq!(get_model_input_limit(&model), 1_050_000.0);
        }
    }
}

#[test]
fn non_azure_gpt_and_provider_lookalikes_keep_the_250k_cap() {
    let settings = default_compaction_settings();
    for provider in [
        "openai",
        "openai-codex",
        "github-copilot",
        "openrouter",
        "faux",
        "azure",
        "azure-other",
        "azure-openai-managed-extra",
        "azure-foundry-managed-extra",
        "Azure-OpenAI-Managed",
        "",
    ] {
        let mut model = model(provider, "gpt-6-astra");
        // Neither the API, display name nor endpoint grants Azure policy.
        model.api = "azure-openai-responses".into();
        model.name = "Azure GPT-6 Astra".into();
        model.base_url = "https://fixture.openai.azure.com/openai/v1".into();
        assert_boundary(&model, &settings, 250_000.0);
    }
}

#[test]
fn azure_non_gpt_models_and_opaque_aliases_keep_the_250k_cap() {
    let settings = default_compaction_settings();
    for provider in AZURE_PROVIDERS {
        for id in [
            "FW-Kimi-K3",
            "FW-GLM-5.3",
            "o1",
            "o3",
            "o4-mini",
            "deployment-astra",
            "openai/gpt-6-astra",
            "not-gpt-6-astra",
            "gpt",
            "gpt6-astra",
            " gpt-6-astra",
            "",
        ] {
            let mut model = model(provider, id);
            model.name = "GPT-6 Astra".into();
            model.context_window = 1_048_576.0;
            assert_boundary(&model, &settings, 250_000.0);
        }
    }
}

#[test]
fn azure_gpt_policy_does_not_depend_on_api_or_native_compaction() {
    let settings = default_compaction_settings();
    for api in [
        "openai-responses",
        "azure-openai-responses",
        "openai-completions",
        "fixture",
    ] {
        let mut model = model("azure-openai-managed", "gpt-6-astra");
        model.api = api.into();
        assert!(model.native_compaction.is_none());
        assert_boundary(&model, &settings, 400_000.0);
    }
}

#[test]
fn smaller_context_and_configured_input_limits_still_win() {
    let settings = default_compaction_settings();
    for provider in AZURE_PROVIDERS {
        let mut model = model(provider, "gpt-6-astra");
        model.context_window = 200_000.0;
        assert_boundary(&model, &settings, 183_616.0);
        model.context_window = 400_000.0;
        assert_boundary(&model, &settings, 383_616.0);
        model.context_window = 416_384.0;
        assert_boundary(&model, &settings, 400_000.0);
        model.context_window = 1_050_000.0;
        model.max_input_tokens = Some(150_000.0);
        assert_eq!(get_model_input_limit(&model), 150_000.0);
        assert_boundary(&model, &settings, 133_616.0);
        model.max_input_tokens = Some(350_000.0);
        assert_boundary(&model, &settings, 333_616.0);
        model.max_input_tokens = Some(1_500_000.0);
        model.context_window = 200_000.0;
        assert_boundary(&model, &settings, 183_616.0);
        for window in [0.0, -1.0] {
            model.context_window = window;
            assert!(!should_compact_for_model(900_000.0, &model, &settings));
        }
    }
}

#[test]
fn custom_reserve_still_bounds_both_policy_caps() {
    let mut settings = default_compaction_settings();
    settings.reserve_tokens = 50_000.0;
    for (provider, id, cap) in [
        ("azure-openai-managed", "gpt-5.6-sol", 400_000.0),
        ("azure-foundry-managed", "gpt-6-astra", 400_000.0),
        ("azure-foundry-managed", "FW-Kimi-K3", 250_000.0),
        ("azure-foundry-managed", "FW-GLM-5.3", 250_000.0),
        ("openai", "gpt-5.6-sol", 250_000.0),
    ] {
        let mut model = model(provider, id);
        assert_boundary(&model, &settings, cap);
        model.context_window = 425_000.0;
        assert_boundary(&model, &settings, f64::min(cap, 375_000.0));
        model.max_input_tokens = Some(225_000.0);
        assert_boundary(&model, &settings, 175_000.0);
    }
}

#[test]
fn disabled_compaction_and_generic_policy_remain_unchanged() {
    let mut settings = default_compaction_settings();
    assert_eq!(MAX_COMPACTION_CONTEXT_TOKENS, 250_000.0);
    assert!(!should_compact(249_999.0, 1_050_000.0, &settings));
    assert!(should_compact(250_000.0, 1_050_000.0, &settings));
    assert!(should_compact(250_001.0, 1_050_000.0, &settings));
    assert!(!should_compact(183_615.0, 200_000.0, &settings));
    assert!(should_compact(183_616.0, 200_000.0, &settings));
    settings.enabled = false;
    for provider in AZURE_PROVIDERS
        .into_iter()
        .chain(["openai", "github-copilot"])
    {
        for id in ["gpt-5.6-sol", "gpt-6-astra", "FW-Kimi-K3", "FW-GLM-5.3"] {
            let mut model = model(provider, id);
            for window in [1_050_000.0, 200_000.0] {
                model.context_window = window;
                for tokens in [250_000.0, 400_000.0, 900_000.0] {
                    assert!(!should_compact_for_model(tokens, &model, &settings));
                }
            }
        }
    }
    assert!(!should_compact(900_000.0, 1_050_000.0, &settings));
}

#[test]
fn selected_model_switches_recompute_the_policy_and_input_bound() {
    let settings = default_compaction_settings();
    let mut selected = model("azure-openai-managed", "gpt-6-astra");
    assert!(!should_compact_for_model(300_000.0, &selected, &settings));
    selected.provider = "github-copilot".into();
    assert!(should_compact_for_model(300_000.0, &selected, &settings));
    selected.provider = "azure-openai-managed".into();
    assert!(!should_compact_for_model(300_000.0, &selected, &settings));
    selected.id = "o3".into();
    assert!(should_compact_for_model(300_000.0, &selected, &settings));
    selected = model("azure-foundry-managed", "FW-Kimi-K3");
    assert!(should_compact_for_model(300_000.0, &selected, &settings));
    selected.id = "gpt-5.6-sol".into();
    assert!(!should_compact_for_model(300_000.0, &selected, &settings));
    selected.max_input_tokens = Some(300_000.0);
    assert!(should_compact_for_model(300_000.0, &selected, &settings));
    selected.max_input_tokens = None;
    assert!(!should_compact_for_model(300_000.0, &selected, &settings));
    assert_boundary(&selected, &settings, 400_000.0);
}
