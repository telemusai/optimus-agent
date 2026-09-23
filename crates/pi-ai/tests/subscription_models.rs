//! Offline metadata/serialization tests. No provider calls or real credentials.
use pi_ai::models::{get_model, get_supported_thinking_levels, supports_fast_mode};
use pi_ai::providers::openai_codex_responses::{
    build_request_body, build_sse_headers, OpenAICodexResponsesOptions,
};
use pi_ai::types::{Context, InputModality};
use serde_json::json;

#[test]
fn codex_sol_luna_catalog_preserves_reviewed_metadata() {
    for (id, input_cost, output_cost) in [
        ("gpt-6-sol", 2.0, 10.0),
        ("gpt-6-luna", 0.1, 0.5),
    ] {
        let model = get_model("openai-codex", id).expect("reviewed subscription model");
        assert_eq!(model.id, id);
        assert_eq!(model.provider, "openai-codex");
        assert_eq!(model.api, "openai-codex-responses");
        assert_eq!(model.base_url, "https://chatgpt.com/backend-api");
        assert_eq!(model.context_window, 272_000.0);
        assert_eq!(model.max_tokens, 128_000.0);
        assert_eq!(model.input, vec![InputModality::Text, InputModality::Image]);
        assert_eq!(model.cost.input, input_cost);
        assert_eq!(model.cost.output, output_cost);
        assert_eq!(get_supported_thinking_levels(model), vec!["low", "medium", "high", "xhigh", "max"]);
        // Model admission does not opt into a new tier/pricing policy.
        assert!(!supports_fast_mode(model));
    }
}

#[test]
fn codex_sol_luna_requests_keep_ids_reasoning_and_existing_auth_shape() {
    for id in ["gpt-6-sol", "gpt-6-luna"] {
        let model = get_model("openai-codex", id).unwrap();
        for effort in ["low", "medium", "high", "xhigh", "max"] {
            let options = OpenAICodexResponsesOptions {
                reasoning_effort: Some(effort.to_string()),
                ..Default::default()
            };
            let body = build_request_body(model, &Context::default(), Some(&options)).unwrap();
            assert_eq!(body["model"], json!(id));
            assert_eq!(body["reasoning"]["effort"], json!(effort));
            assert_eq!(body["store"], json!(false));
            assert_eq!(body["stream"], json!(true));
            assert_eq!(body["tool_choice"], json!("auto"));
            assert!(!body.contains_key("authorization"));
            assert!(!body.contains_key("chatgpt-account-id"));
        }
        let headers = build_sse_headers(None, None, "fixture-account", "synthetic-token", None);
        assert_eq!(headers["Authorization"], "Bearer synthetic-token");
        assert_eq!(headers["chatgpt-account-id"], "fixture-account");
    }
}

#[test]
fn copilot_sol_luna_metadata_matches_sanitized_provider_catalog() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "fixtures/copilot-sol-luna-catalog.json")).unwrap();
    let rows = fixture["models"].as_array().unwrap();
    assert_eq!(rows.len(), 2);
    for row in rows {
        let id = row["id"].as_str().unwrap();
        let model = get_model("github-copilot", id).unwrap();
        assert_eq!(model.id, id);
        assert_eq!(model.api, "openai-responses");
        assert_eq!(model.context_window, row["limits"]["max_context_window_tokens"].as_f64().unwrap());
        assert_eq!(model.max_input_tokens, row["limits"]["max_prompt_tokens"].as_f64());
        assert_eq!(pi_ai::models::get_model_input_limit(model), 272_000.0);
        assert_eq!(model.max_tokens, row["limits"]["max_output_tokens"].as_f64().unwrap());
        assert_eq!(model.input, vec![InputModality::Text, InputModality::Image]);
        assert_eq!(row["supports"]["vision"], true);
        assert_eq!(row["supports"]["tool_calls"], true);
        assert_eq!(row["supports"]["parallel_tool_calls"], true);
        assert_eq!(row["supported_endpoints"], json!(["/responses"]));
        assert_eq!(get_supported_thinking_levels(model), vec!["off", "low", "medium", "high", "xhigh", "max"]);
        let wire_levels: Vec<String> = get_supported_thinking_levels(model).into_iter()
            .map(|level| model.thinking_level_map_get(&level).flatten().unwrap_or(level)).collect();
        assert_eq!(json!(wire_levels), row["supports"]["reasoning_effort"]);
        assert!(!supports_fast_mode(model));
    }
}

#[test]
fn copilot_sol_luna_requests_keep_subscription_auth_routing_and_exact_ids() {
    use pi_ai::copilot_client_version::COPILOT_CLIENT_HEADERS;
    use pi_ai::providers::openai_responses::{build_params, create_client, OpenAIResponsesOptions};
    use pi_ai::types::{StreamOptions, Tool};
    use pi_ai::utils::oauth::github_copilot::github_copilot_oauth_provider;
    use pi_ai::utils::oauth::types::OAuthCredentials;
    let provider = github_copilot_oauth_provider();
    let credentials = OAuthCredentials {
        access: "synthetic;proxy-ep=proxy.business.githubcopilot.com;".to_string(),
        ..Default::default()
    };
    let token = (provider.get_api_key)(&credentials);
    for id in ["gpt-6-sol", "gpt-6-luna"] {
        let built_in = get_model("github-copilot", id).unwrap().clone();
        let routed = provider.modify_models.as_ref().unwrap()(vec![built_in], &credentials);
        let model = &routed[0];
        assert_eq!(model.id, id);
        let context = Context::new(None, Vec::new(), Some(vec![Tool {
            name: "fixture_tool".to_string(), description: "Offline fixture".to_string(),
            parameters: json!({"type": "object", "properties": {}}),
        }]));
        let client = create_client(model, &context, Some(&token), None, None, None).unwrap();
        assert_eq!(client.api_key, token);
        assert_eq!(client.base_url, "https://api.business.githubcopilot.com");
        for (name, value) in COPILOT_CLIENT_HEADERS {
            assert_eq!(client.default_headers.get(name), Some(&Some(value.to_string())));
        }
        for level in ["off", "low", "medium", "high", "xhigh", "max"] {
            let options = OpenAIResponsesOptions {
                reasoning_effort: Some(level.to_string()),
                service_tier: Some(Some("priority".to_string())),
                stream: StreamOptions { max_tokens: Some(128_000.0), ..Default::default() },
                ..Default::default()
            };
            let body = build_params(model, &context, Some(&options)).unwrap();
            assert_eq!(body["model"], id);
            assert_eq!(body["reasoning"]["effort"], if level == "off" { "none" } else { level });
            assert_eq!(body["max_output_tokens"].as_u64(), Some(128_000));
            assert_eq!(body["store"], false);
            assert_eq!(body["stream"], true);
            assert_eq!(body["tools"][0]["name"], "fixture_tool");
            assert!(!body.contains_key("service_tier"));
            assert!(!body.contains_key("authorization"));
            assert!(!serde_json::to_string(&body).unwrap().contains("synthetic"));
        }
    }
}
