use super::{client, response_error, Access, DEFAULT_ENDPOINT};
use crate::types::{InputModality, Model, ModelCost};
use serde_json::{json, Value};
use std::{collections::HashSet, time::Duration};

/// Baseline reported by Kiro CLI 2.25.0 on 2026-09-29. Availability depends on the account.
pub fn built_in_models() -> Vec<Model> {
    serde_json::from_str(include_str!("models.json")).expect("embedded Kiro catalog must be valid")
}

pub fn parse_models(value: &Value, base_url: &str) -> Result<Vec<Model>, String> {
    let items = value["models"]
        .as_array()
        .ok_or("Kiro catalog has no model list")?;
    let mut models = vec![];
    for item in items {
        let id = item["modelId"]
            .as_str()
            .filter(|s| !s.trim().is_empty())
            .ok_or("Kiro catalog has an invalid model ID")?;
        let fallback = built_in_models().into_iter().find(|model| model.id == id);
        let context = item["tokenLimits"]["maxInputTokens"]
            .as_u64()
            .filter(|n| *n > 0)
            .map(|n| n as f64)
            .or_else(|| fallback.as_ref().map(|m| m.context_window))
            .unwrap_or(200_000.0);
        let max_tokens = item["tokenLimits"]["maxOutputTokens"]
            .as_u64()
            .filter(|n| *n > 0)
            .map(|n| n as f64)
            .unwrap_or(8192.0);
        let input = if let Some(types) = item["supportedInputTypes"].as_array() {
            let mut modes = vec![InputModality::Text];
            if types.iter().any(|s| s.as_str() == Some("IMAGE")) {
                modes.push(InputModality::Image);
            }
            modes
        } else {
            fallback
                .as_ref()
                .map(|m| m.input.clone())
                .unwrap_or_else(|| vec![InputModality::Text])
        };
        models.push(Model {
            id: id.into(),
            name: item["modelName"].as_str().unwrap_or(id).into(),
            api: "kiro-api".into(),
            provider: "kiro".into(),
            base_url: base_url.into(),
            reasoning: fallback.is_some_and(|m| m.reasoning),
            input,
            cost: ModelCost::zero(),
            context_window: context,
            max_tokens,
            ..Default::default()
        });
    }
    Ok(models)
}

/// Explicit discovery; errors remain errors (never present an unverified fallback as an entitlement result).
pub async fn discover(key: &str, base_url: Option<&str>) -> Result<Vec<Model>, String> {
    let access = Access::parse(key)?;
    let client = client()?;
    let root = access.root(base_url.unwrap_or(DEFAULT_ENDPOINT))?;
    let mut models = vec![];
    let mut ids = HashSet::new();
    let mut pages = HashSet::new();
    let mut next: Option<String> = None;
    for _ in 0..100 {
        let mut body = json!({"origin":access.origin()});
        if let Some(profile) = &access.credential.profile_arn {
            body["profileArn"] = json!(profile);
        }
        if let Some(next) = &next {
            body["nextToken"] = json!(next);
        }
        let response = access
            .request(
                &client,
                root.clone(),
                "AmazonCodeWhispererService.ListAvailableModels",
            )
            .timeout(Duration::from_secs(15))
            .json(&body)
            .send()
            .await
            .map_err(|_| "Kiro model discovery network error")?;
        if !response.status().is_success() {
            return Err(response_error(response, &access).await);
        }
        let value: Value = response
            .json()
            .await
            .map_err(|_| "Invalid Kiro catalog JSON")?;
        for model in parse_models(&value, root.as_str())? {
            if ids.insert(model.id.clone()) {
                models.push(model);
            }
        }
        next = value["nextToken"]
            .as_str()
            .filter(|s| !s.is_empty())
            .map(str::to_owned);
        let Some(token) = next.as_ref() else {
            return Ok(models);
        };
        if !pages.insert(token.clone()) {
            return Err("Kiro catalog pagination repeated a token".into());
        }
    }
    Err("Kiro catalog exceeded the pagination limit".into())
}
