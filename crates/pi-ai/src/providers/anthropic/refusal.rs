//! Anthropic's server-selected refusal recovery for internal summary requests.
//! https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback

use super::*;
use crate::types::Usage;

const FALLBACK_BETA: &str = "server-side-fallback-2026-07-01";

pub(super) fn enable_default_routing(headers: &mut HeaderBag) {
    let mut betas: Vec<&str> = headers
        .get("anthropic-beta")
        .unwrap_or_default()
        .split(',')
        .map(str::trim)
        .filter(|beta| !beta.is_empty())
        .collect();
    if !betas.contains(&FALLBACK_BETA) {
        betas.push(FALLBACK_BETA);
    }
    headers.set("anthropic-beta", &betas.join(","));
}

pub(super) fn failure(
    raw: Option<&str>,
    request_id: Option<&str>,
    details: Option<&Value>,
) -> StreamFailureError {
    let mut failure = stream_failure_from_stop_reason(raw, request_id);
    if raw == Some("refusal") {
        for key in ["category", "explanation", "recommended_model"] {
            if let Some(text) = details
                .and_then(|details| details.get(key))
                .and_then(Value::as_str)
            {
                let text: String = text
                    .chars()
                    .filter(|ch| !ch.is_control())
                    .take(300)
                    .collect();
                if !text.is_empty() {
                    failure.message.push_str(&format!(" [{key}: {text}]"));
                }
            }
        }
    }
    failure
}

pub(super) fn observe_serving_model(
    output: &mut AssistantMessage,
    requested: &Model,
    model: Option<&Value>,
) {
    let Some(served) = model.and_then(Value::as_str).filter(|id| {
        id.starts_with("claude-")
            && id.len() <= 128
            && id
                .bytes()
                .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, b'-' | b'.'))
    }) else {
        return;
    };
    if served != output.model {
        let mut fields = Map::new();
        fields.insert("requestedModel".into(), Value::String(requested.id.clone()));
        fields.insert("servedModel".into(), Value::String(served.to_string()));
        crate::log::get_logger("ai.provider")
            .info("Anthropic summary recovery model", Some(fields));
        output.model = served.to_string();
    }
}

/// The top-level usage covers only the last attempt. Sum iterations once, using
/// each attempt's model prices, so recovery doesn't undercount summary tokens.
pub(super) fn apply_iteration_usage(
    output: &mut AssistantMessage,
    requested: &Model,
    iterations: Option<&Value>,
    cache: Option<&CacheControlEphemeral>,
) {
    let Some(iterations) = iterations
        .and_then(Value::as_array)
        .filter(|items| !items.is_empty())
    else {
        return;
    };
    let mut total = Usage::zero();
    for iteration in iterations {
        let Some(id) = iteration.get("model").and_then(Value::as_str) else {
            return;
        };
        let mut usage = Usage::zero();
        usage.input = number_or_zero(iteration.get("input_tokens"));
        usage.output = number_or_zero(iteration.get("output_tokens"));
        usage.cache_read = number_or_zero(iteration.get("cache_read_input_tokens"));
        usage.cache_write = number_or_zero(iteration.get("cache_creation_input_tokens"));
        if [
            usage.input,
            usage.output,
            usage.cache_read,
            usage.cache_write,
        ]
        .iter()
        .any(|count| !count.is_finite() || *count < 0.0)
        {
            return;
        }
        let priced_model = if id == requested.id {
            Some(requested)
        } else {
            crate::models::get_model(&requested.provider, id)
        };
        if let Some(model) = priced_model {
            let cache_write = get_anthropic_cache_write_cost(
                model.cost.input,
                if cache.and_then(|cache| cache.ttl.as_deref()) == Some("1h") {
                    "1h"
                } else {
                    "5m"
                },
                parse_cache_creation(iteration.get("cache_creation")).as_ref(),
            );
            calculate_cost(
                model,
                &mut usage,
                Some(&CostOverrides {
                    cache_write: Some(cache_write),
                }),
            );
        } else {
            crate::log::get_logger("ai.provider").warn(
                "Summary recovery model pricing unavailable; cost excludes this attempt",
                None,
            );
        }
        total.input += usage.input;
        total.output += usage.output;
        total.cache_read += usage.cache_read;
        total.cache_write += usage.cache_write;
        total.cost.input += usage.cost.input;
        total.cost.output += usage.cost.output;
        total.cost.cache_read += usage.cost.cache_read;
        total.cost.cache_write += usage.cost.cache_write;
        total.cost.total += usage.cost.total;
    }
    total.total_tokens = total.input + total.output + total.cache_read + total.cache_write;
    output.usage = total;
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn fallback_beta_preserves_oauth_and_existing_betas_once() {
        let mut headers = HeaderBag::default();
        headers.set("Anthropic-Beta", "oauth-2025-04-20,custom-beta");
        enable_default_routing(&mut headers);
        enable_default_routing(&mut headers);
        assert_eq!(
            headers.get("anthropic-beta"),
            Some("oauth-2025-04-20,custom-beta,server-side-fallback-2026-07-01")
        );
    }

    #[test]
    fn refusal_details_are_bounded_without_losing_request_id() {
        let error = failure(
            Some("refusal"),
            Some("req_fixture"),
            Some(&json!({
                "category":"reasoning_extraction", "explanation":"\u{1b}[31m\n".repeat(1000),
                "fallback_credit_token":"never-display-this"
            })),
        );
        assert!(error.message.contains("req_fixture"));
        assert!(error.message.contains("reasoning_extraction"));
        assert!(!error.message.contains("never-display-this"));
        assert!(!error.message.chars().any(char::is_control));
        assert!(error.message.len() < 500);
        let other = failure(
            Some("unknown"),
            None,
            Some(&json!({"category":"ignore-me"})),
        );
        assert!(!other.message.contains("ignore-me"));
    }

    #[test]
    fn iteration_usage_counts_each_model_once_at_its_own_price() {
        let requested = crate::models::get_model("anthropic", "claude-opus-5-5").unwrap();
        let served = crate::models::get_model("anthropic", "claude-opus-4-8").unwrap();
        let mut message = AssistantMessage::default();
        let iterations = json!([
            {"model":requested.id,"input_tokens":100,"output_tokens":3},
            {"model":served.id,"input_tokens":105,"output_tokens":20}
        ]);
        apply_iteration_usage(&mut message, requested, Some(&iterations), None);
        let expected = (100.0 * requested.cost.input
            + 3.0 * requested.cost.output
            + 105.0 * served.cost.input
            + 20.0 * served.cost.output)
            / 1_000_000.0;
        assert!((message.usage.cost.total - expected).abs() < 1e-10);
        assert_eq!(message.usage.total_tokens, 228.0);
        apply_iteration_usage(&mut message, requested, Some(&iterations), None);
        assert_eq!(message.usage.total_tokens, 228.0);
    }

    #[test]
    fn invalid_iteration_usage_keeps_provider_totals() {
        let requested = crate::models::get_model("anthropic", "claude-opus-5-5").unwrap();
        let mut message = AssistantMessage::default();
        message.usage.input = 42.0;
        let original = message.usage.clone();
        for invalid in [
            json!([]),
            json!([{}]),
            json!([{"model":requested.id,"input_tokens":-1}]),
        ] {
            apply_iteration_usage(&mut message, requested, Some(&invalid), None);
            assert_eq!(message.usage, original);
        }
    }
}
