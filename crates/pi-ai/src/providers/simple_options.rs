//! Port of packages/ai/src/providers/simple-options.ts
use crate::types::{Model, SimpleStreamOptions, StreamOptions, ThinkingBudgets, ThinkingLevel};

/// TS: `buildBaseOptions(model, options?, apiKey?)`
pub fn build_base_options(model: &Model, options: Option<&SimpleStreamOptions>, api_key: Option<&str>) -> StreamOptions {
	let mut base = StreamOptions::default();
	base.temperature = options.and_then(|o| o.stream.temperature);
	base.max_tokens = if let Some(default_max_tokens) = model.default_max_tokens {
		// An explicit default opts into a separate default/ceiling policy. Keep
		// invalid values visible to the provider's validation rather than hiding
		// them with f64::min (which would silently replace NaN).
		let requested = options.and_then(|o| o.stream.max_tokens).unwrap_or(default_max_tokens);
		Some(if requested.is_finite() && model.max_tokens.is_finite() && model.max_tokens > 0.0 {
			requested.min(model.max_tokens)
		} else {
			requested
		})
	} else {
		options
			.and_then(|o| o.stream.max_tokens)
			.or_else(|| {
				(model.max_tokens > 0.0).then(|| {
					if model.max_tokens_explicit { model.max_tokens } else { model.max_tokens.min(32000.0) }
				})
			})
	};
	base.signal = options.and_then(|o| o.stream.signal.clone());
	// TS: `apiKey || options?.apiKey`
	base.api_key = match api_key {
		Some(key) if !key.is_empty() => Some(key.to_string()),
		_ => options.and_then(|o| o.stream.api_key.clone()),
	};
	base.transport = options.and_then(|o| o.stream.transport.clone());
	base.service_tier = options.and_then(|o| o.stream.service_tier.clone());
	base.cache_retention = options.and_then(|o| o.stream.cache_retention.clone());
	base.session_id = options.and_then(|o| o.stream.session_id.clone());
	base.headers = options.and_then(|o| o.stream.headers.clone());
	// Local-only callbacks must survive simple -> provider option conversion.
	base.on_usage_observation = options.and_then(|o| o.stream.on_usage_observation.clone());
	base.on_stream_observation = options.and_then(|o| o.stream.on_stream_observation.clone());
	base.on_payload = options.and_then(|o| o.stream.on_payload.clone());
	base.on_response = options.and_then(|o| o.stream.on_response.clone());
	base.timeout_ms = options.and_then(|o| o.stream.timeout_ms);
	base.metadata = options.and_then(|o| o.stream.metadata.clone());
	base
}

/// TS: `clampReasoning(effort)`
pub fn clamp_reasoning(effort: Option<&ThinkingLevel>) -> Option<ThinkingLevel> {
	match effort {
		Some(level) if level == "xhigh" || level == "max" => Some("high".to_string()),
		Some(level) => Some(level.clone()),
		None => None,
	}
}

/// Result of `adjustMaxTokensForThinking`.
#[derive(Debug, Clone, PartialEq)]
pub struct AdjustedMaxTokens {
	pub max_tokens: f64,
	pub thinking_budget: f64,
}

/// TS: `adjustMaxTokensForThinking(baseMaxTokens, modelMaxTokens, reasoningLevel, customBudgets?)`
pub fn adjust_max_tokens_for_thinking(
	base_max_tokens: f64,
	model_max_tokens: f64,
	reasoning_level: &ThinkingLevel,
	custom_budgets: Option<&ThinkingBudgets>,
) -> AdjustedMaxTokens {
	let mut budgets = ThinkingBudgets {
		minimal: Some(1024.0),
		low: Some(2048.0),
		medium: Some(8192.0),
		high: Some(16384.0),
	};
	if let Some(custom) = custom_budgets {
		if custom.minimal.is_some() {
			budgets.minimal = custom.minimal;
		}
		if custom.low.is_some() {
			budgets.low = custom.low;
		}
		if custom.medium.is_some() {
			budgets.medium = custom.medium;
		}
		if custom.high.is_some() {
			budgets.high = custom.high;
		}
	}

	let min_output_tokens = 1024.0;
	let level = clamp_reasoning(Some(reasoning_level)).expect("clamped level");
	let mut thinking_budget = match level.as_str() {
		"minimal" => budgets.minimal.unwrap_or(0.0),
		"low" => budgets.low.unwrap_or(0.0),
		"medium" => budgets.medium.unwrap_or(0.0),
		"high" => budgets.high.unwrap_or(0.0),
		_ => 0.0,
	};
	let max_tokens = (base_max_tokens + thinking_budget).min(model_max_tokens);

	if max_tokens <= thinking_budget {
		thinking_budget = (max_tokens - min_output_tokens).max(0.0);
	}

	AdjustedMaxTokens { max_tokens, thinking_budget }
}

#[cfg(test)]
mod tests {
	use super::*;
	use std::sync::Arc;

	fn model_with_max_tokens(max_tokens: f64) -> Model {
		Model {
			max_tokens,
			..Default::default()
		}
	}

	#[test]
	fn base_options_clamps_max_tokens_to_32000() {
		let model = model_with_max_tokens(100_000.0);
		let options = build_base_options(&model, None, None);
		assert_eq!(options.max_tokens, Some(32000.0));
	}

	#[test]
	fn base_options_honors_configured_model_limit_and_request_override() {
		let model = Model { max_tokens_explicit: true, ..model_with_max_tokens(128_000.0) };
		assert_eq!(build_base_options(&model, None, None).max_tokens, Some(128_000.0));
		let mut simple = SimpleStreamOptions::default();
		simple.stream.max_tokens = Some(64_000.0);
		assert_eq!(build_base_options(&model, Some(&simple), None).max_tokens, Some(64_000.0));
	}

	#[test]
	fn base_options_keeps_smaller_model_max_tokens() {
		let model = model_with_max_tokens(4096.0);
		let options = build_base_options(&model, None, None);
		assert_eq!(options.max_tokens, Some(4096.0));
	}

	#[test]
	fn base_options_omits_max_tokens_when_model_reports_zero() {
		let model = model_with_max_tokens(0.0);
		let options = build_base_options(&model, None, None);
		assert_eq!(options.max_tokens, None);
	}

	#[test]
	fn base_options_forwards_usage_observation_without_serializing_it() {
		let model = model_with_max_tokens(1000.0);
		let mut simple = SimpleStreamOptions::default();
		simple.stream.on_usage_observation = Some(Arc::new(
			|_observation: crate::types::ProviderUsageObservation, _model: &Model| {
				Box::pin(async {}) as crate::types::BoxFuture<()>
			},
		));
		let options = build_base_options(&model, Some(&simple), None);
		assert!(
			options.on_usage_observation.is_some(),
			"buildBaseOptions must forward onUsageObservation"
		);
		assert!(simple.stream.on_usage_observation.is_some());
		let wire_with_callback = serde_json::to_value(&options).unwrap();
		let mut without_callback = options.clone();
		without_callback.on_usage_observation = None;
		assert_eq!(wire_with_callback, serde_json::to_value(without_callback).unwrap());
	}

	#[test]
	fn base_options_prefers_explicit_api_key() {
		let model = model_with_max_tokens(1000.0);
		let mut simple = SimpleStreamOptions::default();
		simple.stream.api_key = Some("from-options".to_string());
		let options = build_base_options(&model, Some(&simple), Some("explicit"));
		assert_eq!(options.api_key.as_deref(), Some("explicit"));
		let options = build_base_options(&model, Some(&simple), None);
		assert_eq!(options.api_key.as_deref(), Some("from-options"));
	}

	#[test]
	fn clamp_reasoning_maps_xhigh_and_max_to_high() {
		assert_eq!(clamp_reasoning(Some(&"xhigh".to_string())), Some("high".to_string()));
		assert_eq!(clamp_reasoning(Some(&"max".to_string())), Some("high".to_string()));
		assert_eq!(clamp_reasoning(Some(&"low".to_string())), Some("low".to_string()));
		assert_eq!(clamp_reasoning(None), None);
	}

	#[test]
	fn adjust_max_tokens_uses_default_budgets() {
		let adjusted = adjust_max_tokens_for_thinking(8000.0, 64000.0, &"medium".to_string(), None);
		assert_eq!(adjusted.max_tokens, 16192.0);
		assert_eq!(adjusted.thinking_budget, 8192.0);
	}

	#[test]
	fn adjust_max_tokens_shrinks_budget_when_model_limit_hits() {
		let adjusted = adjust_max_tokens_for_thinking(1000.0, 4000.0, &"high".to_string(), None);
		assert_eq!(adjusted.max_tokens, 4000.0);
		assert_eq!(adjusted.thinking_budget, 2976.0);
	}

	#[test]
	fn adjust_max_tokens_respects_custom_budgets() {
		let custom = ThinkingBudgets {
			minimal: Some(10.0),
			low: Some(20.0),
			medium: Some(30.0),
			high: Some(40.0),
		};
		let adjusted = adjust_max_tokens_for_thinking(100.0, 100_000.0, &"low".to_string(), Some(&custom));
		assert_eq!(adjusted.max_tokens, 120.0);
		assert_eq!(adjusted.thinking_budget, 20.0);
	}
}
