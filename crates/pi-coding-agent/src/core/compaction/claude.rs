use std::sync::Arc;

use pi_ai::types::{Model, SimpleStreamOptions};
use serde_json::Value;

/// Ask Anthropic to choose its permitted recovery model for a refused summary.
/// This applies to the summary request, never to the session's selected model.
pub(super) fn enable_summary_recovery(model: &Model, options: &mut SimpleStreamOptions) {
    if model.provider != "anthropic"
        || model.api != "anthropic-messages"
        || !matches!(
            model.id.as_str(),
            "claude-opus-5-5"
                | "claude-opus-5"
                | "claude-fable-5"
                | "claude-fable-5-1"
                | "claude-sonnet-5-5"
        )
    {
        return;
    }
    let caller = options.stream.on_payload.clone();
    options.stream.on_payload = Some(Arc::new(move |payload, model| {
        let original = payload.clone();
        let future = caller.as_ref().map(|caller| caller(payload, model));
        Box::pin(async move {
            let mut payload = match future {
                Some(future) => future.await.unwrap_or(original),
                None => original,
            };
            if let Some(object) = payload.as_object_mut() {
                object.insert("fallbacks".into(), Value::String("default".into()));
            }
            Some(payload)
        })
    }));
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn summary_recovery_composes_with_existing_payload_hooks() {
        let model = Model::new(
            "claude-opus-5-5",
            "Fixture",
            "anthropic-messages",
            "anthropic",
            "http://localhost",
        );
        for replacement in [
            None,
            Some(json!({"model":"claude-opus-5-5", "existing":"preserved"})),
        ] {
            let mut options = SimpleStreamOptions::default();
            let supplied = replacement.clone();
            options.stream.on_payload = Some(Arc::new(move |_, _| {
                let supplied = supplied.clone();
                Box::pin(async move { supplied })
            }));
            enable_summary_recovery(&model, &mut options);
            let result = options.stream.on_payload.unwrap()(
                json!({"model":model.id, "original":"preserved"}),
                &model,
            )
            .await
            .unwrap();
            assert_eq!(result["fallbacks"], "default");
            assert_eq!(
                result[if replacement.is_some() {
                    "existing"
                } else {
                    "original"
                }],
                "preserved"
            );
        }
    }
}
