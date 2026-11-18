//! Port of packages/coding-agent/src/core/session-stats.ts

use serde::{Deserialize, Serialize};

/// Local stand-in for `ContextUsage` from `core/extensions/types.ts` (owned by
/// another slice). Field names and absent-vs-null semantics match the TypeScript:
/// `tokens: number | null` and `percent: number | null`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextUsage {
    /// Estimated context tokens, or null if unknown (e.g. right after compaction, before next LLM response).
    pub tokens: Option<f64>,
    #[serde(rename = "contextWindow")]
    pub context_window: f64,
    /// Context usage as percentage of context window, or null if tokens is unknown.
    pub percent: Option<f64>,
}

/// `SessionStats` - `sessionFile` is `undefined` for in-memory sessions, `tokens`
/// fields are plain numbers, and `contextUsage` is optional.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionStats {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_file: Option<String>,
    pub session_id: String,
    pub user_messages: i64,
    pub assistant_messages: i64,
    pub tool_calls: i64,
    pub tool_results: i64,
    pub total_messages: i64,
    pub tokens: SessionStatsTokens,
    pub cost: f64,
    /// Known auxiliary spend is a subtotal, never proof that unreported calls were free.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub textmas_usage: Option<crate::core::occ::textmas::TextMasUsageSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_usage: Option<ContextUsage>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionStatsTokens {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
    pub total: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializes_with_the_typescript_field_names() {
        let stats = SessionStats {
            session_file: None,
            session_id: "abc".to_string(),
            user_messages: 1,
            assistant_messages: 2,
            tool_calls: 3,
            tool_results: 4,
            total_messages: 5,
            tokens: SessionStatsTokens {
                input: 1.0,
                output: 2.0,
                cache_read: 3.0,
                cache_write: 4.0,
                total: 10.0,
            },
            cost: 0.5,
            textmas_usage: None,
            context_usage: None,
        };
        let value = serde_json::to_value(&stats).unwrap();
        assert!(value.get("sessionFile").is_none());
        assert_eq!(value["sessionId"], "abc");
        assert_eq!(value["tokens"]["cacheRead"], 3.0);
        assert!(value.get("contextUsage").is_none());
    }
}
