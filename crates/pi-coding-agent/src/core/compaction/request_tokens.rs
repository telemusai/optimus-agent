//! Request-level token accounting for the model request (defects D17, D16).
//!
//! `compaction::estimate_tokens` scores one message and
//! `compaction::estimate_context_tokens` sums the message list, but the
//! request a provider receives also carries the system prompt and the tool
//! schema JSON. Pressure checks and the context-health score divide that
//! message-list estimate by the context window, so both understate
//! utilisation while no usable assistant usage exists (D17). This module
//! estimates the full request:
//!
//! - `messages_tokens` reuses `compaction::estimate_tokens` per message, so
//!   the message-level and request-level estimators can never drift apart;
//!   the D16 image-block fix lands once in `estimate_tokens` and is
//!   inherited here.
//! - `system_prompt_tokens` counts the prompt text at the same chars/4 rate.
//! - `tool_schema_tokens` serializes each tool definition (the same
//!   serialize-then-count convention `estimate_tokens` applies to assistant
//!   tool-call arguments) and counts chars/4 per tool.
//!
//! Relationship to `estimate_context_tokens`: with no usable assistant
//! usage its `tokens` field equals the plain `messages_tokens` sum below
//! (its no-usage fallback runs the same per-message loop), so the request
//! parts here are exactly what that fallback omits. With a usable usage it
//! anchors on provider-reported usage instead, and that usage already
//! counts the system prompt and tool schemas the provider tokenized, so
//! the parts here must not be added on top of the usage-anchored form
//! (that would double-count).
//!
//! All functions are pure: no I/O, no provider calls, deterministic output.

use pi_agent_core::types::AgentMessage;
use pi_ai::types::Tool;
use serde::Serialize;

use crate::core::compaction::charcount::{count_chars, count_json_chars};
use crate::core::compaction::compaction::estimate_tokens;

/// Char cost of one image block under the shared estimator convention (D16).
///
/// `estimate_tokens` historically scored user-attached image blocks as zero
/// while `find_cut_point`'s `entry_token_cost` charged custom-message image
/// blocks 4096 chars (1024 tokens at the chars/4 rate). The shared
/// convention: one image block costs [`IMAGE_BLOCK_CHARS`] chars, i.e.
/// [`IMAGE_BLOCK_TOKENS`] tokens, in every estimator that adopts it.
pub const IMAGE_BLOCK_CHARS: usize = 4096;

/// Token cost of one image block: `ceil(IMAGE_BLOCK_CHARS / 4)` = 1024.
pub const IMAGE_BLOCK_TOKENS: f64 = 1024.0;

/// Request-level token estimate: what the provider actually receives.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RequestTokenEstimate {
    /// Sum of `compaction::estimate_tokens` over the message list.
    pub messages_tokens: f64,
    /// System prompt text at the chars/4 rate.
    pub system_prompt_tokens: f64,
    /// Serialized tool schema JSON at the chars/4 rate.
    pub tool_schema_tokens: f64,
    /// `messages_tokens + system_prompt_tokens + tool_schema_tokens`.
    pub total: f64,
}

/// `Math.ceil(chars / 4)` with JavaScript's number semantics, mirroring the
/// private `ceil_div4` in `compaction.rs`; duplicated here because that copy
/// is private to its module.
fn ceil_div4(chars: usize) -> f64 {
    (chars as f64 / 4.0).ceil()
}

/// Token cost of the system prompt text at the chars/4 rate.
pub fn system_prompt_tokens(system_prompt: &str) -> f64 {
    ceil_div4(count_chars(system_prompt))
}

/// Token cost of the tool schema JSON. Each definition is serialized and
/// counted at chars/4, the same serialize-then-count convention
/// `estimate_tokens` applies to assistant tool-call arguments. A definition
/// that fails to serialize counts zero, matching that convention.
pub fn tool_schema_tokens<T: Serialize>(tool_definitions: &[T]) -> f64 {
    tool_definitions
        .iter()
        .map(|tool| {
            // Streamed count of `serde_json::to_string(tool)`; 0 on failure.
            count_json_chars(tool).map(ceil_div4).unwrap_or(0.0)
        })
        .sum()
}

/// Estimate the full request: message list, system prompt and tool schemas.
///
/// The message part delegates to `compaction::estimate_tokens` message by
/// message, so the two estimators stay consistent by construction. See the
/// module docs for how this composes with usage-anchored estimates.
pub fn estimate_request_tokens(
    messages: &[AgentMessage],
    system_prompt: &str,
    tool_definitions: &[Tool],
) -> RequestTokenEstimate {
    let messages_tokens = messages.iter().map(estimate_tokens).sum();
    let system_prompt_tokens = system_prompt_tokens(system_prompt);
    let tool_schema_tokens = tool_schema_tokens(tool_definitions);
    RequestTokenEstimate {
        messages_tokens,
        system_prompt_tokens,
        tool_schema_tokens,
        total: messages_tokens + system_prompt_tokens + tool_schema_tokens,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user_text(text: &str) -> AgentMessage {
        AgentMessage::Message(pi_ai::types::Message::User(pi_ai::types::UserMessage::new(
            pi_ai::types::UserContent::Text(text.into()),
            0,
        )))
    }

    fn user_blocks(blocks: Vec<pi_ai::types::ImageOrTextContent>) -> AgentMessage {
        AgentMessage::Message(pi_ai::types::Message::User(pi_ai::types::UserMessage::new(
            pi_ai::types::UserContent::Blocks(blocks),
            0,
        )))
    }

    fn tool(name: &str, description: &str, parameters: serde_json::Value) -> Tool {
        Tool {
            name: name.into(),
            description: description.into(),
            parameters,
        }
    }

    #[test]
    fn empty_inputs_score_zero() {
        let estimate = estimate_request_tokens(&[], "", &[]);
        assert_eq!(estimate.messages_tokens, 0.0);
        assert_eq!(estimate.system_prompt_tokens, 0.0);
        assert_eq!(estimate.tool_schema_tokens, 0.0);
        assert_eq!(estimate.total, 0.0);
        assert_eq!(system_prompt_tokens(""), 0.0);
        assert_eq!(tool_schema_tokens::<serde_json::Value>(&[]), 0.0);
    }

    #[test]
    fn ascii_follows_chars_per_four() {
        // 100 ASCII chars -> ceil(100/4) = 25 tokens.
        let estimate = estimate_request_tokens(&[user_text(&"a".repeat(100))], "1234", &[]);
        assert_eq!(estimate.messages_tokens, 25.0);
        assert_eq!(estimate.system_prompt_tokens, 1.0);
        assert_eq!(estimate.total, 26.0);
        // Ceil semantics: 5 chars -> 2 tokens, not 1.25.
        assert_eq!(system_prompt_tokens("12345"), 2.0);
        assert_eq!(system_prompt_tokens("1234"), 1.0);
    }

    #[test]
    fn unicode_counts_scalars_not_bytes() {
        // 5 CJK scalars (15 UTF-8 bytes): chars/4 counts scalars -> 2 tokens.
        let estimate = estimate_request_tokens(
            &[user_text("\u{65e5}\u{672c}\u{8a9e}\u{30c6}\u{30b9}")],
            "",
            &[],
        );
        assert_eq!(estimate.messages_tokens, 2.0);
        // A byte-based count would score ceil(15/4) = 4.
        assert_eq!(system_prompt_tokens("\u{65e5}\u{672c}\u{8a9e}"), 1.0);
    }

    #[test]
    fn image_blocks_cost_4096_chars_1024_tokens() {
        // D16: the shared convention every estimator adopts. entry_token_cost
        // already charges custom-message image blocks 4096 chars, so this is
        // the value estimate_tokens must adopt for user-attached images (the
        // wiring lives in compaction.rs; see the W3 integration spec).
        assert_eq!(IMAGE_BLOCK_CHARS, 4096);
        assert_eq!(IMAGE_BLOCK_TOKENS, (IMAGE_BLOCK_CHARS as f64 / 4.0).ceil());
        assert_eq!(IMAGE_BLOCK_TOKENS, 1024.0);
    }

    #[test]
    fn image_messages_delegate_to_estimate_tokens() {
        // The request estimator must not drift from the message estimator:
        // its message part is estimate_tokens applied message by message, so
        // the D16 fix lands once in estimate_tokens and is inherited here.
        // (Pre-fix, estimate_tokens scores user-attached images as zero; the
        // relationship below holds before and after that fix.)
        let image_message = user_blocks(vec![
            pi_ai::types::ImageOrTextContent::Text(pi_ai::types::TextContent::new("caption")),
            pi_ai::types::ImageOrTextContent::Image(Default::default()),
        ]);
        let estimate = estimate_request_tokens(std::slice::from_ref(&image_message), "", &[]);
        assert_eq!(estimate.messages_tokens, estimate_tokens(&image_message));
        assert_eq!(estimate.total, estimate.messages_tokens);
    }

    #[test]
    fn tool_schema_tokens_counts_serialized_json_at_chars_per_four() {
        // {"name":"get_weather"} serializes to 22 chars -> ceil(22/4) = 6.
        let tools = vec![serde_json::json!({"name": "get_weather"})];
        assert_eq!(
            serde_json::to_string(&tools[0]).unwrap(),
            r#"{"name":"get_weather"}"#
        );
        assert_eq!(tool_schema_tokens(&tools), 6.0);
    }

    #[test]
    fn tool_schema_tokens_ceils_per_tool_not_once() {
        // {"a":1} = 7 chars -> 2 tokens; {"bb":22} = 9 chars -> 3 tokens.
        // Per-tool ceils sum to 5, while ceil(16/4) = 4 would undercount.
        let tools = vec![
            serde_json::json!({"a": 1}),
            serde_json::json!({"bb": 22}),
        ];
        assert_eq!(serde_json::to_string(&tools[0]).unwrap(), r#"{"a":1}"#);
        assert_eq!(serde_json::to_string(&tools[1]).unwrap(), r#"{"bb":22}"#);
        assert_eq!(tool_schema_tokens(&tools), 5.0);
    }

    #[test]
    fn tool_schema_tokens_counts_json_scalars_not_bytes() {
        // {"n":"日本"} is 10 scalars -> 3 tokens; a byte count (14
        // UTF-8 bytes) would score 4.
        let tools = vec![serde_json::json!({"n": "\u{65e5}\u{672c}"})];
        assert_eq!(tool_schema_tokens(&tools), 3.0);
    }

    #[test]
    fn real_tool_definitions_flow_through_serialization() {
        let tools = vec![tool(
            "get_weather",
            "Get the current weather for a city.",
            serde_json::json!({
                "type": "object",
                "properties": {"city": {"type": "string"}}
            }),
        )];
        let expected = tool_schema_tokens(&tools);
        assert!(expected > 0.0);
        let estimate = estimate_request_tokens(&[], "", &tools);
        assert_eq!(estimate.tool_schema_tokens, expected);
        assert_eq!(estimate.total, expected);
    }

    #[test]
    fn total_is_exactly_the_sum_of_its_parts() {
        let messages = vec![
            user_text(&"a".repeat(40)),
            user_text("\u{65e5}\u{672c}\u{8a9e}"),
        ];
        let tools = vec![tool("t", "d", serde_json::json!({"a": 1}))];
        let estimate = estimate_request_tokens(&messages, "system prompt text!", &tools);
        assert_eq!(
            estimate.total,
            estimate.messages_tokens + estimate.system_prompt_tokens + estimate.tool_schema_tokens
        );
        assert!(estimate.messages_tokens > 0.0);
        assert!(estimate.system_prompt_tokens > 0.0);
        assert!(estimate.tool_schema_tokens > 0.0);
    }
}
