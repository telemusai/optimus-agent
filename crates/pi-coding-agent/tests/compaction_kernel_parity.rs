//! TASK 09 — differential/property guard tests for the local compaction kernels.
//!
//! These tests pin the CURRENT (pristine) semantics of the pure local kernels of the
//! auto-compaction path so the optimization phase is guarded by tests that existed
//! before it. They must pass on unmodified main; any later failure means an
//! optimization changed observable behavior.
//!
//! What is pinned here:
//! - `estimate_tokens`: exact per-variant char accounting (Unicode scalar values via
//!   `chars().count()`), the `ceil(chars / 4)` f64 formula, the 4800-per-image rule,
//!   the `provider_context` override, and the `serde_json::to_string` form used for
//!   assistant tool-call arguments.
//! - `estimate_context_tokens`: last-assistant-usage selection, usage + trailing
//!   arithmetic, and the full-estimate fallback.
//! - `should_compact` / `should_compact_for_model`: inclusive `>=` threshold
//!   boundaries, the 250k cap, the Azure GPT 400k cap, window/reserve edge cases.
//! - `truncate_for_summary` (private): pinned through its only public caller,
//!   `serialize_conversation`, with exact golden strings including the truncation
//!   marker, head/tail selection, and char-vs-byte boundary behavior.
//! - `serialize_conversation`: golden outputs for a deterministic corpus.
//! - `generate_bounded_summary` chunking (private `slice_chars`): pinned through the
//!   public `generate_summary` seam with a recorded fixture provider — chunk covering,
//!   offset advancement, previous-summary chaining, and the exact chunk-budget
//!   formula `floor((min(input_limit, window - retry_max) - 1024) * 3) - suffix -
//!   system prompt - 64`.
//! - Cancellation: the "Aborted" sentinel through `generate_summary` and `compact`
//!   with a pre-cancelled token.
//!
//! Honest gaps (NOT covered here, covered elsewhere or unreachable from tests/):
//! - `should_compact_with_cap` directly (private): exercised only through
//!   `should_compact` (250k cap) and `should_compact_for_model` (400k Azure path).
//! - `truncate_for_summary`'s tail-only fallback branch (`max_chars <= marker_max +
//!   tail_chars`): unreachable through `serialize_conversation` because the only call
//!   site uses the fixed 2000-char limit; needs an in-crate unit test.
//! - Mid-wire cancellation (`tokio::select!` around the summary call) and
//!   `AgentSession::abort_compaction`: covered by `compaction_parity_suite.rs`
//!   (SummaryGate tests); not duplicated here.
//! - The length-retry ladder, native compaction arm, split-turn double summary, and
//!   persistence/restore: covered by the existing compaction_* suites.
//!
//! No production code is modified by this file.

use std::sync::{Arc, Mutex, OnceLock};

use pi_agent_core::types::{
    AgentMessage, ContentBlock as CoreContentBlock, CustomAgentMessage, CustomMessageContent,
};
use pi_ai::api_registry::{register_api_provider_simple, ApiProviderSimple, SimpleStreamFunction};
use pi_ai::types::StreamFunction;
use pi_ai::compaction::ProviderCompactionCheckpoint;
use pi_ai::models::get_model_input_limit;
use pi_ai::types::{
    AssistantMessage, AssistantMessageEvent, ContentBlock as AiContentBlock, ImageContent,
    ImageOrTextContent, Message, Model, TextContent, ThinkingContent, ToolCall, ToolResultMessage,
    UserContent, UserMessage, Usage,
};
use pi_ai::utils::event_stream::AssistantMessageEventStream;
use pi_coding_agent::core::compaction::compaction::{
    build_summarization_prompt, calculate_context_tokens, compact, default_compaction_settings,
    default_summary_call_runner, estimate_context_tokens, estimate_tokens, generate_summary,
    prepare_compaction, should_compact, should_compact_for_model, CompactionSessionEntry,
    CompactionSettings, MAX_COMPACTION_CONTEXT_TOKENS,
};
use pi_coding_agent::core::compaction::utils::{
    serialize_conversation, strip_file_operations, SUMMARIZATION_SYSTEM_PROMPT,
};
use pi_coding_agent::core::messages::convert_to_llm;
use serde_json::{json, Map, Value};

// ---------------------------------------------------------------------------
// Reference implementations (independent mirrors of the pinned semantics)
// ---------------------------------------------------------------------------

/// `(chars as f64 / 4.0).ceil()` — JavaScript number semantics.
fn ceil_div4_reference(chars: usize) -> f64 {
    (chars as f64 / 4.0).ceil()
}

fn reference_estimate_tokens(message: &AgentMessage) -> f64 {
    match message {
        AgentMessage::Message(Message::User(user)) => {
            if let Some(provider_context) = &user.provider_context {
                return provider_context.estimated_tokens;
            }
            let chars = match &user.content {
                UserContent::Text(text) => text.chars().count(),
                UserContent::Blocks(blocks) => blocks
                    .iter()
                    .filter_map(|block| match block {
                        ImageOrTextContent::Text(text) => Some(text.text.chars().count()),
                        ImageOrTextContent::Image(_) => None,
                    })
                    .sum(),
            };
            ceil_div4_reference(chars)
        }
        AgentMessage::Message(Message::Assistant(assistant)) => {
            let mut chars = 0usize;
            for block in &assistant.content {
                match block {
                    AiContentBlock::Text(text) => chars += text.text.chars().count(),
                    AiContentBlock::Thinking(thinking) => {
                        chars += thinking.thinking.chars().count()
                    }
                    AiContentBlock::ToolCall(tool_call) => {
                        chars += tool_call.name.chars().count();
                        chars += serde_json::to_string(&tool_call.arguments)
                            .map(|serialized| serialized.chars().count())
                            .unwrap_or(0);
                    }
                }
            }
            ceil_div4_reference(chars)
        }
        AgentMessage::Message(Message::ToolResult(tool_result)) => {
            let mut chars = 0usize;
            for block in &tool_result.content {
                match block {
                    ImageOrTextContent::Text(text) => chars += text.text.chars().count(),
                    ImageOrTextContent::Image(_) => chars += 4800,
                }
            }
            ceil_div4_reference(chars)
        }
        AgentMessage::Custom(CustomAgentMessage::BashExecution {
            command, output, ..
        }) => ceil_div4_reference(command.chars().count() + output.chars().count()),
        AgentMessage::Custom(CustomAgentMessage::Custom { content, .. }) => {
            let chars = match content {
                CustomMessageContent::Text(text) => text.chars().count(),
                CustomMessageContent::Blocks(blocks) => blocks
                    .iter()
                    .map(|block| match block {
                        CoreContentBlock::Text(text) => text.text.chars().count(),
                        CoreContentBlock::Image(_) => 4800,
                    })
                    .sum(),
            };
            ceil_div4_reference(chars)
        }
        AgentMessage::Custom(CustomAgentMessage::BranchSummary { summary, .. }) => {
            ceil_div4_reference(summary.chars().count())
        }
        AgentMessage::Custom(CustomAgentMessage::CompactionSummary {
            summary,
            provider_context,
            harness_digest,
            ..
        }) => {
            if let Some(provider_context) = provider_context {
                let digest_tokens = harness_digest
                    .as_ref()
                    .map(|digest| ceil_div4_reference(digest.chars().count()))
                    .unwrap_or(0.0);
                return provider_context.estimated_tokens + digest_tokens;
            }
            let mut chars = summary.chars().count();
            if let Some(digest) = harness_digest {
                chars += digest.chars().count();
            }
            ceil_div4_reference(chars)
        }
    }
}

fn reference_calculate_context_tokens(usage: &Usage) -> f64 {
    if usage.total_tokens != 0.0 {
        usage.total_tokens
    } else {
        usage.input + usage.output + usage.cache_read + usage.cache_write
    }
}

fn reference_estimate_context_tokens(messages: &[AgentMessage]) -> (f64, f64, f64, Option<usize>) {
    let last_usage = messages.iter().enumerate().rev().find_map(|(index, message)| {
        let AgentMessage::Message(Message::Assistant(assistant)) = message else {
            return None;
        };
        if assistant.stop_reason != "aborted" && assistant.stop_reason != "error" {
            Some((index, assistant.usage.clone()))
        } else {
            None
        }
    });
    let Some((index, usage)) = last_usage else {
        let mut estimated = 0.0;
        for message in messages {
            estimated += reference_estimate_tokens(message);
        }
        return (estimated, 0.0, estimated, None);
    };
    let usage_tokens = reference_calculate_context_tokens(&usage);
    let mut trailing_tokens = 0.0;
    for message in messages.iter().skip(index + 1) {
        trailing_tokens += reference_estimate_tokens(message);
    }
    (usage_tokens + trailing_tokens, usage_tokens, trailing_tokens, Some(index))
}

// ---------------------------------------------------------------------------
// Message constructors
// ---------------------------------------------------------------------------

fn user_text(content: &str) -> AgentMessage {
    AgentMessage::Message(Message::User(UserMessage::new(
        UserContent::Text(content.to_string()),
        1,
    )))
}

fn user_blocks(blocks: Vec<ImageOrTextContent>, provider_context: Option<f64>) -> AgentMessage {
    let mut message = UserMessage::new(UserContent::Blocks(blocks), 2);
    message.provider_context = provider_context.map(|estimated_tokens| checkpoint(estimated_tokens));
    AgentMessage::Message(Message::User(message))
}

fn assistant(blocks: Vec<AiContentBlock>, stop_reason: &str, usage: Usage) -> AgentMessage {
    AgentMessage::Message(Message::Assistant(AssistantMessage {
        content: blocks,
        stop_reason: stop_reason.to_string(),
        usage,
        ..Default::default()
    }))
}

fn tool_result(id: &str, name: &str, blocks: Vec<ImageOrTextContent>, is_error: bool) -> AgentMessage {
    AgentMessage::Message(Message::ToolResult(ToolResultMessage::new(
        id, name, blocks, is_error, 3,
    )))
}

fn bash_execution(command: &str, output: &str) -> AgentMessage {
    AgentMessage::Custom(CustomAgentMessage::BashExecution {
        command: command.to_string(),
        output: output.to_string(),
        exit_code: Some(0),
        cancelled: false,
        truncated: false,
        full_output_path: None,
        timestamp: 4,
        exclude_from_context: None,
    })
}

fn custom_text(content: &str) -> AgentMessage {
    AgentMessage::Custom(CustomAgentMessage::Custom {
        custom_type: "note".to_string(),
        content: CustomMessageContent::Text(content.to_string()),
        display: true,
        details: None,
        timestamp: 5,
    })
}

fn custom_blocks(blocks: Vec<CoreContentBlock>) -> AgentMessage {
    AgentMessage::Custom(CustomAgentMessage::Custom {
        custom_type: "note".to_string(),
        content: CustomMessageContent::Blocks(blocks),
        display: true,
        details: None,
        timestamp: 5,
    })
}

fn branch_summary(summary: &str) -> AgentMessage {
    AgentMessage::Custom(CustomAgentMessage::BranchSummary {
        summary: summary.to_string(),
        from_id: "from".to_string(),
        timestamp: 6,
    })
}

fn compaction_summary(
    summary: &str,
    provider_context: Option<f64>,
    harness_digest: Option<&str>,
) -> AgentMessage {
    AgentMessage::Custom(CustomAgentMessage::CompactionSummary {
        summary: summary.to_string(),
        provider_context: provider_context.map(checkpoint),
        tokens_before: 100.0,
        retained_message_count: None,
        custom_instructions: None,
        harness_digest: harness_digest.map(str::to_string),
        timestamp: 7,
    })
}

fn checkpoint(estimated_tokens: f64) -> ProviderCompactionCheckpoint {
    ProviderCompactionCheckpoint {
        version: 1,
        provider: "fixture".to_string(),
        api: "fixture-api".to_string(),
        model: "fixture-model".to_string(),
        base_url: "http://fixture.invalid".to_string(),
        endpoint: None,
        items: vec![Map::new()],
        estimated_tokens,
    }
}

fn text_block(text: &str) -> ImageOrTextContent {
    ImageOrTextContent::Text(TextContent::new(text))
}

fn image_block() -> ImageOrTextContent {
    ImageOrTextContent::Image(ImageContent::new("synthetic", "image/png"))
}

fn usage(total: f64, input: f64, output: f64, cache_read: f64, cache_write: f64) -> Usage {
    Usage {
        input,
        output,
        cache_read,
        cache_write,
        total_tokens: total,
        cost: Default::default(),
    }
}

// ---------------------------------------------------------------------------
// Deterministic PRNG (splitmix64; fixed seeds, no external dependency)
// ---------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, bound: u64) -> u64 {
        self.next_u64() % bound.max(1)
    }

    fn string(&mut self, max_chars: usize) -> String {
        // Mixed scripts: 1-, 2-, 3- and 4-byte chars, combining marks, NUL, CRLF.
        const UNITS: [&str; 9] = ["a", "Z", "0", " ", "\n", "\r\n", "\0", "界", "𝕏"];
        let target = self.below(max_chars as u64) as usize + 1;
        let mut text = String::new();
        while text.chars().count() < target {
            let unit = UNITS[self.below(UNITS.len() as u64) as usize];
            text.push_str(unit);
        }
        if self.below(4) == 0 {
            // A combining mark that mutates the previously pushed scalar.
            text.push('\u{0301}');
        }
        text
    }

    fn value(&mut self, depth: u64) -> Value {
        match self.below(if depth == 0 { 4 } else { 6 }) {
            0 => Value::Bool(self.below(2) == 0),
            1 => Value::from(self.next_u64() as i64),
            2 => Value::from((self.next_u64() % 10_000) as f64 / 8.0),
            3 => Value::String(self.string(24)),
            4 => Value::Array((0..self.below(4)).map(|_| self.value(depth - 1)).collect()),
            _ => {
                let mut map = Map::new();
                for _ in 0..self.below(4) {
                    map.insert(format!("k{}", self.below(100)), self.value(depth - 1));
                }
                Value::Object(map)
            }
        }
    }

    fn args_map(&mut self) -> Map<String, Value> {
        let mut map = Map::new();
        for _ in 0..self.below(5) {
            map.insert(format!("arg{}", self.below(50)), self.value(2));
        }
        map
    }

    fn message(&mut self) -> AgentMessage {
        match self.below(11) {
            0 => user_text(&self.string(300)),
            1 => {
                let blocks = (0..self.below(4))
                    .map(|_| {
                        if self.below(3) == 0 {
                            image_block()
                        } else {
                            text_block(&self.string(200))
                        }
                    })
                    .collect();
                user_blocks(blocks, None)
            }
            2 => {
                let blocks = (0..self.below(4))
                    .map(|_| {
                        if self.below(3) == 0 {
                            image_block()
                        } else {
                            text_block(&self.string(200))
                        }
                    })
                    .collect();
                user_blocks(blocks, Some(self.next_u64() as f64 % 90_000.0))
            }
            3 | 4 => {
                let mut blocks = Vec::new();
                for _ in 0..self.below(5) {
                    match self.below(4) {
                        0 => blocks.push(AiContentBlock::Thinking(ThinkingContent::new(
                            self.string(200),
                        ))),
                        1 => blocks.push(AiContentBlock::Text(TextContent::new(
                            self.string(300),
                        ))),
                        _ => blocks.push(AiContentBlock::ToolCall(ToolCall::new(
                            format!("call-{}", self.below(10)),
                            if self.below(2) == 0 { "edit" } else { "ipython" },
                            self.args_map(),
                        ))),
                    }
                }
                assistant(
                    blocks,
                    if self.below(4) == 0 { "stop" } else { "stop" },
                    usage(
                        self.next_u64() as f64 % 5_000.0,
                        self.next_u64() as f64 % 2_000.0,
                        self.next_u64() as f64 % 1_000.0,
                        0.0,
                        0.0,
                    ),
                )
            }
            5 => tool_result(
                &format!("call-{}", self.below(10)),
                "ipython",
                (0..self.below(4))
                    .map(|_| {
                        if self.below(3) == 0 {
                            image_block()
                        } else {
                            text_block(&self.string(400))
                        }
                    })
                    .collect(),
                self.below(2) == 0,
            ),
            6 => bash_execution(&self.string(120), &self.string(500)),
            7 => custom_text(&self.string(300)),
            8 => {
                let blocks = (0..self.below(4))
                    .map(|_| {
                        if self.below(3) == 0 {
                            CoreContentBlock::Image(ImageContent::new("synthetic", "image/png"))
                        } else {
                            CoreContentBlock::text(self.string(200))
                        }
                    })
                    .collect();
                custom_blocks(blocks)
            }
            9 => branch_summary(&self.string(300)),
            _ => {
                let provider_context = if self.below(2) == 0 {
                    Some(self.next_u64() as f64 % 200_000.0)
                } else {
                    None
                };
                let harness_digest = if self.below(2) == 0 {
                    Some(self.string(400))
                } else {
                    None
                };
                compaction_summary(
                    &self.string(300),
                    provider_context,
                    harness_digest.as_deref(),
                )
            }
        }
    }
}

const SEEDS: [u64; 8] = [
    0x00C0FFEE_00000001,
    0x00C0FFEE_00000002,
    0x00C0FFEE_00000003,
    0x00C0FFEE_00000004,
    0x5EED_5EED_00000005,
    0x5EED_5EED_00000006,
    0x5EED_5EED_00000007,
    0x5EED_5EED_00000008,
];

// ---------------------------------------------------------------------------
// 1. estimate_tokens
// ---------------------------------------------------------------------------

#[test]
fn estimate_tokens_pins_exact_values_for_documented_cases() {
    // Empty and 1..=4 chars collapse to the same ceil-div-4 buckets.
    assert_eq!(estimate_tokens(&user_text("")), 0.0);
    assert_eq!(estimate_tokens(&user_text("a")), 1.0);
    assert_eq!(estimate_tokens(&user_text("ab")), 1.0);
    assert_eq!(estimate_tokens(&user_text("abc")), 1.0);
    assert_eq!(estimate_tokens(&user_text("abcd")), 1.0);
    assert_eq!(estimate_tokens(&user_text("abcde")), 2.0);
    assert_eq!(estimate_tokens(&user_text(&"x".repeat(1_000_000))), 250_000.0);
    // A 1M+3 char string crosses the bucket edge: ceil(1000003/4) = 250001.
    assert_eq!(
        estimate_tokens(&user_text(&"x".repeat(1_000_003))),
        250_001.0
    );

    // Multi-byte chars count as Unicode scalar values, not bytes.
    // Three CJK chars (9 bytes) are 3 chars -> ceil(3/4) = 1.
    assert_eq!(estimate_tokens(&user_text("界界界")), 1.0);
    // Five CJK chars -> ceil(5/4) = 2.
    assert_eq!(estimate_tokens(&user_text("界界界界界")), 2.0);
    // A 4-byte emoji is a single char.
    assert_eq!(estimate_tokens(&user_text("𝕏𝕏𝕏")), 1.0);
    assert_eq!(estimate_tokens(&user_text("𝕏𝕏𝕏𝕏")), 1.0);
    assert_eq!(estimate_tokens(&user_text("𝕏𝕏𝕏𝕏𝕏")), 2.0);
    // A combining mark is its own scalar value: 'e' + U+0301 = 2 chars.
    assert_eq!(estimate_tokens(&user_text("e\u{0301}")), 1.0);
    assert_eq!(estimate_tokens(&user_text("e\u{0301}e\u{0301}")), 1.0);
    // 5 pairs = 10 scalars -> ceil(10/4) = 3.
    assert_eq!(estimate_tokens(&user_text("e\u{0301}e\u{0301}e\u{0301}e\u{0301}e\u{0301}")), 3.0);
    // Embedded NUL counts as one char; CRLF is two chars.
    assert_eq!(estimate_tokens(&user_text("a\0b")), 1.0);
    assert_eq!(estimate_tokens(&user_text("\r\n")), 1.0);
    // 5 CRLFs = 10 chars -> ceil(10/4) = 3 (crosses the 2.5 bucket edge).
    assert_eq!(estimate_tokens(&user_text("\r\n\r\n\r\n\r\n\r\n")), 3.0);

    // Reference agreement on all of the above.
    for text in [
        "", "a", "ab", "abc", "abcd", "abcde", "界界界", "界界界界界", "𝕏𝕏𝕏",
        "e\u{0301}", "a\0b", "\r\n",
    ] {
        let message = user_text(text);
        assert_eq!(
            estimate_tokens(&message),
            reference_estimate_tokens(&message),
            "reference disagreement for {text:?}"
        );
    }
}

#[test]
fn estimate_tokens_user_blocks_skip_images_and_provider_context_overrides() {
    // User blocks: only text blocks contribute; images are skipped entirely
    // (no 4800 rule here — that applies to tool results and custom blocks).
    let blocks = user_blocks(
        vec![text_block("abcd"), image_block(), text_block("efgh")],
        None,
    );
    assert_eq!(estimate_tokens(&blocks), 2.0);
    let only_image = user_blocks(vec![image_block()], None);
    assert_eq!(estimate_tokens(&only_image), 0.0);

    // provider_context.estimated_tokens replaces the whole estimate, verbatim,
    // with no ceil-div applied.
    let override_message = user_blocks(vec![text_block(&"x".repeat(40_000))], Some(12345.75));
    assert_eq!(estimate_tokens(&override_message), 12345.75);
    let zero_override = user_blocks(vec![text_block(&"x".repeat(40_000))], Some(0.0));
    assert_eq!(estimate_tokens(&zero_override), 0.0);
    // The override wins even when the text is small.
    let small_override = user_blocks(vec![text_block("ab")], Some(3.25));
    assert_eq!(estimate_tokens(&small_override), 3.25);
}

#[test]
fn estimate_tokens_assistant_counts_text_thinking_and_serialized_arguments() {
    // Text and thinking blocks both contribute their char counts.
    let text_and_thinking = assistant(
        vec![
            AiContentBlock::Text(TextContent::new("abcd")),
            AiContentBlock::Thinking(ThinkingContent::new("efg")),
        ],
        "stop",
        Usage::default(),
    );
    assert_eq!(estimate_tokens(&text_and_thinking), 2.0);

    // Tool calls contribute name chars plus the exact serde_json::to_string form
    // of the arguments map (compact separators, insertion order preserved).
    let mut arguments = Map::new();
    arguments.insert("path".to_string(), json!("a.ts"));
    arguments.insert("count".to_string(), json!(2));
    arguments.insert("nested".to_string(), json!({"k": [1, 2]}));
    let serialized = serde_json::to_string(&arguments).unwrap();
    assert_eq!(serialized, r#"{"path":"a.ts","count":2,"nested":{"k":[1,2]}}"#);
    assert_eq!(serialized.chars().count(), 46);
    // name "edit" (4 chars) + 46 serialized chars = 50 chars -> ceil(50/4) = 13.
    let tool_call_message = assistant(
        vec![AiContentBlock::ToolCall(ToolCall::new("id", "edit", arguments))],
        "stop",
        Usage::default(),
    );
    assert_eq!(estimate_tokens(&tool_call_message), 13.0);

    // Empty arguments serialize to "{}" (2 chars).
    let empty_args = assistant(
        vec![AiContentBlock::ToolCall(ToolCall::new("id", "ab", Map::new()))],
        "stop",
        Usage::default(),
    );
    assert_eq!(estimate_tokens(&empty_args), 1.0);

    // Empty assistant content.
    let empty = assistant(Vec::new(), "stop", Usage::default());
    assert_eq!(estimate_tokens(&empty), 0.0);

    // Usage never influences the estimate.
    let with_usage = assistant(
        vec![AiContentBlock::Text(TextContent::new("abcd"))],
        "stop",
        usage(999_999.0, 1.0, 1.0, 0.0, 0.0),
    );
    assert_eq!(estimate_tokens(&with_usage), 1.0);
}

#[test]
fn estimate_tokens_tool_results_count_images_as_4800_chars() {
    let one_image = tool_result("id", "ipython", vec![image_block()], false);
    assert_eq!(estimate_tokens(&one_image), 1_200.0);
    let two_images = tool_result(
        "id",
        "ipython",
        vec![image_block(), image_block()],
        false,
    );
    assert_eq!(estimate_tokens(&two_images), 2_400.0);
    let images_and_text = tool_result(
        "id",
        "bash",
        vec![image_block(), text_block("abcdefgh"), image_block()],
        true,
    );
    // 4800 + 8 + 4800 = 9608 chars -> ceil(9608/4) = 2402.
    assert_eq!(estimate_tokens(&images_and_text), 2_402.0);
    let text_only = tool_result("id", "bash", vec![text_block("界界界界界")], false);
    assert_eq!(estimate_tokens(&text_only), 2.0);
}

#[test]
fn estimate_tokens_custom_variants_pin_their_own_char_rules() {
    // bashExecution: command + output chars, ceil-div-4.
    let bash = bash_execution("ls -la", "total 0");
    assert_eq!(estimate_tokens(&bash), 4.0);
    let empty_bash = bash_execution("", "");
    assert_eq!(estimate_tokens(&empty_bash), 0.0);

    // custom (text): content chars.
    let custom = custom_text("abcd");
    assert_eq!(estimate_tokens(&custom), 1.0);

    // custom (blocks): text chars + 4800 per image (unlike user blocks).
    let custom_with_image = custom_blocks(vec![
        CoreContentBlock::text("abcd"),
        CoreContentBlock::Image(ImageContent::new("synthetic", "image/png")),
    ]);
    assert_eq!(estimate_tokens(&custom_with_image), 1_201.0);

    // branchSummary: summary chars.
    let branch = branch_summary(&"界".repeat(8));
    assert_eq!(estimate_tokens(&branch), 2.0);

    // compactionSummary without provider_context: summary + harness digest chars.
    let no_digest = compaction_summary("abcd", None, None);
    assert_eq!(estimate_tokens(&no_digest), 1.0);
    let with_digest = compaction_summary("abcd", None, Some("efgh"));
    assert_eq!(estimate_tokens(&with_digest), 2.0);
    let digest_only = compaction_summary("", None, Some("abcd"));
    assert_eq!(estimate_tokens(&digest_only), 1.0);

    // compactionSummary with provider_context: estimated_tokens + ceil(digest/4),
    // with no ceil applied to the sum and no contribution from the summary text.
    let provider_no_digest = compaction_summary(&"x".repeat(100_000), Some(5_000.0), None);
    assert_eq!(estimate_tokens(&provider_no_digest), 5_000.0);
    let provider_with_digest = compaction_summary(&"x".repeat(100_000), Some(5_000.0), Some("abcde"));
    // ceil(5/4) = 2.
    assert_eq!(estimate_tokens(&provider_with_digest), 5_002.0);
}

#[test]
fn estimate_tokens_property_randomized_inputs_match_the_reference() {
    for seed in SEEDS {
        let mut rng = Rng::new(seed);
        for _ in 0..64 {
            let message = rng.message();
            let actual = estimate_tokens(&message);
            let expected = reference_estimate_tokens(&message);
            assert_eq!(
                actual, expected,
                "seed {seed:#x}: estimate mismatch for {message:?}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 2. estimate_context_tokens
// ---------------------------------------------------------------------------

#[test]
fn estimate_context_tokens_falls_back_to_a_full_estimate_without_usage() {
    // No assistant at all.
    let messages = vec![
        user_text("abcdefgh"),
        tool_result("id", "bash", vec![text_block("ijklmnop")], false),
    ];
    let estimate = estimate_context_tokens(&messages);
    assert_eq!(estimate.tokens, 4.0);
    assert_eq!(estimate.usage_tokens, 0.0);
    assert_eq!(estimate.trailing_tokens, 4.0);
    assert_eq!(estimate.last_usage_index, None);

    // Assistants that aborted or errored carry no usage.
    let messages = vec![
        user_text("abcdefgh"),
        assistant(Vec::new(), "aborted", Usage::default()),
        assistant(
            vec![AiContentBlock::Text(TextContent::new("abcd"))],
            "error",
            usage(1234.0, 1.0, 1.0, 0.0, 0.0),
        ),
    ];
    let estimate = estimate_context_tokens(&messages);
    // 8 + 0 + 1 chars -> ceil(8/4) + ceil(0/4) + ceil(4/4) = 2 + 0 + 1.
    assert_eq!(estimate.tokens, 3.0);
    assert_eq!(estimate.usage_tokens, 0.0);
    assert_eq!(estimate.trailing_tokens, 3.0);
    assert_eq!(estimate.last_usage_index, None);

    // Empty list.
    let estimate = estimate_context_tokens(&[]);
    assert_eq!(estimate.tokens, 0.0);
    assert_eq!(estimate.usage_tokens, 0.0);
    assert_eq!(estimate.trailing_tokens, 0.0);
    assert_eq!(estimate.last_usage_index, None);
}

#[test]
fn estimate_context_tokens_uses_the_last_assistant_usage_plus_trailing_estimates() {
    let messages = vec![
        user_text("abcd"),                                  // 0
        assistant(Vec::new(), "stop", usage(1_000.0, 400.0, 100.0, 250.0, 250.0)), // 1
        user_text("efgh"),                                  // 2
        tool_result("id", "bash", vec![image_block()], false), // 3: 1200.0
        assistant(Vec::new(), "error", usage(9_999.0, 9.0, 9.0, 0.0, 0.0)), // 4: skipped
    ];
    let estimate = estimate_context_tokens(&messages);
    assert_eq!(estimate.last_usage_index, Some(1));
    assert_eq!(estimate.usage_tokens, 1_000.0);
    // Trailing: ceil(4/4) + 1200.0 + 0.0 = 1201.0.
    assert_eq!(estimate.trailing_tokens, 1_201.0);
    assert_eq!(estimate.tokens, 2_201.0);

    // With total_tokens == 0 the usage falls back to components.
    let messages = vec![
        assistant(Vec::new(), "stop", usage(0.0, 400.0, 100.0, 250.0, 250.0)),
    ];
    let estimate = estimate_context_tokens(&messages);
    assert_eq!(estimate.usage_tokens, 1_000.0);
    assert_eq!(estimate.tokens, 1_000.0);
    assert_eq!(estimate.trailing_tokens, 0.0);
    assert_eq!(estimate.last_usage_index, Some(0));
}

#[test]
fn calculate_context_tokens_prefers_total_tokens_and_falls_back_to_components() {
    assert_eq!(calculate_context_tokens(&usage(42.0, 1.0, 2.0, 3.0, 4.0)), 42.0);
    assert_eq!(calculate_context_tokens(&usage(0.0, 400.0, 100.0, 250.0, 250.0)), 1_000.0);
    assert_eq!(calculate_context_tokens(&usage(0.0, 0.0, 0.0, 0.0, 0.0)), 0.0);
}

#[test]
fn estimate_context_tokens_property_randomized_sequences_match_the_reference() {
    for seed in SEEDS {
        let mut rng = Rng::new(seed);
        for _ in 0..24 {
            let length = rng.below(12) as usize;
            let messages: Vec<AgentMessage> = (0..length).map(|_| rng.message()).collect();
            let estimate = estimate_context_tokens(&messages);
            let (tokens, usage_tokens, trailing_tokens, last_usage_index) =
                reference_estimate_context_tokens(&messages);
            assert_eq!(estimate.tokens, tokens, "seed {seed:#x}: tokens");
            assert_eq!(estimate.usage_tokens, usage_tokens, "seed {seed:#x}: usage");
            assert_eq!(estimate.trailing_tokens, trailing_tokens, "seed {seed:#x}: trailing");
            assert_eq!(
                estimate.last_usage_index, last_usage_index,
                "seed {seed:#x}: last_usage_index"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 3. should_compact / should_compact_for_model boundaries
// ---------------------------------------------------------------------------

#[test]
fn should_compact_boundary_table_pins_the_inclusive_threshold() {
    let settings = default_compaction_settings();
    assert!(settings.enabled);
    assert_eq!(settings.reserve_tokens, 16_384.0);
    assert_eq!(settings.keep_recent_tokens, 20_000.0);
    assert_eq!(settings.summary_update_policy.as_deref(), Some("off"));
    assert_eq!(MAX_COMPACTION_CONTEXT_TOKENS, 250_000.0);

    // (context_tokens, context_window, reserve, enabled, expected)
    // boundary = min(cap=250_000, window - reserve).
    let rows: [(f64, f64, f64, bool, bool); 30] = [
        // Window 200_000, reserve 16_384 -> boundary 183_616.
        (183_615.0, 200_000.0, 16_384.0, true, false),
        (183_615.999_9, 200_000.0, 16_384.0, true, false),
        (183_616.0, 200_000.0, 16_384.0, true, true),
        (183_617.0, 200_000.0, 16_384.0, true, true),
        // Window 0 and negative windows never compact.
        (0.0, 0.0, 16_384.0, true, false),
        (900_000.0, 0.0, 16_384.0, true, false),
        (900_000.0, -1.0, 16_384.0, true, false),
        // The 250k cap binds for large windows.
        (249_999.0, 1_050_000.0, 16_384.0, true, false),
        (249_999.999_9, 1_050_000.0, 16_384.0, true, false),
        (250_000.0, 1_050_000.0, 16_384.0, true, true),
        (250_001.0, 1_050_000.0, 16_384.0, true, true),
        (900_000.0, 1_050_000.0, 16_384.0, true, true),
        // Infinity window: cap still binds.
        (249_999.0, f64::INFINITY, 16_384.0, true, false),
        (250_000.0, f64::INFINITY, 16_384.0, true, true),
        // Disabled never compacts.
        (900_000.0, 1_050_000.0, 16_384.0, false, false),
        (250_000.0, 200_000.0, 16_384.0, false, false),
        // Reserve larger than the window makes the threshold negative:
        // min(250_000, window - reserve) = -100_000, and the comparison is an
        // inclusive >=, so even a negative token count trips it.
        (0.0, 100_000.0, 200_000.0, true, true),
        (-1.0, 100_000.0, 200_000.0, true, true),
        // Zero reserve: boundary is the window itself (capped at 250k).
        (249_999.0, 250_000.0, 0.0, true, false),
        (250_000.0, 250_000.0, 0.0, true, true),
        // Zero tokens with a normal window.
        (0.0, 1_050_000.0, 16_384.0, true, false),
        // Custom reserves: window 250_001 - 17_001 = 233_000 binds below the cap.
        (232_999.0, 250_001.0, 17_001.0, true, false),
        (233_000.0, 250_001.0, 17_001.0, true, true),
        (99_999.0, 120_000.0, 20_000.0, true, false),
        (100_000.0, 120_000.0, 20_000.0, true, true),
        // Window smaller than the cap: window - reserve binds.
        (33_615.0, 50_000.0, 16_384.0, true, false),
        (33_616.0, 50_000.0, 16_384.0, true, true),
        // Boundary exactly at the cap and just below the window edge.
        (250_000.0, 266_384.0, 16_384.0, true, true),
        (249_999.0, 266_384.0, 16_384.0, true, false),
        // Boundary below zero from a fractional window.
        (0.0, 15_000.0, 16_384.0, true, true),
    ];
    for (context_tokens, context_window, reserve, enabled, expected) in rows {
        let settings = CompactionSettings {
            enabled,
            reserve_tokens: reserve,
            keep_recent_tokens: 20_000.0,
            summary_update_policy: Some("off".to_string()),
        };
        assert_eq!(
            should_compact(context_tokens, context_window, &settings),
            expected,
            "tokens={context_tokens}, window={context_window}, reserve={reserve}, enabled={enabled}"
        );
    }
}

#[test]
fn should_compact_for_model_boundary_table_pins_both_caps() {
    let settings = default_compaction_settings();

    let mut azure_gpt = Model::new(
        "gpt-6-astra",
        "fixture",
        "openai-responses",
        "azure-openai-managed",
        "http://fixture.invalid",
    );
    azure_gpt.context_window = 1_050_000.0;
    azure_gpt.max_tokens = 32_000.0;
    // Azure managed GPT models: 400k cap.
    for (tokens, expected) in [
        (399_999.0, false),
        (400_000.0, true),
        (400_001.0, true),
        (250_000.0, false),
        (250_905.0, false),
        (900_000.0, true),
    ] {
        assert_eq!(
            should_compact_for_model(tokens, &azure_gpt, &settings),
            expected,
            "azure gpt at {tokens}"
        );
    }

    // The same model id on a non-Azure provider keeps the 250k cap.
    let mut openai = azure_gpt.clone();
    openai.provider = "openai".to_string();
    for (tokens, expected) in [(249_999.0, false), (250_000.0, true), (399_999.0, true)] {
        assert_eq!(
            should_compact_for_model(tokens, &openai, &settings),
            expected,
            "openai at {tokens}"
        );
    }

    // Azure non-GPT ids keep the 250k cap.
    let mut azure_other = azure_gpt.clone();
    azure_other.id = "FW-Kimi-K3".to_string();
    for (tokens, expected) in [(249_999.0, false), (250_000.0, true), (399_999.0, true)] {
        assert_eq!(
            should_compact_for_model(tokens, &azure_other, &settings),
            expected,
            "azure non-gpt at {tokens}"
        );
    }

    // The model input limit replaces the raw window: max_input_tokens wins.
    let mut limited = azure_gpt.clone();
    limited.max_input_tokens = Some(150_000.0);
    for (tokens, expected) in [
        (133_615.0, false),
        (133_616.0, true),
        (900_000.0, true),
    ] {
        assert_eq!(
            should_compact_for_model(tokens, &limited, &settings),
            expected,
            "input-limited at {tokens}"
        );
    }
    // A window below the cap binds through get_model_input_limit.
    let mut small_window = azure_gpt.clone();
    small_window.context_window = 200_000.0;
    for (tokens, expected) in [(183_615.0, false), (183_616.0, true)] {
        assert_eq!(
            should_compact_for_model(tokens, &small_window, &settings),
            expected,
            "small window at {tokens}"
        );
    }
    // Window 0 through the model path never compacts.
    let mut zero_window = azure_gpt.clone();
    zero_window.context_window = 0.0;
    assert!(!should_compact_for_model(900_000.0, &zero_window, &settings));
    // Disabled never compacts on the model path either.
    let mut disabled = CompactionSettings {
        enabled: false,
        reserve_tokens: 16_384.0,
        keep_recent_tokens: 20_000.0,
        summary_update_policy: Some("off".to_string()),
    };
    disabled.enabled = false;
    assert!(!should_compact_for_model(900_000.0, &azure_gpt, &disabled));
}

// ---------------------------------------------------------------------------
// 4. truncate_for_summary (through serialize_conversation)
// ---------------------------------------------------------------------------

/// The exact golden for a 2001-char ASCII tool result: 1431 head chars,
/// 70 elided, 500 tail chars, with the marker on its own paragraphs.
#[test]
fn tool_result_truncation_pins_the_exact_marker_and_head_tail_split() {
    // Exactly at the limit: unchanged.
    let at_limit = "H".repeat(2_000) + "T";
    assert_eq!(at_limit.chars().count(), 2_001);
    let messages = vec![Message::ToolResult(ToolResultMessage::new(
        "id",
        "bash",
        vec![text_block(&at_limit)],
        false,
        1,
    ))];
    let _ = messages;
    // 2000 chars: no truncation at all.
    let exactly_2000 = format!("{}{}", "H".repeat(1_500), "T".repeat(500));
    let serialized = serialize_conversation(&[Message::ToolResult(ToolResultMessage::new(
        "id",
        "bash",
        vec![text_block(&exactly_2000)],
        false,
        1,
    ))]);
    assert_eq!(
        serialized,
        format!("[Tool result bash (call outside excerpt)]: {exactly_2000}")
    );

    // 2001 chars: head 1431, tail 500, elided 70.
    let content = format!("{}{}", "H".repeat(1_501), "T".repeat(500));
    assert_eq!(content.chars().count(), 2_001);
    let serialized = serialize_conversation(&[Message::ToolResult(ToolResultMessage::new(
        "id",
        "bash",
        vec![text_block(&content)],
        false,
        1,
    ))]);
    let expected = format!(
        "[Tool result bash (call outside excerpt)]: {}\n\n[... 70 characters truncated; first 1431 and last 500 kept ...]\n\n{}",
        "H".repeat(1_431),
        "T".repeat(500),
    );
    assert_eq!(serialized, expected);
}

#[test]
fn tool_result_truncation_counts_chars_not_bytes() {
    // 1005 CJK chars = 3015 bytes: char count is below the 2000 limit, so no
    // truncation happens even though the byte length is far above it.
    let cjk_ok = "界".repeat(1_005);
    assert_eq!(cjk_ok.len(), 3_015);
    let serialized = serialize_conversation(&[Message::ToolResult(ToolResultMessage::new(
        "id",
        "bash",
        vec![text_block(&cjk_ok)],
        false,
        1,
    ))]);
    assert_eq!(
        serialized,
        format!("[Tool result bash (call outside excerpt)]: {cjk_ok}")
    );
    assert!(!serialized.contains("characters truncated"));

    // 2500 CJK chars: head 1431, tail 500, elided 569 — all char-aligned.
    let cjk_long = "界".repeat(2_500);
    let serialized = serialize_conversation(&[Message::ToolResult(ToolResultMessage::new(
        "id",
        "bash",
        vec![text_block(&cjk_long)],
        false,
        1,
    ))]);
    let expected = format!(
        "[Tool result bash (call outside excerpt)]: {}\n\n[... 569 characters truncated; first 1431 and last 500 kept ...]\n\n{}",
        "界".repeat(1_431),
        "界".repeat(500),
    );
    assert_eq!(serialized, expected);
}

#[test]
fn tool_result_truncation_splits_on_char_boundaries_for_mixed_widths() {
    // Mixed 1/2/3/4-byte scalars with multi-byte chars sitting exactly at the
    // head boundary (char index 1431) and inside the elided/tail ranges.
    let unit = "aé界𝕏"; // 4 chars, 10 bytes per repetition.
    let mut content = String::new();
    while content.chars().count() < 2_900 {
        content.push_str(unit);
    }
    // Pad to exactly 2900 chars with a distinguishable tail.
    let total = 2_900usize;
    let head_target = 1_431usize;
    let content: String = content.chars().take(head_target).collect::<String>()
        + &"x".repeat(total - head_target - 500 - 1)
        + "Z"
        + &"y".repeat(500);
    assert_eq!(content.chars().count(), total);

    let serialized = serialize_conversation(&[Message::ToolResult(ToolResultMessage::new(
        "id",
        "bash",
        vec![text_block(&content)],
        false,
        1,
    ))]);
    let chars: Vec<char> = content.chars().collect();
    let head: String = chars[..1_431].iter().collect();
    let tail: String = chars[total - 500..].iter().collect();
    let elided = total - 1_431 - 500;
    let expected = format!(
        "[Tool result bash (call outside excerpt)]: {head}\n\n[... {elided} characters truncated; first 1431 and last 500 kept ...]\n\n{tail}"
    );
    assert_eq!(serialized, expected);
}

#[test]
fn tool_result_shorter_than_the_limit_and_empty_content() {
    // Short input passes through unchanged.
    let short = "ok";
    let serialized = serialize_conversation(&[Message::ToolResult(ToolResultMessage::new(
        "id",
        "bash",
        vec![text_block(short)],
        false,
        1,
    ))]);
    assert_eq!(serialized, "[Tool result bash (call outside excerpt)]: ok");

    // Empty tool result content is omitted entirely (no empty part, no marker).
    let serialized = serialize_conversation(&[Message::ToolResult(ToolResultMessage::new(
        "id",
        "bash",
        vec![text_block("")],
        false,
        1,
    ))]);
    assert_eq!(serialized, "");

    // Image-only tool result content serializes to nothing visible.
    let serialized = serialize_conversation(&[Message::ToolResult(ToolResultMessage::new(
        "id",
        "bash",
        vec![image_block()],
        false,
        1,
    ))]);
    assert_eq!(serialized, "");
}

// ---------------------------------------------------------------------------
// 5. serialize_conversation goldens
// ---------------------------------------------------------------------------

#[test]
fn serialize_conversation_golden_corpus() {
    let mut arguments = Map::new();
    arguments.insert("path".to_string(), json!("a.ts"));
    arguments.insert("count".to_string(), json!(2));
    arguments.insert("nested".to_string(), json!({"k": [1, 2]}));
    let messages = vec![
        Message::User(UserMessage::new(UserContent::Text("hello".to_string()), 1)),
        Message::User(UserMessage::new(UserContent::Text(String::new()), 2)),
        Message::Assistant(AssistantMessage {
            content: vec![
                AiContentBlock::Thinking(ThinkingContent::new("why")),
                AiContentBlock::Text(TextContent::new("answer")),
                AiContentBlock::Text(TextContent::new("more")),
                AiContentBlock::ToolCall(ToolCall::new("id", "edit", arguments)),
                AiContentBlock::ToolCall(ToolCall::new("id2", "ipython", Map::new())),
            ],
            ..Default::default()
        }),
        Message::ToolResult(ToolResultMessage::new(
            "id",
            "edit",
            vec![text_block("ok")],
            false,
            3,
        )),
        Message::ToolResult(ToolResultMessage::new(
            "missing",
            "bash",
            vec![text_block("orphan")],
            true,
            4,
        )),
        Message::User(UserMessage::new(
            UserContent::Blocks(vec![text_block("block-one"), text_block("block-two")]),
            5,
        )),
    ];
    let serialized = serialize_conversation(&messages);
    let expected = "[User]: hello\n\n\
[Assistant thinking]: why\n\n\
[Assistant]: answer\nmore\n\n\
[Assistant tool calls]: #1 edit(path=\"a.ts\", count=2, nested={\"k\":[1,2]}); #2 ipython()\n\n\
[Tool result #1 edit]: ok\n\n\
[Tool result bash (call outside excerpt) ERROR]: orphan\n\n\
[User]: block-oneblock-two";
    assert_eq!(serialized, expected);
}

#[test]
fn serialize_conversation_preserves_control_characters_and_crlf() {
    let nul = "a\0b";
    let crlf = "line1\r\nline2";
    let messages = vec![
        Message::User(UserMessage::new(UserContent::Text(nul.to_string()), 1)),
        Message::User(UserMessage::new(UserContent::Text(crlf.to_string()), 2)),
    ];
    let serialized = serialize_conversation(&messages);
    assert_eq!(serialized, "[User]: a\0b\n\n[User]: line1\r\nline2");
}

#[test]
fn summarization_system_prompt_is_pinned() {
    // The chunk budget subtracts this prompt's char count; pin both its length
    // and its opening so any edit is caught.
    assert_eq!(SUMMARIZATION_SYSTEM_PROMPT.chars().count(), 317);
    assert!(SUMMARIZATION_SYSTEM_PROMPT.starts_with(
        "You are a context summarization assistant."
    ));
    assert!(SUMMARIZATION_SYSTEM_PROMPT.ends_with(
        "ONLY output the structured summary."
    ));
}

// ---------------------------------------------------------------------------
// 6. generate_bounded_summary chunking (through generate_summary)
// ---------------------------------------------------------------------------

/// A canned section-complete summary that passes validate_summary for the
/// Conversation format.
const VALID_SUMMARY: &str = "## Goal\nKernel parity.\n## Constraints & Preferences\nNo production changes.\n## Progress\nFixtures recorded.\n## Key Decisions\nPinned budgets.\n## Next Steps\nRe-run after optimizing.\n## Critical Context\nFixtures are offline.";

struct RecordedSummaryRequest {
    system_prompt: Option<String>,
    user_text: String,
    max_tokens: Option<f64>,
}

/// The API-provider registry is process-global; every test that registers a
/// provider holds this lock.
fn fixture_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

fn register_recording_provider(api: &'static str, reply: &'static str) -> Arc<Mutex<Vec<RecordedSummaryRequest>>> {
    let recorded = Arc::new(Mutex::new(Vec::<RecordedSummaryRequest>::new()));
    let recorded_for_stream = recorded.clone();
    let stream_simple: SimpleStreamFunction = Arc::new(move |_model, context, options| {
        let user_text = context
            .messages
            .iter()
            .map(|message| match message {
                Message::User(user) => user.content.text(),
                _ => String::new(),
            })
            .collect::<Vec<_>>()
            .join("\n");
        recorded_for_stream.lock().unwrap().push(RecordedSummaryRequest {
            system_prompt: context.system_prompt.clone(),
            user_text,
            max_tokens: options.and_then(|options| options.stream.max_tokens),
        });
        let message = AssistantMessage {
            content: vec![AiContentBlock::Text(TextContent::new(reply))],
            stop_reason: "stop".to_string(),
            ..Default::default()
        };
        let stream = AssistantMessageEventStream::new();
        stream.push(AssistantMessageEvent::Done {
            reason: "stop".to_string(),
            message,
        });
        stream.end(None);
        stream
    });
    let stream: StreamFunction = Arc::new(|_, _, _| {
        panic!("the base stream must not be used by the summary path")
    });
    register_api_provider_simple(
        ApiProviderSimple {
            api: api.to_string(),
            stream,
            stream_simple,
            compact: None,
            supports_compaction: None,
        },
        None,
    );
    recorded
}

fn chunk_fixture_model() -> Model {
    let mut model = Model::new(
        "chunk-fixture",
        "chunk fixture",
        "kernel-parity-chunk-fixture",
        "fixture",
        "http://fixture.invalid",
    );
    model.context_window = 8_192.0;
    model.max_tokens = 100_000.0;
    model
}

/// A deterministic corpus whose serialized conversation mixes 1-4 byte chars so
/// chunk boundaries land inside multi-byte sequences.
fn chunk_corpus(chars_per_turn: usize) -> Vec<AgentMessage> {
    let unit = "a界𝕏e\u{0301}\r\n"; // 7 chars, 13 bytes per repetition.
    let long = |chars: usize| -> String {
        let repetitions = chars / 7 + 1;
        unit.repeat(repetitions).chars().take(chars).collect()
    };
    let mut arguments = Map::new();
    arguments.insert("path".to_string(), json!("src/a.rs"));
    vec![
        user_text(&format!("first turn {}", long(chars_per_turn))),
        assistant(
            vec![
                AiContentBlock::Thinking(ThinkingContent::new("plan")),
                AiContentBlock::Text(TextContent::new("ack")),
                AiContentBlock::ToolCall(ToolCall::new("call-1", "edit", arguments)),
            ],
            "stop",
            Usage::default(),
        ),
        tool_result("call-1", "edit", vec![text_block("ok")], false),
        user_text(&format!("second turn {}", long(chars_per_turn))),
        assistant(
            vec![AiContentBlock::Text(TextContent::new("done"))],
            "stop",
            Usage::default(),
        ),
        user_text(&format!("third turn {}", long(chars_per_turn))),
        assistant(
            vec![AiContentBlock::Text(TextContent::new("fin"))],
            "stop",
            Usage::default(),
        ),
    ]
}

#[tokio::test]
async fn generate_summary_pins_chunk_covering_budget_formula_and_previous_summary_chaining() {
    let _guard = fixture_lock().lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let recorded = register_recording_provider("kernel-parity-chunk-fixture", VALID_SUMMARY);

    let model = chunk_fixture_model();
    let reserve_tokens: f64 = 1_000.0;

    // Pinned budget derivation (summary_output_budgets + chunk budget):
    //   input_limit = 8192 (context window; no max_input_tokens, not Astra)
    //   ceiling = min(max_tokens=100_000, floor(8192/4)=2048) = 2048
    //   requested = floor(0.8 * 1000) = 800
    //   initial = min(800, 2048) = 800
    //   retry_max = min(2*800, 2048, 65_536) = 1600
    //   budget_i = floor((min(8192, 8192-1600) - 1024) * 3) - suffix_i - SYS_PROMPT - 64
    let input_limit = get_model_input_limit(&model);
    assert_eq!(input_limit, 8_192.0);
    let ceiling = model.max_tokens.min((input_limit / 4.0).floor());
    assert_eq!(ceiling, 2_048.0);
    let requested = (0.8 * reserve_tokens).floor();
    assert_eq!(requested, 800.0);
    let initial = requested.min(ceiling).floor().max(1.0);
    assert_eq!(initial, 800.0);
    let retry_max_tokens = (initial * 2.0).min(ceiling).min(65_536.0).floor().max(initial);
    assert_eq!(retry_max_tokens, 1_600.0);
    let budget_base = (input_limit.min(model.context_window - retry_max_tokens) - 1_024.0) * 3.0;
    assert_eq!(budget_base, 16_704.0);

    let messages = chunk_corpus(12_000);
    let conversation = serialize_conversation(&convert_to_llm(&messages, &Default::default()));
    let conversation_chars = conversation.chars().count();
    assert!(
        conversation_chars > budget_base as usize,
        "corpus must span more than one chunk"
    );

    let policy: String = "off".to_string();
    let result = generate_summary(
        &messages,
        &model,
        reserve_tokens,
        "unused",
        None,
        None,
        None,
        None,
        None,
        default_summary_call_runner(None),
        &policy,
    )
    .await
    .expect("generate_summary must succeed against the fixture provider");

    let requests = recorded.lock().unwrap();
    assert!(
        requests.len() >= 2,
        "expected a multi-chunk summary loop, got {} requests",
        requests.len()
    );

    let mut offset = 0usize;
    for (index, request) in requests.iter().enumerate() {
        // Every wire call carries the exact summarization system prompt and a
        // single user message shaped as "<conversation>\n{chunk}\n</conversation>\n\n{suffix}".
        assert_eq!(
            request.system_prompt.as_deref(),
            Some(SUMMARIZATION_SYSTEM_PROMPT),
            "request {index}"
        );
        let body = request
            .user_text
            .strip_prefix("<conversation>\n")
            .unwrap_or_else(|| panic!("request {index} lost the conversation wrapper"));
        let (chunk, suffix) = body
            .split_once("\n</conversation>\n\n")
            .unwrap_or_else(|| panic!("request {index} lost the suffix separator"));

        // Previous-summary chaining: chunk 1 has no block; every later chunk's
        // suffix starts with the previous chunk's summary in the exact block.
        let expected_suffix = if index == 0 {
            build_summarization_prompt(None, None, &policy)
        } else {
            format!(
                "<previous-summary>\n{VALID_SUMMARY}\n</previous-summary>\n\n{}",
                build_summarization_prompt(None, Some(VALID_SUMMARY), &policy)
            )
        };
        assert_eq!(suffix, expected_suffix, "request {index} suffix");

        // The pinned budget formula: chunk length is exactly the budget left
        // after the suffix, the system prompt, and the 64-char headroom.
        let budget = budget_base.floor()
            - suffix.chars().count() as f64
            - SUMMARIZATION_SYSTEM_PROMPT.chars().count() as f64
            - 64.0;
        let remaining = conversation_chars - offset;
        let expected_len = (budget as usize).min(remaining);
        assert_eq!(
            chunk.chars().count(),
            expected_len,
            "request {index} chunk length (budget {budget})"
        );
        let expected_chunk: String = conversation.chars().skip(offset).take(expected_len).collect();
        assert_eq!(chunk, &expected_chunk, "request {index} chunk content");

        // Output budget: every non-retried attempt asks for the initial budget.
        assert_eq!(request.max_tokens, Some(initial), "request {index} max_tokens");

        offset += chunk.chars().count();
    }
    // Covering: the chunks reassemble the whole conversation exactly.
    assert_eq!(offset, conversation_chars);

    // The final summary is the last chunk's reply.
    assert_eq!(result.summary, VALID_SUMMARY);
    assert!(result.usage.is_some());
}

#[tokio::test]
async fn generate_summary_seeds_the_first_request_with_the_stripped_previous_summary() {
    let _guard = fixture_lock().lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let recorded = register_recording_provider("kernel-parity-chunk-fixture", VALID_SUMMARY);

    let model = chunk_fixture_model();
    let policy: String = "off".to_string();

    // A previous summary with a trailing file inventory: the inventory is
    // stripped before the first request, and the update prompt is used.
    let previous = format!("{VALID_SUMMARY}\n\n<read-files>\nold.rs\n</read-files>");
    let stripped = strip_file_operations(&previous);
    assert_eq!(stripped, VALID_SUMMARY);

    let messages = chunk_corpus(200);
    generate_summary(
        &messages,
        &model,
        1_000.0,
        "unused",
        None,
        None,
        Some(&previous),
        None,
        None,
        default_summary_call_runner(None),
        &policy,
    )
    .await
    .unwrap();

    let requests = recorded.lock().unwrap();
    assert_eq!(requests.len(), 1, "small corpus fits in one chunk");
    let body = requests[0]
        .user_text
        .strip_prefix("<conversation>\n")
        .unwrap();
    let (_chunk, suffix) = body.split_once("\n</conversation>\n\n").unwrap();
    let expected_suffix = format!(
        "<previous-summary>\n{stripped}\n</previous-summary>\n\n{}",
        build_summarization_prompt(None, Some(&stripped), &policy)
    );
    assert_eq!(suffix, expected_suffix);
    assert!(!suffix.contains("<read-files>"));

    // An empty previous summary is dropped: the base prompt, no block.
    let recorded = register_recording_provider("kernel-parity-chunk-fixture", VALID_SUMMARY);
    generate_summary(
        &messages,
        &model,
        1_000.0,
        "unused",
        None,
        None,
        Some(""),
        None,
        None,
        default_summary_call_runner(None),
        &policy,
    )
    .await
    .unwrap();
    let requests = recorded.lock().unwrap();
    assert_eq!(requests.len(), 1);
    let body = requests[0]
        .user_text
        .strip_prefix("<conversation>\n")
        .unwrap();
    let (_chunk, suffix) = body.split_once("\n</conversation>\n\n").unwrap();
    assert_eq!(suffix, build_summarization_prompt(None, None, &policy));
    assert!(!suffix.contains("<previous-summary>"));
}

#[tokio::test]
async fn generate_summary_with_a_precancelled_token_returns_the_abort_sentinel() {
    let _guard = fixture_lock().lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let recorded = register_recording_provider("kernel-parity-chunk-fixture", VALID_SUMMARY);

    let model = chunk_fixture_model();
    let token = tokio_util::sync::CancellationToken::new();
    token.cancel();
    let policy: String = "off".to_string();
    let messages = chunk_corpus(200);
    let error = generate_summary(
        &messages,
        &model,
        1_000.0,
        "unused",
        Some(&token),
        None,
        None,
        None,
        None,
        default_summary_call_runner(None),
        &policy,
    )
    .await
    .expect_err("a cancelled signal must abort the summary loop");
    assert_eq!(error, "Aborted");
    assert!(
        recorded.lock().unwrap().is_empty(),
        "no wire call may happen after cancellation"
    );
}

// ---------------------------------------------------------------------------
// 7. Cancellation smoke through compact()
// ---------------------------------------------------------------------------

fn cancel_entries() -> Vec<CompactionSessionEntry> {
    let entry = |id: &str, assistant: bool, text: &str| CompactionSessionEntry::Message {
        id: id.into(),
        parent_id: None,
        message: if assistant {
            AgentMessage::Message(Message::Assistant(AssistantMessage {
                content: vec![AiContentBlock::Text(TextContent::new(text))],
                ..Default::default()
            }))
        } else {
            AgentMessage::Message(Message::User(UserMessage::new(
                UserContent::Text(text.to_string()),
                0,
            )))
        },
    };
    vec![
        entry("e1", false, "Keep user data. Investigate the build failure."),
        entry("e2", true, "Build still fails."),
        entry("e3", false, "Please recheck."),
        entry("e4", true, "Build passes now."),
    ]
}

#[tokio::test]
async fn compact_with_a_precancelled_token_aborts_without_a_wire_call() {
    let _guard = fixture_lock().lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let recorded = register_recording_provider("kernel-parity-chunk-fixture", VALID_SUMMARY);

    let mut settings = default_compaction_settings();
    settings.keep_recent_tokens = 1.0;
    let entries = cancel_entries();
    let preparation = prepare_compaction(&entries, &settings, &|_| 0.0).expect("preparation");

    let model = chunk_fixture_model();
    let token = tokio_util::sync::CancellationToken::new();
    token.cancel();
    let error = compact(
        &preparation,
        &model,
        "unused",
        None,
        Some(&token),
        None,
        default_summary_call_runner(None),
        None,
        None,
    )
    .await
    .expect_err("a cancelled signal must abort compaction");
    assert_eq!(error, "Aborted");
    assert!(
        recorded.lock().unwrap().is_empty(),
        "no wire call may happen after cancellation"
    );
}
