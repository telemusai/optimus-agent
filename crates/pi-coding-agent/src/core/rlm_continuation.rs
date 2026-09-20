//! Port of packages/coding-agent/src/core/rlm-continuation.ts

use pi_ai::types::{AssistantMessage, ImageOrTextContent, StopReason, UserContent, UserMessage};

pub const RLM_CHILD_MAX_CONTINUATIONS: i64 = 3;
pub const RLM_CONTINUATION_STATE_CUSTOM_TYPE: &str = "prime-agent.rlm-continuation-state";
pub type RlmTerminalStatus = String;

pub const TERMINAL_COMPLETE: &str = "complete";
pub const TERMINAL_BLOCKED: &str = "blocked";
pub const TERMINAL_FAILED: &str = "failed";

/// `interface RlmParentTask`.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct RlmParentTask {
    pub id: String,
    pub received_at: f64,
    pub replied: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<RlmPendingResult>,
    /// Any marker blocks automatic replay, including after a process interruption.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result_delivery: Option<RlmResultDelivery>,
}

#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct RlmResultDelivery {
    pub attempt_id: String,
    /// Content-free evidence; the retained task result is never discarded.
    pub reason: String,
}

/// `interface RlmPendingContinuation`.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct RlmPendingContinuation {
    pub source_key: String,
    pub attempt: f64,
    pub previous_stop_reason: StopReason,
    pub message_timestamp: f64,
    pub message_text: String,
    pub phase: String,
}

/// `interface RlmPendingResult`.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct RlmPendingResult {
    pub status: RlmTerminalStatus,
    pub text: String,
    pub partial: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<StopReason>,
}

/// `interface RlmContinuationState`.
///
/// `version: 1` is a required literal, so the parser rejects anything else
/// before constructing this value.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct RlmContinuationState {
    pub version: f64,
    pub tasks: Vec<RlmParentTask>,
    pub continuation_count: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_stop_reason: Option<StopReason>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_source_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub terminal_status: Option<RlmTerminalStatus>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compaction_reason: Option<String>,
    pub task_had_length: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pending_continuation: Option<RlmPendingContinuation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pending_result: Option<RlmPendingResult>,
}

impl Default for RlmContinuationState {
    fn default() -> Self {
        empty_rlm_continuation_state()
    }
}

/// `emptyRlmContinuationState()`.
pub fn empty_rlm_continuation_state() -> RlmContinuationState {
    RlmContinuationState {
        version: 1.0,
        tasks: Vec::new(),
        continuation_count: 0.0,
        last_stop_reason: None,
        last_source_key: None,
        terminal_status: None,
        compaction_reason: None,
        task_had_length: false,
        pending_continuation: None,
        pending_result: None,
    }
}

/// `readRlmVisibleText(message)`.
pub fn read_rlm_visible_text(message: &AssistantMessage) -> String {
    message
        .content
        .iter()
        .filter_map(|block| match block {
            pi_ai::types::ContentBlock::Text(text) => Some(text.text.clone()),
            _ => None,
        })
        .collect::<Vec<String>>()
        .join("")
        .trim()
        .to_string()
}

/// `classifyRlmChildTerminal(message)` result.
#[derive(Debug, Clone, PartialEq)]
pub struct RlmChildTerminalClassification {
    pub terminal: bool,
    pub status: Option<RlmTerminalStatus>,
    pub text: String,
    pub can_continue: bool,
}

/// `classifyRlmChildTerminal(message)`.
pub fn classify_rlm_child_terminal(message: &AssistantMessage) -> RlmChildTerminalClassification {
    let text = read_rlm_visible_text(message);
    // An output-limited response cannot certify completion, even if it happened
    // to emit a terminal marker before the provider cut it off.
    if message.stop_reason == pi_ai::types::STOP_REASON_ERROR
        || message.stop_reason == pi_ai::types::STOP_REASON_ABORTED
    {
        return RlmChildTerminalClassification {
            terminal: true,
            status: Some(TERMINAL_FAILED.to_string()),
            text,
            can_continue: false,
        };
    }
    let matched = if message.stop_reason == pi_ai::types::STOP_REASON_STOP {
        terminal_marker_regex().captures(&text).map(|captures| {
            captures
                .get(1)
                .map(|value| value.as_str().to_lowercase())
                .unwrap_or_default()
        })
    } else {
        None
    };
    if let Some(status) = matched {
        return RlmChildTerminalClassification {
            terminal: true,
            status: Some(status),
            text,
            can_continue: false,
        };
    }
    RlmChildTerminalClassification {
        terminal: false,
        status: None,
        text,
        can_continue: message.stop_reason == pi_ai::types::STOP_REASON_STOP
            || message.stop_reason == pi_ai::types::STOP_REASON_LENGTH,
    }
}

/// `/(?:^|\n)RLM_CHILD_STATUS:\s*(complete|blocked|failed)\s*$/i`.
fn terminal_marker_regex() -> &'static regex::Regex {
    static REGEX: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    REGEX.get_or_init(|| {
        regex::Regex::new(r"(?i)(?:^|\n)RLM_CHILD_STATUS:\s*(complete|blocked|failed)\s*$").expect("static pattern")
    })
}

/// `createRlmChildContinuationMessage(attempt, previousStopReason, timestamp)`.
pub fn create_rlm_child_continuation_message(
    attempt: i64,
    previous_stop_reason: &str,
    timestamp: i64,
) -> UserMessage {
    let lines: Vec<String> = if previous_stop_reason == pi_ai::types::STOP_REASON_LENGTH {
        vec![
            format!("RLM child length recovery {attempt}/{RLM_CHILD_MAX_CONTINUATIONS}."),
            "Your previous response reached its output limit. Stop further research, exploration, file reads, code changes, background work, and ordinary tool use now.".to_string(),
            "Immediately send the parent one concise partial-result report with agent_message.send, then end with exactly one terminal line: RLM_CHILD_STATUS: complete, RLM_CHILD_STATUS: blocked, or RLM_CHILD_STATUS: failed.".to_string(),
        ]
    } else {
        vec![
            format!("RLM child continuation {attempt}/{RLM_CHILD_MAX_CONTINUATIONS}."),
            "Your previous response ended without the required terminal status. Continue the same task without repeating completed work.".to_string(),
            "Before ending, send the substantive result or blocker to the parent when appropriate, then end with exactly one terminal line: RLM_CHILD_STATUS: complete, RLM_CHILD_STATUS: blocked, or RLM_CHILD_STATUS: failed.".to_string(),
        ]
    };
    UserMessage {
        role: pi_ai::types::ROLE_USER.to_string(),
        content: UserContent::Blocks(vec![ImageOrTextContent::Text(pi_ai::types::TextContent::new(
            lines.join("\n"),
        ))]),
        provider_context: None,
        timestamp,
    }
}

/// `boundedRlmVisibleText(text, maxChars = 6000)`.
pub fn bounded_rlm_visible_text(text: &str, max_chars: Option<usize>) -> String {
    let max_chars = max_chars.unwrap_or(6000);
    let normalized = text.trim();
    if normalized.chars().count() <= max_chars {
        return normalized.to_string();
    }
    // `slice(0, maxChars - 32)` counts UTF-16 units; the port counts chars.
    let keep = max_chars.saturating_sub(32);
    let head: String = normalized.chars().take(keep).collect();
    format!("{}\n[truncated by Prime runtime]", head.trim_end())
}

fn is_record(value: &serde_json::Value) -> bool {
    value.is_object()
}

/// `isStopReason(value)`.
fn is_stop_reason(value: &serde_json::Value) -> bool {
    matches!(
        value.as_str(),
        Some(pi_ai::types::STOP_REASON_STOP)
            | Some(pi_ai::types::STOP_REASON_LENGTH)
            | Some(pi_ai::types::STOP_REASON_TOOL_USE)
            | Some(pi_ai::types::STOP_REASON_ERROR)
            | Some(pi_ai::types::STOP_REASON_ABORTED)
    )
}

/// `isTerminalStatus(value)`.
fn is_terminal_status(value: &serde_json::Value) -> bool {
    matches!(
        value.as_str(),
        Some(TERMINAL_COMPLETE) | Some(TERMINAL_BLOCKED) | Some(TERMINAL_FAILED)
    )
}

/// `Number.isSafeInteger(value)`.
fn is_safe_integer(value: &serde_json::Value) -> bool {
    match value.as_f64() {
        Some(number) => number.is_finite() && number.fract() == 0.0 && number.abs() <= 9_007_199_254_740_991.0,
        None => false,
    }
}

/// `isPendingResult(value)`.
fn is_pending_result(value: &serde_json::Value) -> bool {
    if !is_record(value) {
        return false;
    }
    is_terminal_status(&value["status"])
        && value.get("text").map(|text| text.is_string()).unwrap_or(false)
        && value.get("partial").map(|partial| partial.is_boolean()).unwrap_or(false)
        && (value.get("reason").is_none() || value["reason"].is_null() || value["reason"].is_string())
        && (value.get("stopReason").is_none() || value["stopReason"].is_null() || is_stop_reason(&value["stopReason"]))
}

/// `parseRlmContinuationState(value)`.
///
/// Returns `None` for any malformed or unrecognised state, exactly like the
/// TypeScript parser returning `undefined`.
pub fn parse_rlm_continuation_state(value: &serde_json::Value) -> Option<RlmContinuationState> {
    if !is_record(value) {
        return None;
    }
    if value.get("version").and_then(|version| version.as_f64()) != Some(1.0) {
        return None;
    }
    let tasks = value.get("tasks")?.as_array()?;
    let continuation_count = value.get("continuationCount")?;
    if !is_safe_integer(continuation_count) {
        return None;
    }
    let continuation_count_value = continuation_count.as_f64()?;
    if continuation_count_value < 0.0 || continuation_count_value > RLM_CHILD_MAX_CONTINUATIONS as f64 {
        return None;
    }
    if !value.get("taskHadLength")?.is_boolean() {
        return None;
    }
    if !tasks.iter().all(|task| {
        is_record(task)
            && task.get("id").map(|id| id.is_string()).unwrap_or(false)
            && task
                .get("receivedAt")
                .and_then(|received_at| received_at.as_f64())
                .map(|received_at| received_at.is_finite())
                .unwrap_or(false)
            && task.get("replied").map(|replied| replied.is_boolean()).unwrap_or(false)
            && (task.get("result").is_none() || task["result"].is_null() || is_pending_result(&task["result"]))
    }) {
        return None;
    }
    if let Some(last_stop_reason) = value.get("lastStopReason") {
        if !last_stop_reason.is_null() && !is_stop_reason(last_stop_reason) {
            return None;
        }
    }
    if let Some(last_source_key) = value.get("lastSourceKey") {
        if !last_source_key.is_null() && !last_source_key.is_string() {
            return None;
        }
    }
    if let Some(terminal_status) = value.get("terminalStatus") {
        if !terminal_status.is_null() && !is_terminal_status(terminal_status) {
            return None;
        }
    }
    if let Some(compaction_reason) = value.get("compactionReason") {
        if !compaction_reason.is_null() && !compaction_reason.is_string() {
            return None;
        }
    }
    if let Some(pending) = value.get("pendingContinuation") {
        if !pending.is_null() {
            let valid = is_record(pending)
                && pending.get("sourceKey").map(|key| key.is_string()).unwrap_or(false)
                && is_safe_integer(&pending["attempt"])
                && pending["attempt"].as_f64().map(|attempt| attempt >= 1.0).unwrap_or(false)
                && pending["attempt"]
                    .as_f64()
                    .map(|attempt| attempt <= RLM_CHILD_MAX_CONTINUATIONS as f64)
                    .unwrap_or(false)
                && is_stop_reason(&pending["previousStopReason"])
                && pending
                    .get("messageTimestamp")
                    .and_then(|timestamp| timestamp.as_f64())
                    .map(|timestamp| timestamp.is_finite())
                    .unwrap_or(false)
                && pending.get("messageText").map(|text| text.is_string()).unwrap_or(false)
                && matches!(
                    pending.get("phase").and_then(|phase| phase.as_str()),
                    Some("reserved") | Some("queued") | Some("started")
                );
            if !valid {
                return None;
            }
        }
    }
    if let Some(result) = value.get("pendingResult") {
        if !result.is_null() && !is_pending_result(result) {
            return None;
        }
    }
    // `structuredClone(value)`: re-serialize through the same shape.
    // `JSON.stringify` omits members whose value is `undefined`; the parser has
    // already rejected every malformed member, so a plain deserialize keeps the
    // observable state.
    let cloned = value.clone();
    serde_json::from_value::<RlmContinuationState>(normalize_state_value(&cloned)).ok()
}

/// Keeps the declared optional members only when present, matching the
/// TypeScript `structuredClone` result fed to `RlmContinuationState`.
fn normalize_state_value(value: &serde_json::Value) -> serde_json::Value {
    let mut normalized = value.clone();
    if let Some(object) = normalized.as_object_mut() {
        for key in [
            "lastStopReason",
            "lastSourceKey",
            "terminalStatus",
            "compactionReason",
            "pendingContinuation",
            "pendingResult",
        ] {
            if object.get(key).map(|member| member.is_null()).unwrap_or(false) {
                object.remove(key);
            }
        }
    }
    normalized
}

/// Only authoritative receipt shapes acknowledge a result. No transport error is
/// assumed to be a rejection unless it matches a known pre-admission route.
pub(crate) fn rlm_result_receipt_is_valid(
    receipt: &crate::core::agent_messages::AgentSessionMessageReceipt,
    target: &str,
    message: &str,
) -> bool {
    use crate::core::agent_messages::{AGENT_MESSAGE_SOURCE, DELIVERY_STATUS_DELIVERED, DELIVERY_STATUS_QUEUED};
    let timestamp = match receipt.delivery_status.as_str() {
        DELIVERY_STATUS_DELIVERED if receipt.queued_at.is_none() => receipt.delivered_at.as_deref(),
        DELIVERY_STATUS_QUEUED if receipt.delivered_at.is_none() => receipt.queued_at.as_deref(),
        _ => None,
    };
    !receipt.id.trim().is_empty()
        && receipt.source == AGENT_MESSAGE_SOURCE
        && !receipt.target.session_id.trim().is_empty()
        && !receipt.target.active_session_id.trim().is_empty()
        && (receipt.target.session_id == target
            || receipt.target.active_session_id == target
            || receipt.target.session_name.as_deref() == Some(target))
        && receipt.message == message.trim()
        && timestamp.is_some_and(|at| chrono::DateTime::parse_from_rfc3339(at).is_ok())
}

pub(crate) fn rlm_result_rejected_before_admission(error: &str) -> bool {
    if error == "Agent messaging is paused" {
        return true;
    }
    error
        .strip_prefix("Agent messaging rate limit exceeded; retry after ")
        .and_then(|rest| rest.strip_suffix("ms"))
        .is_some_and(|delay| !delay.is_empty() && delay.bytes().all(|byte| byte.is_ascii_digit()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn assistant(stop_reason: &str, text: &str) -> AssistantMessage {
        AssistantMessage {
            stop_reason: stop_reason.to_string(),
            content: vec![pi_ai::types::ContentBlock::Text(pi_ai::types::TextContent::new(text))],
            ..Default::default()
        }
    }


    #[test]
    fn result_delivery_marker_is_additive_and_survives_round_trip() {
        let mut value = json!({
            "version": 1, "tasks": [{ "id": "t", "receivedAt": 1, "replied": false }],
            "continuationCount": 0, "taskHadLength": false
        });
        assert!(parse_rlm_continuation_state(&value).unwrap().tasks[0].result_delivery.is_none());
        value["tasks"][0]["resultDelivery"] = json!({ "attemptId": "attempt-1", "reason": "acknowledgement_pending" });
        let state = parse_rlm_continuation_state(&value).unwrap();
        let round_trip = parse_rlm_continuation_state(&serde_json::to_value(&state).unwrap()).unwrap();
        assert_eq!(state, round_trip);
        assert_eq!(round_trip.tasks[0].result_delivery.as_ref().unwrap().attempt_id, "attempt-1");
        assert!(!round_trip.tasks[0].replied);
    }

    #[test]
    fn retry_allowlist_matches_only_exact_known_pre_admission_errors() {
        assert!(rlm_result_rejected_before_admission("Agent messaging is paused"));
        assert!(rlm_result_rejected_before_admission("Agent messaging rate limit exceeded; retry after 10ms"));
        for unknown in [
            "timeout: Agent messaging is paused", "Agent messaging is paused after submission",
            "Agent messaging rate limit exceeded; retry after ms",
            "Agent messaging rate limit exceeded; retry after 1ms; response lost", "connection closed",
        ] {
            assert!(!rlm_result_rejected_before_admission(unknown), "{unknown}");
        }
    }

    #[test]
    fn result_receipt_validation_rejects_incomplete_or_mismatched_acknowledgements() {
        use crate::core::agent_messages::{AgentSessionMessageEndpoint, AgentSessionMessageReceipt};
        let valid = AgentSessionMessageReceipt {
            id: "agentmsg_fixture".into(), source: "agent_message".into(), message: "answer".into(),
            target: AgentSessionMessageEndpoint {
                session_id: "parent".into(), active_session_id: "active".into(), ..Default::default()
            },
            delivery_status: "queued".into(), queued_at: Some("2026-09-20T00:00:00Z".into()),
            ..Default::default()
        };
        assert!(rlm_result_receipt_is_valid(&valid, "parent", "answer"));
        for change in 0..8 {
            let mut invalid = valid.clone();
            match change {
                0 => invalid.id.clear(),
                1 => invalid.source.clear(),
                2 => invalid.target.session_id = "other".into(),
                3 => invalid.message = "different".into(),
                4 => invalid.delivery_status = "unknown".into(),
                5 => invalid.queued_at = None,
                6 => invalid.queued_at = Some("not a timestamp".into()),
                _ => invalid.delivered_at = invalid.queued_at.clone(),
            }
            assert!(!rlm_result_receipt_is_valid(&invalid, "parent", "answer"), "mutation {change}");
        }
    }
    #[test]
    fn empty_state_matches_the_typescript_literal() {
        let state = empty_rlm_continuation_state();
        assert_eq!(state.version, 1.0);
        assert!(state.tasks.is_empty());
        assert_eq!(state.continuation_count, 0.0);
        assert!(!state.task_had_length);
    }

    #[test]
    fn visible_text_joins_text_blocks_and_ignores_thinking() {
        let message = AssistantMessage {
            content: vec![
                pi_ai::types::ContentBlock::Thinking(pi_ai::types::ThinkingContent::new("private")),
                pi_ai::types::ContentBlock::Text(pi_ai::types::TextContent::new(" hello")),
                pi_ai::types::ContentBlock::Text(pi_ai::types::TextContent::new(" world ")),
            ],
            ..Default::default()
        };
        assert_eq!(read_rlm_visible_text(&message), "hello world");
    }

    #[test]
    fn terminal_classification_follows_stop_reason_and_marker() {
        let classified = classify_rlm_child_terminal(&assistant("stop", "RLM_CHILD_STATUS: blocked"));
        assert!(classified.terminal);
        assert_eq!(classified.status.as_deref(), Some("blocked"));
        assert!(!classified.can_continue);

        let classified = classify_rlm_child_terminal(&assistant("error", "RLM_CHILD_STATUS: complete"));
        assert_eq!(classified.status.as_deref(), Some("failed"));

        let classified = classify_rlm_child_terminal(&assistant("length", "partial work"));
        assert!(!classified.terminal);
        assert!(classified.can_continue);

        let classified = classify_rlm_child_terminal(&assistant("toolUse", "still working"));
        assert!(!classified.terminal);
        assert!(!classified.can_continue);
    }

    #[test]
    fn continuation_messages_use_the_length_and_generic_wording() {
        let message = create_rlm_child_continuation_message(2, "length", 42);
        assert_eq!(message.timestamp, 42);
        let text = match &message.content {
            UserContent::Blocks(blocks) => match &blocks[0] {
                ImageOrTextContent::Text(content) => content.text.clone(),
                _ => panic!("expected text"),
            },
            _ => panic!("expected blocks"),
        };
        assert!(text.starts_with("RLM child length recovery 2/3."));
        assert!(text.contains("RLM_CHILD_STATUS: failed."));

        let message = create_rlm_child_continuation_message(1, "stop", 1);
        let text = match &message.content {
            UserContent::Blocks(blocks) => match &blocks[0] {
                ImageOrTextContent::Text(content) => content.text.clone(),
                _ => panic!("expected text"),
            },
            _ => panic!("expected blocks"),
        };
        assert!(text.starts_with("RLM child continuation 1/3."));
    }

    #[test]
    fn bounded_text_truncates_with_the_runtime_marker() {
        assert_eq!(bounded_rlm_visible_text("  short  ", None), "short");
        let long = "a".repeat(7000);
        let bounded = bounded_rlm_visible_text(&long, None);
        assert!(bounded.ends_with("[truncated by Prime runtime]"));
        assert_eq!(bounded.chars().count(), 6000 - 32 + "\n[truncated by Prime runtime]".chars().count());
    }

    #[test]
    fn parser_accepts_a_valid_state_and_round_trips_it() {
        let value = json!({
            "version": 1,
            "tasks": [{ "id": "task-1", "receivedAt": 10, "replied": false }],
            "continuationCount": 1,
            "lastStopReason": "length",
            "lastSourceKey": "20:length:7:partial",
            "terminalStatus": "blocked",
            "compactionReason": "overflow",
            "taskHadLength": true,
            "pendingContinuation": {
                "sourceKey": "k",
                "attempt": 1,
                "previousStopReason": "length",
                "messageTimestamp": 30,
                "messageText": "text",
                "phase": "queued"
            },
            "pendingResult": { "status": "failed", "text": "t", "partial": true, "stopReason": "length" }
        });
        let state = parse_rlm_continuation_state(&value).expect("valid state");
        assert_eq!(state.tasks[0].id, "task-1");
        assert_eq!(state.pending_continuation.as_ref().unwrap().phase, "queued");
        assert_eq!(state.pending_result.as_ref().unwrap().status, "failed");
    }

    #[test]
    fn parser_rejects_every_malformed_shape() {
        let base = json!({
            "version": 1,
            "tasks": [],
            "continuationCount": 0,
            "taskHadLength": false
        });
        assert!(parse_rlm_continuation_state(&json!(null)).is_none());
        assert!(parse_rlm_continuation_state(&json!({})).is_none());
        assert!(parse_rlm_continuation_state(&json!({
            "version": 2, "tasks": [], "continuationCount": 0, "taskHadLength": false
        }))
        .is_none());
        let mut over = base.clone();
        over["continuationCount"] = json!(4);
        assert!(parse_rlm_continuation_state(&over).is_none());
        let mut bad_task = base.clone();
        bad_task["tasks"] = json!([{ "id": "t", "receivedAt": 1 }]);
        assert!(parse_rlm_continuation_state(&bad_task).is_none());
        let mut bad_status = base.clone();
        bad_status["terminalStatus"] = json!("done");
        assert!(parse_rlm_continuation_state(&bad_status).is_none());
        let mut bad_pending = base.clone();
        bad_pending["pendingContinuation"] = json!({
            "sourceKey": "k", "attempt": 0, "previousStopReason": "stop",
            "messageTimestamp": 1, "messageText": "t", "phase": "reserved"
        });
        assert!(parse_rlm_continuation_state(&bad_pending).is_none());
        assert!(parse_rlm_continuation_state(&base).is_some());
    }
}
