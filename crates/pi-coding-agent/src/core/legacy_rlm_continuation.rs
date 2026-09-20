//! Port of packages/coding-agent/src/core/legacy-rlm-continuation.ts

use pi_agent_core::types::AgentMessage;
use serde_json::Value;

use crate::core::rlm_continuation::{
    bounded_rlm_visible_text, parse_rlm_continuation_state, read_rlm_visible_text,
    RlmContinuationState, RlmParentTask, RlmPendingContinuation, RlmPendingResult,
};

pub const LEGACY_RLM_CONTINUATION_STATE_CUSTOM_TYPE: &str =
    "prime-agent.rlm-continuation-state-v091";

fn is_record(value: &Value) -> bool {
    value.is_object()
}

/// Read the pre-source-integration Windows ledger without rewriting saved
/// sessions. A missing execution checkpoint cannot safely authorize replay of
/// old tool work.
pub fn parse_legacy_rlm_continuation_state(
    value: &Value,
    messages: &[AgentMessage],
) -> Option<RlmContinuationState> {
    if !is_record(value) || value.get("version").and_then(|version| version.as_f64()) != Some(1.0) {
        return None;
    }
    if !value
        .get("replySent")
        .map(|sent| sent.is_boolean())
        .unwrap_or(false)
    {
        return None;
    }
    let delivered_task_ids = value.get("deliveredTaskIds")?.as_array()?;
    if !delivered_task_ids.iter().all(|id| id.is_string()) {
        return None;
    }
    // legacy-rlm-continuation.ts:26-27 and :36-42 test `!== undefined`, not
    // "not null": a JSON null member is PRESENT and invalid, so the whole v091
    // ledger is rejected instead of silently authorizing legacy replay.
    let task = value.get("currentTask");
    if let Some(task) = task {
        if !is_record(task)
            || !task.get("id").map(|id| id.is_string()).unwrap_or(false)
            || task
                .get("id")
                .and_then(|id| id.as_str())
                .map(str::is_empty)
                .unwrap_or(true)
            || !task
                .get("receivedAt")
                .and_then(|received_at| received_at.as_f64())
                .map(|received_at| received_at.is_finite())
                .unwrap_or(false)
        {
            return None;
        }
    }
    let pending = value.get("pendingContinuation");
    if let Some(pending) = pending {
        let valid = is_record(pending)
            && matches!(
                pending
                    .get("phase")
                    .map(|phase| phase.to_string())
                    .unwrap_or_default()
                    .trim_matches('"'),
                "reserved" | "queued" | "continuation_hook"
            );
        if !valid {
            return None;
        }
    }
    // Without task identity, a pending recovery cannot be correlated or delivered.
    if pending.is_some() && task.is_none() {
        return None;
    }
    if let Some(pending) = pending {
        // legacy-rlm-continuation.ts:45: `pending.taskId !== undefined` then a
        // strict comparison against `task.id`; null can never equal a task id.
        if let Some(task_id) = pending.get("taskId") {
            if Some(task_id) != task.and_then(|task| task.get("id")) {
                return None;
            }
        }
    }
    let started = match pending {
        Some(pending) => messages.iter().any(|message| {
            if message.role() != "user"
                || Some(agent_message_timestamp(message))
                    != pending
                        .get("messageTimestamp")
                        .and_then(|value| value.as_f64())
            {
                return false;
            }
            let text = match message {
                AgentMessage::Message(pi_ai::types::Message::User(user)) => match &user.content {
                    pi_ai::types::UserContent::Text(text) => text.clone(),
                    pi_ai::types::UserContent::Blocks(blocks) => blocks
                        .iter()
                        .filter_map(|block| match block {
                            pi_ai::types::ImageOrTextContent::Text(text) => Some(text.text.clone()),
                            pi_ai::types::ImageOrTextContent::Image(_) => None,
                        })
                        .collect::<Vec<String>>()
                        .join("\n"),
                },
                _ => return false,
            };
            Some(text.as_str()) == pending.get("messageText").and_then(|text| text.as_str())
        }),
        None => false,
    };
    let task_id = task
        .and_then(|task| task.get("id"))
        .and_then(|id| id.as_str());
    let tasks: Vec<RlmParentTask> = match task {
        None => Vec::new(),
        Some(task) => {
            let replied = value
                .get("replySent")
                .and_then(|sent| sent.as_bool())
                .unwrap_or(false)
                || delivered_task_ids.iter().any(|id| id.as_str() == task_id);
            vec![RlmParentTask {
                id: task_id.unwrap_or_default().to_string(),
                received_at: task["receivedAt"].as_f64().unwrap_or(0.0),
                replied,
                result: None,
                result_delivery: None,
            }]
        }
    };
    let pending_phase = pending.map(|pending| {
        if started {
            "started".to_string()
        } else if pending.get("phase").and_then(|phase| phase.as_str()) == Some("queued") {
            "queued".to_string()
        } else {
            "reserved".to_string()
        }
    });
    let state_value = serde_json::json!({
        "version": 1,
        "tasks": tasks.iter().map(|task| serde_json::json!({
            "id": task.id,
            "receivedAt": task.received_at,
            "replied": task.replied,
        })).collect::<Vec<Value>>(),
        "continuationCount": value.get("continuationCount").cloned().unwrap_or(Value::Null),
        "lastStopReason": value.get("lastStopReason").cloned().unwrap_or(Value::Null),
        "lastSourceKey": pending.and_then(|pending| pending.get("sourceKey")).cloned().unwrap_or(Value::Null),
        "terminalStatus": value.get("terminalStatus").cloned().unwrap_or(Value::Null),
        "compactionReason": value.get("compactionReason").cloned().unwrap_or(Value::Null),
        "taskHadLength": value.get("taskHadLength").cloned().unwrap_or(Value::Null),
        "pendingContinuation": match pending {
            None => Value::Null,
            Some(pending) => serde_json::json!({
                "sourceKey": pending.get("sourceKey").cloned().unwrap_or(Value::Null),
                "attempt": pending.get("attempt").cloned().unwrap_or(Value::Null),
                "previousStopReason": pending.get("previousStopReason").cloned().unwrap_or(Value::Null),
                "messageTimestamp": pending.get("messageTimestamp").cloned().unwrap_or(Value::Null),
                "messageText": pending.get("messageText").cloned().unwrap_or(Value::Null),
                "phase": pending_phase.clone().unwrap_or_default(),
            }),
        },
    });
    let state = parse_rlm_continuation_state(&state_value)?;
    if let Some(pending) = &state.pending_continuation {
        if pending.attempt != state.continuation_count {
            return None;
        }
    }
    let mut state = state;
    let Some(current_task) = state.tasks.first().cloned() else {
        return Some(state);
    };
    if current_task.replied || state.pending_continuation.is_some() {
        return Some(state);
    }
    let mut visible_text = String::new();
    for message in messages.iter().rev() {
        if let AgentMessage::Message(pi_ai::types::Message::Assistant(assistant)) = message {
            if agent_message_timestamp(message) >= current_task.received_at {
                visible_text = read_rlm_visible_text(assistant);
                break;
            }
        }
    }
    let incomplete = state.terminal_status.is_none()
        || state.last_stop_reason.as_deref() == Some(pi_ai::types::STOP_REASON_LENGTH);
    let result = RlmPendingResult {
        status: if incomplete {
            "failed".to_string()
        } else {
            state
                .terminal_status
                .clone()
                .unwrap_or_else(|| "failed".to_string())
        },
        text: bounded_rlm_visible_text(
            if visible_text.is_empty() {
                "No visible result was saved for this legacy parent task."
            } else {
                &visible_text
            },
            None,
        ),
        partial: incomplete || state.task_had_length,
        reason: if incomplete {
            Some(
                "Legacy session has no durable continuation checkpoint; parent direction is required before replaying work."
                    .to_string(),
            )
        } else {
            None
        },
        stop_reason: state.last_stop_reason.clone(),
    };
    state.terminal_status = Some(result.status.clone());
    if let Some(task) = state.tasks.first_mut() {
        task.result = Some(result.clone());
    }
    state.pending_result = Some(result);
    Some(state)
}

/// `message.timestamp` for the three message shapes the legacy reader inspects.
fn agent_message_timestamp(message: &AgentMessage) -> f64 {
    match message {
        AgentMessage::Message(pi_ai::types::Message::User(user)) => user.timestamp as f64,
        AgentMessage::Message(pi_ai::types::Message::Assistant(assistant)) => {
            assistant.timestamp as f64
        }
        AgentMessage::Message(pi_ai::types::Message::ToolResult(tool_result)) => {
            tool_result.timestamp as f64
        }
        AgentMessage::Custom(_) => f64::NAN,
    }
}

/// Kept private: the legacy ledger never builds a pending continuation itself.
#[allow(dead_code)]
fn legacy_pending_continuation_placeholder() -> RlmPendingContinuation {
    RlmPendingContinuation::default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn assistant(timestamp: i64, stop_reason: &str, text: &str) -> AgentMessage {
        AgentMessage::Message(pi_ai::types::Message::Assistant(
            pi_ai::types::AssistantMessage {
                stop_reason: stop_reason.to_string(),
                timestamp,
                content: vec![
                    pi_ai::types::ContentBlock::Thinking(pi_ai::types::ThinkingContent::new(
                        "private reasoning",
                    )),
                    pi_ai::types::ContentBlock::Text(pi_ai::types::TextContent::new(text)),
                ],
                ..Default::default()
            },
        ))
    }

    fn user(timestamp: i64, text: &str) -> AgentMessage {
        AgentMessage::Message(pi_ai::types::Message::User(pi_ai::types::UserMessage::new(
            pi_ai::types::UserContent::Text(text.to_string()),
            timestamp,
        )))
    }

    fn record() -> Value {
        json!({
            "version": 1,
            "currentTask": { "id": "task-1", "receivedAt": 10 },
            "continuationCount": 1,
            "lastStopReason": "length",
            "taskHadLength": true,
            "replySent": false,
            "deliveredTaskIds": [],
            "pendingContinuation": {
                "sourceKey": "20:length:7:partial",
                "attempt": 1,
                "previousStopReason": "length",
                "messageTimestamp": 30,
                "messageText": "RLM child length recovery 1/3.",
                "taskId": "task-1",
                "phase": "queued"
            }
        })
    }

    /// `pendingContinuation` absent, as the TypeScript object literal passes it
    /// (`pendingContinuation: undefined` at legacy-rlm-continuation.ts:79-88).
    /// A JSON null member is PRESENT and invalid, so it must not be used here.
    fn without_member(value: &Value, key: &str) -> Value {
        let mut value = value.clone();
        value.as_object_mut().expect("record").remove(key);
        value
    }

    #[test]
    fn unstarted_recovery_phases_are_preserved_without_spending_an_attempt() {
        for phase in ["reserved", "queued", "continuation_hook"] {
            let mut value = record();
            value["pendingContinuation"]["phase"] = json!(phase);
            let state =
                parse_legacy_rlm_continuation_state(&value, &[assistant(20, "length", "partial")])
                    .expect("valid state");
            assert_eq!(state.continuation_count, 1.0);
            assert_eq!(
                state.last_source_key.as_deref(),
                Some("20:length:7:partial")
            );
            let expected = if phase == "continuation_hook" {
                "reserved"
            } else {
                phase
            };
            assert_eq!(state.pending_continuation.as_ref().unwrap().phase, expected);
            assert_eq!(state.tasks.len(), 1);
            assert!(!state.tasks[0].replied);
        }
    }

    #[test]
    fn persisted_recovery_is_correlated_and_marked_started() {
        let messages = vec![
            assistant(20, "length", "partial"),
            user(30, "RLM child length recovery 1/3."),
        ];
        let state = parse_legacy_rlm_continuation_state(&record(), &messages).unwrap();
        assert_eq!(
            state.pending_continuation.as_ref().unwrap().phase,
            "started"
        );

        let messages = vec![
            assistant(20, "length", "partial"),
            user(31, "RLM child length recovery 1/3."),
        ];
        let state = parse_legacy_rlm_continuation_state(&record(), &messages).unwrap();
        assert_eq!(state.pending_continuation.as_ref().unwrap().phase, "queued");
    }

    #[test]
    fn explicit_and_runtime_delivered_receipts_both_count() {
        for reply_sent in [true, false] {
            let mut value = record();
            value["replySent"] = json!(reply_sent);
            value["deliveredTaskIds"] = if reply_sent {
                json!([])
            } else {
                json!(["task-1"])
            };
            let value = without_member(&value, "pendingContinuation");
            let state =
                parse_legacy_rlm_continuation_state(&value, &[assistant(20, "length", "partial")])
                    .expect("valid state");
            assert!(state.tasks[0].replied);
            assert!(state.pending_result.is_none());
        }
    }

    #[test]
    fn missing_recovery_reports_a_bounded_partial_failure() {
        let mut value = record();
        let value = without_member(&value, "pendingContinuation");
        let state =
            parse_legacy_rlm_continuation_state(&value, &[assistant(20, "length", "partial")])
                .expect("valid state");
        assert_eq!(state.terminal_status.as_deref(), Some("failed"));
        let result = state.pending_result.as_ref().expect("pending result");
        assert_eq!(result.status, "failed");
        assert!(result.partial);
        assert_eq!(result.text, "partial");
        let serialized = serde_json::to_string(&state).unwrap();
        assert!(!serialized.contains("private reasoning"));
        assert!(state.pending_continuation.is_none());
    }

    #[test]
    fn valid_terminal_status_is_retained_for_this_task_only() {
        let mut value = record();
        let value = without_member(&value, "pendingContinuation");
        let mut value = value;
        value["terminalStatus"] = json!("blocked");
        value["lastStopReason"] = json!("stop");
        let state =
            parse_legacy_rlm_continuation_state(&value, &[assistant(20, "stop", "partial")])
                .expect("valid state");
        let result = state.pending_result.expect("pending result");
        assert_eq!(result.status, "blocked");
        assert_eq!(result.text, "partial");
        assert!(result.partial);

        let mut other = value.clone();
        other["currentTask"] = json!({ "id": "new-task", "receivedAt": 21 });
        let state =
            parse_legacy_rlm_continuation_state(&other, &[assistant(20, "stop", "partial")])
                .expect("valid state");
        assert_ne!(state.pending_result.unwrap().text, "partial");
    }

    #[test]
    fn malformed_or_uncorrelatable_state_is_rejected() {
        let mut uncorrelated = record();
        uncorrelated["pendingContinuation"]["taskId"] = json!("other-task");
        let mut unknown_phase = record();
        unknown_phase["pendingContinuation"]["phase"] = json!("unknown");
        // legacy-rlm-continuation.ts:36-44: `pendingContinuation !== undefined`
        // with no task identity is rejected.
        let missing_task = without_member(&record(), "currentTask");
        let mut bad_count = record();
        bad_count["continuationCount"] = json!(2);
        let mut bad_ids = record();
        bad_ids["deliveredTaskIds"] = json!([3]);
        let mut bad_version = record();
        bad_version["version"] = json!(2);
        for value in [
            Value::Null,
            json!({}),
            bad_version,
            missing_task,
            bad_count,
            bad_ids,
            uncorrelated,
            unknown_phase,
        ] {
            assert!(
                parse_legacy_rlm_continuation_state(&value, &[]).is_none(),
                "{value}"
            );
        }
    }

    /// The TypeScript tests every optional member with `!== undefined`
    /// (legacy-rlm-continuation.ts:26, :36, :45), so a JSON null member is
    /// PRESENT and invalid and must reject the whole v091 ledger. Rust used to
    /// filter null to "absent" and authorize legacy continuation/replay.
    #[test]
    fn a_null_member_is_present_and_invalid_like_the_typescript() {
        for key in ["currentTask", "pendingContinuation"] {
            let mut value = record();
            value[key] = Value::Null;
            assert!(
                parse_legacy_rlm_continuation_state(&value, &[assistant(20, "length", "partial")])
                    .is_none(),
                "null {key} must reject the ledger, got {value}"
            );
        }

        let mut null_task_id = record();
        null_task_id["pendingContinuation"]["taskId"] = Value::Null;
        assert!(
            parse_legacy_rlm_continuation_state(&null_task_id, &[]).is_none(),
            "null pendingContinuation.taskId must reject the ledger"
        );

        // Absent members are still valid: this is the control that proves the
        // test above fails on the null case and not on the shape.
        let control = without_member(&record(), "pendingContinuation");
        assert!(parse_legacy_rlm_continuation_state(&control, &[]).is_some());
    }
}
