//! Port of packages/coding-agent/src/core/messages.ts
//!
//! Custom message types and transformers for the coding agent.
//!
//! Type mapping notes:
//! - `custom` messages are carried by `pi_agent_core::types::CustomAgentMessage::Custom`,
//!   so `details` stays a `serde_json::Value` (the TypeScript generic `T`).
//! - `timestamp: string` parameters (`createBranchSummaryMessage` and friends) take
//!   an ISO string and are converted with `Date.parse` semantics; `new Date(x).getTime()`
//!   returns `NaN` for an unparsable value, which maps to `0` here.
//! - `AgentMessage` is an enum in this port, so the message factories return the
//!   specific custom message variant and callers wrap it into `AgentMessage`.

use pi_agent_core::types::{AgentMessage, CustomAgentMessage, CustomMessageContent};
use pi_ai::compaction::ProviderCompactionCheckpoint;
use pi_ai::types::{ImageOrTextContent, Message, TextContent, UserContent, UserMessage, ROLE_USER};
use serde_json::{Map, Value};

use crate::core::model_tool_output_policy::{
    apply_model_tool_output_policy, ModelToolOutputPolicyOptions,
};
use crate::core::refinement::refinement::{
    format_refinement_notice_body, AppliedRefinementEdit, HarnessScope, RefinementResult,
};
use crate::core::slash_commands::{
    is_session_slash_command_name, parse_session_slash_command, SessionSlashCommand,
};

pub const COMPACTION_SUMMARY_PREFIX: &str = "The conversation history before this point was compacted into the following summary:\n\n<summary>\n";

pub const COMPACTION_SUMMARY_SUFFIX: &str = "\n</summary>";

pub const BRANCH_SUMMARY_PREFIX: &str =
    "The following is a summary of a branch that this conversation came back from:\n\n<summary>\n";

pub const BRANCH_SUMMARY_SUFFIX: &str = "</summary>";

pub const HEARTBEAT_PROMPT_CUSTOM_TYPE: &str = "heartbeat_prompt";
pub const HEARTBEAT_PROMPT_PREVIEW_LABEL: &str = "Heartbeat prompt";
pub const IPYTHON_STATE_RESTORED_CUSTOM_TYPE: &str = "ipython_state_restored";
pub const SESSION_SLASH_COMMAND_CUSTOM_TYPE: &str = "session_slash_command";
pub const SESSION_SLASH_COMMAND_RESULT_CUSTOM_TYPE: &str = "session_slash_command_result";
pub const COMPACTION_OUTCOME_CUSTOM_TYPE: &str = "compaction_outcome";
pub const REFINEMENT_OUTCOME_CUSTOM_TYPE: &str = "refinement_outcome";
pub const REFINEMENT_NOTICE_CUSTOM_TYPE: &str = "refinement_notice";
pub const HARNESS_DIGEST_CUSTOM_TYPE: &str = "harness_digest";
pub const RLM_CHILD_FAILURE_CUSTOM_TYPE: &str = "rlm_child_failure";
pub const RLM_CHILD_TERMINAL_NOTICE_CUSTOM_TYPE: &str = "rlm_child_terminal_notice";
pub const ASYNC_BASH_COMPLETION_CUSTOM_TYPE: &str = "async_bash_completion";
pub const ASYNC_BASH_COMPLETION_PREVIEW_LABEL: &str = "Shell message received";

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionSlashCommandDetails {
    pub command: SessionSlashCommand,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub command_entry_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionSlashCommandResultDetails {
    pub command: SessionSlashCommand,
    pub success: bool,
    pub severity: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub command_entry_id: Option<String>,
}

pub type CompactionOutcomeReason = String;
pub type CompactionOutcome = String;

pub const COMPACTION_OUTCOME_REASON_THRESHOLD: &str = "threshold";
pub const COMPACTION_OUTCOME_REASON_OVERFLOW: &str = "overflow";
pub const COMPACTION_OUTCOME_REASON_REQUESTED: &str = "requested";

pub const COMPACTION_OUTCOME_SKIPPED: &str = "skipped";
pub const COMPACTION_OUTCOME_CANCELLED: &str = "cancelled";
pub const COMPACTION_OUTCOME_FAILED: &str = "failed";

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactionOutcomeDetails {
    pub reason: CompactionOutcomeReason,
    pub outcome: CompactionOutcome,
    /// Sanitized classification of a failed summary attempt. Absent on
    /// cancelled/skipped outcomes and older history entries.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure_kind: Option<String>,
    /// First-kept boundary entry UUID of the attempt, when preparation ran.
    /// Entry IDs only; conversation content is never recorded here.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub boundary_entry_id: Option<String>,
    /// Request shape of the attempt: "splitTurnPrefix" or "historyOnly".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary_shape: Option<String>,
    /// Automatic-threshold failure streak depth after this failure.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub consecutive_failures: Option<u32>,
}

impl Default for CompactionOutcomeDetails {
    fn default() -> Self {
        Self {
            reason: String::new(),
            outcome: String::new(),
            failure_kind: None,
            boundary_entry_id: None,
            summary_shape: None,
            consecutive_failures: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RefinementOutcomeDetails {
    pub refinement_id: String,
    pub summary: String,
    pub scope: HarnessScope,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rollback_of: Option<String>,
    pub edits: Vec<AppliedRefinementEdit>,
}

/// How a refinement was initiated: reviewer-triggered auto-refine, the /refine slash command, or the model's own refine.run().
pub type RefinementSource = String;

pub const REFINEMENT_SOURCE_AUTO: &str = "auto";
pub const REFINEMENT_SOURCE_USER: &str = "user";
pub const REFINEMENT_SOURCE_SELF: &str = "self";

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RefinementNoticeDetails {
    pub refinement_id: String,
    pub summary: String,
    pub scope: HarnessScope,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rollback_of: Option<String>,
    pub edits: Vec<AppliedRefinementEdit>,
    pub source: RefinementSource,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct HarnessDigestDetails {
    pub digest: String,
}

pub const HARNESS_DIGEST_PREFIX: &str =
    "The persistent memories produced across this session so far:\n\n<harness_state>\n";

pub const HARNESS_DIGEST_SUFFIX: &str = "\n</harness_state>";

pub fn create_harness_digest_message(digest: String, timestamp: i64) -> CustomAgentMessage {
    CustomAgentMessage::Custom {
        custom_type: HARNESS_DIGEST_CUSTOM_TYPE.to_string(),
        content: CustomMessageContent::Text(
            HARNESS_DIGEST_PREFIX.to_string() + &digest + HARNESS_DIGEST_SUFFIX,
        ),
        display: false,
        details: Some(serde_json::json!({ "digest": digest })),
        timestamp,
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RlmChildFailureDetails {
    pub child_id: String,
    pub session_name: String,
    pub error: String,
}

/// `RlmChildTerminalNoticeDetails`; `kind` selects which optional fields are present.
#[derive(Debug, Clone, PartialEq)]
pub struct RlmChildTerminalNoticeDetails {
    pub kind: String,
    pub child_id: String,
    pub session_name: String,
    pub reason: Option<String>,
    pub last_assistant_text_preview: Option<String>,
}

impl RlmChildTerminalNoticeDetails {
    pub const KIND_CANCELLED: &'static str = "cancelled";
    pub const KIND_COMPLETED_WITHOUT_REPLY: &'static str = "completed_without_reply";

    pub fn to_json(&self) -> Value {
        let mut object = Map::new();
        object.insert("kind".to_string(), Value::String(self.kind.clone()));
        object.insert("childId".to_string(), Value::String(self.child_id.clone()));
        object.insert(
            "sessionName".to_string(),
            Value::String(self.session_name.clone()),
        );
        if self.kind == Self::KIND_CANCELLED {
            if let Some(reason) = &self.reason {
                object.insert("reason".to_string(), Value::String(reason.clone()));
            }
        } else if let Some(preview) = &self.last_assistant_text_preview {
            object.insert(
                "lastAssistantTextPreview".to_string(),
                Value::String(preview.clone()),
            );
        }
        Value::Object(object)
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AsyncBashCompletionDetails {
    pub pid: i64,
    pub command: String,
    pub exit_code: i64,
}

pub fn create_async_bash_completion_message(
    details: AsyncBashCompletionDetails,
    timestamp: i64,
) -> CustomAgentMessage {
    CustomAgentMessage::Custom {
        custom_type: ASYNC_BASH_COMPLETION_CUSTOM_TYPE.to_string(),
        content: CustomMessageContent::Text(format!(
            "{ASYNC_BASH_COMPLETION_PREVIEW_LABEL}.\nSource: bash\nCommand completed (pid {}, exit code {}).\nCommand: {}\n\nInspect the saved BashHandle with .poll(), .output(), or .tail(), then continue the task.",
            details.pid,
            details.exit_code,
            serde_json::to_string(&details.command).unwrap_or_default()
        )),
        display: true,
        details: serde_json::to_value(&details).ok(),
        timestamp,
    }
}

pub fn create_rlm_child_failure_message(
    details: RlmChildFailureDetails,
    timestamp: i64,
) -> CustomAgentMessage {
    CustomAgentMessage::Custom {
        custom_type: RLM_CHILD_FAILURE_CUSTOM_TYPE.to_string(),
        content: CustomMessageContent::Text(format!(
            "RLM child {} ({}) failed: {}",
            details.session_name, details.child_id, details.error
        )),
        display: true,
        details: serde_json::to_value(&details).ok(),
        timestamp,
    }
}

pub fn create_rlm_child_terminal_notice_message(
    details: RlmChildTerminalNoticeDetails,
    timestamp: i64,
) -> CustomAgentMessage {
    let content = if details.kind == RlmChildTerminalNoticeDetails::KIND_CANCELLED {
        match &details.reason {
            Some(reason) => format!(
                "RLM child {} ({}) was cancelled: {}",
                details.session_name, details.child_id, reason
            ),
            None => format!(
                "RLM child {} ({}) was cancelled",
                details.session_name, details.child_id
            ),
        }
    } else {
        match &details.last_assistant_text_preview {
            Some(preview) => format!(
                "RLM child {} ({}) completed without sending a reply. Last assistant text: {}",
                details.session_name, details.child_id, preview
            ),
            None => format!(
                "RLM child {} ({}) completed without sending a reply",
                details.session_name, details.child_id
            ),
        }
    };
    CustomAgentMessage::Custom {
        custom_type: RLM_CHILD_TERMINAL_NOTICE_CUSTOM_TYPE.to_string(),
        content: CustomMessageContent::Text(content),
        display: true,
        details: Some(details.to_json()),
        timestamp,
    }
}

/// Message type for bash executions via the ! command.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BashExecutionMessage {
    pub role: String,
    pub command: String,
    pub output: String,
    pub exit_code: Option<i64>,
    pub cancelled: bool,
    pub truncated: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub full_output_path: Option<String>,
    pub timestamp: i64,
    /// If true, this message is excluded from LLM context (!! prefix)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exclude_from_context: Option<bool>,
}

/// Message type for extension-injected messages via sendMessage().
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CustomMessage {
    pub role: String,
    pub custom_type: String,
    pub content: CustomMessageContent,
    pub display: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
    pub timestamp: i64,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HeartbeatPromptDetails {
    pub job_id: String,
    pub schedule: String,
    pub status: String,
    pub run_count: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_run_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_run_at: Option<String>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct IpythonStateRestoredDetails {
    pub restored: bool,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BranchSummaryMessage {
    pub role: String,
    pub summary: String,
    pub from_id: String,
    pub timestamp: i64,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactionSummaryMessage {
    pub role: String,
    pub summary: String,
    /// Complete opaque provider window, used instead of the display summary.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_context: Option<ProviderCompactionCheckpoint>,
    pub tokens_before: f64,
    /// Number of retained messages that precede this summary in transcript presentation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retained_message_count: Option<f64>,
    /// User instructions that guided the summary (from `/compact <instructions>`)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub custom_instructions: Option<String>,
    /// Harness digest snapshot rendered before the summary in LLM context. Attached mechanically at compaction, never summarized.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub harness_digest: Option<String>,
    pub timestamp: i64,
}

/// `Date.parse(timestamp)`; an unparsable value becomes `NaN` in JavaScript, which
/// this port reports as `0` because the field is an integer millisecond count.
fn parse_timestamp_millis(timestamp: &str) -> i64 {
    match chrono::DateTime::parse_from_rfc3339(timestamp) {
        Ok(value) => value.timestamp_millis(),
        Err(_) => 0,
    }
}

/// Format bash output for LLM context. The fence must be longer than any
/// backtick run in the output so command output cannot terminate it early.
pub fn bash_output_to_text(msg: &BashExecutionMessage) -> String {
    let mut text = String::new();
    if !msg.output.is_empty() {
        let mut longest_backtick_run = 0usize;
        let bytes = msg.output.as_bytes();
        let mut index = 0usize;
        while index < bytes.len() {
            if bytes[index] == b'`' {
                let start = index;
                while index < bytes.len() && bytes[index] == b'`' {
                    index += 1;
                }
                longest_backtick_run = longest_backtick_run.max(index - start);
            } else {
                index += 1;
            }
        }
        let fence = "`".repeat(std::cmp::max(3, longest_backtick_run + 1));
        text += &format!("{fence}\n{}\n{fence}", msg.output);
    } else {
        text += "(no output)";
    }
    if msg.cancelled {
        text += "\n\n(command cancelled)";
    } else if let Some(exit_code) = msg.exit_code {
        if exit_code != 0 {
            text += &format!("\n\nCommand exited with code {exit_code}");
        }
    }
    if msg.truncated {
        // The formatted suffix is bound to a local: a `&format!(..)` arm would borrow a temporary
        // that is freed at the end of the `match`.
        let suffix = match &msg.full_output_path {
            Some(path) => format!("\n\n[Output truncated. Full output: {path}]"),
            None => "\n\n[Output truncated.]".to_string(),
        };
        text += &suffix;
    }
    text
}

/// Convert a BashExecutionMessage to user message text for LLM context.
pub fn bash_execution_to_text(msg: &BashExecutionMessage) -> String {
    format!("Ran `{}`\n{}", msg.command, bash_output_to_text(msg))
}

// ---------------------------------------------------------------------------
// Structural-typing bridges
// ---------------------------------------------------------------------------
//
// In TypeScript `CustomMessage`, `BranchSummaryMessage` and
// `CompactionSummaryMessage` are members of the `AgentMessage` union by shape.
// The Rust port models the union as an enum, so each factory result is wrapped
// explicitly. These are crate-internal plumbing, not new behaviour.

pub(crate) fn custom_message_to_agent_message(message: CustomMessage) -> AgentMessage {
    AgentMessage::Custom(CustomAgentMessage::Custom {
        custom_type: message.custom_type,
        content: message.content,
        display: message.display,
        details: message.details,
        timestamp: message.timestamp,
    })
}

pub(crate) fn branch_summary_to_agent_message(message: BranchSummaryMessage) -> AgentMessage {
    AgentMessage::Custom(CustomAgentMessage::BranchSummary {
        summary: message.summary,
        from_id: message.from_id,
        timestamp: message.timestamp,
    })
}

pub(crate) fn compaction_summary_to_agent_message(
    message: CompactionSummaryMessage,
) -> AgentMessage {
    AgentMessage::Custom(CustomAgentMessage::CompactionSummary {
        summary: message.summary,
        provider_context: message.provider_context,
        tokens_before: message.tokens_before,
        retained_message_count: message.retained_message_count,
        custom_instructions: message.custom_instructions,
        harness_digest: message.harness_digest,
        timestamp: message.timestamp,
    })
}

pub(crate) fn harness_digest_to_agent_message(message: CustomAgentMessage) -> AgentMessage {
    AgentMessage::Custom(message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn empty_options() -> ModelToolOutputPolicyOptions {
        ModelToolOutputPolicyOptions {
            policy: None,
            scope: None,
        }
    }

    #[test]
    fn bash_output_fence_outgrows_backtick_runs() {
        let msg = BashExecutionMessage {
            role: "bashExecution".to_string(),
            command: "echo hi".to_string(),
            output: "a ``` b".to_string(),
            exit_code: Some(1),
            cancelled: false,
            truncated: false,
            full_output_path: None,
            timestamp: 0,
            exclude_from_context: None,
        };
        let text = bash_output_to_text(&msg);
        assert!(text.starts_with("````\na ``` b\n````"));
        assert!(text.ends_with("\n\nCommand exited with code 1"));
    }

    #[test]
    fn bash_output_without_output_reports_no_output() {
        let msg = BashExecutionMessage {
            role: "bashExecution".to_string(),
            command: "true".to_string(),
            output: String::new(),
            exit_code: Some(0),
            cancelled: true,
            truncated: true,
            full_output_path: Some("C:/tmp/out.txt".to_string()),
            timestamp: 0,
            exclude_from_context: None,
        };
        let text = bash_output_to_text(&msg);
        assert_eq!(
            text,
            "(no output)\n\n(command cancelled)\n\n[Output truncated. Full output: C:/tmp/out.txt]"
        );
    }

    #[test]
    fn convert_to_llm_drops_session_only_custom_messages() {
        let messages = vec![
            AgentMessage::Custom(CustomAgentMessage::Custom {
                custom_type: SESSION_SLASH_COMMAND_CUSTOM_TYPE.to_string(),
                content: CustomMessageContent::Text("/compact".to_string()),
                display: true,
                details: None,
                timestamp: 1,
            }),
            AgentMessage::Custom(CustomAgentMessage::Custom {
                custom_type: REFINEMENT_OUTCOME_CUSTOM_TYPE.to_string(),
                content: CustomMessageContent::Text("done".to_string()),
                display: true,
                details: None,
                timestamp: 2,
            }),
            AgentMessage::Custom(CustomAgentMessage::Custom {
                custom_type: "other".to_string(),
                content: CustomMessageContent::Text("keep".to_string()),
                display: true,
                details: None,
                timestamp: 3,
            }),
        ];
        let converted = convert_to_llm(&messages, &empty_options());
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].as_user().unwrap().content.text(), "keep");
    }

    #[test]
    fn convert_to_llm_skips_excluded_bash_execution() {
        let messages = vec![AgentMessage::Custom(CustomAgentMessage::BashExecution {
            command: "ls".to_string(),
            output: "file".to_string(),
            exit_code: Some(0),
            cancelled: false,
            truncated: false,
            full_output_path: None,
            timestamp: 7,
            exclude_from_context: Some(true),
        })];
        assert!(convert_to_llm(&messages, &empty_options()).is_empty());
    }

    #[test]
    fn convert_to_llm_splits_digest_out_of_provider_context_carrier() {
        let checkpoint = ProviderCompactionCheckpoint {
            version: 1,
            provider: "openai".to_string(),
            api: "openai-responses".to_string(),
            model: "gpt-5".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            endpoint: None,
            items: vec![Map::new()],
            estimated_tokens: 1.0,
        };
        let messages = vec![AgentMessage::Custom(
            CustomAgentMessage::CompactionSummary {
                summary: "sum".to_string(),
                provider_context: Some(checkpoint),
                tokens_before: 10.0,
                retained_message_count: None,
                custom_instructions: None,
                harness_digest: Some("digest".to_string()),
                timestamp: 11,
            },
        )];
        let converted = convert_to_llm(&messages, &empty_options());
        assert_eq!(converted.len(), 2);
        let first = converted[0].as_user().unwrap();
        assert!(first.provider_context.is_some());
        assert_eq!(
            first.content.text(),
            format!("{COMPACTION_SUMMARY_PREFIX}sum{COMPACTION_SUMMARY_SUFFIX}")
        );
        let second = converted[1].as_user().unwrap();
        assert_eq!(
            second.content.text(),
            format!("{HARNESS_DIGEST_PREFIX}digest{HARNESS_DIGEST_SUFFIX}")
        );
    }

    #[test]
    fn convert_to_llm_keeps_digest_before_summary_without_provider_context() {
        let messages = vec![AgentMessage::Custom(
            CustomAgentMessage::CompactionSummary {
                summary: "sum".to_string(),
                provider_context: None,
                tokens_before: 10.0,
                retained_message_count: None,
                custom_instructions: None,
                harness_digest: Some("digest".to_string()),
                timestamp: 11,
            },
        )];
        let converted = convert_to_llm(&messages, &empty_options());
        assert_eq!(converted.len(), 1);
        assert_eq!(
            converted[0].as_user().unwrap().content.text(),
            format!("{HARNESS_DIGEST_PREFIX}digest{HARNESS_DIGEST_SUFFIX}\n\n{COMPACTION_SUMMARY_PREFIX}sum{COMPACTION_SUMMARY_SUFFIX}")
        );
    }

    #[test]
    fn without_harness_digests_removes_snapshots() {
        let messages = vec![
            harness_digest_to_agent_message(create_harness_digest_message("digest".to_string(), 1)),
            AgentMessage::Custom(CustomAgentMessage::CompactionSummary {
                summary: "sum".to_string(),
                provider_context: None,
                tokens_before: 1.0,
                retained_message_count: None,
                custom_instructions: None,
                harness_digest: Some("digest".to_string()),
                timestamp: 2,
            }),
        ];
        let filtered = without_harness_digests_for_compaction(&messages);
        assert_eq!(filtered.len(), 1);
        match &filtered[0] {
            AgentMessage::Custom(CustomAgentMessage::CompactionSummary {
                harness_digest, ..
            }) => {
                assert!(harness_digest.is_none());
            }
            other => panic!("unexpected message {other:?}"),
        }
    }

    #[test]
    fn compaction_outcome_validation_matches_typescript() {
        assert!(is_compaction_outcome_message(&json!({
            "role": "custom",
            "customType": COMPACTION_OUTCOME_CUSTOM_TYPE,
            "content": "skipped",
            "display": true,
            "timestamp": 1,
            "details": {"reason": "threshold", "outcome": "skipped"},
        })));
        assert!(!is_compaction_outcome_message(&json!({
            "role": "custom",
            "customType": COMPACTION_OUTCOME_CUSTOM_TYPE,
            "content": "skipped",
            "display": true,
            "timestamp": 1,
            "details": {"reason": "nope", "outcome": "skipped"},
        })));
    }

    #[test]
    fn session_slash_command_messages_are_validated_by_shape() {
        let message = json!({
            "role": "custom",
            "customType": SESSION_SLASH_COMMAND_CUSTOM_TYPE,
            "content": "/compact focus",
            "display": true,
            "timestamp": 5,
            "details": {
                "command": {"name": "compact", "args": "focus", "text": "/compact focus"},
            },
        });
        assert!(is_session_slash_command_message(&message));
        let mismatched = json!({
            "role": "custom",
            "customType": SESSION_SLASH_COMMAND_CUSTOM_TYPE,
            "content": "/compact other",
            "display": true,
            "timestamp": 5,
            "details": {
                "command": {"name": "compact", "args": "focus", "text": "/compact focus"},
            },
        });
        assert!(!is_session_slash_command_message(&mismatched));
    }

    #[test]
    fn slash_command_result_rejects_a_json_null_error() {
        // `(message.details.error === undefined || typeof message.details.error === "string")`
        // (`core/messages.ts:529`): `null` is neither, so the message is not a
        // terminal slash-command result.
        let base = serde_json::json!({
            "role": "custom",
            "customType": SESSION_SLASH_COMMAND_RESULT_CUSTOM_TYPE,
            "content": "/compact focus",
            "display": true,
            "timestamp": 5,
            "details": {
                "command": {"name": "compact", "args": "focus", "text": "/compact focus"},
                "success": true,
                "severity": "info",
                "commandEntryId": "entry-1",
            },
        });
        assert!(is_session_slash_command_result_message(&base));
        let with_string_error = serde_json::json!({
            "role": "custom",
            "customType": SESSION_SLASH_COMMAND_RESULT_CUSTOM_TYPE,
            "content": "/compact focus",
            "display": true,
            "timestamp": 5,
            "details": {
                "command": {"name": "compact", "args": "focus", "text": "/compact focus"},
                "success": false,
                "severity": "error",
                "error": "Goal unavailable",
                "commandEntryId": "entry-1",
            },
        });
        assert!(is_session_slash_command_result_message(&with_string_error));
        let mut with_null_error = base.clone();
        with_null_error["details"]["error"] = serde_json::Value::Null;
        assert!(!is_session_slash_command_result_message(&with_null_error));
    }

    #[test]
    fn refinement_outcome_validation_checks_edits() {
        assert!(is_refinement_outcome_message(&json!({
            "role": "custom",
            "customType": REFINEMENT_OUTCOME_CUSTOM_TYPE,
            "content": "Refinement complete: x",
            "display": true,
            "timestamp": 1,
            "details": {
                "summary": "x",
                "scope": "local",
                "edits": [{"action": "update", "kind": "memory", "id": "a", "applied": true}],
            },
        })));
        assert!(!is_refinement_outcome_message(&json!({
            "role": "custom",
            "customType": REFINEMENT_OUTCOME_CUSTOM_TYPE,
            "content": "Refinement complete: x",
            "display": true,
            "timestamp": 1,
            "details": {
                "summary": "x",
                "scope": "local",
                "edits": [{"action": "nope", "kind": "memory", "id": "a", "applied": true}],
            },
        })));
    }

    #[test]
    fn timestamp_parsing_matches_date_parse() {
        assert_eq!(parse_timestamp_millis("1970-01-01T00:00:01.000Z"), 1000);
        assert_eq!(parse_timestamp_millis("not a date"), 0);
    }
}

// ---------------------------------------------------------------------------
// Recovered by the lead: these functions were written by a worker that mistakenly
// pasted them into crates/pi-coding-agent/Cargo.toml instead of this module
// (the real port target for packages/coding-agent/src/core/messages.ts).
// ---------------------------------------------------------------------------

pub fn create_branch_summary_message(
    summary: String,
    from_id: String,
    timestamp: &str,
) -> BranchSummaryMessage {
    BranchSummaryMessage {
        role: "branchSummary".to_string(),
        summary,
        from_id,
        timestamp: parse_timestamp_millis(timestamp),
    }
}

pub fn create_compaction_summary_message(
    summary: String,
    tokens_before: f64,
    timestamp: &str,
    custom_instructions: Option<String>,
    retained_message_count: Option<f64>,
    provider_context: Option<ProviderCompactionCheckpoint>,
    harness_digest: Option<String>,
) -> CompactionSummaryMessage {
    CompactionSummaryMessage {
        role: "compactionSummary".to_string(),
        summary,
        tokens_before,
        retained_message_count,
        provider_context,
        custom_instructions,
        harness_digest,
        timestamp: parse_timestamp_millis(timestamp),
    }
}

/// Convert CustomMessageEntry to AgentMessage format
pub fn create_custom_message(
    custom_type: String,
    content: CustomMessageContent,
    display: bool,
    details: Option<Value>,
    timestamp: &str,
) -> CustomMessage {
    CustomMessage {
        role: "custom".to_string(),
        custom_type,
        content,
        display,
        details,
        timestamp: parse_timestamp_millis(timestamp),
    }
}

pub fn create_session_slash_command_message(
    command: SessionSlashCommand,
    details: SessionSlashCommandDetails,
    display: bool,
    timestamp: i64,
) -> CustomMessage {
    let details = SessionSlashCommandDetails {
        command: command.clone(),
        command_entry_id: details.command_entry_id,
    };
    CustomMessage {
        role: "custom".to_string(),
        custom_type: SESSION_SLASH_COMMAND_CUSTOM_TYPE.to_string(),
        content: CustomMessageContent::Text(command.text),
        display,
        details: serde_json::to_value(&details).ok(),
        timestamp,
    }
}

pub fn create_session_slash_command_result_message(
    content: String,
    details: SessionSlashCommandResultDetails,
    display: bool,
    timestamp: i64,
) -> CustomMessage {
    CustomMessage {
        role: "custom".to_string(),
        custom_type: SESSION_SLASH_COMMAND_RESULT_CUSTOM_TYPE.to_string(),
        content: CustomMessageContent::Text(content),
        display,
        details: serde_json::to_value(&details).ok(),
        timestamp,
    }
}

pub fn create_compaction_outcome_message(
    content: String,
    details: CompactionOutcomeDetails,
    display: bool,
    timestamp: i64,
) -> CustomMessage {
    CustomMessage {
        role: "custom".to_string(),
        custom_type: COMPACTION_OUTCOME_CUSTOM_TYPE.to_string(),
        content: CustomMessageContent::Text(content),
        display,
        details: serde_json::to_value(&details).ok(),
        timestamp,
    }
}

fn refinement_outcome_details(
    result: &RefinementResult,
    source: Option<RefinementSource>,
) -> Value {
    let mut object = Map::new();
    object.insert("refinementId".to_string(), Value::String(result.id.clone()));
    object.insert("summary".to_string(), Value::String(result.summary.clone()));
    object.insert(
        "scope".to_string(),
        Value::String(
            result
                .scope
                .unwrap_or(HarnessScope::Local)
                .as_str()
                .to_string(),
        ),
    );
    if let Some(rollback_of) = &result.rollback_of {
        object.insert("rollbackOf".to_string(), Value::String(rollback_of.clone()));
    }
    object.insert(
        "edits".to_string(),
        serde_json::to_value(&result.applied_edits).unwrap_or(Value::Null),
    );
    if let Some(source) = source {
        object.insert("source".to_string(), Value::String(source));
    }
    Value::Object(object)
}

pub fn create_refinement_outcome_message(
    result: &RefinementResult,
    display: bool,
    timestamp: i64,
) -> CustomMessage {
    CustomMessage {
        role: "custom".to_string(),
        custom_type: REFINEMENT_OUTCOME_CUSTOM_TYPE.to_string(),
        content: CustomMessageContent::Text(format!("Refinement complete: {}", result.summary)),
        display,
        details: Some(refinement_outcome_details(result, None)),
        timestamp,
    }
}

/// Model-facing refinement notice: passes convertToLlm (unlike the refinement_outcome audit entry); display false because the TUI renders the outcome message.
pub fn create_refinement_notice_message(
    result: &RefinementResult,
    source: RefinementSource,
    timestamp: i64,
) -> CustomMessage {
    CustomMessage {
        role: "custom".to_string(),
        custom_type: REFINEMENT_NOTICE_CUSTOM_TYPE.to_string(),
        content: CustomMessageContent::Text(format!(
            "[{source}-refinement]\n\n{}",
            format_refinement_notice_body(result)
        )),
        display: false,
        details: Some(refinement_outcome_details(result, Some(source))),
        timestamp,
    }
}

fn is_record(value: &Value) -> bool {
    value.is_object()
}

fn has_valid_custom_message_envelope(message: &Map<String, Value>, custom_type: &str) -> bool {
    message.get("role").and_then(Value::as_str) == Some("custom")
        && message.get("customType").and_then(Value::as_str) == Some(custom_type)
        && message
            .get("content")
            .map(Value::is_string)
            .unwrap_or(false)
        && message
            .get("display")
            .map(Value::is_boolean)
            .unwrap_or(false)
        && message
            .get("timestamp")
            .and_then(Value::as_f64)
            .map(f64::is_finite)
            .unwrap_or(false)
}

pub fn is_session_slash_command(value: &Value) -> bool {
    let Some(object) = value.as_object() else {
        return false;
    };
    let name_ok = object
        .get("name")
        .and_then(Value::as_str)
        .map(is_session_slash_command_name)
        .unwrap_or(false);
    if !name_ok
        || !object.get("args").map(Value::is_string).unwrap_or(false)
        || !object.get("text").map(Value::is_string).unwrap_or(false)
    {
        return false;
    }
    let name = object
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let args = object
        .get("args")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let text = object
        .get("text")
        .and_then(Value::as_str)
        .unwrap_or_default();
    match parse_session_slash_command(text) {
        Some(parsed) => parsed.name == name && parsed.args == args && parsed.text == text,
        None => false,
    }
}

fn is_valid_command_entry_id(value: Option<&Value>) -> bool {
    // `value === undefined || (typeof value === "string" && value.length > 0)`
    // (`core/messages.ts:504-506`): a missing key passes, a non-empty string passes,
    // and JSON `null` must NOT - `null !== undefined` and `typeof null !== "string"`.
    match value {
        None => true,
        Some(Value::String(text)) => !text.is_empty(),
        Some(_) => false,
    }
}

pub fn is_session_slash_command_message(message: &Value) -> bool {
    let Some(object) = message.as_object() else {
        return false;
    };
    if !has_valid_custom_message_envelope(object, SESSION_SLASH_COMMAND_CUSTOM_TYPE)
        || !object.get("content").map(Value::is_string).unwrap_or(false)
    {
        return false;
    }
    let Some(details) = object.get("details").and_then(Value::as_object) else {
        return false;
    };
    let Some(command) = details.get("command") else {
        return false;
    };
    if !is_session_slash_command(command) {
        return false;
    }
    let content = object
        .get("content")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let command_text = command
        .as_object()
        .and_then(|command| command.get("text"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    content == command_text && is_valid_command_entry_id(details.get("commandEntryId"))
}

pub fn is_session_slash_command_result_message(message: &Value) -> bool {
    let Some(object) = message.as_object() else {
        return false;
    };
    if !has_valid_custom_message_envelope(object, SESSION_SLASH_COMMAND_RESULT_CUSTOM_TYPE) {
        return false;
    }
    let Some(details) = object.get("details").and_then(Value::as_object) else {
        return false;
    };
    let Some(command) = details.get("command") else {
        return false;
    };
    if !is_session_slash_command(command) {
        return false;
    }
    let success_ok = details
        .get("success")
        .map(Value::is_boolean)
        .unwrap_or(false);
    let severity_ok = matches!(
        details.get("severity").and_then(Value::as_str),
        Some("info") | Some("warning") | Some("error")
    );
    // `(message.details.error === undefined || typeof message.details.error === "string")`
    // (`core/messages.ts:529`): a missing key passes, a string passes, and JSON
    // `null` must NOT - `typeof null !== "string"` and `null !== undefined`.
    let error_ok = match details.get("error") {
        None => true,
        Some(Value::String(_)) => true,
        Some(_) => false,
    };
    success_ok
        && severity_ok
        && error_ok
        && is_valid_command_entry_id(details.get("commandEntryId"))
}

pub fn is_compaction_outcome_message(message: &Value) -> bool {
    let Some(object) = message.as_object() else {
        return false;
    };
    if !has_valid_custom_message_envelope(object, COMPACTION_OUTCOME_CUSTOM_TYPE) {
        return false;
    }
    let Some(details) = object.get("details").and_then(Value::as_object) else {
        return false;
    };
    matches!(
        details.get("reason").and_then(Value::as_str),
        Some("threshold") | Some("overflow") | Some("requested")
    ) && matches!(
        details.get("outcome").and_then(Value::as_str),
        Some("skipped") | Some("cancelled") | Some("failed")
    )
}

fn is_applied_refinement_edit(value: &Value) -> bool {
    let Some(object) = value.as_object() else {
        return false;
    };
    matches!(
        object.get("action").and_then(Value::as_str),
        Some("create") | Some("update") | Some("delete")
    ) && object.get("kind").map(Value::is_string).unwrap_or(false)
        && object.get("id").map(Value::is_string).unwrap_or(false)
        && object
            .get("applied")
            .map(Value::is_boolean)
            .unwrap_or(false)
}

pub fn is_refinement_outcome_message(message: &Value) -> bool {
    let Some(object) = message.as_object() else {
        return false;
    };
    if !has_valid_custom_message_envelope(object, REFINEMENT_OUTCOME_CUSTOM_TYPE) {
        return false;
    }
    let Some(details) = object.get("details").and_then(Value::as_object) else {
        return false;
    };
    details
        .get("summary")
        .map(Value::is_string)
        .unwrap_or(false)
        && matches!(
            details.get("scope").and_then(Value::as_str),
            Some("local") | Some("global")
        )
        && match details.get("edits") {
            Some(Value::Array(edits)) => edits.iter().all(is_applied_refinement_edit),
            _ => false,
        }
}

/// Port of `createHeartbeatPromptMessage`; the `AgentCronJob` fields are passed
/// explicitly so this module does not depend on the cron-job module.
pub fn create_heartbeat_prompt_message(
    job_id: String,
    schedule_expression: String,
    status: String,
    run_count: f64,
    prompt: String,
    next_run_at: Option<String>,
    last_run_at: Option<String>,
    timestamp: i64,
) -> CustomMessage {
    let details = HeartbeatPromptDetails {
        job_id,
        schedule: schedule_expression,
        status,
        run_count,
        next_run_at,
        last_run_at,
    };
    CustomMessage {
        role: "custom".to_string(),
        custom_type: HEARTBEAT_PROMPT_CUSTOM_TYPE.to_string(),
        content: CustomMessageContent::Text(prompt),
        display: true,
        details: serde_json::to_value(&details).ok(),
        timestamp,
    }
}

/// Transform AgentMessages (including custom types) to LLM-compatible Messages.
///
/// This is used by:
/// - Agent's transormToLlm option (for prompt calls and queued messages)
/// - Compaction's generateSummary (for summarization)
/// - Custom extensions and tools
/// Mechanical memory snapshots are regenerated after compaction, never summarized.
pub fn without_harness_digests_for_compaction(messages: &[AgentMessage]) -> Vec<AgentMessage> {
    messages
        .iter()
        .filter(|message| {
            !matches!(
                message,
                AgentMessage::Custom(CustomAgentMessage::Custom { custom_type, .. })
                    if custom_type == HARNESS_DIGEST_CUSTOM_TYPE
            )
        })
        .map(|message| match message {
            AgentMessage::Custom(CustomAgentMessage::CompactionSummary {
                summary,
                provider_context,
                tokens_before,
                retained_message_count,
                custom_instructions,
                harness_digest,
                timestamp,
            }) if harness_digest.is_some() => {
                AgentMessage::Custom(CustomAgentMessage::CompactionSummary {
                    summary: summary.clone(),
                    provider_context: provider_context.clone(),
                    tokens_before: *tokens_before,
                    retained_message_count: *retained_message_count,
                    custom_instructions: custom_instructions.clone(),
                    harness_digest: None,
                    timestamp: *timestamp,
                })
            }
            other => other.clone(),
        })
        .collect()
}

/// One `convertToLlm` result item: a message, or the compaction summary pair.
enum ConvertedItem {
    One(Message),
    Two(Message, Message),
    None,
}

pub fn convert_to_llm(
    messages: &[AgentMessage],
    options: &ModelToolOutputPolicyOptions,
) -> Vec<Message> {
    let mut converted: Vec<Message> = Vec::new();
    for message in apply_model_tool_output_policy(messages, options) {
        let item = match &message {
            AgentMessage::Custom(CustomAgentMessage::BashExecution {
                command,
                output,
                exit_code,
                cancelled,
                truncated,
                full_output_path,
                timestamp,
                exclude_from_context,
            }) => {
                if *exclude_from_context == Some(true) {
                    ConvertedItem::None
                } else {
                    let bash = BashExecutionMessage {
                        role: "bashExecution".to_string(),
                        command: command.clone(),
                        output: output.clone(),
                        exit_code: *exit_code,
                        cancelled: *cancelled,
                        truncated: *truncated,
                        full_output_path: full_output_path.clone(),
                        timestamp: *timestamp,
                        exclude_from_context: *exclude_from_context,
                    };
                    ConvertedItem::One(Message::User(UserMessage {
                        role: ROLE_USER.to_string(),
                        content: UserContent::Blocks(vec![ImageOrTextContent::Text(
                            TextContent::new(bash_execution_to_text(&bash)),
                        )]),
                        provider_context: None,
                        timestamp: *timestamp,
                    }))
                }
            }
            AgentMessage::Custom(CustomAgentMessage::Custom {
                custom_type,
                content,
                timestamp,
                ..
            }) => {
                if custom_type == SESSION_SLASH_COMMAND_CUSTOM_TYPE
                    || custom_type == SESSION_SLASH_COMMAND_RESULT_CUSTOM_TYPE
                    || custom_type == COMPACTION_OUTCOME_CUSTOM_TYPE
                    || custom_type == REFINEMENT_OUTCOME_CUSTOM_TYPE
                {
                    ConvertedItem::None
                } else {
                    let blocks = match content {
                        CustomMessageContent::Text(text) => {
                            vec![ImageOrTextContent::Text(TextContent::new(text.clone()))]
                        }
                        CustomMessageContent::Blocks(blocks) => blocks
                            .iter()
                            .map(|block| match block {
                                pi_agent_core::types::ContentBlock::Text(text) => {
                                    ImageOrTextContent::Text(text.clone())
                                }
                                pi_agent_core::types::ContentBlock::Image(image) => {
                                    ImageOrTextContent::Image(image.clone())
                                }
                            })
                            .collect(),
                    };
                    ConvertedItem::One(Message::User(UserMessage {
                        role: ROLE_USER.to_string(),
                        content: UserContent::Blocks(blocks),
                        provider_context: None,
                        timestamp: *timestamp,
                    }))
                }
            }
            AgentMessage::Custom(CustomAgentMessage::BranchSummary {
                summary, timestamp, ..
            }) => ConvertedItem::One(Message::User(UserMessage {
                role: ROLE_USER.to_string(),
                content: UserContent::Blocks(vec![ImageOrTextContent::Text(TextContent::new(
                    format!("{BRANCH_SUMMARY_PREFIX}{summary}{BRANCH_SUMMARY_SUFFIX}"),
                ))]),
                provider_context: None,
                timestamp: *timestamp,
            })),
            AgentMessage::Custom(CustomAgentMessage::CompactionSummary {
                summary,
                provider_context,
                harness_digest,
                timestamp,
                ..
            }) => {
                let digest_block = match harness_digest {
                    Some(digest) => {
                        format!("{HARNESS_DIGEST_PREFIX}{digest}{HARNESS_DIGEST_SUFFIX}\n\n")
                    }
                    None => String::new(),
                };
                let text = (if provider_context.is_some() {
                    String::new()
                } else {
                    digest_block.clone()
                }) + COMPACTION_SUMMARY_PREFIX
                    + summary
                    + COMPACTION_SUMMARY_SUFFIX;
                let carrier = Message::User(UserMessage {
                    role: ROLE_USER.to_string(),
                    content: UserContent::Blocks(vec![ImageOrTextContent::Text(TextContent::new(
                        text,
                    ))]),
                    provider_context: provider_context.clone(),
                    timestamp: *timestamp,
                });
                // Opaque checkpoint replay replaces the carrier's text; memories must be a separate message.
                if provider_context.is_some() && !digest_block.is_empty() {
                    let memory = Message::User(UserMessage {
                        role: ROLE_USER.to_string(),
                        content: UserContent::Blocks(vec![ImageOrTextContent::Text(
                            TextContent::new(digest_block.trim_end().to_string()),
                        )]),
                        provider_context: None,
                        timestamp: *timestamp,
                    });
                    ConvertedItem::Two(carrier, memory)
                } else {
                    ConvertedItem::One(carrier)
                }
            }
            AgentMessage::Message(message) => ConvertedItem::One(message.clone()),
        };
        match item {
            ConvertedItem::One(message) => converted.push(message),
            ConvertedItem::Two(first, second) => {
                converted.push(first);
                converted.push(second);
            }
            ConvertedItem::None => {}
        }
    }
    converted
}
