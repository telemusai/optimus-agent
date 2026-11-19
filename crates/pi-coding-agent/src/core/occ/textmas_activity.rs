//! Provider-neutral, content-free OCC3.1 activity for host presentation.
//!
//! This is not accounting. Real invocations and token usage remain in
//! `TextMasUsageRecord`; queued, skipped and stale are ephemeral display states.
use super::textmas::{TextMasCallStatus, TextMasMode, TextMasRole, TextMasUsageLedger, TextMasUsageRecord};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const TEXTMAS_ACTIVITY_SCHEMA: &str = "occ31-textmas-activity-v1";
const MAX_ACTIVITY_CALLS: usize = 130; // Three roles, 64 mappers and at most 63 reducers.
const MAX_ACTIVITY_DEPENDENCIES: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TextMasActivityStatus {
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
    TimedOut,
    /// A real plan ended before this invocation; no usage reservation was made.
    Skipped,
    /// A durable Started reservation has no live owner after an interruption.
    Stale,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextMasActivityCall {
    pub call_id: String,
    pub role: TextMasRole,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub call_index: Option<u32>,
    pub status: TextMasActivityStatus,
    /// Zero for mappers, one-based for reducer levels; unavailable after restart.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub level: Option<u32>,
    /// Exact input call IDs known at planning time; never inferred at render time.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub depends_on: Vec<String>,
}

impl TextMasActivityCall {
    pub fn queued(
        action_id: &str,
        role: TextMasRole,
        call_index: Option<u32>,
        level: Option<u32>,
        depends_on: Vec<String>,
    ) -> Self {
        Self {
            call_id: textmas_call_id(action_id, role, call_index),
            role,
            call_index,
            status: TextMasActivityStatus::Queued,
            level,
            depends_on,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextMasActivitySnapshot {
    pub schema: String,
    pub session_id: String,
    pub action_id: String,
    /// The actual run mode. It is unknown for disk-only recovery after restart.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<TextMasMode>,
    pub active: bool,
    /// True only when at least one durable Started reservation became stale.
    pub stale: bool,
    pub calls: Vec<TextMasActivityCall>,
}

pub fn textmas_call_id(action_id: &str, role: TextMasRole, call_index: Option<u32>) -> String {
    match call_index {
        Some(index) => format!("{action_id}:{}:{index}", role.as_str()),
        None => format!("{action_id}:{}", role.as_str()),
    }
}

fn role_order(role: TextMasRole) -> u8 {
    match role {
        TextMasRole::Planner => 0,
        TextMasRole::Critic => 1,
        TextMasRole::Refiner => 2,
        TextMasRole::Mapper => 3,
        TextMasRole::Reducer => 4,
    }
}

fn identifier_valid(value: &str, max: usize) -> bool {
    !value.is_empty() && value.len() <= max && !value.chars().any(char::is_control)
}

impl TextMasActivitySnapshot {
    /// Unknown optional activity is ignored by readers; malformed known activity
    /// is rejected instead of being presented as real execution evidence.
    pub fn is_valid(&self) -> bool {
        if self.schema != TEXTMAS_ACTIVITY_SCHEMA
            || !identifier_valid(&self.session_id, 256)
            || !identifier_valid(&self.action_id, 256)
            || self.calls.len() > MAX_ACTIVITY_CALLS
            || self.calls.iter().map(|call| call.depends_on.len()).sum::<usize>() > MAX_ACTIVITY_DEPENDENCIES
        {
            return false;
        }
        let stale = self.calls.iter().any(|call| call.status == TextMasActivityStatus::Stale);
        if self.stale != stale || (self.active && stale) {
            return false;
        }
        let mut ids = BTreeSet::new();
        for call in &self.calls {
            let auxiliary = matches!(call.role, TextMasRole::Mapper | TextMasRole::Reducer);
            if !identifier_valid(&call.call_id, 512)
                || call.call_id != textmas_call_id(&self.action_id, call.role, call.call_index)
                || auxiliary != call.call_index.is_some()
                || !ids.insert(call.call_id.as_str())
                || call.depends_on.iter().any(|id| !identifier_valid(id, 512) || id == &call.call_id)
            {
                return false;
            }
            let status_valid = match call.status {
                TextMasActivityStatus::Queued | TextMasActivityStatus::Running => self.active,
                TextMasActivityStatus::Skipped | TextMasActivityStatus::Stale => !self.active,
                _ => true,
            };
            if !status_valid
                || (call.role == TextMasRole::Mapper && call.level.is_some() && call.level != Some(0))
                || (call.role == TextMasRole::Reducer && call.level == Some(0))
                || (!auxiliary && call.level.is_some())
            {
                return false;
            }
            if matches!(call.status, TextMasActivityStatus::Queued | TextMasActivityStatus::Running | TextMasActivityStatus::Skipped)
                && self.mode.is_none()
            {
                return false;
            }
        }
        let ids: BTreeSet<&str> = self.calls.iter().map(|call| call.call_id.as_str()).collect();
        self.calls.iter().all(|call| call.depends_on.iter().all(|id| ids.contains(id.as_str())))
    }
}

/// Merge a genuinely planned call set with the deduplicated durable call ledger.
/// Plans carry queued calls only. A terminal durable record always wins. An
/// orphaned Started reservation is stale, never relabelled as a cancellation.
pub fn project_textmas_activity(
    session_id: &str,
    action_id: &str,
    mode: Option<TextMasMode>,
    active: bool,
    planned: Vec<TextMasActivityCall>,
    records: Vec<TextMasUsageRecord>,
) -> TextMasActivitySnapshot {
    let mut calls: BTreeMap<String, TextMasActivityCall> = planned
        .into_iter()
        .map(|mut call| {
            call.status = if active { TextMasActivityStatus::Queued } else { TextMasActivityStatus::Skipped };
            (call.call_id.clone(), call)
        })
        .collect();
    let mut ledger = TextMasUsageLedger::default();
    for record in records { ledger.observe(record); }
    for record in ledger.records() {
        let status = match record.status {
            TextMasCallStatus::Started if active => TextMasActivityStatus::Running,
            TextMasCallStatus::Started => TextMasActivityStatus::Stale,
            TextMasCallStatus::Completed => TextMasActivityStatus::Completed,
            TextMasCallStatus::Failed => TextMasActivityStatus::Failed,
            TextMasCallStatus::Cancelled => TextMasActivityStatus::Cancelled,
            TextMasCallStatus::TimedOut => TextMasActivityStatus::TimedOut,
        };
        calls.entry(record.call_id())
            .and_modify(|call| call.status = status)
            .or_insert_with(|| TextMasActivityCall {
                call_id: record.call_id(),
                role: record.role,
                call_index: record.call_index,
                status,
                level: None,
                depends_on: Vec::new(),
            });
    }
    let mut calls: Vec<_> = calls.into_values().collect();
    calls.sort_by_key(|call| (role_order(call.role), call.level, call.call_index, call.call_id.clone()));
    let stale = calls.iter().any(|call| call.status == TextMasActivityStatus::Stale);
    TextMasActivitySnapshot {
        schema: TEXTMAS_ACTIVITY_SCHEMA.into(),
        session_id: session_id.into(),
        action_id: action_id.into(),
        mode,
        active,
        stale,
        calls,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn usage(action: &str, role: TextMasRole, index: Option<u32>, status: TextMasCallStatus) -> TextMasUsageRecord {
        TextMasUsageRecord::new_indexed(action, role, index, status, None)
    }

    #[test]
    fn lifecycle_is_content_free_mode_aware_and_never_duplicates_usage() {
        let planned = TextMasRole::ALL.into_iter().enumerate().map(|(index, role)| {
            let depends = if index == 0 { vec![] } else { vec![textmas_call_id("a", TextMasRole::ALL[index - 1], None)] };
            TextMasActivityCall::queued("a", role, None, None, depends)
        }).collect();
        let activity = project_textmas_activity("s", "a", Some(TextMasMode::Sequential), true, planned, vec![
            usage("a", TextMasRole::Planner, None, TextMasCallStatus::Started),
        ]);
        assert!(activity.is_valid());
        assert!(activity.active);
        assert_eq!(activity.mode, Some(TextMasMode::Sequential));
        assert_eq!(activity.calls[0].status, TextMasActivityStatus::Running);
        assert!(activity.calls[1..].iter().all(|call| call.status == TextMasActivityStatus::Queued));
        let encoded = serde_json::to_value(&activity).unwrap();
        assert!(encoded.get("usage").is_none());
        assert!(encoded["calls"].as_array().unwrap().iter().all(|call| call.get("usage").is_none()));
        assert!(!encoded.to_string().contains("draft"));
    }

    #[test]
    fn failed_run_skips_only_uninvoked_plans_and_keeps_real_terminal_states() {
        let planned = TextMasRole::ALL.into_iter().map(|role| TextMasActivityCall::queued("a", role, None, None, vec![])).collect();
        let activity = project_textmas_activity("s", "a", Some(TextMasMode::Hierarchical), false, planned, vec![
            usage("a", TextMasRole::Planner, None, TextMasCallStatus::Failed),
            usage("a", TextMasRole::Mapper, Some(0), TextMasCallStatus::Cancelled),
            usage("a", TextMasRole::Reducer, Some(0), TextMasCallStatus::TimedOut),
        ]);
        assert!(activity.is_valid());
        assert_eq!(activity.calls.iter().find(|call| call.role == TextMasRole::Planner).unwrap().status, TextMasActivityStatus::Failed);
        assert!(activity.calls.iter().filter(|call| matches!(call.role, TextMasRole::Critic | TextMasRole::Refiner)).all(|call| call.status == TextMasActivityStatus::Skipped));
        assert_eq!(activity.calls.iter().find(|call| call.role == TextMasRole::Mapper).unwrap().status, TextMasActivityStatus::Cancelled);
        assert_eq!(activity.calls.iter().find(|call| call.role == TextMasRole::Reducer).unwrap().status, TextMasActivityStatus::TimedOut);
    }

    #[test]
    fn orphaned_started_is_stale_not_cancelled_and_terminal_ledger_entry_wins() {
        let recovered = project_textmas_activity("s", "a", None, false, vec![], vec![
            usage("a", TextMasRole::Mapper, Some(10), TextMasCallStatus::Started),
            usage("a", TextMasRole::Planner, None, TextMasCallStatus::Started),
            usage("a", TextMasRole::Planner, None, TextMasCallStatus::Completed),
            usage("a", TextMasRole::Planner, None, TextMasCallStatus::Completed),
        ]);
        assert!(recovered.is_valid());
        assert!(recovered.stale);
        assert_eq!(recovered.mode, None);
        assert_eq!(recovered.calls.iter().find(|call| call.role == TextMasRole::Mapper).unwrap().status, TextMasActivityStatus::Stale);
        assert_eq!(recovered.calls.iter().find(|call| call.role == TextMasRole::Planner).unwrap().status, TextMasActivityStatus::Completed);
        assert_eq!(recovered.calls.iter().filter(|call| call.role == TextMasRole::Planner).count(), 1);
        assert!(recovered.calls.iter().all(|call| call.status != TextMasActivityStatus::Cancelled));
    }

    #[test]
    fn exact_reduction_metadata_is_retained_without_inventing_dependencies() {
        let plan = vec![
            TextMasActivityCall::queued("a", TextMasRole::Mapper, Some(0), Some(0), vec![]),
            TextMasActivityCall::queued("a", TextMasRole::Mapper, Some(1), Some(0), vec![]),
            TextMasActivityCall::queued("a", TextMasRole::Reducer, Some(0), Some(1), vec![
                textmas_call_id("a", TextMasRole::Mapper, Some(0)),
                textmas_call_id("a", TextMasRole::Mapper, Some(1)),
            ]),
        ];
        let activity = project_textmas_activity("s", "a", Some(TextMasMode::Hierarchical), true, plan, vec![]);
        assert!(activity.is_valid());
        let reducer = activity.calls.iter().find(|call| call.role == TextMasRole::Reducer).unwrap();
        assert_eq!(reducer.level, Some(1));
        assert_eq!(reducer.depends_on.len(), 2);
        let mut malformed = activity.clone();
        malformed.calls[0].depends_on.push("unknown".into());
        assert!(!malformed.is_valid());
    }
}
