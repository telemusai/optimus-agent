//! Bounded, process-local usage for the current session; no prompts or credentials.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::types::Usage;

const MAX_SESSIONS: usize = 64;

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionUsage {
    pub requests: u64,
    pub attempts: u64,
    pub in_flight: u64,
    pub completed: u64,
    pub failed: u64,
    pub cancelled: u64,
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub incomplete_usage: bool,
    pub activity: String,
    pub last_latency_ms: Option<u64>,
}

#[derive(Default)]
struct Entry {
    usage: SessionUsage,
    active: HashMap<u64, &'static str>,
}

type Entries = HashMap<String, Arc<Mutex<Entry>>>;

fn entries() -> &'static Mutex<Entries> {
    static ENTRIES: OnceLock<Mutex<Entries>> = OnceLock::new();
    ENTRIES.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn session_usage(session_id: &str) -> Option<SessionUsage> {
    let entry = entries()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(session_id)
        .cloned()?;
    let usage = entry
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .usage
        .clone();
    Some(usage)
}

fn activity(stage: &str) -> &'static str {
    match stage {
        "compaction" | "context_compaction" => "compacting",
        "tool_call" | "tool_requirement" | "provider_request" | "before_provider_request" => {
            "checking tools"
        }
        "code_search" | "code_search_rerank" | "code_line_find" => "searching",
        "skill_suggestion" => "selecting skills",
        "context_relevance" | "memory_relevance" | "retrieval" => "checking context",
        "agent_end" | "turn_end" | "verification" | "citation_check" | "retrieval_safety" => {
            "verifying"
        }
        "guardrails_input" | "guardrails_output" => "checking safety",
        "model_select" => "checking effort",
        _ => "evaluating",
    }
}

/// A dropped future clears in-flight state and records unknown usage.
pub(crate) struct RequestUsage {
    entry: Option<Arc<Mutex<Entry>>>,
    id: u64,
    started: Instant,
    finished: bool,
}

impl RequestUsage {
    pub(crate) fn begin(session_id: &str, stage: &str) -> Self {
        let mut all = entries().lock().unwrap_or_else(|e| e.into_inner());
        let valid = !session_id.is_empty() && session_id.len() <= 128;
        if valid && !all.contains_key(session_id) && all.len() >= MAX_SESSIONS {
            let idle = all.iter().find_map(|(id, entry)| {
                (entry
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .usage
                    .in_flight
                    == 0)
                    .then(|| id.clone())
            });
            if let Some(id) = idle {
                all.remove(&id);
            }
        }
        let entry = if valid && (all.contains_key(session_id) || all.len() < MAX_SESSIONS) {
            Some(all.entry(session_id.to_owned()).or_default().clone())
        } else {
            None
        };
        let mut id = 0;
        if let Some(entry) = &entry {
            let mut entry = entry.lock().unwrap_or_else(|e| e.into_inner());
            entry.usage.requests = entry.usage.requests.saturating_add(1);
            id = entry.usage.requests;
            entry.usage.in_flight = entry.usage.in_flight.saturating_add(1);
            entry.active.insert(id, activity(stage));
            entry.usage.activity = activity(stage).into();
        }
        drop(all);
        Self {
            entry,
            id,
            started: Instant::now(),
            finished: false,
        }
    }

    pub(crate) fn attempt(&self) {
        if let Some(entry) = &self.entry {
            let mut entry = entry.lock().unwrap_or_else(|e| e.into_inner());
            entry.usage.attempts = entry.usage.attempts.saturating_add(1);
        }
    }

    pub(crate) fn unknown_attempt(&self) {
        if let Some(entry) = &self.entry {
            entry
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .usage
                .incomplete_usage = true;
        }
    }

    pub(crate) fn finish(&mut self, usage: Option<&Usage>) {
        self.finished = true;
        if let Some(entry) = &self.entry {
            let mut entry = entry.lock().unwrap_or_else(|e| e.into_inner());
            let status = &mut entry.usage;
            status.last_latency_ms =
                Some(self.started.elapsed().as_millis().min(u64::MAX as u128) as u64);
            if let Some(usage) = usage {
                status.completed = status.completed.saturating_add(1);
                for (total, used) in [
                    (&mut status.input_tokens, usage.input_tokens),
                    (&mut status.output_tokens, usage.output_tokens),
                ] {
                    if let Some(used) = used {
                        *total = Some(total.unwrap_or(0).saturating_add(used));
                    }
                }
                status.incomplete_usage |=
                    usage.input_tokens.is_none() || usage.output_tokens.is_none();
            } else {
                status.failed = status.failed.saturating_add(1);
                status.incomplete_usage = true;
            }
        }
    }
}

impl Drop for RequestUsage {
    fn drop(&mut self) {
        if let Some(entry) = &self.entry {
            let mut entry = entry.lock().unwrap_or_else(|e| e.into_inner());
            entry.active.remove(&self.id);
            entry.usage.in_flight = entry.usage.in_flight.saturating_sub(1);
            if !self.finished {
                entry.usage.cancelled = entry.usage.cancelled.saturating_add(1);
                entry.usage.incomplete_usage = true;
            }
            entry.usage.activity = entry
                .active
                .iter()
                .max_by_key(|(id, _)| *id)
                .map(|(_, label)| *label)
                .unwrap_or("idle")
                .into();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usage_is_session_scoped_and_preserves_unknown_and_measured_zero() {
        let session = uuid::Uuid::new_v4().to_string();
        assert!(session_usage(&session).is_none());
        let mut call = RequestUsage::begin(&session, "compaction");
        call.attempt();
        assert_eq!(session_usage(&session).unwrap().activity, "compacting");
        call.finish(Some(&Usage {
            input_tokens: Some(0),
            output_tokens: Some(12),
        }));
        drop(call);
        let mut call = RequestUsage::begin(&session, "tool_call");
        call.attempt();
        call.attempt();
        call.finish(Some(&Usage {
            input_tokens: Some(30),
            output_tokens: None,
        }));
        drop(call);
        let usage = session_usage(&session).unwrap();
        assert_eq!((usage.requests, usage.attempts, usage.in_flight), (2, 3, 0));
        assert_eq!(
            (usage.input_tokens, usage.output_tokens),
            (Some(30), Some(12))
        );
        assert!(usage.incomplete_usage);
        assert_eq!(usage.activity, "idle");
        assert!(session_usage(&uuid::Uuid::new_v4().to_string()).is_none());
    }

    #[test]
    fn cancelled_request_does_not_leave_checking_or_hide_other_in_flight_work() {
        let session = uuid::Uuid::new_v4().to_string();
        let earlier = RequestUsage::begin(&session, "compaction");
        let later = RequestUsage::begin(&session, "skill_suggestion");
        drop(later);
        let usage = session_usage(&session).unwrap();
        assert_eq!((usage.in_flight, usage.cancelled), (1, 1));
        assert_eq!(usage.activity, "compacting");
        assert_eq!(usage.input_tokens, None);
        drop(earlier);
        assert_eq!(session_usage(&session).unwrap().activity, "idle");
    }
}
