use super::*;
use crate::core::rlm_runtime::RlmCollectResultEntry;

pub(super) struct RlmChildDeletion {
    pub subagent: RlmSubagentRegistryEntry,
    completion: AgentMessageDeferred,
}

impl RlmChildDeletion {
    pub fn new(subagent: &RlmSubagentRegistryEntry) -> Self {
        Self {
            subagent: subagent.clone(),
            completion: create_agent_message_deferred(),
        }
    }
}

impl std::ops::Deref for RlmChildDeletion {
    type Target = AgentMessageDeferred;
    fn deref(&self) -> &Self::Target {
        &self.completion
    }
}

#[derive(Clone)]
pub(super) struct DeletedRlmChild {
    pub subagent: RlmSubagentRegistryEntry,
    pub result: RlmCollectResultEntry,
}

impl AgentSession {
    pub(super) fn remember_deleted_rlm_child(
        &self,
        subagent: &RlmSubagentRegistryEntry,
        run: Option<&RlmChildRun>,
    ) {
        // Retain only bounded receipt data, never a live session or its closures.
        let result = RlmCollectResultEntry {
            rlm_child_id: subagent.rlm_child_id.clone(),
            session_name: Some(subagent.session_name.clone()),
            session_dir: subagent.session_dir.clone(),
            status: RLM_CHILD_AGENT_STATUS_CANCELLED.to_string(),
            settled: true,
            answer_preview: run
                .and_then(|run| run.answer_preview.as_deref())
                .map(|text| compact_rlm_text(text, 160)),
            error: Some(
                run.and_then(|run| run.error.as_deref())
                    .map(|text| compact_rlm_text(text, 2000))
                    .unwrap_or_else(|| "Deleted by parent orchestrator".to_string()),
            ),
            duration_ms: run.and_then(|run| run.duration_ms),
            tool_use_count: run.map(|run| run.tool_use_count),
            replied_since_task: run
                .and_then(|run| run.session.as_ref())
                .and_then(|child| child.replied_to_parent_since_task()),
        };
        self.deleted_rlm_children.lock().unwrap().insert(
            subagent.rlm_child_id.clone(),
            DeletedRlmChild {
                subagent: subagent.clone(),
                result,
            },
        );
    }

    pub(super) fn deleted_rlm_collect_match(
        &self,
        target: &str,
    ) -> Result<Option<RlmCollectResultEntry>, String> {
        let deleted = self.deleted_rlm_children.lock().unwrap().clone();
        let reserved = self
            .deleting_rlm_children
            .lock()
            .unwrap()
            .values()
            .cloned()
            .collect::<Vec<_>>();
        // A replacement still in delete preflight can survive that preflight.
        // Its name must not resolve to an older generation's cancellation.
        if reserved.iter().any(|entry| {
            !deleted.contains_key(&entry.subagent.rlm_child_id)
                && self.rlm_subagent_matches_target(&entry.subagent, target)
        }) {
            return Ok(None);
        }
        let retained = self.rlm_child_sessions.lock().unwrap().clone();
        if retained.iter().any(|(id, child)| {
            !deleted.contains_key(id)
                && child.run.is_none()
                && (id == target
                    || child.session.session_id() == target
                    || child.session.session_name().as_deref() == Some(target))
        }) {
            return Ok(None);
        }
        let mut matches = deleted
            .values()
            .filter(|entry| self.rlm_subagent_matches_target(&entry.subagent, target));
        let result = matches.next().map(|entry| entry.result.clone());
        if matches.next().is_some() {
            return Err(format!(
                "RLM child selector {target:?} is ambiguous in the current parent session"
            ));
        }
        Ok(result)
    }
}
