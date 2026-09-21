use super::*;

pub(super) fn is_family_message_action(action: &QueuedSessionAction) -> bool {
    primary_delivery_record(action).ok().is_some_and(|record| {
        matches!(record.message, DeliveryMessage::Custom(custom)
            if custom.custom_type == "agent_message"
                && matches!(custom.details.as_ref().and_then(|details| details.get("fromRelationship"))
                    .and_then(Value::as_str), Some("parent" | "sibling" | "child")))
    })
}

pub(super) fn compatible_family_message_actions(
    first: &QueuedSessionAction,
    next: &QueuedSessionAction,
) -> bool {
    let (QueuedActionPayload::Turn(first_turn), QueuedActionPayload::Turn(next_turn)) =
        (&first.payload, &next.payload)
    else {
        return false;
    };
    // Batch delivery, not message contents: each ordered record keeps its own
    // durable receipt and (for parent instructions) continuation-ledger task.
    is_family_message_action(first)
        && is_family_message_action(next)
        && next.delivery == first.delivery
        && next.wake == first.wake
        && next.effective_priority() == first.effective_priority()
        && next.suppress_autonomous_continuation == first.suppress_autonomous_continuation
        && turn_execution_policies_equal(&first_turn.execution_policy, &next_turn.execution_policy)
}

impl AgentSession {
    pub(super) fn is_dispatched_input(&self, message: &AgentMessage) -> bool {
        if !matches!(message.role(), "user" | "custom") {
            return false;
        }
        self.action_store
            .lock()
            .unwrap()
            .actions_for_message(&delivery_message_of(message))
            .iter()
            .any(|action| {
                matches!(
                    action.lifecycle.state(),
                    ActionLifecycleState::Committing
                        | ActionLifecycleState::Running
                        | ActionLifecycleState::Failed
                )
            })
    }

    // Initial input persistence runs inside the awaited lower-agent callback,
    // before it may request a model response. The queued event carries the same
    // result so later extension/event processing cannot append it a second time.
    pub(super) fn persist_dispatch_message(
        &self,
        message: &AgentMessage,
    ) -> Result<String, String> {
        let mut manager = self.session_manager.lock().unwrap();
        let appended = match message {
            AgentMessage::Custom(CustomAgentMessage::Custom {
                custom_type,
                content,
                display,
                details,
                ..
            }) => {
                use crate::core::session_manager::CustomMessageEntryContent;
                let content = match content {
                    CustomMessageContent::Text(text) => {
                        CustomMessageEntryContent::Text(text.clone())
                    }
                    CustomMessageContent::Blocks(blocks) => CustomMessageEntryContent::Blocks(
                        blocks
                            .iter()
                            .map(|block| match block {
                                pi_agent_core::types::ContentBlock::Text(text) => {
                                    serde_json::json!({"type":"text", "text":text.text})
                                }
                                pi_agent_core::types::ContentBlock::Image(image) => {
                                    serde_json::to_value(image).unwrap_or(Value::Null)
                                }
                            })
                            .collect(),
                    ),
                };
                manager.append_custom_message_entry(
                    custom_type,
                    &content,
                    *display,
                    details.clone(),
                )
            }
            AgentMessage::Message(
                Message::User(_) | Message::Assistant(_) | Message::ToolResult(_),
            ) => manager.append_message(message.clone()),
            _ => return Ok(String::new()),
        };
        let entry_id = appended.and_then(|entry_id| manager.flush_now().map(|()| entry_id))?;
        drop(manager);
        // Parent work must be registered durably before the delivery receipt or
        // provider/tool work, not later in the extension event-processing queue.
        self.begin_rlm_parent_task(message)?;
        Ok(entry_id)
    }

    pub(super) fn record_dispatch_persistence(
        &self,
        message: &AgentMessage,
        persisted: &Result<String, String>,
    ) {
        let mut delivered = Vec::new();
        let mut failed = Vec::new();
        {
            let mut store = self.action_store.lock().unwrap();
            let actions = if persisted.is_err() {
                store.active_actions(None)
            } else if matches!(message.role(), "user" | "custom") {
                store.actions_for_message(&delivery_message_of(message))
            } else {
                Vec::new()
            };
            let key = agent_message_key_of(message);
            for mut action in actions {
                if !matches!(
                    action.lifecycle.state(),
                    ActionLifecycleState::Committing | ActionLifecycleState::Running
                ) {
                    continue;
                }
                let QueuedActionPayload::Turn(turn) = &mut action.payload else {
                    continue;
                };
                match persisted {
                    Ok(_) => {
                        let mut primary = false;
                        for record in &mut turn.base.records {
                            if delivery_message_key_of(&record.message) == key {
                                record.durable = true;
                                primary |= record.role == DeliveryRecordRole::Primary;
                            }
                        }
                        if primary {
                            if action.lifecycle.state() == ActionLifecycleState::Committing {
                                let _ = transition_session_action(
                                    &mut action,
                                    ActionLifecycle::Running {
                                        execution: ActionExecutionAlias::AgentTurn,
                                    },
                                    &TransitionOptions::default(),
                                );
                            }
                            delivered.push((action.clone(), store.ticket_for(&action).ok()));
                        }
                    }
                    Err(error) => {
                        let error = format!("Session transcript persistence failed for action {}: {error}. Unsaved transcript entries remain in memory; this entry was not confirmed saved and will not be replayed automatically.", action.id);
                        let _ = transition_session_action(
                            &mut action,
                            ActionLifecycle::Failed {
                                error: error.clone(),
                            },
                            &TransitionOptions::default(),
                        );
                        failed.push((action.clone(), error, store.ticket_for(&action).ok()));
                    }
                }
                let _ = store.update_action(&action);
            }
        }
        for (action, ticket) in delivered {
            if let Some(ticket) = ticket {
                ticket.settle_delivered(DeliveryOutcome::Delivered);
            }
            self.settle_agent_message(action.agent_message_id.as_deref(), "delivery", None);
        }
        for (action, error, ticket) in &failed {
            if let Some(ticket) = ticket {
                ticket.reject_delivered(error.clone());
            }
            self.settle_agent_message(action.agent_message_id.as_deref(), "delivery", Some(error));
        }
        if !failed.is_empty() {
            self.agent.abort();
            self.notify_session_input_checkpoint_change();
            self.emit_queue_update();
        }
    }

    pub(super) fn has_failed_dispatch_persistence(&self) -> bool {
        self.failed_dispatch_persistence_error().is_some()
    }

    pub(super) fn failed_dispatch_persistence_error(&self) -> Option<String> {
        self.action_store
            .lock()
            .unwrap()
            .owned_actions()
            .iter()
            .find_map(|action| match &action.lifecycle {
                ActionLifecycle::Failed { error }
                    if error.starts_with("Session transcript persistence failed for action ") =>
                {
                    Some(error.clone())
                }
                _ => None,
            })
    }

    pub(super) fn settled_turn_delivery_error(
        &self,
        actions: &[QueuedSessionAction],
    ) -> Option<String> {
        let store = self.action_store.lock().unwrap();
        let current = store.owned_actions();
        for dispatched in actions {
            let action = current.iter().find(|action| action.id == dispatched.id);
            if let Some(action) = action {
                if action.lifecycle.state() == ActionLifecycleState::Cancelled {
                    continue;
                }
                if let ActionLifecycle::Failed { error } = &action.lifecycle {
                    return Some(error.clone());
                }
                // Read the live record by stable identity. The dispatch snapshot
                // predates persistence; active model context can be compacted.
                if let (Ok(original), Ok(receipt)) = (
                    primary_delivery_record(dispatched),
                    primary_delivery_record(action),
                ) {
                    if original.id == receipt.id && receipt.durable {
                        continue;
                    }
                }
            }
            return Some(format!(
                "Session input dispatch settled without durable delivery (action {})",
                dispatched.id
            ));
        }
        None
    }
}
