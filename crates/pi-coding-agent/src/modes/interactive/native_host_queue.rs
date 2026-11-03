//! Native host wiring for TypeScript's serialized queued-message editing flow.

use super::{wire, CustomEditor, InteractiveMode, QueueSelectionItem};
use serde_json::{json, Value};
use std::{
    cell::RefCell,
    collections::VecDeque,
    rc::Rc,
    sync::{mpsc, Arc},
};

enum Mutation {
    Move(i64),
    Edit(String, bool),
}
struct Reply {
    generation: u64,
    revision: u64,
    session: String,
    selected: QueueSelectionItem,
    mutation: Mutation,
    editor_before: String,
    result: Result<(String, wire::AgentConnectionQueueState), String>,
}

pub(super) struct QueueRuntime {
    generation: u64,
    revision: u64,
    mode: Rc<RefCell<InteractiveMode>>,
    editor: Rc<RefCell<CustomEditor>>,
    connection: Arc<dyn wire::AgentConnection>,
    send: mpsc::Sender<Reply>,
    receive: mpsc::Receiver<Reply>,
    session: String,
    pending: bool,
    queued: VecDeque<Mutation>,
    /// The last queue this runtime saw in `connection_state.session_actions`.
    ///
    /// A mutation reply carries a queue snapshot taken when the daemon handled
    /// the request. Any newer state that reaches `session_actions` through a
    /// path that does not emit `session_action_update` - `HostEvent::Completed`
    /// re-applies a full snapshot (`native_host.rs`) - must invalidate older
    /// snapshots, so `poll` compares the live queue against this before it
    /// drains replies.
    observed_queue: crate::modes::interactive::interactive_mode_services::AgentConnectionQueueState,
}

impl QueueRuntime {
    pub(super) fn new(
        mode: Rc<RefCell<InteractiveMode>>,
        editor: Rc<RefCell<CustomEditor>>,
        connection: Arc<dyn wire::AgentConnection>,
    ) -> Self {
        let header = mode.clone();
        editor.borrow_mut().get_header_line = Some(Box::new(move || {
            header.borrow().get_queue_selection_header()
        }));
        let (send, receive) = mpsc::channel();
        let session = mode
            .borrow()
            .connection_state
            .as_ref()
            .map(|s| s.session_id.clone())
            .unwrap_or_default();
        let observed_queue = mode.borrow().get_connection_queue();
        Self {
            generation: 0,
            revision: 0,
            mode,
            editor,
            connection,
            send,
            receive,
            session,
            pending: false,
            queued: VecDeque::new(),
            observed_queue,
        }
    }

    pub(super) fn observe_queue_change(&mut self) {
        self.revision = self.revision.wrapping_add(1);
    }

    pub(super) fn reset_session(&mut self, session: String) {
        self.generation = self.generation.wrapping_add(1);
        self.session = session;
        self.pending = false;
        self.queued.clear();
        self.observed_queue = self.mode.borrow().get_connection_queue();
        self.mode.borrow_mut().queue_selection.reset();
    }

    pub(super) fn handle_input(&mut self, data: &str) -> bool {
        let keys = pi_tui::keybindings::get_keybindings();
        for (key, direction) in [
            ("app.message.navigateOlder", -1),
            ("app.message.navigateNewer", 1),
        ] {
            if keys.matches(data, key) {
                if !self.pending {
                    let text = self.editor.borrow().editor().get_text();
                    let next = self
                        .mode
                        .borrow_mut()
                        .browse_queue_selection(&text, direction);
                    if let Some(next) = next {
                        self.editor.borrow_mut().editor_mut().set_text(&next);
                    }
                }
                return true;
            }
        }
        if !self.mode.borrow().queue_selection.is_browsing() {
            return false;
        }
        for (key, direction) in [
            ("app.message.moveEarlier", -1),
            ("app.message.moveLater", 1),
        ] {
            if keys.matches(data, key) {
                self.queued.push_back(Mutation::Move(direction));
                self.start_next();
                return true;
            }
        }
        if self.pending
            && (keys.matches(data, "tui.input.submit")
                || keys.matches(data, "app.message.followUp"))
        {
            return true;
        }
        if keys.matches(data, "app.input.clear") && !self.mode.borrow().has_interruptible_work() {
            let draft = self.mode.borrow_mut().queue_selection.reset();
            self.editor.borrow_mut().editor_mut().set_text(&draft);
            return true;
        }
        false
    }

    pub(super) fn submit(&mut self, text: &str, follow_up: bool) -> bool {
        if !self.mode.borrow().queue_selection.is_browsing() {
            return false;
        }
        if !self.pending {
            self.queued
                .push_back(Mutation::Edit(text.to_string(), follow_up));
            self.editor.borrow_mut().editor_mut().set_text("");
            self.start_next();
        }
        true
    }

    fn start_next(&mut self) {
        if self.pending {
            return;
        }
        let Some(mutation) = self.queued.pop_front() else {
            return;
        };
        let Some(selected) = self.mode.borrow().queue_selection.selected().cloned() else {
            self.queued.clear();
            return;
        };
        self.pending = true;
        let generation = self.generation;
        let revision = self.revision;
        let editor_before = self.editor.borrow().editor().get_text();
        let session = self.session.clone();
        let connection = self.connection.clone();
        let send = self.send.clone();
        tokio::spawn(async move {
            let result = async {
                if connection.get_state().await?.session_id != session {
                    return Err("Session changed".into());
                }
                let payload = mutation_payload(&mutation);
                let status = connection
                    .mutate_queued_message(
                        selected.lane.as_str(),
                        selected.index as i64,
                        &selected.text,
                        payload,
                    )
                    .await?;
                let queue = connection.get_queue().await?;
                Ok((status, queue))
            }
            .await;
            let _ = send.send(Reply {
                generation,
                revision,
                session,
                selected,
                mutation,
                editor_before,
                result,
            });
        });
    }

    pub(super) fn poll(&mut self) {
        let session = self
            .mode
            .borrow()
            .connection_state
            .as_ref()
            .map(|s| s.session_id.clone())
            .unwrap_or_default();
        if session != self.session {
            self.reset_session(session);
        }
        // A queue change that reaches `session_actions` without a
        // `session_action_update` - `HostEvent::Completed` re-applies a whole
        // snapshot (`native_host.rs`, `apply_connection_state_snapshot`) - must
        // still invalidate reply snapshots taken before it. Otherwise a late
        // mutation reply would overwrite the newer queue with its stale copy.
        let live = self.mode.borrow().get_connection_queue();
        if live != self.observed_queue {
            self.observe_queue_change();
            self.observed_queue = live;
        }
        while let Ok(reply) = self.receive.try_recv() {
            if reply.generation != self.generation || reply.session != self.session {
                continue;
            }
            self.pending = false;
            let untouched = self.editor.borrow().editor().get_text() == reply.editor_before;
            let mut mode = self.mode.borrow_mut();
            let status = match reply.result {
                Ok((status, queue)) => {
                    if reply.revision == self.revision {
                        mode.patch_connection_queue(|state| {
                            state.session_actions.queued_count =
                                queue.steering.len() + queue.follow_up.len();
                            state.session_actions.steering = queue.steering;
                            state.session_actions.follow_ups = queue.follow_up;
                        });
                    }
                    status
                }
                Err(error) => {
                    mode.show_error(&error);
                    "error".into()
                }
            };
            match reply.mutation {
                Mutation::Move(direction) => {
                    let queue = mode.get_connection_queue();
                    let index = if status == "applied" {
                        reply
                            .selected
                            .index
                            .saturating_add_signed(direction as isize)
                    } else {
                        reply.selected.index
                    };
                    let dropped = mode.queue_selection.refresh_at(
                        &queue,
                        reply.selected.lane,
                        index,
                        &reply.selected.text,
                    );
                    if untouched {
                        if let Some(draft) = dropped {
                            self.editor.borrow_mut().editor_mut().set_text(&draft);
                        }
                    }
                    if status != "applied" {
                        mode.show_status(
                            if status == "unsupported" {
                                "Queue editing requires a newer daemon"
                            } else {
                                "Queue changed; reorder not applied"
                            },
                            "dim",
                        );
                    }
                }
                Mutation::Edit(text, _) => {
                    if status == "applied" {
                        if !text.trim().is_empty() {
                            self.editor
                                .borrow_mut()
                                .editor_mut()
                                .add_to_history(text.trim());
                        }
                        let draft = mode.queue_selection.reset();
                        if untouched {
                            self.editor.borrow_mut().editor_mut().set_text(&draft);
                        }
                    } else {
                        if untouched {
                            self.editor.borrow_mut().editor_mut().set_text(&text);
                        }
                        mode.show_status(match status.as_str() {
                            "invalid" => "Edited command is not a valid session command; edit kept in the editor",
                            "unsupported" => "Queue editing requires a newer daemon",
                            _ => "Queue changed; edit kept in the editor",
                        }, "dim");
                    }
                }
            }
        }
        // Whatever is live now has been accounted for; a later comparison must
        // not re-count the change this poll already handled.
        self.observed_queue = self.mode.borrow().get_connection_queue();
        if !self.pending {
            let mut mode = self.mode.borrow_mut();
            if let Some(selected) = mode.queue_selection.selected().cloned() {
                let queue = mode.get_connection_queue();
                let draft = mode.queue_selection.refresh_at(
                    &queue,
                    selected.lane,
                    selected.index,
                    &selected.text,
                );
                if let Some(draft) = draft {
                    if self.editor.borrow().editor().get_text() == selected.text {
                        self.editor.borrow_mut().editor_mut().set_text(&draft);
                    }
                }
            }
        }
        self.start_next();
    }
}

fn mutation_payload(mutation: &Mutation) -> Value {
    match mutation {
        Mutation::Move(direction) => json!({"type": "move", "direction": direction}),
        Mutation::Edit(text, follow_up) if !text.trim().is_empty() => {
            // Lane choice mirrors submit semantics: Enter re-files the
            // edited item as an in-order follow-up, the follow-up key steers.
            let mut value = json!({"type": "replace", "text": text.trim(), "lane": if *follow_up { "steering" } else { "followUp" }});
            // Missing images preserves attachments that this client cannot resolve.
            if crate::modes::interactive::image_markers::image_marker_ids(text).is_empty() {
                value["images"] = json!([]);
            }
            value
        }
        Mutation::Edit(_, _) => json!({"type": "delete"}),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::modes::interactive::components::custom_editor::{CustomEditor, CustomEditorOptions};
    use crate::modes::interactive::interactive_mode::{InteractiveMode, InteractiveModeOptions};
    use crate::modes::interactive::interactive_mode_services::{
        InteractiveModeUiServices, ModelRegistry, SettingsManager,
    };
    use crate::modes::interactive::queue_selection::QueueLane;
    use crate::modes::interactive::theme::theme::init_theme;
    use pi_agent_core::types::{AgentMessage, ThinkingLevel};
    use pi_ai::types::{ImageContent, ServiceTier, Transport};
    use pi_tui::tui::TUI;
    use serde_json::Value;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    /// Test-only `AgentConnection`.
    ///
    /// It is the smallest double the queue and heartbeat paths need: a mutable
    /// queue, a mutable session id, a recorded mutation log and a controllable
    /// heartbeat catalog. Every other method is stubbed generically from the
    /// trait, so the double cannot drift from the contract it stands in for.
    #[derive(Default)]
    struct FakeConnection {
        session_id: Mutex<String>,
        queue: Mutex<wire::AgentConnectionQueueState>,
        mutations: Mutex<Vec<(String, i64, String, Value)>>,
        /// A test-held gate that parks a mutation inside the daemon so reply
        /// ordering can be driven deliberately.
        mutation_gate: Mutex<Option<std::sync::mpsc::Receiver<()>>>,
    }

    fn unsupported<T: Send + 'static>(method: &str) -> pi_ai::types::BoxFuture<Result<T, String>> {
        let method = method.to_string();
        Box::pin(async move { Err(format!("FakeConnection does not implement {method}")) })
    }

    impl FakeConnection {
        fn set_session(&self, session_id: &str) {
            *self.session_id.lock().unwrap() = session_id.to_string();
        }

        fn set_queue(&self, steering: &[&str], follow_up: &[&str]) {
            *self.queue.lock().unwrap() = wire::AgentConnectionQueueState {
                steering: steering.iter().map(|s| (*s).to_string()).collect(),
                follow_up: follow_up.iter().map(|s| (*s).to_string()).collect(),
            };
        }

        fn queue(&self) -> wire::AgentConnectionQueueState {
            self.queue.lock().unwrap().clone()
        }

        fn mutations(&self) -> Vec<(String, i64, String, Value)> {
            self.mutations.lock().unwrap().clone()
        }
    }

    fn wire_state(session_id: &str) -> wire::AgentConnectionState {
        wire::AgentConnectionState {
            session_id: session_id.to_string(),
            active_session_id: Some(format!("active-{session_id}")),
            cwd: "/tmp".into(),
            ..Default::default()
        }
    }

    impl wire::AgentConnection for FakeConnection {
        fn subscribe(
            &self,
            _listener: wire::AgentConnectionEventListener,
        ) -> Box<dyn Fn() + Send + Sync> {
            Box::new(|| {})
        }

        fn on_before_session_invalidate(
            &self,
            _listener: wire::AgentConnectionBeforeSessionInvalidateListener,
        ) -> Box<dyn Fn() + Send + Sync> {
            Box::new(|| {})
        }

        fn get_state(&self) -> pi_ai::types::BoxFuture<Result<wire::AgentConnectionState, String>> {
            let session_id = self.session_id.lock().unwrap().clone();
            Box::pin(async move { Ok(wire_state(&session_id)) })
        }

        fn get_initial_snapshot(
            &self,
        ) -> pi_ai::types::BoxFuture<Result<wire::AgentConnectionSnapshot, String>> {
            let session_id = self.session_id.lock().unwrap().clone();
            Box::pin(async move {
                Ok(wire::AgentConnectionSnapshot {
                    state: wire_state(&session_id),
                    ..Default::default()
                })
            })
        }

        fn get_queue(
            &self,
        ) -> pi_ai::types::BoxFuture<Result<wire::AgentConnectionQueueState, String>> {
            let queue = self.queue.lock().unwrap().clone();
            Box::pin(async move { Ok(queue) })
        }

        fn mutate_queued_message(
            &self,
            lane: &str,
            index: i64,
            expected_text: &str,
            mutation: Value,
        ) -> pi_ai::types::BoxFuture<Result<String, String>> {
            self.mutations.lock().unwrap().push((
                lane.to_string(),
                index,
                expected_text.to_string(),
                mutation.clone(),
            ));
            let gate = self.mutation_gate.lock().unwrap().take();
            let result = {
                let mut queue = self.queue.lock().unwrap();
                let texts = if lane == "steering" {
                    &mut queue.steering
                } else {
                    &mut queue.follow_up
                };
                match texts.get(index as usize) {
                    Some(actual) if actual == expected_text => match mutation["type"].as_str() {
                        Some("replace") => {
                            texts[index as usize] =
                                mutation["text"].as_str().unwrap_or("").to_string();
                            Ok("applied".to_string())
                        }
                        Some("delete") => {
                            texts.remove(index as usize);
                            Ok("applied".to_string())
                        }
                        Some("move") => {
                            let target = index + mutation["direction"].as_i64().unwrap_or(0);
                            if target < 0 || target >= texts.len() as i64 {
                                Ok("rejected".to_string())
                            } else {
                                texts.swap(index as usize, target as usize);
                                Ok("applied".to_string())
                            }
                        }
                        _ => Ok("rejected".to_string()),
                    },
                    _ => Ok("rejected".to_string()),
                }
            };
            Box::pin(async move {
                if let Some(gate) = gate {
                    // The test releases this after delivering the newer event.
                    tokio::task::spawn_blocking(move || gate.recv().ok())
                        .await
                        .ok();
                }
                result
            })
        }

        fn manage_heartbeat(
            &self,
            _active_session_id: &str,
            _job_id: &str,
            _action: Value,
        ) -> pi_ai::types::BoxFuture<Result<Value, String>> {
            Box::pin(async { Ok(Value::Null) })
        }

        fn list_heartbeats(
            &self,
        ) -> pi_ai::types::BoxFuture<Result<Vec<wire::AgentConnectionHeartbeat>, String>> {
            unsupported("list_heartbeats")
        }

        fn dispose(&self) -> pi_ai::types::BoxFuture<Result<(), String>> {
            Box::pin(async { Ok(()) })
        }

        fn get_rlm_child_snapshots(
            &self,
        ) -> pi_ai::types::BoxFuture<Result<Vec<wire::AgentConnectionRlmChildAgentSnapshot>, String>>
        {
            unsupported("get_rlm_child_snapshots")
        }
        fn get_messages(&self) -> pi_ai::types::BoxFuture<Result<Vec<AgentMessage>, String>> {
            unsupported("get_messages")
        }
        fn get_session_header(
            &self,
        ) -> pi_ai::types::BoxFuture<Result<Option<wire::AgentConnectionSessionHeader>, String>>
        {
            unsupported("get_session_header")
        }
        fn get_commands(
            &self,
        ) -> pi_ai::types::BoxFuture<Result<Vec<wire::AgentConnectionSlashCommand>, String>>
        {
            unsupported("get_commands")
        }
        fn get_resource_snapshot(
            &self,
        ) -> pi_ai::types::BoxFuture<Result<wire::AgentConnectionResourceSnapshot, String>>
        {
            unsupported("get_resource_snapshot")
        }
        fn get_model_catalog(
            &self,
        ) -> pi_ai::types::BoxFuture<Result<wire::AgentConnectionModelCatalog, String>> {
            unsupported("get_model_catalog")
        }
        fn get_available_models(
            &self,
        ) -> pi_ai::types::BoxFuture<Result<Vec<wire::AgentConnectionModel>, String>> {
            unsupported("get_available_models")
        }
        fn get_session_stats(&self) -> pi_ai::types::BoxFuture<Result<Value, String>> {
            unsupported("get_session_stats")
        }
        fn get_context_tree(&self) -> pi_ai::types::BoxFuture<Result<Value, String>> {
            unsupported("get_context_tree")
        }
        fn get_session_context(
            &self,
        ) -> pi_ai::types::BoxFuture<Result<wire::AgentConnectionSessionContext, String>> {
            unsupported("get_session_context")
        }
        fn get_session_tree(
            &self,
        ) -> pi_ai::types::BoxFuture<Result<wire::AgentConnectionWatchSessionTree, String>>
        {
            unsupported("get_session_tree")
        }
        fn list_saved_sessions(
            &self,
            _scope: &str,
        ) -> pi_ai::types::BoxFuture<Result<Vec<wire::AgentConnectionSavedSessionInfo>, String>>
        {
            unsupported("list_saved_sessions")
        }
        fn clear_queue(
            &self,
        ) -> pi_ai::types::BoxFuture<Result<wire::AgentConnectionQueueState, String>> {
            unsupported("clear_queue")
        }
        fn abort_and_clear_queue(
            &self,
        ) -> pi_ai::types::BoxFuture<Result<wire::AgentConnectionQueueState, String>> {
            unsupported("abort_and_clear_queue")
        }
        fn acquire_session_input_pause(
            &self,
            _lease_key: &str,
        ) -> pi_ai::types::BoxFuture<Result<wire::AgentConnectionSessionInputPause, String>>
        {
            unsupported("acquire_session_input_pause")
        }
        fn list_cron_jobs(
            &self,
            _include_inactive: bool,
        ) -> pi_ai::types::BoxFuture<Result<Vec<Value>, String>> {
            unsupported("list_cron_jobs")
        }
        fn add_cron_job(
            &self,
            _schedule: &str,
            _prompt: &str,
        ) -> pi_ai::types::BoxFuture<Result<Value, String>> {
            unsupported("add_cron_job")
        }
        fn cancel_cron_job(&self, _job_id: &str) -> pi_ai::types::BoxFuture<Result<Value, String>> {
            unsupported("cancel_cron_job")
        }
        fn get_heartbeat(&self) -> pi_ai::types::BoxFuture<Result<Option<Value>, String>> {
            unsupported("get_heartbeat")
        }
        fn set_heartbeat(
            &self,
            _schedule: &str,
            _instruction: &str,
            _delivery_mode: Option<&str>,
        ) -> pi_ai::types::BoxFuture<Result<Value, String>> {
            unsupported("set_heartbeat")
        }
        fn update_heartbeat(
            &self,
            _action: Value,
        ) -> pi_ai::types::BoxFuture<Result<Option<Value>, String>> {
            unsupported("update_heartbeat")
        }
        fn send_agent_message(
            &self,
            _target_active_session_id: &str,
            _message: &str,
        ) -> pi_ai::types::BoxFuture<Result<Value, String>> {
            unsupported("send_agent_message")
        }
        fn get_agent_message_status(&self) -> pi_ai::types::BoxFuture<Result<Value, String>> {
            unsupported("get_agent_message_status")
        }
        fn pause_agent_messages(&self) -> pi_ai::types::BoxFuture<Result<Value, String>> {
            unsupported("pause_agent_messages")
        }
        fn resume_agent_messages(&self) -> pi_ai::types::BoxFuture<Result<Value, String>> {
            unsupported("resume_agent_messages")
        }
        fn clear_agent_messages(&self) -> pi_ai::types::BoxFuture<Result<f64, String>> {
            unsupported("clear_agent_messages")
        }
        fn get_user_messages_for_forking(
            &self,
        ) -> pi_ai::types::BoxFuture<Result<Vec<wire::AgentConnectionUserMessage>, String>>
        {
            unsupported("get_user_messages_for_forking")
        }
        fn get_last_assistant_text(
            &self,
        ) -> pi_ai::types::BoxFuture<Result<Option<String>, String>> {
            unsupported("get_last_assistant_text")
        }
        fn get_system_prompt(&self) -> pi_ai::types::BoxFuture<Result<String, String>> {
            unsupported("get_system_prompt")
        }
        fn get_tool_definition(
            &self,
            _name: &str,
        ) -> pi_ai::types::BoxFuture<Result<Option<wire::AgentConnectionToolDefinition>, String>>
        {
            unsupported("get_tool_definition")
        }
        fn set_session_entry_label(
            &self,
            _entry_id: &str,
            _label: Option<&str>,
        ) -> pi_ai::types::BoxFuture<Result<(), String>> {
            unsupported("set_session_entry_label")
        }
        fn respond_to_extension_ui_request(
            &self,
            _request_id: &str,
            _response: wire::AgentConnectionExtensionUiResponse,
        ) -> pi_ai::types::BoxFuture<Result<(), String>> {
            unsupported("respond_to_extension_ui_request")
        }
        fn prompt(
            &self,
            _message: &str,
            _options: Option<wire::AgentConnectionPromptOptions>,
        ) -> pi_ai::types::BoxFuture<Result<(), String>> {
            unsupported("prompt")
        }
        fn prompt_and_wait(
            &self,
            _message: &str,
            _options: Option<wire::AgentConnectionPromptOptions>,
        ) -> pi_ai::types::BoxFuture<Result<(), String>> {
            unsupported("prompt_and_wait")
        }
        fn start_side_question(
            &self,
            _id: &str,
            _question: &str,
            _previous_turns: Option<Vec<wire::AgentConnectionSideQuestionTurn>>,
        ) -> pi_ai::types::BoxFuture<Result<(), String>> {
            unsupported("start_side_question")
        }
        fn abort_side_question(&self, _id: &str) -> pi_ai::types::BoxFuture<Result<bool, String>> {
            unsupported("abort_side_question")
        }
        fn steer(
            &self,
            _message: &str,
            _images: Option<Vec<ImageContent>>,
        ) -> pi_ai::types::BoxFuture<Result<(), String>> {
            unsupported("steer")
        }
        fn follow_up(
            &self,
            _message: &str,
            _images: Option<Vec<ImageContent>>,
        ) -> pi_ai::types::BoxFuture<Result<(), String>> {
            unsupported("follow_up")
        }
        fn abort(&self) -> pi_ai::types::BoxFuture<Result<(), String>> {
            unsupported("abort")
        }
        fn cancel_rlm_child(
            &self,
            _child_id: &str,
        ) -> pi_ai::types::BoxFuture<Result<bool, String>> {
            unsupported("cancel_rlm_child")
        }
        fn wait_for_idle(&self) -> pi_ai::types::BoxFuture<Result<(), String>> {
            unsupported("wait_for_idle")
        }
        fn wait_for_headless_completion(
            &self,
            _options: Option<wire::AgentConnectionHeadlessCompletionOptions>,
        ) -> pi_ai::types::BoxFuture<Result<wire::AgentAutonomousStatus, String>> {
            unsupported("wait_for_headless_completion")
        }
        fn execute_bash(
            &self,
            _command: &str,
            _options: Option<wire::AgentConnectionExecuteBashOptions>,
        ) -> pi_ai::types::BoxFuture<Result<(), String>> {
            unsupported("execute_bash")
        }
        fn execute_bash_and_wait(
            &self,
            _command: &str,
        ) -> pi_ai::types::BoxFuture<Result<Value, String>> {
            unsupported("execute_bash_and_wait")
        }
        fn abort_bash(&self) -> pi_ai::types::BoxFuture<Result<(), String>> {
            unsupported("abort_bash")
        }
        fn set_model(
            &self,
            _provider: &str,
            _model_id: &str,
        ) -> pi_ai::types::BoxFuture<Result<wire::AgentConnectionModel, String>> {
            unsupported("set_model")
        }
        fn cycle_model(
            &self,
            _direction: Option<&str>,
        ) -> pi_ai::types::BoxFuture<Result<Option<wire::AgentConnectionModelCycleResult>, String>>
        {
            unsupported("cycle_model")
        }
        fn set_scoped_models(
            &self,
            _scoped_models: Vec<wire::AgentConnectionScopedModel>,
        ) -> pi_ai::types::BoxFuture<Result<(), String>> {
            unsupported("set_scoped_models")
        }
        fn set_thinking_level(
            &self,
            _level: ThinkingLevel,
        ) -> pi_ai::types::BoxFuture<Result<(), String>> {
            unsupported("set_thinking_level")
        }
        fn set_service_tier(
            &self,
            _service_tier: ServiceTier,
        ) -> pi_ai::types::BoxFuture<Result<(), String>> {
            unsupported("set_service_tier")
        }
        fn cycle_thinking_level(
            &self,
        ) -> pi_ai::types::BoxFuture<Result<Option<ThinkingLevel>, String>> {
            unsupported("cycle_thinking_level")
        }
        fn set_transport(
            &self,
            _transport: Transport,
        ) -> pi_ai::types::BoxFuture<Result<(), String>> {
            unsupported("set_transport")
        }
        fn set_steering_mode(&self, _mode: &str) -> pi_ai::types::BoxFuture<Result<(), String>> {
            unsupported("set_steering_mode")
        }
        fn set_follow_up_mode(&self, _mode: &str) -> pi_ai::types::BoxFuture<Result<(), String>> {
            unsupported("set_follow_up_mode")
        }
        fn set_auto_compaction_enabled(
            &self,
            _enabled: bool,
        ) -> pi_ai::types::BoxFuture<Result<(), String>> {
            unsupported("set_auto_compaction_enabled")
        }
        fn set_auto_retry_enabled(
            &self,
            _enabled: bool,
        ) -> pi_ai::types::BoxFuture<Result<(), String>> {
            unsupported("set_auto_retry_enabled")
        }
        fn compact(
            &self,
            _custom_instructions: Option<&str>,
        ) -> pi_ai::types::BoxFuture<Result<Value, String>> {
            unsupported("compact")
        }
        fn refine(&self, _options: Value) -> pi_ai::types::BoxFuture<Result<Value, String>> {
            unsupported("refine")
        }
        fn abort_compaction(&self) -> pi_ai::types::BoxFuture<Result<(), String>> {
            unsupported("abort_compaction")
        }
        fn abort_branch_summary(&self) -> pi_ai::types::BoxFuture<Result<(), String>> {
            unsupported("abort_branch_summary")
        }
        fn abort_retry(&self) -> pi_ai::types::BoxFuture<Result<(), String>> {
            unsupported("abort_retry")
        }
        fn reload(&self) -> pi_ai::types::BoxFuture<Result<(), String>> {
            unsupported("reload")
        }
        fn new_session(
            &self,
            _options: Option<wire::AgentConnectionNewSessionOptions>,
        ) -> pi_ai::types::BoxFuture<Result<bool, String>> {
            unsupported("new_session")
        }
        fn switch_session(
            &self,
            _session_path: &str,
            _options: Option<wire::AgentConnectionSwitchSessionOptions>,
        ) -> pi_ai::types::BoxFuture<Result<bool, String>> {
            unsupported("switch_session")
        }
        fn fork(
            &self,
            _entry_id: &str,
            _options: Option<wire::AgentConnectionForkOptions>,
        ) -> pi_ai::types::BoxFuture<Result<Value, String>> {
            unsupported("fork")
        }
        fn navigate_tree(
            &self,
            _target_id: &str,
            _options: Option<wire::AgentConnectionNavigateTreeOptions>,
        ) -> pi_ai::types::BoxFuture<Result<wire::AgentConnectionNavigateTreeResult, String>>
        {
            unsupported("navigate_tree")
        }
        fn import_from_jsonl(
            &self,
            _input_path: &str,
            _cwd_override: Option<&str>,
        ) -> pi_ai::types::BoxFuture<Result<bool, String>> {
            unsupported("import_from_jsonl")
        }
        fn export_to_html(
            &self,
            _output_path: Option<&str>,
        ) -> pi_ai::types::BoxFuture<Result<String, String>> {
            unsupported("export_to_html")
        }
        fn export_to_jsonl(
            &self,
            _output_path: Option<&str>,
        ) -> pi_ai::types::BoxFuture<Result<String, String>> {
            unsupported("export_to_jsonl")
        }
        fn set_session_name(&self, _name: &str) -> pi_ai::types::BoxFuture<Result<(), String>> {
            unsupported("set_session_name")
        }
        fn get_rlm_max_depth_status(&self) -> pi_ai::types::BoxFuture<Result<Value, String>> {
            unsupported("get_rlm_max_depth_status")
        }
        fn set_rlm_max_depth(
            &self,
            _max_depth: f64,
            _options: Option<Value>,
        ) -> pi_ai::types::BoxFuture<Result<Value, String>> {
            unsupported("set_rlm_max_depth")
        }
        fn rename_saved_session(
            &self,
            _session_path: &str,
            _name: &str,
        ) -> pi_ai::types::BoxFuture<Result<(), String>> {
            unsupported("rename_saved_session")
        }
        fn delete_saved_session(
            &self,
            _session_path: &str,
        ) -> pi_ai::types::BoxFuture<Result<Value, String>> {
            unsupported("delete_saved_session")
        }
        fn watch_session(
            &self,
            _active_session_id: &str,
        ) -> pi_ai::types::BoxFuture<
            Result<Option<Box<dyn wire::AgentConnectionSessionWatcher>>, String>,
        > {
            unsupported("watch_session")
        }
    }
    fn setup() {
        init_theme(Some("prime"), false);
        crate::core::keybindings::KeybindingsManager::new(Default::default(), None).install();
    }

    fn test_mode(session_id: &str) -> InteractiveMode {
        let services = InteractiveModeUiServices {
            settings_manager: Arc::new(Mutex::new(SettingsManager::in_memory(
                serde_json::Map::new(),
            ))),
            model_registry: Arc::new(Mutex::new(ModelRegistry::in_memory())),
            get_initial_cwd: Box::new(|| "/initial".to_string()),
            get_initial_session_name: Box::new(|| Some("initial".to_string())),
            get_themes: Box::new(Vec::new),
            refresh_mcp_providers: None,
        };
        let mut mode = InteractiveMode::new(InteractiveModeOptions {
            migrated_providers: None,
            model_fallback_message: None,
            startup_notice: None,
            initial_message: None,
            initial_images: None,
            initial_messages: None,
            initial_prompts: None,
            verbose: false,
            agent_connection: Arc::new(()),
            daemon_socket_path: None,
            local_session_host: None,
            bind_local_session_extensions: false,
            ui_services: Some(services),
            on_shutdown: None,
            return_to_agents_view: false,
            force_fullscreen: false,
            agents_view_owns_startup_notices: false,
            session_depth: None,
            session_has_children: false,
            prompt_stash_store: None,
            prompt_stash_session_id: None,
        })
        .expect("mode");
        mode.apply_connection_state_snapshot(
            crate::modes::interactive::interactive_mode_services::AgentConnectionState {
                session_id: session_id.to_string(),
                active_session_id: Some(format!("active-{session_id}")),
                ..Default::default()
            },
        );
        mode
    }

    struct Harness {
        runtime: QueueRuntime,
        mode: Rc<RefCell<InteractiveMode>>,
        editor: Rc<RefCell<CustomEditor>>,
        connection: Arc<FakeConnection>,
    }

    fn harness(queue: (&[&str], &[&str])) -> Harness {
        setup();
        let connection = Arc::new(FakeConnection::default());
        connection.set_session("s1");
        connection.set_queue(queue.0, queue.1);
        let mode = Rc::new(RefCell::new(test_mode("s1")));
        {
            let mut borrowed = mode.borrow_mut();
            borrowed.patch_connection_queue(|state| {
                state.session_actions.steering = queue.0.iter().map(|s| (*s).to_string()).collect();
                state.session_actions.follow_ups =
                    queue.1.iter().map(|s| (*s).to_string()).collect();
                state.session_actions.queued_count = queue.0.len() + queue.1.len();
            });
        }
        let tui = Rc::new(RefCell::new(TUI::new(
            Box::new(pi_tui::terminal::ProcessTerminal::new()),
            None,
        )));
        let editor = Rc::new(RefCell::new(CustomEditor::new(
            tui,
            pi_tui::components::editor::EditorTheme {
                border_color: Rc::new(|text| text.to_string()),
                background_color: None,
                autocomplete_background_color: None,
                select_list: super::super::select_theme(),
                command_color: None,
            },
            CustomEditorOptions::default(),
        )));
        let runtime = QueueRuntime::new(
            mode.clone(),
            editor.clone(),
            connection.clone() as Arc<dyn wire::AgentConnection>,
        );
        Harness {
            runtime,
            mode,
            editor,
            connection,
        }
    }

    impl Harness {
        fn editor_text(&self) -> String {
            self.editor.borrow().editor().get_text()
        }

        fn set_editor_text(&self, text: &str) {
            self.editor.borrow_mut().editor_mut().set_text(text);
        }

        fn pending(&self) -> bool {
            self.runtime.pending
        }

        /// `session_action_update` with a fresh queue snapshot, matching
        /// `apply_event` in `native_host.rs`.
        fn deliver_queue_event(&mut self, steering: &[&str], follow_up: &[&str]) {
            self.runtime.observe_queue_change();
            let transcript = Rc::new(RefCell::new(super::super::Transcript::new(self.mode.clone())));
            super::super::apply_event(&self.mode, &transcript, wire::AgentConnectionSessionEvent::SessionActionUpdate {
                actions: json!({"steering": steering, "followUps": follow_up, "queuedCount": steering.len() + follow_up.len()}),
            });
        }

        fn painted(&self) -> String {
            super::super::ui_tests::painted_transcript(self.mode.clone(), self.editor.clone())
        }

        fn queue(&self) -> (Vec<String>, Vec<String>) {
            let queue = self.mode.borrow().get_connection_queue();
            (queue.steering, queue.follow_up)
        }

        fn selected(&self) -> Option<QueueSelectionItem> {
            self.mode.borrow().queue_selection.selected().cloned()
        }

        fn is_browsing(&self) -> bool {
            self.mode.borrow().queue_selection.is_browsing()
        }

        /// Every rendered chat line, so a status message can be asserted without
        /// depending on private component internals.
        fn statuses(&self) -> Vec<String> {
            self.mode
                .borrow()
                .chat_container
                .children
                .iter()
                .flat_map(|child| child.render(200))
                .collect()
        }
    }

    /// Polls the runtime for a fixed number of iterations without any
    /// condition, so a test can assert that nothing happened.
    async fn poll_for(harness: &mut Harness, iterations: usize) {
        for _ in 0..iterations {
            harness.runtime.poll();
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    }

    /// Waits for a predicate on the polling loop, then polls once more so the
    /// runtime observes the reply.
    async fn settle(harness: &mut Harness, predicate: impl Fn(&Harness) -> bool) {
        for _ in 0..2_000 {
            harness.runtime.poll();
            if predicate(harness) {
                return;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        harness.runtime.poll();
        panic!("condition was never reached");
    }

    /// The terminal bytes for a binding's first default key. The helper checks
    /// itself against the keybinding matcher, so a wrong byte sequence fails
    /// here with the real reason instead of as a confusing assertion later.
    fn key_data(binding: &str) -> String {
        let keys = pi_tui::keybindings::get_keybindings();
        let name = keys
            .get_keys(binding)
            .first()
            .cloned()
            .unwrap_or_else(|| panic!("no key configured for {binding}"));
        let parts: Vec<&str> = name.split('+').collect();
        let base = *parts.last().expect("key name");
        let alt = parts.contains(&"alt");
        let ctrl = parts.contains(&"ctrl");
        let data = match base {
            "up" | "down" => {
                let letter = if base == "up" { 'A' } else { 'B' };
                if alt || ctrl {
                    // Kitty encodes the modifier field as flags + 1, where
                    // shift = 1, alt = 2, ctrl = 4 (`pi-tui/src/keys.rs`).
                    let flags = 2 * i32::from(alt) + 4 * i32::from(ctrl);
                    format!("\x1b[1;{}{letter}", flags + 1)
                } else {
                    format!("\x1b[{letter}")
                }
            }
            "escape" | "esc" => "\x1b".to_string(),
            "enter" | "return" => {
                if alt {
                    "\x1b\r".to_string()
                } else {
                    "\r".to_string()
                }
            }
            other => panic!("key_data has no byte sequence for {other:?} ({binding})"),
        };
        assert!(
            keys.matches(&data, binding),
            "{binding} ({name:?}) must accept its own key data {data:?}"
        );
        data
    }

    fn browse_to_follow_up(harness: &mut Harness, draft: &str) {
        harness.set_editor_text(draft);
        let older = key_data("app.message.navigateOlder");
        assert!(harness.runtime.handle_input(&older));
    }

    /// `interactive-mode.ts:7326-7342` and `test/interactive-queue-edit.test.ts:527-540`:
    /// a move mirrors locally and the reply must not undo it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_move_mirrors_the_queue_and_keeps_the_selection() {
        let mut harness = harness((&["s1", "s2"], &[]));
        browse_to_follow_up(&mut harness, "draft");
        assert_eq!(harness.editor_text(), "s2");
        assert_eq!(
            harness.selected(),
            Some(QueueSelectionItem {
                lane: QueueLane::Steering,
                index: 1,
                text: "s2".into()
            })
        );

        let earlier = key_data("app.message.moveEarlier");
        assert!(harness.runtime.handle_input(&earlier));
        settle(&mut harness, |h| !h.pending()).await;

        assert_eq!(harness.connection.queue().steering, vec!["s2", "s1"]);
        assert_eq!(harness.queue(), (vec!["s2".into(), "s1".into()], vec![]));
        let painted = harness.painted();
        assert!(painted.find("Steering: s2").unwrap() < painted.find("Steering: s1").unwrap(), "{painted}");
        assert_eq!(
            harness.selected(),
            Some(QueueSelectionItem {
                lane: QueueLane::Steering,
                index: 0,
                text: "s2".into()
            })
        );
        assert_eq!(
            harness.connection.mutations()[0].3,
            json!({"type": "move", "direction": -1})
        );
    }

    /// The flagged risk: a NEWER queue event that arrives before the LATE reply
    /// of an older mutation must win. `interactive-mode.ts:7326-7327` only
    /// mirrors when no event has replaced `sessionActions`, and :7338-7342
    /// re-derives the selection from whatever is live.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_newer_queue_event_survives_a_late_mutation_snapshot() {
        let mut harness = harness((&["s1", "s2"], &[]));
        browse_to_follow_up(&mut harness, "draft");
        // Gate the mutation so the test controls reply ordering.
        let (release, gate) = std::sync::mpsc::channel();
        *harness.connection.mutation_gate.lock().unwrap() = Some(gate);

        let earlier = key_data("app.message.moveEarlier");
        harness.runtime.handle_input(&earlier);
        // The worker thread records the call, then blocks on the gate.
        for _ in 0..2_000 {
            if !harness.connection.mutations().is_empty() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        assert_eq!(harness.connection.mutations().len(), 1);

        // A newer queue arrives first: the user submitted the queued item and it
        // is gone, while a fresh message joined the steering lane.
        harness.deliver_queue_event(&["fresh"], &[]);

        // Now the late reply lands with its stale snapshot.
        release.send(()).unwrap();
        settle(&mut harness, |h| !h.pending()).await;

        // The newer state must survive: the stale snapshot must not restore
        // ["s2", "s1"] and the selection must drop because "s2" is gone.
        assert_eq!(
            harness.queue(),
            (vec!["fresh".into()], vec![]),
            "the stale mutation snapshot overwrote the newer queue"
        );
        assert!(
            !harness.is_browsing(),
            "the selection must drop once its item is gone"
        );
        assert_eq!(harness.editor_text(), "draft", "the draft is restored");
        let painted = harness.painted();
        assert!(painted.contains("Steering: fresh"), "{painted}");
        assert!(!painted.contains("Steering: s1") && !painted.contains("Steering: s2"));
    }

    /// The same guard must not fire when the live queue simply *is* the reply
    /// snapshot: a normal successful move must still be applied.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_reply_snapshot_applies_when_no_newer_event_arrived() {
        let mut harness = harness((&["s1", "s2"], &[]));
        browse_to_follow_up(&mut harness, "draft");
        let earlier = key_data("app.message.moveEarlier");
        harness.runtime.handle_input(&earlier);
        settle(&mut harness, |h| !h.pending()).await;
        assert_eq!(harness.queue(), (vec!["s2".into(), "s1".into()], vec![]));
        assert_eq!(
            harness.selected(),
            Some(QueueSelectionItem {
                lane: QueueLane::Steering,
                index: 0,
                text: "s2".into()
            })
        );
    }

    /// `interactive-mode.ts:7350-7353`: a failed mutation refreshes from live
    /// state instead of applying its own stale snapshot.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_rejected_move_keeps_the_live_queue_and_reports_the_failure() {
        let mut harness = harness((&["s1", "s2"], &[]));
        browse_to_follow_up(&mut harness, "draft");
        // The entry moved while the request was in flight: index 1 no longer
        // holds "s2", so the daemon rejects the move.
        harness.connection.set_queue(&["s1"], &[]);
        harness.deliver_queue_event(&["s1"], &[]);

        let earlier = key_data("app.message.moveEarlier");
        harness.runtime.handle_input(&earlier);
        settle(&mut harness, |h| !h.pending()).await;

        assert_eq!(
            harness.queue(),
            (vec!["s1".into()], vec![]),
            "no stale restore"
        );
        assert!(harness
            .statuses()
            .iter()
            .any(|line| line.contains("Queue changed; reorder not applied")));
    }

    /// A session switch must drop the selection, the stash and any queued
    /// mutation, and must ignore a reply from the previous session
    /// (`interactive-mode.ts:2985-2989`, :7322-7324).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_session_switch_discards_the_selection_and_ignores_old_replies() {
        let mut harness = harness((&["s1", "s2"], &[]));
        browse_to_follow_up(&mut harness, "draft");
        let (release, gate) = std::sync::mpsc::channel();
        *harness.connection.mutation_gate.lock().unwrap() = Some(gate);

        let earlier = key_data("app.message.moveEarlier");
        harness.runtime.handle_input(&earlier);
        for _ in 0..2_000 {
            if !harness.connection.mutations().is_empty() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }

        // The session is replaced before the reply lands.
        harness.connection.set_session("s2");
        harness.connection.set_queue(&["new"], &[]);
        {
            let mut mode = harness.mode.borrow_mut();
            mode.apply_connection_state_snapshot(
                crate::modes::interactive::interactive_mode_services::AgentConnectionState {
                    session_id: "s2".into(),
                    active_session_id: Some("active-s2".into()),
                    session_actions:
                        crate::modes::interactive::interactive_mode_services::SessionActionSnapshot {
                            steering: vec!["new".into()],
                            follow_ups: vec![],
                            queued_count: 1,
                            active: None,
                        },
                    ..Default::default()
                },
            );
        }
        harness.runtime.reset_session("s2".to_string());
        // The host clears every editor on a session reset; stand in for that so
        // a revived draft would be visible.
        harness.set_editor_text("");
        release.send(()).unwrap();
        // Poll long enough that the gated reply is certainly delivered and
        // consumed, so "nothing happened" is a real observation.
        poll_for(&mut harness, 200).await;

        assert!(
            !harness.is_browsing(),
            "the old selection must be discarded"
        );
        assert_eq!(harness.queue(), (vec!["new".into()], vec![]));
        let painted = harness.painted();
        assert!(painted.contains("Steering: new"), "{painted}");
        assert!(!painted.contains("Steering: s1") && !painted.contains("Steering: s2"));
        assert_eq!(
            harness.editor_text(),
            "",
            "a reply from the previous session must not restore the old draft"
        );
    }

    /// `interactive-mode.ts:7422-7425`: an applied edit resets the selection and
    /// restores the stashed draft.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_applied_edit_restores_the_stashed_draft() {
        let mut harness = harness((&["s1"], &["f1"]));
        browse_to_follow_up(&mut harness, "typed draft");
        assert_eq!(harness.editor_text(), "f1");
        assert!(harness.mode.borrow().queue_selection.has_draft());

        harness.set_editor_text("f1 edited");
        assert!(harness.runtime.submit("f1 edited", false));
        settle(&mut harness, |h| !h.pending()).await;

        assert_eq!(
            harness.connection.mutations()[0].3,
            json!({"type": "replace", "text": "f1 edited", "lane": "followUp", "images": []})
        );
        assert_eq!(harness.editor_text(), "typed draft");
        assert!(!harness.is_browsing());
        assert!(!harness.mode.borrow().queue_selection.has_draft());
        let painted = harness.painted();
        assert!(painted.contains("Follow-up: f1 edited"), "{painted}");
        assert!(!painted.lines().any(|line| line.trim() == "Follow-up: f1"));
    }

    /// Empty text deletes the selected entry
    /// (`interactive-mode.ts:7329-7341`; `queue-selection.ts:72`).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_empty_edit_deletes_the_selected_entry() {
        let mut harness = harness((&["s1"], &["f1"]));
        browse_to_follow_up(&mut harness, "draft");
        harness.set_editor_text("   ");
        assert!(harness.runtime.submit("   ", false));
        settle(&mut harness, |h| !h.pending()).await;

        assert_eq!(
            harness.connection.mutations()[0].3,
            json!({"type": "delete"})
        );
        assert_eq!(harness.connection.queue().follow_up, Vec::<String>::new());
        assert_eq!(harness.queue(), (vec!["s1".into()], vec![]));
        assert_eq!(harness.editor_text(), "draft");
        let painted = harness.painted();
        assert!(painted.contains("Steering: s1"), "{painted}");
        assert!(!painted.contains("Follow-up: f1"));
    }

    /// `interactive-mode.ts:7426-7436`: a rejected edit is kept in the editor so
    /// the work is never swallowed.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_rejected_edit_is_kept_in_the_editor() {
        let mut harness = harness((&["s1"], &[]));
        browse_to_follow_up(&mut harness, "draft");
        // The entry was replaced while the edit was in flight, so the expected
        // text at index 0 no longer matches and the daemon rejects the edit.
        harness.connection.set_queue(&["changed"], &[]);
        harness.deliver_queue_event(&["changed"], &[]);

        assert!(harness.runtime.submit("s1 edited", false));
        settle(&mut harness, |h| !h.pending()).await;

        assert_eq!(harness.editor_text(), "s1 edited");
        assert!(harness
            .statuses()
            .iter()
            .any(|line| line.contains("Queue changed; edit kept in the editor")));
    }

    /// A reply must not clobber text the user typed while it was in flight
    /// (`interactive-mode.ts:7406-7410`, :7420-7429).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_reply_does_not_clobber_newer_typing() {
        let mut harness = harness((&["s1"], &[]));
        browse_to_follow_up(&mut harness, "draft");
        harness.set_editor_text("s1 edited");
        assert!(harness.runtime.submit("s1 edited", false));
        // The user keeps typing before the reply lands.
        harness.set_editor_text("s1 edited plus more");
        settle(&mut harness, |h| !h.pending()).await;

        assert_eq!(harness.editor_text(), "s1 edited plus more");
    }

    /// `queue-selection.ts:42-61`: browse order is newest-first and the draft is
    /// stashed on the first step only.
    #[test]
    fn browsing_preserves_the_draft_and_stops_at_the_boundary() {
        let mut harness = harness((&["s1"], &["f1", "f2"]));
        browse_to_follow_up(&mut harness, "draft");
        assert_eq!(harness.editor_text(), "f2");
        let older = key_data("app.message.navigateOlder");
        harness.runtime.handle_input(&older);
        assert_eq!(harness.editor_text(), "f1");
        harness.runtime.handle_input(&older);
        assert_eq!(harness.editor_text(), "s1");
        // At the oldest entry, older is a noop.
        harness.runtime.handle_input(&older);
        assert_eq!(harness.editor_text(), "s1");

        let newer = key_data("app.message.navigateNewer");
        harness.runtime.handle_input(&newer);
        assert_eq!(harness.editor_text(), "f1");
        harness.runtime.handle_input(&newer);
        assert_eq!(harness.editor_text(), "f2");
        harness.runtime.handle_input(&newer);
        assert_eq!(harness.editor_text(), "draft");
        assert!(!harness.is_browsing());
    }

    /// Queued-only work is interruptible even before streaming starts. Escape
    /// must reach the host's interrupt path instead of silently clearing input.
    #[test]
    fn clearing_queued_only_work_defers_to_interrupt_without_deleting() {
        let mut harness = harness((&["s1"], &[]));
        browse_to_follow_up(&mut harness, "typed draft");
        assert_eq!(harness.editor_text(), "s1");
        let clear = key_data("app.input.clear");
        assert!(harness.mode.borrow().has_interruptible_work());
        assert!(!harness.runtime.handle_input(&clear));
        assert_eq!(harness.editor_text(), "s1");
        assert!(harness.is_browsing());
        assert!(harness.mode.borrow().queue_selection.has_draft());
        assert!(
            harness.connection.mutations().is_empty(),
            "no delete was sent"
        );
    }

    /// Attachments: a marker this client can resolve is sent, an unresolvable
    /// one leaves `images` absent so the server keeps its own copy
    /// (`interactive-mode.ts:7454-7461`).
    #[test]
    fn a_replace_sends_resolvable_images_and_omits_unresolvable_ones() {
        assert_eq!(
            mutation_payload(&Mutation::Edit("[image #7] look".into(), false)),
            json!({"type": "replace", "text": "[image #7] look", "lane": "followUp"})
        );
        // No marker at all means the queue's own images are cleared.
        assert_eq!(
            mutation_payload(&Mutation::Edit("plain".into(), false)),
            json!({"type": "replace", "text": "plain", "lane": "followUp", "images": []})
        );
    }

    fn heartbeat(
        id: &str,
        active_session_id: &str,
        session_id: &str,
    ) -> wire::AgentConnectionHeartbeat {
        wire::AgentConnectionHeartbeat {
            job: json!({
                "id": id,
                "status": "active",
                "source": "heartbeat",
                "activeSessionId": active_session_id,
                "sessionId": session_id,
                "sessionFile": format!("/tmp/{id}.jsonl"),
                "cwd": "/tmp",
                "prompt": "check the session",
                "schedule": {"kind": "interval", "expression": "every 5m", "intervalMs": 300000},
                "createdAt": "2026-01-01T00:00:00.000Z",
                "updatedAt": "2026-01-01T00:00:00.000Z",
                "nextRunAt": "2026-01-01T00:05:00.000Z",
                "runCount": 0.0
            }),
            session_name: None,
            first_message: None,
        }
    }

    #[test]
    fn queue_edits_keep_the_typescript_mutation_contract() {
        assert_eq!(
            mutation_payload(&Mutation::Edit(" \n ".into(), false)),
            json!({"type":"delete"})
        );
        assert_eq!(
            mutation_payload(&Mutation::Edit(" updated ".into(), true)),
            json!({"type":"replace","text":"updated","lane":"steering","images":[]})
        );
        assert!(
            mutation_payload(&Mutation::Edit("[image #1] inspect".into(), false))
                .get("images")
                .is_none()
        );
        assert_eq!(
            mutation_payload(&Mutation::Move(-1)),
            json!({"type":"move","direction":-1})
        );
    }
}
