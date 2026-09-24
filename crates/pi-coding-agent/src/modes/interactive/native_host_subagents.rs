//! TypeScript's mounted subagent summary, roster subscription and scoped handoff.

use super::*;
use crate::modes::agent_connection::daemon_agent_connection::AgentConnectionRosterStore;
use crate::modes::interactive::components::subagent_summary_line::{
    RosterParent, SubagentSummaryCounts, SubagentSummaryLine,
};
use pi_tui::tui::Focusable;

#[path = "jev_activity.rs"]
mod jev_activity;

#[derive(Default)]
struct JevBarState {
    session_id: String,
    status: Option<crate::modes::interactive::components::subagent_summary_line::RightStatus>,
}

impl JevBarState {
    fn select(&mut self, session_id: String) {
        if self.session_id != session_id {
            self.session_id = session_id;
            self.status = None;
        }
    }

    fn update(&mut self, session_id: &str, status: Option<crate::modes::interactive::components::subagent_summary_line::RightStatus>) -> bool {
        if self.session_id != session_id || self.status == status { return false; }
        self.status = status;
        true
    }
}

pub(super) struct Bar {
    mode: Rc<RefCell<InteractiveMode>>,
    editor: Option<Rc<RefCell<CustomEditor>>>,
    line: SubagentSummaryLine,
    roster: Arc<Mutex<Option<Arc<dyn AgentConnectionRosterStore>>>>,
    jev: Arc<Mutex<JevBarState>>,
    jev_task: RefCell<Option<tokio::task::JoinHandle<()>>>,
}

impl Bar {
    pub(super) fn new(mode: Rc<RefCell<InteractiveMode>>) -> Self {
        let mut line = SubagentSummaryLine::default();
        line.set_always_visible(true);
        // The bar exists only in the workspace-attached terminal, where the
        // sessions sidebar always serves the roster the old agents view did.
        // `options.return_to_agents_view` describes that retired hand-off and is
        // forced false by the workspace loop, so keying `openable` on it
        // disabled the editor's Down-arrow path (UI006).
        line.set_openable(true);
        Self {
            mode,
            editor: None,
            line,
            roster: Arc::new(Mutex::new(None)),
            jev: Arc::new(Mutex::new(JevBarState::default())),
            jev_task: RefCell::new(None),
        }
    }

    /// Reuse the bar's cached status; never trigger another poll for the header.
    pub(super) fn compact_jev_status(&self) -> Option<String> {
        let mode = self.mode.borrow();
        let session_id = mode.connection_state.as_ref()?.session_id.as_str();
        let state = self.jev.lock().unwrap_or_else(|e| e.into_inner());
        if state.session_id != session_id { return None; }
        state.status.as_ref().map(|status| status.minimal.clone())
    }

    pub(super) fn subscribe(
        &self,
        connection: Arc<dyn wire::AgentConnection>,
        send: mpsc::Sender<HostEvent>,
    ) {
        let jev = self.jev.clone();
        let status_connection = connection.clone();
        let status_send = send.clone();
        let task = tokio::spawn(async move {
            loop {
                let session_id = jev.lock().unwrap_or_else(|e| e.into_inner()).session_id.clone();
                if !session_id.is_empty() {
                    let settings = pi_jev::config::JevSettingsStore::new(crate::config::get_agent_dir()).load();
                    let mode = settings.effective_mode(&session_id);
                    let compaction = settings.effective_compaction_enabled(&session_id);
                    let response = if mode.is_enabled() || compaction {
                        tokio::time::timeout(Duration::from_secs(2), status_connection.get_jev_status())
                            .await.ok().and_then(Result::ok).flatten()
                    } else { None };
                    let status = jev_activity::status(mode, settings.full_jev_active(), compaction, response.as_ref());
                    let changed = {
                        let mut state = jev.lock().unwrap_or_else(|e| e.into_inner());
                        state.update(&session_id, status)
                    };
                    if changed && status_send.send(HostEvent::Render).is_err() { break; }
                }
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        });
        if let Some(previous) = self.jev_task.borrow_mut().replace(task) { previous.abort(); }
        let slot = self.roster.clone();
        tokio::spawn(async move {
            let update = send.clone();
            if let Ok(store) = connection
                .subscribe_agent_roster(Arc::new(move || {
                    let _ = update.send(HostEvent::Render);
                }))
                .await
            {
                *slot.lock().unwrap() = Some(store);
                let _ = send.send(HostEvent::Render);
            }
        });
    }

    fn counts(&self) -> SubagentSummaryCounts {
        use crate::modes::interactive::components::subagent_summary_line::count_roster_subagent_statuses;
        let mode = self.mode.borrow();
        let summaries = self
            .roster
            .lock()
            .unwrap()
            .as_ref()
            .map(|store| store.summaries());
        if let (Some(values), Some(state)) = (summaries, &mode.connection_state) {
            let rows: Vec<crate::modes::daemon::daemon_session_list::SessionSummary> = values
                .into_iter()
                .filter_map(|row| serde_json::from_value(crate::modes::agents_view::native_wire::normalize_browser_numbers(row)).ok())
                .collect();
            if rows.iter().any(|row| row.session_id == state.session_id) {
                return count_roster_subagent_statuses(
                    &rows,
                    &RosterParent {
                        active_session_id: state.active_session_id.clone(),
                        session_id: Some(state.session_id.clone()),
                        session_file: state.session_file.clone(),
                    },
                );
            }
        }
        let children: Vec<wire::AgentConnectionRlmChildAgentSnapshot> = mode.subagent_snapshots.values().map(|child| {
            wire::AgentConnectionRlmChildAgentSnapshot {
                id: child.id.clone(), parent_id: child.parent_id.clone(),
                active_session_id: child.active_session_id.clone(), status: child.status.clone(),
                activity: child.activity.as_ref().map(|kind| wire::AgentConnectionRlmChildAgentActivity { kind: kind.clone(), ..Default::default() }),
                ..Default::default()
            }
        }).collect();
        crate::modes::interactive::components::subagent_summary_line::count_direct_subagent_statuses(
            &children, mode.rlm_node_id.as_deref(),
        )
    }

    /// Returns true when the tray consumed input; all other input stays in chat.
    pub(super) fn input(
        &mut self,
        data: &str,
        editor: &Rc<RefCell<CustomEditor>>,
        actions: &Rc<RefCell<Vec<InputAction>>>,
    ) -> bool {
        self.editor = Some(editor.clone());
        self.line.set_subagent_counts(self.counts());
        let keys = pi_tui::keybindings::get_keybindings();
        if self.line.focused() {
            if !self.line.is_selectable() {
                self.line.set_focused(false);
                editor.borrow_mut().editor_mut().set_focused(true);
                return false;
            }
            if keys.matches(data, "tui.select.confirm") || keys.matches(data, "app.agents.open") {
                self.line.set_focused(false);
                actions.borrow_mut().push(InputAction::Subagents);
                return true;
            }
            self.line.set_focused(false);
            editor.borrow_mut().editor_mut().set_focused(true);
            return keys.matches(data, "tui.select.up")
                || keys.matches(data, "tui.select.cancel")
                || keys.matches(data, "app.agents.back");
        }
        if self.line.is_selectable() && keys.matches(data, "tui.editor.cursorDown") {
            let at_end = {
                let editor = editor.borrow();
                let text = editor.editor().get_text();
                editor.editor().get_cursor().0 + 1 >= text.lines().count().max(1)
                    && !editor.editor().is_showing_autocomplete()
            };
            if at_end {
                self.line.set_focused(true);
                editor.borrow_mut().editor_mut().set_focused(false);
                return true;
            }
        }
        false
    }
}

impl TuiComponent for Bar {
    fn render(&mut self, width: f64) -> Vec<String> {
        let session_id = self.mode.borrow().connection_state.as_ref().map(|state| state.session_id.clone()).unwrap_or_default();
        let status = {
            let mut state = self.jev.lock().unwrap_or_else(|e| e.into_inner());
            state.select(session_id);
            state.status.clone()
        };
        self.line.set_right_status(status);
        let counts = self.counts();
        self.line.set_subagent_counts(counts);
        if !self.line.is_selectable() && self.line.focused() {
            self.line.set_focused(false);
            if let Some(editor) = &self.editor {
                editor.borrow_mut().editor_mut().set_focused(true);
            }
        }
        let mut lines = self.line.render(width);
        if counts.running > 0 && !self.mode.borrow().is_agent_streaming() {
            lines.push(truncate_to_width(
                &theme().fg(
                    "muted",
                    &format!(
                        "Waiting for {} subagent{}",
                        counts.running,
                        if counts.running == 1 { "" } else { "s" }
                    ),
                ),
                width,
                "…",
                false,
            ));
        }
        lines
    }
    fn invalidate(&mut self) {
        self.line.invalidate();
    }
}

impl Drop for Bar {
    fn drop(&mut self) {
        if let Some(task) = self.jev_task.get_mut().take() { task.abort(); }
    }
}

pub(super) fn seed(mode: &Rc<RefCell<InteractiveMode>>, snapshot: &wire::AgentConnectionSnapshot) {
    let mut mode = mode.borrow_mut();
    mode.rlm_node_id = snapshot
        .parent
        .as_ref()
        .and_then(|parent| parent.child_id.clone());
    let children = snapshot
        .children
        .as_ref()
        .into_iter()
        .flatten()
        .map(project_child)
        .collect::<Vec<_>>();
    mode.replace_subagent_summary(Some(&children));
}

pub(super) fn project_child(
    child: &wire::AgentConnectionRlmChildAgentSnapshot,
) -> local::AgentConnectionRlmChildAgentSnapshot {
    local::AgentConnectionRlmChildAgentSnapshot {
        id: child.id.clone(),
        parent_id: child.parent_id.clone(),
        active_session_id: child.active_session_id.clone(),
        session_name: child.session_name.clone(),
        model: child.model.clone(),
        label: child.label.clone(),
        status: child.status.clone(),
        duration_ms: child.duration_ms,
        answer_preview: child.answer_preview.clone(),
        replied_since_task: child.replied_since_task,
        tool_use_count: child.tool_use_count,
        token_count: child.token_count,
        recap: child.recap.clone(),
        session_dir: child.session_dir.clone(),
        activity: child
            .activity
            .as_ref()
            .map(|activity| activity.kind.clone()),
        error: child.error.clone(),
    }
}

#[cfg(test)]
mod jev_bar_tests {
    use super::*;

    #[test]
    fn neon_header_uses_only_the_current_sessions_cached_status() {
        let mode = Rc::new(RefCell::new(super::super::tests::stash_mode("neon-bar")));
        mode.borrow_mut().apply_connection_state_snapshot(local::AgentConnectionState { session_id: "a".into(), ..Default::default() });
        let bar = Bar::new(mode.clone());
        assert!(bar.compact_jev_status().is_none());
        {
            let mut state = bar.jev.lock().unwrap();
            state.select("a".into());
            state.update("a", jev_activity::status(pi_jev::config::JevMode::Compare, false, false, None));
        }
        assert!(bar.compact_jev_status().unwrap().contains("Jev"));
        mode.borrow_mut().apply_connection_state_snapshot(local::AgentConnectionState { session_id: "b".into(), ..Default::default() });
        assert!(bar.compact_jev_status().is_none());
    }

    #[test]
    fn switching_sessions_clears_usage_and_rejects_the_old_poll() {
        crate::modes::interactive::theme::theme::init_theme(Some("dark"), false);
        let mut state = JevBarState::default();
        state.select("a".into());
        let status = jev_activity::status(pi_jev::config::JevMode::Compare, false, false, None);
        assert!(state.update("a", status.clone()));
        assert!(!state.update("a", status.clone()));
        state.select("b".into());
        assert!(state.status.is_none());
        assert!(!state.update("a", status.clone()));
        assert!(state.update("b", status));
        assert!(state.update("b", None));
        assert!(state.status.is_none());
    }
}
