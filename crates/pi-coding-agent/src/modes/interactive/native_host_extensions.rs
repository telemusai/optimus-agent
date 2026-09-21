//! UI state owned by the terminal thread. Daemon notifications carry data;
//! components and their lifetime stay here, as in the TypeScript interactive host.
use super::*;
use crate::core::messages::{bash_output_to_text, BashExecutionMessage};
use crate::core::tools::truncate::{truncate_tail, TruncationOptions};
use crate::modes::interactive::components::bash_execution::{
    BashExecutionComponent, BashExecutionOptions,
};
use crate::modes::interactive::components::side_question::SideQuestionComponent;
use super::native_commands::jev_menu::{JEV_COMPACT_STATUS_KEY, JEV_STATUS_KEY};

/// One extension status entry.
///
/// `text` is the full labelled form. `compact` is the OPTIONAL narrow form the
/// tray row uses when the whole row does not fit (`statusCompactText` in the
/// `setStatus` payload). Senders that do not provide one — old daemons, generic
/// extensions — leave it absent, and the row then falls back to truncating the
/// left label instead of collapsing the segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ExtensionStatus {
    pub text: String,
    pub compact: Option<String>,
}

#[derive(Default)]
pub(super) struct Surfaces {
    statuses: indexmap::IndexMap<String, ExtensionStatus>,
    widgets: indexmap::IndexMap<String, (bool, Box<dyn TuiComponent>)>,
    pub header: Option<Box<dyn TuiComponent>>,
    pub footer: Option<Box<dyn TuiComponent>>,
}
struct Lines(Vec<String>);
impl TuiComponent for Lines {
    fn render(&mut self, width: f64) -> Vec<String> {
        self.0
            .iter()
            .flat_map(|line| TuiText::new(line.clone(), 1, 0, None).render(width))
            .collect()
    }
    fn invalidate(&mut self) {}
}
impl Surfaces {
    pub fn set_status(&mut self, key: String, text: Option<String>, compact: Option<String>) {
        if let Some(text) = text {
            self.statuses.insert(key, ExtensionStatus { text, compact });
        } else {
            self.statuses.shift_remove(&key);
        }
    }
    pub fn set_widget(&mut self, key: String, lines: Option<Vec<String>>, below: bool) {
        self.widgets.shift_remove(&key);
        if let Some(mut lines) = lines {
            if lines.len() > 10 {
                lines.truncate(10);
                lines.push(theme().fg("muted", "... (widget truncated)"));
            }
            self.widgets.insert(key, (below, Box::new(Lines(lines))));
        }
    }
    pub fn set_widget_component(
        &mut self,
        key: String,
        component: Option<Box<dyn TuiComponent>>,
        below: bool,
    ) {
        self.widgets.shift_remove(&key);
        if let Some(component) = component {
            self.widgets.insert(key, (below, component));
        }
    }
    pub fn reset(&mut self) {
        self.widgets.clear();
        self.statuses.clear();
        self.footer = None;
        self.header = None;
    }

    /// Blanket reset that RETAINS the two host-published Jev segments.
    ///
    /// For SAME-SESSION resets (`/reload`, an extension Reset event, the update
    /// path): the session and its settings are unchanged, so the segments stay
    /// truthful and are refreshed by key on the next publish. A SESSION SWITCH
    /// must use [`Surfaces::reset`]: old-session labels are never kept, and an
    /// attached daemon pushes the new session's authoritative footer itself.
    pub fn reset_keeping_jev(&mut self) {
        let keep_decision = self.statuses.shift_remove(JEV_STATUS_KEY);
        let keep_compaction = self.statuses.shift_remove(JEV_COMPACT_STATUS_KEY);
        self.reset();
        if let Some(decision) = keep_decision {
            self.statuses.insert(JEV_STATUS_KEY.to_string(), decision);
        }
        if let Some(compaction) = keep_compaction {
            self.statuses.insert(JEV_COMPACT_STATUS_KEY.to_string(), compaction);
        }
    }
}
pub(super) struct Widgets(pub Rc<RefCell<Surfaces>>, pub bool);
impl TuiComponent for Widgets {
    fn render(&mut self, width: f64) -> Vec<String> {
        let mut lines = Vec::new();
        for (below, content) in self.0.borrow_mut().widgets.values_mut() {
            if *below == self.1 {
                lines.extend(content.render(width));
            }
        }
        lines
    }
    fn invalidate(&mut self) {}
}
/// Splits the extension status map for ONE paint: the two Jev segments for the
/// tray row (immutable lookups by key, the map is never mutated or reordered)
/// and every OTHER status for the plain line below. Returning the remaining
/// line together with the row segments is what keeps each Jev label painted
/// exactly once: the plain line excludes the two keys the row already shows.
pub(super) fn split_status_row(
    statuses: &indexmap::IndexMap<String, ExtensionStatus>,
) -> (
    Option<(&str, Option<&str>)>,
    Option<(&str, Option<&str>)>,
    Vec<String>,
) {
    let decision_row = statuses
        .get(JEV_STATUS_KEY)
        .map(|status| (status.text.as_str(), status.compact.as_deref()));
    let compaction_row = statuses
        .get(JEV_COMPACT_STATUS_KEY)
        .map(|status| (status.text.as_str(), status.compact.as_deref()));
    let other_lines = statuses
        .iter()
        .filter(|(key, _)| *key != JEV_STATUS_KEY && *key != JEV_COMPACT_STATUS_KEY)
        .map(|(_, status)| status.text.clone())
        .collect();
    (decision_row, compaction_row, other_lines)
}

pub(super) struct Statuses(pub Rc<RefCell<Surfaces>>, pub(super) Tray);
impl TuiComponent for Statuses {
    fn render(&mut self, width: f64) -> Vec<String> {
        let mut state = self.0.borrow_mut();
        if let Some(footer) = &mut state.footer {
            return footer.render(width);
        }
        // The two Jev segments ride ON the tray row, on the same baseline as the
        // navigation/model/effort label, so the context counter can be
        // right-aligned beside them. The map is only READ here (immutable key
        // lookups, no shift_remove/extend per paint): a status refresh still
        // overwrites the segments by key, the key order is untouched, and the
        // plain status line below EXCLUDES the two keys the row already shows,
        // so each Jev label is painted exactly once.
        let (decision_row, compaction_row, other_lines) = split_status_row(&state.statuses);
        let mut lines = self.1.render_row(width, decision_row, compaction_row);
        if !other_lines.is_empty() {
            lines.extend(TuiText::new(other_lines.join(" "), 1, 0, None).render(width));
        }
        lines
    }
    fn invalidate(&mut self) {}
}

#[derive(Default)]
pub(super) struct SidePane {
    component: Option<SideQuestionComponent>,
    turns: Vec<wire::AgentConnectionSideQuestionEvent>,
    commands: Vec<String>,
    bash: Option<SideBash>,
    bash_components: Vec<Rc<RefCell<BashExecutionComponent>>>,
    hidden_bash: Option<String>,
    discarded_bash: std::collections::HashSet<String>,
    expanded: bool,
}
struct SideBash {
    id: String,
    input: String,
    seed_transcript: bool,
    component: Option<Rc<RefCell<BashExecutionComponent>>>,
}
impl SidePane {
    pub fn is_open(&self) -> bool {
        self.component.is_some()
    }
    pub fn running(&self) -> bool {
        self.turns.iter().any(|t| t.status == "running")
    }
    pub fn submit(
        &mut self,
        text: String,
        mode: &Rc<RefCell<InteractiveMode>>,
        editor: &Rc<RefCell<CustomEditor>>,
        connection: Arc<dyn wire::AgentConnection>,
        send: mpsc::Sender<HostEvent>,
    ) {
        let text = text.trim();
        if text.is_empty() {
            return;
        }
        let command = crate::core::slash_commands::parse_slash_command(text);
        if command.is_some_and(|command| {
            mode.borrow().is_recognized_slash_command(&command.name)
                || self.commands.contains(&command.name)
        }) {
            editor.borrow_mut().editor_mut().add_to_history(text);
            editor.borrow_mut().editor_mut().set_text("");
            self.notice(text, "Slash commands are not available in side conversations. Press esc to return to the main thread.");
            return;
        }
        let bash_command = text.strip_prefix("!!").or_else(|| text.strip_prefix('!'));
        if bash_command.is_some_and(|command| command.trim().is_empty()) {
            editor.borrow_mut().editor_mut().set_text("");
            return;
        }
        let blocked =
            if self.bash.is_some() || (bash_command.is_some() && mode.borrow().is_bash_running()) {
                Some("Wait for the running command to finish or cancel it first.")
            } else if self.running() {
                Some("Wait for the current side question to finish or cancel it first.")
            } else {
                None
            };
        if let Some(message) = blocked {
            // The editor clears before onSubmit; skipping dispatch alone loses the draft.
            editor.borrow_mut().editor_mut().set_text(text);
            mode.borrow_mut().show_warning(message);
            return;
        }
        if bash_command.is_none()
            && !crate::modes::interactive::prompt_stash_state::stash_images(
                &mode.borrow().pasted_images,
                text,
            )
            .is_empty()
        {
            editor.borrow_mut().editor_mut().set_text(text);
            self.notice(text, "Images are not supported in side conversations. Press esc to return to the main thread.");
            return;
        }
        editor.borrow_mut().editor_mut().add_to_history(text);
        editor.borrow_mut().editor_mut().set_text("");
        if let Some(command) = bash_command {
            let id = uuid::Uuid::new_v4().to_string();
            self.bash = Some(SideBash {
                id: id.clone(),
                input: text.into(),
                seed_transcript: !text.starts_with("!!"),
                component: None,
            });
            let command = command.trim().to_string();
            tokio::spawn(async move {
                let result = connection
                    .execute_bash(
                        &command,
                        Some(wire::AgentConnectionExecuteBashOptions {
                            exclude_from_context: Some(true),
                            transient: Some(true),
                            run_id: Some(id.clone()),
                        }),
                    )
                    .await;
                if let Err(error) = result {
                    let _ = send.send(HostEvent::SideBashFailed(id, error));
                }
            });
        } else {
            let padding = mode
                .borrow()
                .settings_manager()
                .lock()
                .map(|settings| settings.get_editor_padding_x() as usize)
                .unwrap_or(2);
            self.start(text.into(), padding, connection, send);
        }
    }
    fn notice(&mut self, question: &str, answer: &str) {
        if let Some(component) = &mut self.component {
            component.add_turn(wire::AgentConnectionSideQuestionEvent {
                id: format!("side-notice-{}", uuid::Uuid::new_v4()),
                question: question.into(),
                answer: answer.into(),
                status: "complete".into(),
                error_message: None,
            });
        }
    }
    pub fn set_commands(&mut self, id: &str, commands: Vec<String>) {
        if self.turns.first().is_some_and(|turn| turn.id == id) {
            self.commands = commands;
        }
    }
    pub fn start(
        &mut self,
        question: String,
        padding: usize,
        connection: Arc<dyn wire::AgentConnection>,
        send: mpsc::Sender<HostEvent>,
    ) {
        if question.trim().is_empty() {
            let _ = send.send(HostEvent::Warning("Usage: /btw <question>".into()));
            return;
        }
        if self.running() {
            let _ = send.send(HostEvent::Warning(
                "Wait for the current side question to finish or cancel it first.".into(),
            ));
            return;
        }
        let previous: Vec<_> = self
            .turns
            .iter()
            .filter(|t| !t.answer.is_empty())
            .map(|t| wire::AgentConnectionSideQuestionTurn {
                question: t.question.clone(),
                answer: t.answer.clone(),
            })
            .collect();
        let event = wire::AgentConnectionSideQuestionEvent {
            id: uuid::Uuid::new_v4().to_string(),
            question,
            answer: String::new(),
            status: "running".into(),
            error_message: None,
        };
        match &mut self.component {
            Some(component) => component.add_turn(event.clone()),
            None => {
                self.component = Some(SideQuestionComponent::new(event.clone(), Some(padding)));
                let (connection, send, id) = (connection.clone(), send.clone(), event.id.clone());
                tokio::spawn(async move {
                    if let Ok(Ok(commands)) =
                        tokio::time::timeout(Duration::from_secs(30), connection.get_commands())
                            .await
                    {
                        let _ = send.send(HostEvent::SideCommands(
                            id,
                            commands.into_iter().map(|command| command.name).collect(),
                        ));
                    }
                });
            }
        }
        self.turns.push(event.clone());
        tokio::spawn(async move {
            if let Err(error) = connection
                .start_side_question(
                    &event.id,
                    &event.question,
                    (!previous.is_empty()).then_some(previous),
                )
                .await
            {
                let _ = send.send(HostEvent::Connection(
                    wire::AgentConnectionEvent::SideQuestionEvent {
                        event: wire::AgentConnectionSideQuestionEvent {
                            status: "error".into(),
                            error_message: Some(error),
                            ..event
                        },
                    },
                ));
            }
        });
    }
    pub fn update(&mut self, mut event: wire::AgentConnectionSideQuestionEvent) {
        if let Some(turn) = self.turns.iter_mut().find(|t| t.id == event.id) {
            if turn.status != "running" {
                return;
            }
            if event.status == "error" && event.answer.is_empty() {
                event.answer = turn.answer.clone();
            }
            *turn = event.clone();
            if let Some(component) = &mut self.component {
                component.update(event);
            }
        }
    }
    pub fn set_expanded(&mut self, expanded: bool) {
        self.expanded = expanded;
        for component in &self.bash_components {
            component.borrow_mut().set_expanded(expanded);
        }
    }
    pub fn bash_failed(&mut self, id: &str, error: &str) -> bool {
        self.discarded_bash.remove(id);
        if !self.bash.as_ref().is_some_and(|bash| bash.id == id) {
            return false;
        }
        let bash = self.bash.take().expect("matching side bash");
        let started = bash.component.is_some();
        if let Some(component) = bash.component {
            component.borrow_mut().set_failed(error);
            // A delayed bash_end must not escape into the main transcript.
            self.hidden_bash = Some(id.into());
        } else {
            self.notice(&bash.input, error);
        }
        if let Some(component) = &mut self.component {
            component.finish_bash();
        }
        started
    }
    /// Routes transient shell events to their owning pane, matching the TS
    /// bash_start/output/end handlers. Output has no run id, so its start owns
    /// the session's single bash slot until bash_end.
    pub fn handle_bash_event(
        &mut self,
        event: &wire::AgentConnectionSessionEvent,
        ui: &Rc<RefCell<TUI>>,
        connection: &Arc<dyn wire::AgentConnection>,
    ) -> bool {
        match event {
            wire::AgentConnectionSessionEvent::BashStart {
                command,
                exclude_from_context,
                transient,
                run_id,
            } => {
                self.hidden_bash = None;
                if run_id
                    .as_ref()
                    .is_some_and(|id| self.discarded_bash.contains(id))
                {
                    self.hidden_bash = run_id.clone();
                    Self::abort_bash(connection.clone());
                    return true;
                }
                if let Some(bash) = self
                    .bash
                    .as_mut()
                    .filter(|bash| run_id.as_ref() == Some(&bash.id))
                {
                    let component = Rc::new(RefCell::new(BashExecutionComponent::new(
                        command,
                        ui.clone(),
                        *exclude_from_context,
                        BashExecutionOptions::default(),
                    )));
                    component.borrow_mut().set_expanded(self.expanded);
                    if let Some(pane) = &mut self.component {
                        pane.add_bash(Box::new(SharedComponent(component.clone())));
                    }
                    self.bash_components.push(component.clone());
                    bash.component = Some(component);
                    return true;
                }
                if transient == &Some(true) {
                    self.hidden_bash = Some(run_id.clone().unwrap_or_default());
                    return true;
                }
                false
            }
            wire::AgentConnectionSessionEvent::BashOutput { chunk } => {
                if self.hidden_bash.is_some() {
                    return true;
                }
                if let Some(component) = self.bash.as_ref().and_then(|bash| bash.component.as_ref())
                {
                    component.borrow_mut().append_output(chunk);
                    return true;
                }
                false
            }
            wire::AgentConnectionSessionEvent::BashEnd {
                exit_code,
                cancelled,
                truncated,
                full_output_path,
                error_message,
                transient,
                run_id,
            } => {
                if let Some(id) = run_id {
                    self.discarded_bash.remove(id);
                }
                if self
                    .hidden_bash
                    .as_ref()
                    .is_some_and(|id| run_id.as_deref().unwrap_or_default() == id)
                {
                    self.hidden_bash = None;
                    return true;
                }
                if !self
                    .bash
                    .as_ref()
                    .is_some_and(|bash| run_id.as_ref() == Some(&bash.id))
                {
                    return transient == &Some(true);
                }
                let bash = self.bash.take().expect("matching side bash");
                if let Some(pane) = &mut self.component {
                    pane.finish_bash();
                }
                if let Some(component) = bash.component {
                    let mut component = component.borrow_mut();
                    let mut truncation =
                        truncate_tail(&component.get_output(), TruncationOptions::default());
                    truncation.truncated |= truncated;
                    if let Some(error) = error_message {
                        component.set_failed(error);
                    } else {
                        component.set_complete(
                            exit_code.map(|code| code as f64),
                            *cancelled,
                            Some(truncation.clone()),
                            full_output_path.clone(),
                        );
                    }
                    if bash.seed_transcript && !cancelled && error_message.is_none() {
                        self.turns.push(wire::AgentConnectionSideQuestionEvent {
                            id: format!("side-bash-{}", bash.id),
                            question: bash.input,
                            answer: bash_output_to_text(&BashExecutionMessage {
                                role: "bashExecution".into(),
                                command: component.get_command().into(),
                                output: truncation.content.trim_end_matches('\n').into(),
                                exit_code: *exit_code,
                                cancelled: false,
                                truncated: truncation.truncated,
                                full_output_path: full_output_path.clone(),
                                timestamp: now_ms() as i64,
                                exclude_from_context: Some(true),
                            }),
                            status: "complete".into(),
                            error_message: None,
                        });
                    }
                } else if let Some(error) = error_message {
                    self.notice(&bash.input, error);
                }
                true
            }
            _ => false,
        }
    }
    fn abort_bash(connection: Arc<dyn wire::AgentConnection>) {
        tokio::spawn(async move {
            let _ = connection.abort_bash().await;
        });
    }
    pub fn close(&mut self, connection: Arc<dyn wire::AgentConnection>) {
        if let Some(bash) = self.bash.take() {
            self.discarded_bash.insert(bash.id.clone());
            // abort_bash is session-scoped. Only abort after our matching start,
            // otherwise another client's command may still own the slot.
            if let Some(component) = bash.component {
                component.borrow_mut().set_complete(None, true, None, None);
                self.hidden_bash = Some(bash.id);
                Self::abort_bash(connection.clone());
            }
        }
        for turn in self.turns.drain(..).filter(|t| t.status == "running") {
            let connection = connection.clone();
            tokio::spawn(async move {
                let _ = connection.abort_side_question(&turn.id).await;
            });
        }
        self.component = None;
        self.commands.clear();
        self.bash_components.clear();
    }
}
impl TuiComponent for SidePane {
    fn render(&mut self, width: f64) -> Vec<String> {
        let Some(component) = &mut self.component else {
            return Vec::new();
        };
        let mut lines = vec![String::new()];
        lines.extend(component.render(width));
        lines
    }
    fn invalidate(&mut self) {
        if let Some(component) = &mut self.component {
            component.invalidate();
        }
    }
}
