//! Native terminal ownership for the interactive-mode controller.
//!
//! The UI stays on one thread because pi-tui components use Rc/RefCell. Session
//! work runs through AgentConnection on the Tokio runtime; event callbacks only
//! enqueue owned data and never borrow the UI while it is rendering.

use super::*;
// `tray_row` is a sibling module of the parent that owns this file; the bare
// `tray_row::` paths below resolve through this explicit import (the glob
// above only reaches the owning parent's own items).
use crate::modes::interactive::tray_row;
use crate::main_entry::InteractiveModeSeamOptions;
use crate::modes::agent_connection::types as wire;
use crate::modes::interactive::components::{
    assistant_message::{AssistantMessageComponent, AssistantMessageComponentOptions},
    custom_editor::{CustomEditor, CustomEditorOptions},
    extension_editor::{AppKeybindingsManager, ExtensionEditorComponent},
    extension_input::{ExtensionInputComponent, ExtensionInputOptions},
    extension_selector::{ExtensionSelectorComponent, ExtensionSelectorOptions},
    login_dialog::LoginDialogComponent,
    model_selector::{
        ModelItemModel, ModelSelectorComponent, ModelSelectorOptions, ScopedModelItem,
    },
    prime_onboarding_splash::{PrimeOnboardingSplashComponent, PrimeOnboardingSplashOptions},
    thinking_selector::ThinkingSelectorComponent,
    tool_execution::{ToolExecutionComponent, ToolExecutionOptions, ToolExecutionResult},
    refinement_outcome_message::RefinementOutcomeMessageComponent,
    user_message::UserMessageComponent,
};
use crate::modes::interactive::interactive_mode_services as local;
use crate::modes::interactive::prompt_stash_state::{
    PromptStashCapture, PromptStashEditorEffect, PromptStashOutcome, PromptStashSession,
};
use pi_tui::components::text::Text as TuiText;
use pi_tui::tui::{Component as TuiComponent, InputListenerResult, TuiStopOptions, TUI};
use std::cell::{Cell, RefCell};
use std::hash::{Hash, Hasher};
use std::io::IsTerminal;
use std::rc::Rc;
use std::sync::mpsc;
use std::time::{Duration, Instant};

const MAX_HOST_EVENTS_PER_FRAME: usize = 64;
const MAX_HOST_EVENT_TIME_PER_FRAME: Duration = Duration::from_millis(8);

struct HostEventBudget {
    started: Instant,
    remaining: usize,
}
impl HostEventBudget {
    fn new() -> Self { Self { started: Instant::now(), remaining: MAX_HOST_EVENTS_PER_FRAME } }
    fn next<T>(&mut self, receive: &mpsc::Receiver<T>) -> Option<T> {
        self.next_at(receive, Instant::now())
    }
    fn next_at<T>(&mut self, receive: &mpsc::Receiver<T>, now: Instant) -> Option<T> {
        if self.remaining == 0 || now.saturating_duration_since(self.started) >= MAX_HOST_EVENT_TIME_PER_FRAME { return None; }
        let event = receive.try_recv().ok()?;
        self.remaining -= 1;
        Some(event)
    }
}

#[path = "native_host_autocomplete.rs"]
mod native_autocomplete;
#[path = "native_host_configuration.rs"]
mod native_configuration;
#[path = "native_host_heartbeats.rs"]
mod native_heartbeats;
#[path = "native_host_history.rs"]
mod native_history;
#[path = "native_host_queue.rs"]
mod native_queue;
#[path = "native_host_neon.rs"]
mod native_neon;
#[path = "native_host_settings.rs"]
mod native_settings;
#[path = "native_host_state.rs"]
mod native_state;
#[path = "native_host_status.rs"]
mod native_status;
#[path = "native_host_clipboard.rs"]
mod native_clipboard;
#[path = "native_host_commands.rs"]
mod native_commands;
// Child-session mode inheritance (integration hunk C-4) uses the /jev mode
// bridge from core paths that cannot see the private `native_commands` module.
// The footer helpers are re-exported for the same reason: the daemon-side
// footer text in core/jev_bridge.rs must use the exact labels and colours the
// interactive tray renders (full + compact forms included).
pub(crate) use native_commands::jev_menu::{
    compaction_state, footer_color_key, footer_compact_text, footer_compaction_compact_text,
    footer_compaction_text, footer_text, JevFooterState, JevModeBridge,
    JEV_FULL_JEV_EMERGENCY_EXIT_NOTICE, JEV_FULL_JEV_REJECTION,
};
#[path = "native_host_extensions.rs"]
mod native_extensions;
#[path = "native_host_extension_bridge.rs"]
mod native_extension_bridge;
#[path = "native_host_subagents.rs"]
mod native_subagents;
#[path = "native_host_recovery_notice.rs"]
mod native_recovery_notice;
#[path = "native_host_metrics.rs"]
mod native_metrics;
#[cfg(test)]
#[path = "native_host_ui_tests.rs"]
mod ui_tests;

#[cfg(test)]
#[path = "native_host_chat_detail_tests.rs"]
mod chat_detail_tests;

pub(crate) async fn run_interactive_mode(
    options: InteractiveModeSeamOptions,
) -> Result<(), String> {
    launch(options, false).await.map(|_| ())
}

pub(crate) async fn init_interactive_mode(
    options: InteractiveModeSeamOptions,
) -> Result<(), String> {
    launch(options, true).await.map(|_| ())
}

pub(crate) async fn run_interactive_mode_for_agents(
    options: InteractiveModeSeamOptions,
) -> Result<Option<InteractiveModeRunResult>, String> {
    launch(options, false).await
}

async fn launch(
    options: InteractiveModeSeamOptions,
    benchmark: bool,
) -> Result<Option<InteractiveModeRunResult>, String> {
    let handle = tokio::runtime::Handle::current();
    tokio::task::spawn_blocking(move || handle.block_on(run_terminal(options, benchmark)))
        .await
        .map_err(|error| format!("Interactive terminal failed: {error}"))?
}

struct SharedComponent<T>(Rc<RefCell<T>>);
impl<T: TuiComponent> TuiComponent for SharedComponent<T> {
    fn render(&mut self, width: f64) -> Vec<String> {
        self.0.borrow_mut().render(width)
    }
    fn invalidate(&mut self) {
        self.0.borrow_mut().invalidate();
    }
    fn get_selection_regions(&self) -> Vec<pi_tui::selection_metadata::TableCellSelectionRegion> {
        self.0.borrow().get_selection_regions()
    }
}

struct Transcript {
    clipboard_notice: native_clipboard::Notice,
    recovery_notices: Vec<Rc<RefCell<native_recovery_notice::RecoveryNotice>>>,
    refinement_outcomes: Vec<Rc<RefCell<RefinementOutcomeMessageComponent>>>,
    subagents: Option<Rc<RefCell<native_subagents::Bar>>>,
    agent_messages: Vec<Rc<RefCell<crate::modes::interactive::components::agent_message::AgentMessageComponent>>>,
    extension_surfaces: Option<Rc<RefCell<native_extensions::Surfaces>>>,
    side_pane: Option<Rc<RefCell<native_extensions::SidePane>>>,
    stats_panel: native_commands::StatsDock,
    history: Option<Box<Transcript>>,
    mode: Rc<RefCell<InteractiveMode>>,
    rows: Vec<Box<dyn TuiComponent>>,
    row_keys: HashMap<usize, Rc<str>>,
    row_metadata: HashMap<Rc<str>, native_neon::RowMeta>,
    assistant_row: Option<usize>,
    timeline: Option<native_neon::Timeline>,
    selection_columns: Vec<Option<(usize, usize)>>,
    connection_status: String,
    refinement_progress: Option<String>,
    viewport_anchors: Vec<Option<pi_tui::fullscreen::ViewportAnchor>>,
    assistant: Option<Rc<RefCell<AssistantMessageComponent>>>,
    assistants: Vec<Rc<RefCell<AssistantMessageComponent>>>,
    tools: HashMap<String, Rc<RefCell<ToolExecutionComponent>>>,
    wrapped_lines: TranscriptWrapCache,
}

#[derive(Default)]
struct TranscriptWrapCache {
    width: usize,
    lines: Vec<(String, Vec<String>, Vec<usize>)>,
}
impl TranscriptWrapCache {
    fn render(&mut self, lines: Vec<String>, width: usize) -> Vec<String> {
        if self.width != width {
            self.lines.clear();
            self.width = width;
        }
        self.lines.truncate(lines.len());
        let mut output = Vec::new();
        for (index, line) in lines.into_iter().enumerate() {
            if let Some((previous, wrapped, counts)) = self.lines.get_mut(index) {
                if *previous != line {
                    *wrapped = pi_tui::utils::wrap_text_with_ansi(&line, width);
                    *counts = Self::content_counts(wrapped);
                    *previous = line;
                }
                output.extend(wrapped.iter().cloned());
            } else {
                let wrapped = pi_tui::utils::wrap_text_with_ansi(&line, width);
                output.extend(wrapped.iter().cloned());
                let counts = Self::content_counts(&wrapped);
                self.lines.push((line, wrapped, counts));
            }
        }
        output
    }

    fn content_counts(lines: &[String]) -> Vec<usize> {
        lines.iter().map(|line| pi_tui::utils::strip_ansi(line).chars().filter(|ch| !ch.is_whitespace()).count()).collect()
    }

    fn anchors(&self, keys: &[Option<Rc<str>>]) -> Vec<Option<pi_tui::fullscreen::ViewportAnchor>> {
        let mut anchors = Vec::new();
        let mut previous = None;
        let mut offset = 0;
        for ((_, _, counts), key) in self.lines.iter().zip(keys) {
            if previous != *key {
                offset = 0;
                previous = key.clone();
            }
            for count in counts {
                anchors.push(key.as_ref().filter(|_| *count > 0).map(|key| pi_tui::fullscreen::ViewportAnchor { key: key.clone(), offset }));
                offset += count;
            }
        }
        anchors
    }
}
struct ToolRow(Rc<RefCell<ToolExecutionComponent>>);
impl TuiComponent for ToolRow {
    fn render(&mut self, width: f64) -> Vec<String> {
        self.0.borrow_mut().render_lines(width)
    }
    fn invalidate(&mut self) {
        self.0.borrow_mut().content_panel().invalidate();
    }
}
impl Transcript {
    /// Port of `echoLocalCommand` (interactive-mode.ts:6405-6413): the submitted
    /// command is echoed as the user's own message.
    fn echo_local(&mut self, text: &str) {
        let mode = self.mode.borrow();
        self.rows
            .push(Box::new(pi_tui::components::spacer::Spacer::new(1)));
        self.rows.push(Box::new(UserMessageComponent::new(
            text,
            mode.get_markdown_theme_with_settings(),
            &|name| crate::core::slash_commands::is_builtin_slash_command_name(name),
        )));
        drop(mode);
    }

    /// The `chatContainer.addChild(new Spacer(1)); addChild(new Text(info, 1, 0))`
    /// pair every local command panel uses (`/session`
    /// interactive-mode.ts:9499-9501, `/logs` :9531-9533).
    fn panel(&mut self, message: &str) {
        self.rows
            .push(Box::new(pi_tui::components::spacer::Spacer::new(1)));
        self.rows
            .push(Box::new(TuiText::new(message.into(), 1, 0, None)));
    }

    fn new(mode: Rc<RefCell<InteractiveMode>>) -> Self {
        Self {
            clipboard_notice: native_clipboard::Notice::default(),
            recovery_notices: Vec::new(),
            refinement_outcomes: Vec::new(),
            subagents: None,
            agent_messages: Vec::new(),
            extension_surfaces: None,
            side_pane: None,
            stats_panel: native_commands::StatsDock::default(),
            history: None,
            mode,
            rows: Vec::new(),
            row_keys: HashMap::new(),
            row_metadata: HashMap::new(),
            assistant_row: None,
            timeline: None,
            selection_columns: Vec::new(),
            connection_status: String::new(),
            refinement_progress: None,
            viewport_anchors: Vec::new(),
            assistant: None,
            assistants: Vec::new(),
            tools: HashMap::new(),
            wrapped_lines: TranscriptWrapCache::default(),
        }
    }
    fn replace(&mut self, messages: Vec<AgentMessage>) {
        self.history = None;
        self.clipboard_notice = native_clipboard::Notice::default();
        self.rows.clear();
        self.row_keys.clear();
        self.row_metadata.clear();
        self.assistant_row = None;
        self.viewport_anchors.clear();
        self.tools.clear();
        self.assistant = None;
        self.assistants.clear();
        self.agent_messages.clear();
        self.recovery_notices.clear();
        self.refinement_outcomes.clear();
        self.refinement_progress = None;
        for message in initial_render_messages(messages) {
            self.message(message, false);
        }
    }
    fn replace_history(&mut self, messages: Vec<AgentMessage>, total: f64) {
        self.replace_history_with_ids(messages, total, &[]);
    }
    fn replace_history_with_ids(&mut self, messages: Vec<AgentMessage>, total: f64, entry_ids: &[String]) {
        let mut history = Self::new(self.mode.clone());
        let start = (total as usize).saturating_sub(messages.len());
        if (messages.len() as f64) < total {
            history.rows.push(Box::new(TuiText::new(theme().fg("dim", &format!("Showing {} of {total} messages. Scroll up or use PageUp to load earlier history.", messages.len())), 1, 0, None)));
            history
                .rows
                .push(Box::new(pi_tui::components::spacer::Spacer::new(1)));
        }
        for (index, message) in messages.into_iter().enumerate() {
            let key = entry_ids.get(index).map(|id| format!("entry:{id}"))
                .unwrap_or_else(|| {
                    let mut hash = std::collections::hash_map::DefaultHasher::new();
                    serde_json::to_vec(&message).unwrap_or_default().hash(&mut hash);
                    format!("history:{}:{:x}", start + index, hash.finish())
                });
            history.message_anchored(message, false, &key);
        }
        self.history = Some(Box::new(history));
    }
    fn all_tools(&self) -> Vec<Rc<RefCell<ToolExecutionComponent>>> {
        self.tools
            .values()
            .cloned()
            .chain(self.history.iter().flat_map(|h| h.all_tools()))
            .collect()
    }
    fn set_recovery_notices_expanded(&mut self, expanded: bool) {
        for notice in &self.recovery_notices { notice.borrow_mut().set_expanded(expanded); }
        for outcome in &self.refinement_outcomes { outcome.borrow_mut().set_expanded(expanded); }
        if let Some(history) = &mut self.history { history.set_recovery_notices_expanded(expanded); }
    }
    /// Refresh both restored and live components from the view's detail settings.
    fn apply_chat_detail(&mut self) {
        let (tools, messages, diffs, hide_thinking) = {
            let mode = self.mode.borrow();
            (mode.tool_output_expanded, mode.agent_messages_expanded,
                mode.edit_diffs_expanded, mode.hide_thinking_block)
        };
        self.set_recovery_notices_expanded(tools);
        if let Some(pane) = &self.side_pane { pane.borrow_mut().set_expanded(tools); }
        for component in self.all_tools() {
            let mut component = component.borrow_mut();
            component.set_expanded(tools);
            component.set_agent_messages_expanded(messages);
            component.set_edit_diffs_expanded(diffs);
        }
        for component in self.all_assistants() {
            let mut component = component.borrow_mut();
            component.set_hide_thinking_block(hide_thinking);
            component.set_expanded(messages);
        }
        for component in self.all_agent_messages() {
            component.borrow_mut().set_expanded(messages);
        }
    }
    fn sent_agent_message(&mut self, tool_call_id: &str, message: wire::KernelSentAgentMessage) {
        if let Some(tool) = self.tools.get(tool_call_id) {
            tool.borrow_mut().append_sent_agent_message(message);
        } else if let Some(history) = &mut self.history {
            history.sent_agent_message(tool_call_id, message);
        }
    }
    fn all_agent_messages(&self) -> Vec<Rc<RefCell<crate::modes::interactive::components::agent_message::AgentMessageComponent>>> {
        self.agent_messages.iter().cloned().chain(self.history.iter().flat_map(|history| history.all_agent_messages())).collect()
    }
    fn all_assistants(&self) -> Vec<Rc<RefCell<AssistantMessageComponent>>> {
        self.assistants
            .iter()
            .cloned()
            .chain(self.history.iter().flat_map(|h| h.all_assistants()))
            .collect()
    }
    fn message(&mut self, message: AgentMessage, streaming: bool) {
        let key = format!("live:{}", self.rows.len());
        self.message_anchored(message, streaming, &key);
    }
    fn message_anchored(&mut self, message: AgentMessage, streaming: bool, key: &str) {
        let start = self.rows.len();
        let meta = native_neon::RowMeta::message(&message);
        if message.role() == "assistant" {
            if let Some(key) = self.assistant_row.and_then(|row| self.row_keys.get(&row)) {
                self.row_metadata.insert(key.clone(), meta.clone());
            }
        }
        self.append_message(message, streaming);
        for index in start..self.rows.len() {
            let key = self.row_keys.entry(index).or_insert_with(|| Rc::from(format!("{key}:{}", index - start)));
            let row = self.row_metadata.entry(key.clone()).or_insert_with(|| meta.clone());
            if row.timestamp.is_none() { row.timestamp = meta.timestamp; }
        }
    }
    fn append_message(&mut self, message: AgentMessage, streaming: bool) {
        match message {
            AgentMessage::Message(pi_ai::types::Message::User(user)) => {
                let mode = self.mode.borrow();
                self.rows.push(Box::new(UserMessageComponent::new(
                    &mode.get_user_message_text(&user),
                    mode.get_markdown_theme_with_settings(),
                    &|name| crate::core::slash_commands::is_session_slash_command_name(name),
                )));
            }
            AgentMessage::Message(pi_ai::types::Message::Assistant(message)) => {
                let component = if let Some(component) = &self.assistant {
                    component.clone()
                } else {
                    let mode = self.mode.borrow();
                    let component = Rc::new(RefCell::new(AssistantMessageComponent::new(
                        None,
                        mode.hide_thinking_block,
                        mode.get_markdown_theme_with_settings(),
                        &mode.hidden_thinking_label,
                        AssistantMessageComponentOptions {
                            cwd: Some(mode.get_current_cwd()),
                            expanded: mode.agent_messages_expanded,
                            mermaid_transform: Some(Rc::new(crate::modes::interactive::components::mermaid::create_mermaid_markdown_transform(
                                crate::modes::interactive::components::mermaid::MermaidTransformOptions {
                                    get_mode: { let settings = mode.settings_manager().clone(); Box::new(move || settings.lock().map(|s| s.get_mermaid_rendering_mode()).unwrap_or_else(|_| "off".into())) },
                                    theme: Some(theme()),
                                },
                            ))),
                            ..Default::default()
                        },
                    )));
                    self.assistant_row = Some(self.rows.len());
                    self.rows.push(Box::new(SharedComponent(component.clone())));
                    self.assistants.push(component.clone());
                    self.assistant = Some(component.clone());
                    component
                };
                component
                    .borrow_mut()
                    .update_content(message.clone(), streaming);
                for block in &message.content {
                    if let pi_ai::types::ContentBlock::ToolCall(call) = block {
                        self.tool_start(
                            &call.id,
                            &call.name,
                            serde_json::Value::Object(call.arguments.clone()),
                        );
                    }
                }
                if !streaming {
                    self.assistant = None;
                    self.assistant_row = None;
                }
            }
            AgentMessage::Message(pi_ai::types::Message::ToolResult(result)) => {
                let value = serde_json::to_value(&result).unwrap_or_default();
                self.tool_result(&result.tool_call_id, &value, result.is_error, false);
            }
            AgentMessage::Custom(message) => {
                if matches!(&message, pi_agent_core::types::CustomAgentMessage::Custom { display: false, .. }) {
                    return;
                }
                if let Some(notice) = crate::modes::interactive::components::shell_completion::ShellCompletion::from_message(&message) {
                    self.rows.push(Box::new(notice));
                    return;
                }
                if let pi_agent_core::types::CustomAgentMessage::Custom { custom_type, content: pi_agent_core::types::CustomMessageContent::Text(text), .. } = &message {
                    if custom_type == crate::core::messages::IPYTHON_STATE_RESTORED_CUSTOM_TYPE {
                        let component = Rc::new(RefCell::new(native_recovery_notice::RecoveryNotice::new(text.clone(), self.mode.borrow().tool_output_expanded)));
                        self.rows.push(Box::new(SharedComponent(component.clone())));
                        self.recovery_notices.push(component);
                        return;
                    }
                }
                if let pi_agent_core::types::CustomAgentMessage::Custom { custom_type, details: Some(details), .. } = &message {
                    if custom_type == crate::core::messages::REFINEMENT_OUTCOME_CUSTOM_TYPE {
                        if let Ok(details) = serde_json::from_value(details.clone()) {
                            let component = Rc::new(RefCell::new(RefinementOutcomeMessageComponent::new(details)));
                            component.borrow_mut().set_expanded(self.mode.borrow().tool_output_expanded);
                            self.rows.push(Box::new(SharedComponent(component.clone())));
                            self.refinement_outcomes.push(component);
                            return;
                        }
                    }
                    if custom_type == crate::core::agent_messages::AGENT_MESSAGE_CUSTOM_TYPE {
                        if let Ok(details) = serde_json::from_value(details.clone()) {
                            let component = Rc::new(RefCell::new(crate::modes::interactive::components::agent_message::AgentMessageComponent::new(details, false)));
                            component.borrow_mut().set_expanded(self.mode.borrow().agent_messages_expanded);
                            self.rows.push(Box::new(SharedComponent(component.clone())));
                            self.agent_messages.push(component);
                            return;
                        }
                    }
                }
                let value = serde_json::to_value(message).unwrap_or_default();
                let text = value
                    .get("content")
                    .and_then(|value| value.as_str())
                    .or_else(|| value.get("text").and_then(|value| value.as_str()));
                if let Some(text) = text {
                    self.rows
                        .push(Box::new(TuiText::new(text.into(), 1, 1, None)));
                }
            }
        }
    }
    fn tool_start(&mut self, id: &str, name: &str, args: serde_json::Value) {
        if let Some(component) = self.tools.get(id) {
            component.borrow_mut().update_args(args);
            return;
        }
        let mode = self.mode.borrow();
        let mut component = ToolExecutionComponent::new(
            name,
            id,
            args,
            ToolExecutionOptions {
                show_images: mode
                    .settings_manager()
                    .lock()
                    .ok()
                    .map(|s| s.get_show_images()),
                ..Default::default()
            },
            None,
            &mode.get_current_cwd(),
        );
        component.mark_execution_started();
        component.set_expanded(mode.tool_output_expanded);
        component.set_agent_messages_expanded(mode.agent_messages_expanded);
        component.set_edit_diffs_expanded(mode.edit_diffs_expanded);
        let component = Rc::new(RefCell::new(component));
        let key: Rc<str> = Rc::from(format!("tool:{id}"));
        self.row_metadata.insert(key.clone(), native_neon::RowMeta::new(native_neon::Kind::Tool, None));
        self.row_keys.insert(self.rows.len(), key);
        self.rows.push(Box::new(ToolRow(component.clone())));
        self.tools.insert(id.to_string(), component);
    }
    fn tool_result(&mut self, id: &str, result: &serde_json::Value, is_error: bool, partial: bool) {
        let Some(component) = self.tools.get(id) else {
            return;
        };
        let content = result
            .get("content")
            .and_then(|value| value.as_array())
            .into_iter()
            .flatten()
            .map(
                |block| crate::core::tools::render_utils::RenderContentBlock {
                    r#type: string(block, "type"),
                    text: optional_string(block, "text"),
                    data: optional_string(block, "data"),
                    mime_type: optional_string(block, "mimeType"),
                },
            )
            .collect();
        if !partial {
            if let Some(meta) = self.row_metadata.get_mut(format!("tool:{id}").as_str()) {
                meta.kind = if is_error { native_neon::Kind::Error } else { native_neon::Kind::Success };
                if let Some(started) = meta.started.take() { meta.elapsed = Some(started.elapsed()); }
            }
        }
        component.borrow_mut().update_result(
            ToolExecutionResult {
                content,
                is_error,
                details: result.get("details").cloned(),
            },
            partial,
        );
    }
}
impl TuiComponent for Transcript {
    fn render(&mut self, width: f64) -> Vec<String> {
        self.timeline = (native_neon::active() && self.mode.borrow().fullscreen_enabled).then(|| native_neon::Timeline::new(width.max(1.0) as usize));
        let width = self.timeline.map_or(width, |timeline| timeline.content_width() as f64);
        let mode = self.mode.borrow();
        let mut lines = Vec::new();
        let mut keys = Vec::new();
        let mut custom_header = false;
        if let Some(surfaces) = &self.extension_surfaces {
            if let Some(header) = &mut surfaces.borrow_mut().header { lines.extend(header.render(width)); custom_header = true; }
        }
        if self.timeline.is_some() && !custom_header && self.rows.is_empty() && self.history.as_ref().is_none_or(|h| h.rows.is_empty()) {
            lines.push(theme().fg("accent", "Ready when you are."));
            lines.push(theme().fg("dim", "Describe a task, or use /help to explore commands."));
        }
        if self.timeline.is_none() && !custom_header && self.rows.is_empty() && self.history.as_ref().is_none_or(|h| h.rows.is_empty()) {
            lines.push(theme().fg("muted", "Ready when you are."));
            lines.push(theme().fg("dim", "Describe a task, or choose a chat from Sessions."));
        }
        drop(mode);
        keys.resize(lines.len(), None);
        if let Some(history) = &mut self.history {
            for (index, row) in history.rows.iter_mut().enumerate() {
                let rendered = row.render(width);
                keys.extend(std::iter::repeat_n(history.row_keys.get(&index).cloned(), rendered.len()));
                lines.extend(rendered);
            }
        }
        for (index, row) in self.rows.iter_mut().enumerate() {
            let rendered = row.render(width);
            keys.extend(std::iter::repeat_n(self.row_keys.get(&index).cloned(), rendered.len()));
            lines.extend(rendered);
        }
        let mode = self.mode.borrow();
        // The `mainViewContainer` child order (interactive-mode.ts:1268-1272), then
        // the prompt-context containers below it (TS:1575-1577).
        for container in mode.get_main_view_containers() {
            lines.extend(local::Component::render(container, width.max(1.0) as usize));
        }
        for container in mode.get_prompt_context_containers() {
            lines.extend(local::Component::render(container, width.max(1.0) as usize));
        }
        if let Some(message) = self.clipboard_notice.text() {
            lines.push(theme().fg("dim", message));
        }
        if mode.restored_draft_notice.borrow().is_some() {
            lines.push(theme().fg("dim", "Draft restored"));
        }
        if let Some(progress) = &self.refinement_progress {
            lines.push(truncate_to_width(&theme().fg("accent", progress), width, "…", false));
        }
        if mode.should_show_working_loader() {
            // `createWorkingLoader` mounts the animated pi-tui `Loader`
            // (interactive-mode.ts:3305-3313); the native host renders the same
            // component's frame cycle in place, advanced by the host ticker.
            let custom = mode.working_indicator_options.as_ref();
            let default_frames = pi_tui::components::loader::DEFAULT_FRAMES.iter().map(|s| s.to_string()).collect::<Vec<_>>();
            let frames = custom.and_then(|c| c.frames.as_ref()).filter(|f| !f.is_empty()).unwrap_or(&default_frames);
            let interval = custom.and_then(|c| c.interval_ms).filter(|n| n.is_finite() && *n > 0.0)
                .unwrap_or(pi_tui::components::loader::DEFAULT_INTERVAL_MS as f64).max(16.0);
            // The pi-tui `Loader` advances one frame per `DEFAULT_INTERVAL_MS`
            // (crates/pi-tui/src/components/loader.rs:11-12); driving the index from
            // the wall clock reproduces that cadence without a second timer.
            let frame = ((now_ms() / interval).floor()
                as i64)
                .rem_euclid(frames.len() as i64) as usize;
            lines.push(format!(
                "{}{}",
                theme().fg("accent", &frames[frame]),
                theme().fg("muted", &format!(" {}", mode.get_working_loader_message())),
            ));
        }
        drop(mode);
        // TypeScript includes the side-question container in fullscreen scroll
        // content as well as inline content (getPromptContextContainers).
        if let Some(side_pane) = &self.side_pane {
            lines.extend(side_pane.borrow_mut().render(width));
        }
        // Editor and overlay repaints must not reparse unchanged history's ANSI
        // and Unicode on every key. Compare rendered bytes so external component
        // updates, theme changes and expansion still invalidate precisely.
        keys.resize(lines.len(), None);
        let rendered = self.wrapped_lines.render(lines, width.max(1.0) as usize);
        self.viewport_anchors = self.wrapped_lines.anchors(&keys);
        self.selection_columns = if let Some(t) = self.timeline {
            rendered.iter().map(|line| Some((t.left, t.left + pi_tui::utils::visible_width(pi_tui::utils::strip_ansi(line).trim_end()).min(t.content_width())))).collect()
        } else { Vec::new() };
        if let Some(timeline) = self.timeline {
            rendered.iter().zip(&self.viewport_anchors).map(|(line, anchor)| {
                let meta = anchor.as_ref().and_then(|anchor| self.row_metadata.get(&anchor.key)
                    .or_else(|| self.history.as_ref().and_then(|h| h.row_metadata.get(&anchor.key))));
                timeline.line(line, meta, anchor.as_ref().is_some_and(|a| a.offset == 0))
            }).collect()
        } else { rendered }
    }
    fn get_selection_columns(&self) -> Vec<Option<(usize, usize)>> {
        self.selection_columns.clone()
    }
    fn get_fullscreen_padding(&self) -> String {
        self.timeline.map(|t| t.padding()).unwrap_or_default()
    }
    fn get_viewport_anchors(&self) -> Vec<Option<pi_tui::fullscreen::ViewportAnchor>> {
        self.viewport_anchors.clone()
    }
    fn bottom_align_in_fullscreen(&self) -> bool {
        !self.rows.is_empty() || self.history.as_ref().is_some_and(|history| !history.rows.is_empty())
    }
    fn invalidate(&mut self) {
        if let Some(side_pane) = &self.side_pane {
            side_pane.borrow_mut().invalidate();
        }
        if let Some(history) = &mut self.history {
            history.invalidate();
        }
        for row in &mut self.rows {
            row.invalidate();
        }
    }
}

struct Tray(Rc<RefCell<InteractiveMode>>, Rc<RefCell<CustomEditor>>);
impl Tray {
    /// The bottom row. The navigation/model/effort label, the two Jev segments
    /// (decision mode + independent compaction, published through the extension
    /// status surface) and the context usage counter share ONE baseline: the
    /// counter is right-aligned on the same row, never on a separate lower-left
    /// line and never hidden behind a full-width left string. The narrow
    /// fallback ladder (full -> compact segments -> left truncation) lives in
    /// the pure `tray_row` module.
    fn render_row(
        &mut self,
        width: f64,
        jev_decision: Option<(&str, Option<&str>)>,
        jev_compaction: Option<(&str, Option<&str>)>,
    ) -> Vec<String> {
        let mode = self.0.borrow();
        let left = mode
            .get_tray_override_label(&self.1.borrow().editor().get_text())
            .or_else(|| mode.get_tray_location_label())
            .unwrap_or_default();
        let usage = mode.get_tray_context_usage_text();
        let left_themed = theme().fg("dim", &left);
        let usage_themed = usage.as_deref().map(|usage| if native_neon::active() {
            native_neon::context_meter(mode.get_connection_context_usage().as_ref(), usage, width as usize)
        } else { theme().fg("dim", usage) });
        let right = native_status::right_status(&mode, usage_themed.as_deref(), width as usize);
        let composed = tray_row::compose_tray_row(
            tray_row::TrayRow {
                left: &left_themed,
                jev_decision: jev_decision
                    .map(|(full, compact)| tray_row::TraySegment { full, compact }),
                jev_compaction: jev_compaction
                    .map(|(full, compact)| tray_row::TraySegment { full, compact }),
                right: right.as_deref(),
            },
            width as usize,
        );
        let mut lines = Vec::new();
        if let Some(context) = mode.get_tray_context_label() {
            // Keep goal/manager detail above the bottom-right heartbeat/context status.
            lines.push(truncate_to_width(
                &theme().fg("dim", &context),
                width,
                "…",
                false,
            ));
        }
        lines.push(composed);
        lines
    }
}
impl TuiComponent for Tray {
    fn render(&mut self, width: f64) -> Vec<String> {
        // A bare Tray (e.g. the settings view) has no extension status surface,
        // so no Jev segments are available here; the row still carries the
        // right-aligned context counter.
        self.render_row(width, None, None)
    }
    fn invalidate(&mut self) {}
}

struct TerminalGuard(Rc<RefCell<TUI>>);
impl Drop for TerminalGuard {
    fn drop(&mut self) {
        if let Ok(mut ui) = self.0.try_borrow_mut() {
            ui.stop(TuiStopOptions::default());
        }
    }
}

#[derive(Debug)]
enum InputAction {
    Submit(String, bool),
    /// A chosen `/effort` level (`applyThinkingLevel`, interactive-mode.ts:8309-8321).
    ThinkingLevel(pi_agent_core::types::ThinkingLevel),
    /// The `/effort` picker's `onCancel` (interactive-mode.ts:8301-8304).
    DismissSelector,
    Interrupt,
    Escape,
    Exit,
    Heartbeats,
    ToggleTools,
    ToggleExecutionMode,
    ToggleThinking,
    ToggleMessages,
    Model,
    AgentsBack,
    Subagents,
    Shortcuts,
    Suspend,
    PromptStash,
    /// `app.edits.expand` (interactive-mode.ts:4300).
    ToggleEditDiffs,
    /// `app.editor.external` (interactive-mode.ts:4306).
    OpenExternalEditor,
    /// `app.session.new|tree|fork|resume` (interactive-mode.ts:4313-4322).
    SessionNew,
    SessionTree,
    SessionFork,
    SessionResume,
}
enum HostEvent {
    MenuTiming(Instant),
    Extension(native_extension_bridge::Event),
    Shutdown,
    CommandDialog(native_commands::Dialog),
    CloseCommandDialog,
    CommandBusy(String, tokio_util::sync::CancellationToken),
    RefreshSnapshot(wire::AgentConnectionSnapshot),
    EditorText(String),
    AuthChanged,
    ReloadSettings,
    ScopeChanged(String, Vec<wire::AgentConnectionScopedModel>),
    RunUpdate(Vec<String>),
    SideQuestion(String),
    SideCommands(String, Vec<String>),
    SideBashFailed(String, String),
    Debug,
    Connection(wire::AgentConnectionEvent),
    Completed(Result<(), String>),
    Status(String),
    ClipboardNotice(String),
    /// A local command's reply that lands in the chat as a message
    /// (`chatContainer.addChild(new Text(info, 1, 0))`, e.g. `/session`
    /// interactive-mode.ts:9499-9501).
    Panel(String),
    /// The submitted command echoed as the user's own message
    /// (`echoLocalCommand`, interactive-mode.ts:6405-6413).
    EchoLocal(String),
    /// The raw `getContextTree()` reply, formatted and mounted on the owner loop
    /// so the width is the LIVE terminal width (`handleContextCommand`,
    /// interactive-mode.ts:9799-9808).
    ContextTree(serde_json::Value),
    /// `showWarning` (interactive-mode.ts:7678-7682).
    Warning(String),
    /// `/effort` with no argument opens `ThinkingSelectorComponent`
    /// (`showThinkingSelector`, interactive-mode.ts:8285-8307).
    ThinkingLevels {
        current: pi_agent_core::types::ThinkingLevel,
        levels: Vec<pi_agent_core::types::ThinkingLevel>,
    },
    Render,
    Heartbeats(Vec<wire::AgentConnectionHeartbeat>, bool),
    HeartbeatUpdated(wire::AgentConnectionHeartbeat, serde_json::Value),
    CloseHeartbeats,
    Settings(wire::AgentConnectionState),
    Setting(native_settings::Change),
    /// The daemon accepted a settings change, so its `.then` continuation can
    /// run (`onThinkingLevelChange` patches the state only after the remote call
    /// resolves, interactive-mode.ts:7852-7856). The owner loop must not await
    /// the RPC itself: the continuation travels back as this event.
    SettingAccepted(native_settings::Change),
    Models(wire::AgentConnectionModelCatalog, Option<String>, Instant),
    ModelSelected {
        session_id: String,
        model: wire::AgentConnectionModel,
        result: Result<wire::AgentConnectionState, String>,
    },
    Configuration(
        wire::AgentConnectionModelCatalog,
        &'static str,
        Option<String>,
    ),
    BeginLogin(String, bool),
    LoginAuth(String, Option<String>),
    LoginProgress(String),
    LoginPrompt(String, Option<String>, tokio::sync::oneshot::Sender<String>),
    /// The generation identifies the login that finished, so a login that
    /// completes after a newer login began cannot touch the newer login's dialog.
    ///
    /// TypeScript gives every login its own `LoginDialogComponent` with its own
    /// `abortController` (login-dialog.ts:77, read at :136, aborted at :140), so
    /// one login's cancellation is independent of every sibling's. The port
    /// collapsed that per-dialog state into shared single slots
    /// (`login_cancel`/`overlay`), which is what let a stale login steal them.
    LoginFinished(LoginGeneration, Result<(), String>),
    /// `requestAgentsView()` - `/resume` without arguments and the
    /// `app.agents.open` handoff (interactive-mode.ts:8736-8738).
    AgentsView,
    /// `/reload` succeeded: rebuild the autocomplete provider and refetch the
    /// command catalogue (interactive-mode.ts:9185-9186).
    ReconfigureAutocomplete,
    /// The `/new <prompt>` text, handed to the owner loop so it can collect
    /// pasted images, record history and prompt verbatim
    /// (interactive-mode.ts:10243-10248).
    PromptSession { text: String },
    /// `showError` (interactive-mode.ts:7672-7676) raised by a local handler
    /// that runs off the UI thread and must not render from there.
    Error(String),
    /// `/fullscreen [on|off]` (interactive-mode.ts:5014-5023). `Some(value)` is
    /// an explicit `on`/`off`; `None` resolves against the live state.
    Fullscreen(Option<bool>),
}

/// Monotonic identity for one login attempt.
///
/// TypeScript scopes cancellation to a single `LoginDialogComponent` instance
/// (`private abortController = new AbortController()`, login-dialog.ts:77);
/// the port carries the equivalent identity on the event so `LoginFinished`
/// can tell its own dialog from a newer one.
pub(crate) type LoginGeneration = u64;

/// The state one in-flight login owns.
///
/// Mirrors the per-dialog fields of `LoginDialogComponent`
/// (login-dialog.ts:77 `abortController`) plus the overlay it is shown in.
pub(crate) struct LoginSlot {
    pub generation: LoginGeneration,
    pub token: tokio_util::sync::CancellationToken,
    pub overlay: pi_tui::tui::OverlayHandle,
    pub dialog: Rc<RefCell<LoginDialogComponent>>,
}

/// Owns the one login the host may have in flight.
///
/// TypeScript keeps this state on each `LoginDialogComponent`
/// (login-dialog.ts:77). The port keeps it here so a superseded login can be
/// told from the current one; without an owner the two shared slots
/// (`login_cancel`/`overlay`) let a stale login cancel its successor.
#[derive(Default)]
pub(crate) struct LoginCoordinator {
    next_generation: LoginGeneration,
    current: Option<LoginSlot>,
}

impl LoginCoordinator {
    /// Retires the login in flight and mints the generation for the next one.
    ///
    /// Port of showing a fresh dialog: the previous dialog is torn down with
    /// `abortController.abort()` (login-dialog.ts:140) before the new one is
    /// shown, so the previous login can never cancel its successor.
    pub fn begin(&mut self) -> LoginGeneration {
        if let Some(previous) = self.current.take() {
            previous.token.cancel();
            previous.overlay.hide();
        }
        self.next_generation += 1;
        self.next_generation
    }

    /// Records the login that generation `generation` owns.
    pub fn install(&mut self, slot: LoginSlot) {
        self.current = Some(slot);
    }

    /// The generation currently in flight.
    pub fn current_generation(&self) -> Option<LoginGeneration> {
        self.current.as_ref().map(|slot| slot.generation)
    }

    /// Whether a finishing login still owns the host's login state.
    ///
    /// False for a superseded generation, so a stale completion cannot cancel
    /// or hide the newer dialog.
    pub fn is_current(&self, finishing: LoginGeneration) -> bool {
        self.current_generation() == Some(finishing)
    }

    /// Consumes the slot when `finishing` still owns it.
    ///
    /// Returns the retired slot for the current generation, and nothing for a
    /// stale one, so the caller can cancel/hide only its own dialog.
    pub fn finish(&mut self, finishing: LoginGeneration) -> Option<LoginSlot> {
        if self.is_current(finishing) {
            self.current.take()
        } else {
            None
        }
    }

    /// The dialog the current login shows, if one is open.
    pub fn dialog(&self) -> Option<&Rc<RefCell<LoginDialogComponent>>> {
        self.current.as_ref().map(|slot| &slot.dialog)
    }

    /// Whether any login is in flight. Shutdown uses this in place of the old
    /// `login_dialog.is_none()` shared slot.
    pub fn is_active(&self) -> bool {
        self.current.is_some()
    }

    /// Cancels the login in flight without hiding its overlay.
    ///
    /// Shutdown uses this to abort the pending sign-in before tearing the
    /// terminal down, matching the old `login_cancel.take(); cancel.cancel()`.
    pub fn cancel_current(&mut self) {
        if let Some(slot) = self.current.take() {
            slot.token.cancel();
        }
    }
}

/// Applies one `LoginFinished` to the host's login state.
///
/// Returns `true` when `generation` still owned the dialog, so the caller may
/// continue with the mode/UI effects; `false` for a superseded login, which must
/// be ignored. This is the whole decision the `HostEvent::LoginFinished` arm
/// makes, so the arm and its tests share one path.
pub(crate) fn apply_login_finished(
    logins: &mut LoginCoordinator,
    generation: LoginGeneration,
) -> bool {
    match logins.finish(generation) {
        Some(finished) => {
            // `cancel()` aborts this dialog's own controller and rejects its
            // pending input promise (login-dialog.ts:139-146); the overlay is
            // the one this dialog was shown in.
            finished.token.cancel();
            finished.overlay.hide();
            true
        }
        // Superseded: the newer `BeginLogin` already cancelled and hid this
        // dialog, so a late completion must not touch the newer one
        // (login-dialog.ts:77 gives each login an independent controller).
        None => false,
    }
}

struct ExtensionDialog {
    request: wire::AgentConnectionExtensionUiRequest,
    component: Rc<RefCell<dyn TuiComponent>>,
    overlay: pi_tui::tui::OverlayHandle,
    deadline: Option<Instant>,
}

type ExtensionReply = (String, wire::AgentConnectionExtensionUiResponse);

fn respond_extension(
    connection: &Arc<dyn wire::AgentConnection>,
    send: &mpsc::Sender<HostEvent>,
    reply: ExtensionReply,
    local: Option<&Arc<native_extension_bridge::Bridge>>,
) {
    if local.is_some_and(|bridge| bridge.response(&reply)) { return; }
    let (connection, send) = (connection.clone(), send.clone());
    tokio::spawn(async move {
        if let Err(error) = connection
            .respond_to_extension_ui_request(&reply.0, reply.1)
            .await
        {
            let _ = send.send(HostEvent::Completed(Err(error)));
        }
    });
}

fn cancel_extension_response(method: &str) -> wire::AgentConnectionExtensionUiResponse {
    if method == "confirm" {
        wire::AgentConnectionExtensionUiResponse::Confirmed { confirmed: false }
    } else {
        wire::AgentConnectionExtensionUiResponse::Cancelled { cancelled: true }
    }
}

fn extension_dialog(
    request: wire::AgentConnectionExtensionUiRequest,
    ui: &Rc<RefCell<TUI>>,
    send: &mpsc::Sender<ExtensionReply>,
) -> Option<ExtensionDialog> {
    let title = optional_string(&request.payload, "title").filter(|title| !title.is_empty())?;
    let selected_id = request.id.clone();
    let selected = send.clone();
    let cancelled_id = request.id.clone();
    let cancelled = send.clone();
    let cancel_response = cancel_extension_response(&request.method);
    let on_cancel = Box::new(move || {
        let _ = cancelled.send((cancelled_id.clone(), cancel_response.clone()));
    });
    let component: Rc<RefCell<dyn TuiComponent>> = match request.method.as_str() {
        "select" | "confirm" => {
            let confirm = request.method == "confirm";
            let (title, choices) = if confirm {
                let message = optional_string(&request.payload, "message")?;
                (
                    format!("{title}\n{message}"),
                    vec!["Yes".into(), "No".into()],
                )
            } else {
                let values = request.payload.get("options")?.as_array()?;
                let choices = values
                    .iter()
                    .map(|value| value.as_str().map(str::to_string))
                    .collect::<Option<Vec<_>>>()?;
                (title, choices)
            };
            Rc::new(RefCell::new(ExtensionSelectorComponent::new(
                &title,
                choices,
                Box::new(move |value| {
                    let response = if confirm {
                        wire::AgentConnectionExtensionUiResponse::Confirmed {
                            confirmed: value == "Yes",
                        }
                    } else {
                        wire::AgentConnectionExtensionUiResponse::Value {
                            value: value.into(),
                        }
                    };
                    let _ = selected.send((selected_id.clone(), response));
                }),
                on_cancel,
                ExtensionSelectorOptions::default(),
            )))
        }
        "input" => Rc::new(RefCell::new(ExtensionInputComponent::new(
            &title,
            optional_string(&request.payload, "placeholder"),
            Box::new(move |value| {
                let _ = selected.send((
                    selected_id.clone(),
                    wire::AgentConnectionExtensionUiResponse::Value {
                        value: value.into(),
                    },
                ));
            }),
            on_cancel,
            ExtensionInputOptions::default(),
        ))),
        "editor" => Rc::new(RefCell::new(ExtensionEditorComponent::new(
            ui.clone(),
            Arc::new(AppKeybindingsManager),
            &title,
            request
                .payload
                .get("prefill")
                .and_then(|value| value.as_str()),
            Box::new(move |value| {
                let _ = selected.send((
                    selected_id.clone(),
                    wire::AgentConnectionExtensionUiResponse::Value { value },
                ));
            }),
            on_cancel,
            Default::default(),
        ))),
        _ => return None,
    };
    let overlay = ui
        .borrow_mut()
        .show_overlay(component.clone(), Default::default());
    let deadline = number(&request.payload, "timeout")
        .filter(|timeout| timeout.is_finite() && *timeout > 0.0)
        .and_then(|timeout| Duration::try_from_secs_f64(timeout / 1000.0).ok())
        .and_then(|duration| Instant::now().checked_add(duration));
    Some(ExtensionDialog {
        request,
        component,
        overlay,
        deadline,
    })
}

/// The stash session for this host. `open` resolves the same store state for the
/// session id, so a stash survives the agents-view handoff and a reopen.
fn stash_session(mode: &InteractiveMode, session_id: &str) -> PromptStashSession {
    let store =
        mode.options.prompt_stash_store.clone().unwrap_or_else(
            crate::modes::interactive::prompt_stash_state::shared_prompt_stash_store,
        );
    PromptStashSession::open(store, session_id)
}

/// Port of `snapshotPromptStash` (interactive-mode.ts:4354-4356) over the host's editor.
fn snapshot_editor_prompt_stash(
    mode: &InteractiveMode,
    editor: &CustomEditor,
) -> PromptStashCapture {
    let text = editor.editor().get_text();
    let images =
        crate::modes::interactive::prompt_stash_state::stash_images(&mode.pasted_images, &text);
    // `editor.getPasteSnapshot?.()` (interactive-mode.ts:4344): the pi-tui `Editor`
    // implements it, so the snapshot is always present on this path.
    let paste_snapshot = editor.editor().get_paste_snapshot();
    PromptStashCapture {
        expanded_text: editor.editor().get_expanded_text(),
        text,
        paste_snapshot: Some(paste_snapshot),
        images,
    }
}

/// The app actions the host editor dispatches, in `editor.onAction` order
/// (`interactive-mode.ts:4290-4320`). `app.prompt.stash` is the Ctrl+S stash.
fn bind_editor_actions(
    editor: &Rc<RefCell<CustomEditor>>,
    actions: &Rc<RefCell<Vec<InputAction>>>,
) {
    for (binding, make) in [
        (
            "app.clear",
            (|| InputAction::Interrupt) as fn() -> InputAction,
        ),
        ("app.heartbeats.open", || InputAction::Heartbeats),
        ("app.tools.expand", || InputAction::ToggleTools),
        ("app.executionMode.toggle", || InputAction::ToggleExecutionMode),
        ("app.thinking.toggle", || InputAction::ToggleThinking),
        ("app.messages.expand", || InputAction::ToggleMessages),
        ("app.model.select", || InputAction::Model),
        ("app.shortcuts", || InputAction::Shortcuts),
        ("app.suspend", || InputAction::Suspend),
        ("app.prompt.stash", || InputAction::PromptStash),
        ("app.edits.expand", || InputAction::ToggleEditDiffs),
        ("app.editor.external", || InputAction::OpenExternalEditor),
        // Focusing the in-session summary must not navigate away from this chat.
        ("app.session.new", || InputAction::SessionNew),
        ("app.session.tree", || InputAction::SessionTree),
        ("app.session.fork", || InputAction::SessionFork),
        ("app.session.resume", || InputAction::SessionResume),
    ] {
        let actions = actions.clone();
        editor
            .borrow_mut()
            .on_action(binding, Box::new(move || actions.borrow_mut().push(make())));
    }
}

/// Port of the escape-repeat half of `handleEscape`
/// (interactive-mode.ts:6924-6953). Returns true when the repeated press was
/// consumed (tree selector or input clear), so the caller must not run the
/// interrupt flow. A first press arms the repeat and returns false; the
/// interrupt flow then runs like the TypeScript's `interruptOrClearInput`.
async fn escape_repeat_step(
    mode: &Rc<RefCell<InteractiveMode>>,
    editor: &Rc<RefCell<CustomEditor>>,
    ui: &Rc<RefCell<TUI>>,
    connection: &Arc<dyn wire::AgentConnection>,
    send: &mpsc::Sender<HostEvent>,
) -> bool {
    mode.borrow_mut().clear_ctrl_c_exit_hint(true);
    // Bind before matching: the scrutinee temporary would otherwise hold the
    // mode borrow across the arms.
    let repeat_action = mode.borrow_mut().take_escape_repeat_action();
    match repeat_action {
        Some("tree") => {
            // The tree command waits for a dialog response. The owner loop must
            // remain free to display that dialog and accept its input.
            submit(connection, send, "/tree".into(), false, None);
            return true;
        }
        Some("clear") => {
            let draft = mode.borrow_mut().queue_selection.reset();
            editor.borrow_mut().editor_mut().set_text(&draft);
            ui.borrow_mut().request_render();
            return true;
        }
        _ => {}
    }
    let arm_tree =
        mode.borrow().has_interruptible_work() || editor.borrow().editor().get_text().is_empty();
    mode.borrow_mut().arm_escape_repeat(if arm_tree { "tree" } else { "clear" });
    false
}

/// Port of `openExternalEditor` (interactive-mode.ts:7611-7661) for the host
/// editor. The extension editor owns the same flow
/// (components/extension_editor.rs:189); the host editor only lacked the
/// binding.
fn open_external_editor_for(
    mode: &Rc<RefCell<InteractiveMode>>,
    editor: &Rc<RefCell<CustomEditor>>,
    tui: &Rc<RefCell<TUI>>,
) {
    let Some(editor_cmd) =
        crate::modes::interactive::components::extension_editor::process_env_visual_editor()
    else {
        mode.borrow_mut().show_warning(
            "No editor configured. Set $VISUAL or $EDITOR environment variable.",
        );
        return;
    };
    let current_text = editor.borrow().editor().get_text().to_string();
    let tmp_file = std::env::temp_dir().join(format!("pi-editor-{}.pi.md", now_ms()));
    if std::fs::write(&tmp_file, &current_text).is_err() {
        return;
    }
    tui.borrow_mut().stop(TuiStopOptions::default());
    let parts: Vec<String> = editor_cmd.split(' ').map(|part| part.to_string()).collect();
    let editor_program = parts.first().cloned().unwrap_or_default();
    let mut command = std::process::Command::new(&editor_program);
    for arg in parts.iter().skip(1) {
        command.arg(arg);
    }
    command.arg(&tmp_file);
    if let Ok(status) = command.status() {
        if status.success() {
            if let Ok(content) = std::fs::read_to_string(&tmp_file) {
                let new_content =
                crate::modes::interactive::components::extension_editor::strip_trailing_newline(
                    &content,
                );
                editor.borrow_mut().editor_mut().set_text(&new_content);
            }
        }
    }
    let _ = std::fs::remove_file(&tmp_file);
    tui.borrow_mut().start();
    tui.borrow_mut().request_render_forced();
}

/// Port of `handlePromptStash` (interactive-mode.ts:4379-4394).
///
/// The host editor is a `CustomEditor` over the pi-tui `Editor`, which always
/// offers `restorePasteSnapshot` (`editor-component.ts:52`), so a paste snapshot
/// can always be restored on this path.
fn handle_prompt_stash_action(
    mode: &Rc<RefCell<InteractiveMode>>,
    editor: &Rc<RefCell<CustomEditor>>,
    session_id: &str,
) {
    let session = stash_session(&mode.borrow(), session_id);
    let capture = snapshot_editor_prompt_stash(&mode.borrow(), &editor.borrow());
    let outcome = session.handle_prompt_stash(&capture, true);
    apply_prompt_stash_outcome(mode, editor, &outcome);
}

fn bind_draft_restore_notice(mode: &InteractiveMode, editor: &mut CustomEditor) {
    let notice = mode.restored_draft_notice.clone();
    let mut previous = editor.editor_mut().on_change.take();
    editor.editor_mut().on_change = Some(Box::new(move |text| {
        if notice.borrow().as_deref().is_some_and(|draft| draft != text) {
            notice.borrow_mut().take();
        }
        if let Some(callback) = previous.as_mut() {
            callback(text);
        }
    }));
}

/// Port of `stashDraftForAgentsView` (interactive-mode.ts:4367-4377).
fn stash_editor_draft_for_agents_view(
    mode: &Rc<RefCell<InteractiveMode>>,
    editor: &Rc<RefCell<CustomEditor>>,
    session_id: &str,
) {
    let session = stash_session(&mode.borrow(), session_id);
    let capture = snapshot_editor_prompt_stash(&mode.borrow(), &editor.borrow());
    session.stash_draft_for_agents_view(&capture);
}

/// Applies a stash outcome to the host editor and posts its notice.
fn apply_prompt_stash_outcome(
    mode: &Rc<RefCell<InteractiveMode>>,
    editor: &Rc<RefCell<CustomEditor>>,
    outcome: &PromptStashOutcome,
) {
    match &outcome.editor {
        PromptStashEditorEffect::None => {}
        PromptStashEditorEffect::Clear => editor.borrow_mut().editor_mut().set_text(""),
        PromptStashEditorEffect::SetText {
            text,
            paste_snapshot,
        } => {
            let mut editor = editor.borrow_mut();
            editor.editor_mut().set_text(text);
            if let Some(snapshot) = paste_snapshot {
                editor.editor_mut().restore_paste_snapshot(snapshot.clone());
            }
            *mode.borrow().restored_draft_notice.borrow_mut() = Some(text.clone());
        }
    }
    if let Some(status) = outcome.status {
        if !matches!(outcome.editor, PromptStashEditorEffect::SetText { .. }) {
            mode.borrow_mut().show_status(status, "dim");
        }
    }
}

fn editor_theme() -> pi_tui::components::editor::EditorTheme {
    pi_tui::components::editor::EditorTheme {
        border_color: Rc::new(|text| theme().fg("borderMuted", text)),
        background_color: Some(Rc::new(|text| {
            theme()
                .get_editor_background_color()
                .map(|color| color(text))
                .unwrap_or_else(|| text.into())
        })),
        autocomplete_background_color: Some(Rc::new(|text| {
            (theme().get_popup_background_color())(text)
        })),
        command_color: Some(Rc::new(|text| theme().fg("accent", text))),
        select_list: select_theme(),
    }
}
/// Applies one session snapshot to the transcript: history reset first, then the
/// in-flight assistant message.
///
/// This is the ordering both TypeScript entry points use.
/// `renderInitialMessages` renders the transcript and only afterwards calls
/// `restoreStreamingMessageFromSnapshot` (interactive-mode.ts:6865-6871), and
/// `renderResyncedSession` does the same (:3066-3067). Adding the streaming row
/// before the reset loses it, because the reset replaces the transcript.
///
/// Returns the error to report when optional recent-first history metadata could
/// not be used. The reference reports and continues instead of failing the
/// attachment: the interactive event handler catches the throw and calls only
/// `showError` (interactive-mode.ts:5302-5304), the no-window branch renders the
/// plain transcript (:6759-6767), and the daemon refuses to attach metadata it
/// cannot build ("retain the full legacy snapshot rather than dropping or
/// misidentifying it", modes/daemon/daemon-mode.ts:5507-5509).
fn apply_history_snapshot(
    history: Option<wire::AgentConnectionHistoryWindow>,
    messages: Vec<AgentMessage>,
    streaming_message: Option<AgentMessage>,
    transcript: &Rc<RefCell<Transcript>>,
    editor: &Rc<RefCell<CustomEditor>>,
    history_runtime: &mut native_history::HistoryRuntime,
    viewport: Option<native_history::ViewportFill>,
) -> Option<String> {
    let error = history_runtime.reset(history, messages, transcript, editor, viewport);
    if let Some(message) = streaming_message {
        transcript.borrow_mut().message(message, true);
    }
    error
}

fn select_theme() -> pi_tui::components::select_list::SelectListTheme {
    pi_tui::components::select_list::SelectListTheme {
        selected_prefix: Box::new(|s| theme().fg("accent", s)),
        selected_text: Box::new(|s| theme().fg("accent", s)),
        description: Box::new(|s| theme().fg("muted", s)),
        argument_hint: None,
        source_tag: None,
        scroll_info: Box::new(|s| theme().fg("dim", s)),
        no_match: Box::new(|s| theme().fg("muted", s)),
    }
}

async fn run_terminal(
    options: InteractiveModeSeamOptions,
    benchmark: bool,
) -> Result<Option<InteractiveModeRunResult>, String> {
    let opened_at = Instant::now();
    let mut in_process_connection = None;
    let connection: Arc<dyn wire::AgentConnection> = match options.connection.clone() {
        Some(connection) => connection,
        None => {
            let runtime = options
                .runtime
                .clone()
                .ok_or("Interactive mode requires a session runtime or connection")?;
            let local = Arc::new(crate::modes::agent_connection::in_process_agent_connection::InProcessAgentConnection::new(
                Arc::new(crate::core::agent_session_runtime::InProcessRuntimeHostAdapter::new(runtime)),
            ));
            in_process_connection = Some(local.clone());
            local
        }
    };
    let snapshot = connection.get_initial_snapshot().await?;
    let mut current_session_id = snapshot.state.session_id.clone();
    let mut ui_metrics = native_metrics::UiMetrics::new(&current_session_id);
    let mut state_refresh = native_state::StateRefresh::new();
    let services = if let Some(runtime) = &options.runtime {
        local::create_interactive_mode_ui_services(&runtime.session())
    } else {
        let cwd = snapshot.state.cwd.clone();
        let name = snapshot.state.session_name.clone();
        local::InteractiveModeUiServices {
            settings_manager: Arc::new(Mutex::new(local::SettingsManager::create(&cwd, None))),
            model_registry: Arc::new(Mutex::new(local::ModelRegistry::in_memory())),
            get_initial_cwd: Box::new(move || cwd.clone()),
            get_initial_session_name: Box::new(move || name.clone()),
            get_themes: Box::new(Vec::new),
            refresh_mcp_providers: None,
        }
    };
    let selected_theme = services
        .settings_manager
        .lock()
        .map_err(|e| e.to_string())?
        .get_theme();
    crate::modes::interactive::theme::theme::init_theme(selected_theme.as_deref(), false);
    crate::core::keybindings::KeybindingsManager::create(None).install();
    let initial_message = options.initial_message.clone();
    let initial_images = options.initial_images.clone();
    let initial_messages = options.initial_messages.clone();
    let mut controller = InteractiveMode::new(InteractiveModeOptions {
        migrated_providers: Some(options.migrated_providers.clone()),
        model_fallback_message: options.model_fallback_message.clone(),
        startup_notice: None,
        initial_message: None,
        initial_images: None,
        initial_messages: None,
        initial_prompts: None,
        verbose: options.verbose,
        agent_connection: Arc::new(connection.clone()),
        daemon_socket_path: options.daemon_socket_path.clone(),
        local_session_host: None,
        bind_local_session_extensions: false,
        ui_services: Some(services),
        on_shutdown: None,
        return_to_agents_view: options.return_to_agents_view,
        force_fullscreen: false,
        agents_view_owns_startup_notices: false,
        session_depth: options.session_depth,
        session_has_children: options.session_has_children,
        // A real store, so Ctrl+S and the agents-view handoff keep a draft.
        // TypeScript creates one `ClientPromptStashStore` per process and shares it
        // across chat views (main.ts:1448, 1544).
        prompt_stash_store: Some(
            crate::modes::interactive::prompt_stash_state::shared_prompt_stash_store(),
        ),
        prompt_stash_session_id: Some(snapshot.state.session_id.clone()),
    })?;
    controller.apply_connection_state_snapshot(project_state(snapshot.state.clone()));
    controller.init().await?;
    if controller.get_current_model().is_none() {
        controller.show_status("Welcome to Optimus. Connect a provider with /login, then choose a model with /model. Type /help for commands.", "accent");
    }
    if let Some(warning) = &options.model_fallback_message {
        if controller.get_model_fallback_warning_action(Some(warning)) == ModelFallbackWarningAction::Show {
            controller.show_warning(warning);
        }
    }
    let mode = Rc::new(RefCell::new(controller));
    native_subagents::seed(&mode, &snapshot);
    let subagents = Rc::new(RefCell::new(native_subagents::Bar::new(mode.clone())));
    let transcript = Rc::new(RefCell::new(Transcript::new(mode.clone())));
    transcript.borrow_mut().subagents = Some(subagents.clone());
    let initial_history = snapshot.history.clone();
    let initial_history_messages = snapshot.messages.clone();
    let initial_streaming_message = snapshot.streaming_message.clone();
    let ui = Rc::new(RefCell::new(TUI::new(
        Box::new(pi_tui::terminal::ProcessTerminal::new()),
        None,
    )));
    // The chat view owns the title from startup on, so a stale view title never
    // survives opening a chat (TS rebindCurrentSession -> updateTerminalTitle).
    refresh_terminal_title(&mode, &ui);
    let editor = Rc::new(RefCell::new(CustomEditor::new(
        ui.clone(),
        editor_theme(),
        CustomEditorOptions {
            placeholder: Some(mode.borrow().start_hint.into()),
            ..Default::default()
        },
    )));
    let actions = Rc::new(RefCell::new(Vec::<InputAction>::new()));
    native_autocomplete::configure(
        &mut editor.borrow_mut(),
        mode.clone(),
        &mode.borrow().get_current_cwd(),
    );
    {
        let actions = actions.clone();
        editor.borrow_mut().editor_mut().on_submit = Some(Box::new(move |text| {
            actions
                .borrow_mut()
                .push(InputAction::Submit(text.to_string(), false))
        }));
    }
    bind_editor_actions(&editor, &actions);
    bind_draft_restore_notice(&mode.borrow(), &mut editor.borrow_mut());
    let mut history_runtime = native_history::HistoryRuntime::new(connection.clone());
    // `renderInitialMessages` renders the transcript first and restores the
    // in-flight assistant message afterwards (interactive-mode.ts:6865-6871), the
    // same reset-then-streaming order the `SessionResynced` arm uses
    // (:3057-3067). `reset` replaces the transcript, so the streaming row must be
    // added after it or the attach loses the in-flight response.
    if let Some(error) = apply_history_snapshot(
        initial_history,
        initial_history_messages,
        initial_streaming_message,
        &transcript,
        &editor,
        &mut history_runtime,
        Some(native_history::ViewportFill::from_tui(&ui.borrow())),
    ) {
        // Optional history metadata never aborts attachment. The reference
        // reports and continues instead: the interactive event handler catches
        // the throw and only calls `showError` (interactive-mode.ts:5302-5304), so
        // the session stays attached and usable, and the no-window branch renders
        // the plain transcript (:6759-6767).
        mode.borrow_mut().show_error(&error);
    }
    let mut queue_runtime =
        native_queue::QueueRuntime::new(mode.clone(), editor.clone(), connection.clone());
    {
        let actions = actions.clone();
        editor.borrow_mut().on_escape = Some(Box::new(move || {
            actions.borrow_mut().push(InputAction::Escape)
        }));
    }
    {
        let actions = actions.clone();
        editor.borrow_mut().on_ctrl_d = Some(Box::new(move || {
            actions.borrow_mut().push(InputAction::Exit)
        }));
    }
    if mode.borrow().options.return_to_agents_view {
        let actions = actions.clone();
        editor.borrow_mut().on_agents_back = Some(Box::new(move || {
            actions.borrow_mut().push(InputAction::AgentsBack);
            true
        }));
    }
    // Automatic draft restore is session-scoped feedback, not transcript history.
    {
        let session = stash_session(&mode.borrow(), &current_session_id);
        if session.restore_on_open_pending() {
            let editor_text = editor.borrow().editor().get_text();
            if let Some(outcome) =
                session.restore_prompt_stash_if_editor_empty(None, &editor_text, true)
            {
                apply_prompt_stash_outcome(&mode, &editor, &outcome);
            }
        }
    }
    let input = Rc::new(RefCell::new(Vec::<String>::new()));
    let input_received = Rc::new(Cell::new(None::<Instant>));
    let viewport_input = Rc::new(Cell::new(false));
    let history_requested = Rc::new(Cell::new(false));
    {
        let input = input.clone();
        let mode = mode.clone();
        let viewport_input = viewport_input.clone();
        let history_requested = history_requested.clone();
        let input_received = input_received.clone();
        // Dispatch component input after releasing the TUI borrow: Editor owns
        // the same TUI handle and requests rendering from its input handlers.
        ui.borrow_mut().add_input_listener(Box::new(move |data| {
            if !pi_tui::keys::is_key_release(data) && input_received.get().is_none() {
                input_received.set(Some(Instant::now()));
            }
            let keys = pi_tui::keybindings::get_keybindings();
            if keys.matches(data, "tui.viewport.pageUp")
                || keys.matches(data, "tui.viewport.top")
                || data.starts_with("\x1b[<64;")
            {
                history_requested.set(true);
            }
            // Keep terminal replies and mouse scrolling in TUI's own handlers.
            if (data.starts_with("\x1b[6;") && data.ends_with('t'))
                || (mode.borrow().fullscreen_enabled && pi_tui::mouse::is_mouse_sequence(data))
                || (viewport_input.get()
                    && [
                        "tui.viewport.pageUp",
                        "tui.viewport.pageDown",
                        "tui.viewport.top",
                        "tui.viewport.follow",
                    ]
                    .iter()
                    .any(|key| pi_tui::keybindings::get_keybindings().matches(data, key)))
            {
                return InputListenerResult::default();
            }
            if pi_tui::keys::is_key_release(data) || pi_tui::mouse::is_mouse_sequence(data) {
                return InputListenerResult {
                    consume: true,
                    data: None,
                };
            }
            input.borrow_mut().push(data.to_string());
            InputListenerResult {
                consume: true,
                data: None,
            }
        }));
    }
    let extension_surfaces = Rc::new(RefCell::new(native_extensions::Surfaces::default()));
    transcript.borrow_mut().extension_surfaces = Some(extension_surfaces.clone());
    let side_pane = Rc::new(RefCell::new(native_extensions::SidePane::default()));
    transcript.borrow_mut().side_pane = Some(side_pane.clone());
    ui.borrow_mut().add_child(transcript.clone());
    ui.borrow_mut().add_child(Rc::new(RefCell::new(native_extensions::Widgets(extension_surfaces.clone(), false))));
    ui.borrow_mut().add_child(Rc::new(RefCell::new(transcript.borrow().stats_panel.clone())));
    ui.borrow_mut().add_child(editor.clone());
    ui.borrow_mut().add_child(Rc::new(RefCell::new(native_extensions::Widgets(extension_surfaces.clone(), true))));
    ui.borrow_mut().add_child(subagents.clone());
    ui.borrow_mut().add_child(Rc::new(RefCell::new(native_extensions::Statuses(extension_surfaces.clone(), Tray(mode.clone(), editor.clone())))));
    ui.borrow_mut().set_focus(Some(editor.clone()));
    ui.borrow_mut().start();
    let guard = TerminalGuard(ui.clone());
    {
        let settings = mode.borrow().settings_manager().clone();
        let settings = settings.lock().map_err(|e| e.to_string())?;
        ui.borrow_mut()
            .set_show_hardware_cursor(settings.get_show_hardware_cursor());
        ui.borrow_mut()
            .set_clear_on_shrink(settings.get_clear_on_shrink());
        editor
            .borrow_mut()
            .editor_mut()
            .set_padding_x(settings.get_editor_padding_x());
        editor
            .borrow_mut()
            .editor_mut()
            .set_autocomplete_max_visible(settings.get_autocomplete_max_visible());
    }
    let fullscreen = mode.borrow().fullscreen_enabled;
    native_settings::fullscreen(fullscreen, &mode, &editor, &ui, &transcript);
    ui.borrow_mut().run_pending_render(now_ms());
    ui_metrics.first_frame(opened_at);
    if benchmark {
        mode.borrow_mut().shutdown().await;
        drop(guard);
        connection.dispose().await?;
        return Ok(None);
    }
    let (send, receive) = mpsc::channel();
    {
        let send = send.clone();
        ui.borrow_mut().on_copy = Some(Box::new(move |text| {
            let (text, send) = (text.to_owned(), send.clone());
            tokio::spawn(async move {
                let event = match crate::utils::clipboard::copy_to_clipboard(&text).await {
                    Ok(outcome) => HostEvent::ClipboardNotice(outcome.status().to_string()),
                    Err(error) => HostEvent::Error(error.to_string()),
                };
                let _ = send.send(event);
            });
        }));
    }
    subagents.borrow().subscribe(connection.clone(), send.clone());
    let local_extension_bridge = in_process_connection.as_ref().map(|local| {
        let bridge = native_extension_bridge::Bridge::new(send.clone(), &mode.borrow().get_current_cwd());
        let shutdown = send.clone();
        let bindings = crate::modes::agent_connection::in_process_agent_connection::InProcessHeadlessExtensionOptions {
            ui_context: Some(bridge.clone()), shutdown_handler: Some(Arc::new(move || { let _ = shutdown.send(HostEvent::Shutdown); })),
        };
        let future = local.bind_headless_extensions(bindings.clone());
        let ready = send.clone();
        tokio::spawn(async move { if let Err(error) = future.await { let _ = ready.send(HostEvent::Error(error)); } });
        // Rebind the same typed UI after a fork/new/resume changes the runtime.
        let weak = Arc::downgrade(local);
        let reset = send.clone();
        local.runtime_host().runtime_set_rebind_session(Some(Arc::new(move || {
            // A rebind means fork / new / resume: the SESSION changed, so the
            // receiver must blanket-reset instead of keeping the old session's
            // Jev segments (Event::Reset stays the same-session reload path).
            let _ = reset.send(HostEvent::Extension(
                native_extension_bridge::Event::RuntimeRebound,
            ));
            let connection = weak.upgrade(); let bindings = bindings.clone();
            Box::pin(async move { if let Some(connection) = connection { let _ = connection.bind_headless_extensions(bindings).await; } })
        })));
        bridge
    });
    let mut custom_extension: Option<(String, Rc<RefCell<dyn TuiComponent>>, pi_tui::tui::OverlayHandle, tokio::sync::oneshot::Sender<Option<serde_json::Value>>)> = None;
    let mut early_custom_results = HashMap::<String, serde_json::Value>::new();
    let mut command_dialog: Option<(Rc<RefCell<dyn TuiComponent>>, Option<pi_tui::tui::OverlayHandle>)> = None;
    let mut command_cancel: Option<tokio_util::sync::CancellationToken> = None;
    let mut login_provider = String::new();
    let mut pending_relaunch = None;
    let event_send = send.clone();
    let unsubscribe = connection.subscribe(Arc::new(move |event| {
        let _ = event_send.send(HostEvent::Connection(event));
        Box::pin(async {})
    }));
    if let Some(message) = initial_message {
        submit(&connection, &send, message, false, initial_images);
    }
    for message in initial_messages {
        submit(&connection, &send, message, true, None);
    }
    let mut last_tick = Instant::now();
    let mut terminal_progress = false;
    let mut last_loader_tick = Instant::now();
    let mut selector: Option<Rc<RefCell<pi_tui::components::select_list::SelectList>>> = None;
    let mut overlay: Option<pi_tui::tui::OverlayHandle> = None;
    let (selection_send, selection_receive) = mpsc::channel::<Option<String>>();
    let mut models = Vec::<wire::AgentConnectionModel>::new();
    let mut configured_providers = std::collections::HashSet::<String>::new();
    let mut configuration: Option<Rc<RefCell<crate::modes::interactive::components::configuration_menu::ConfigurationMenuComponent>>> = None;
    let mut configuration_overlay: Option<pi_tui::tui::OverlayHandle> = None;
    let mut model_selector: Option<Rc<RefCell<ModelSelectorComponent>>> = None;
    let heartbeat_catalog = Rc::new(RefCell::new(Vec::<wire::AgentConnectionHeartbeat>::new()));
    let mut heartbeat_manager: Option<
        Rc<
            RefCell<
                crate::modes::interactive::components::heartbeat_manager::HeartbeatManagerComponent,
            >,
        >,
    > = None;
    let mut heartbeat_refresh_at: Option<Instant> = None;
    let mut heartbeat_status_refresh = native_status::HeartbeatRefresh::new();
    heartbeat_status_refresh.request(connection.clone(), &current_session_id, false);
    let mut settings_selector: Option<
        Rc<
            RefCell<
                crate::modes::interactive::components::settings_selector::SettingsSelectorComponent,
            >,
        >,
    > = None;
    let mut thinking_selector: Option<Rc<RefCell<ThinkingSelectorComponent>>> = None;
    let model_rows = Rc::new(Cell::new(ui.borrow().terminal_rows() as f64));
    let mut pending_login_model: Option<String> = None;
    // TS keeps this per `LoginDialogComponent` (`abortController`,
    // login-dialog.ts:77); the port owns it here so a superseded login is
    // distinguishable from the current one.
    let mut logins = LoginCoordinator::default();
    let mut extension: Option<ExtensionDialog> = None;
    let mut extension_queue =
        std::collections::VecDeque::<wire::AgentConnectionExtensionUiRequest>::new();
    let (extension_send, extension_receive) = mpsc::channel::<ExtensionReply>();
    // `await this.runStartupOnboarding()` (interactive-mode.ts:1803): persists
    // `onboardingShown` before the flow opens and shows the splash overlay for
    // first-run users. `PrimeOnboardingSplashComponent` swallows input while a
    // progress message is active (components/prime_onboarding_splash.rs:617-620).
    let mut onboarding_splash: Option<Rc<RefCell<PrimeOnboardingSplashComponent>>> = None;
    let mut onboarding_overlay: Option<pi_tui::tui::OverlayHandle> = None;
    let mut onboarding_settled: Option<Rc<Cell<i8>>> = None;
    if mode.borrow_mut().run_startup_onboarding() {
        let prime_cli_splash = mode.borrow().onboarding_uses_prime_cli_splash();
        let splash_rows = model_rows.clone();
        let splash_ui = ui.clone();
        // 0 = still open, 1 = accepted, -1 = cancelled
        // (`showOnboardingSplash`'s `settle`/`dismiss`, TS:8805-8832).
        let settled = Rc::new(Cell::new(0i8));
        let accepted_flag = settled.clone();
        let cancelled_flag = settled.clone();
        let component = Rc::new(RefCell::new(PrimeOnboardingSplashComponent::new(
            Box::new(move || accepted_flag.set(1)),
            Box::new(move || cancelled_flag.set(-1)),
            PrimeOnboardingSplashOptions {
                get_rows: Some(Box::new(move || splash_rows.get())),
                request_render: Some(Box::new(move || splash_ui.borrow_mut().request_render())),
                continue_action_label: prime_cli_splash.then(|| "choose a model".to_string()),
                ..Default::default()
            },
        )));
        let handle = ui.borrow_mut().show_overlay(
            component.clone(),
            pi_tui::tui::OverlayOptions {
                // `showOverlay(selector, { width: "100%", maxHeight: "100%",
                // row: 0, col: 0 })` (interactive-mode.ts:8839-8844).
                width: Some(pi_tui::tui::SizeValue::Percent("100%".into())),
                max_height: Some(pi_tui::tui::SizeValue::Percent("100%".into())),
                row: Some(pi_tui::tui::SizeValue::Number(0.0)),
                col: Some(pi_tui::tui::SizeValue::Number(0.0)),
                ..Default::default()
            },
        );
        onboarding_splash = Some(component);
        onboarding_overlay = Some(handle);
        onboarding_settled = Some(settled);
    }
    let mut exit_error = None;
    // Publish the CURRENT effective Jev footer segments (decision mode +
    // independent compaction) once, so the tray row shows the real per-session
    // settings from the first frame, before any `/jev` command runs. Only the
    // in-process connection publishes here: an attached daemon pushes its own
    // footer for the session it binds, so the UI never overrides it.
    if in_process_connection.is_some() {
        native_commands::jev_host::publish_session_footer(&send, &current_session_id);
    }
    loop {
        match ui.borrow_mut().terminal.poll_input() {
            Ok(true) => {}
            Ok(false) => break,
            Err(error) => {
                exit_error = Some(error.to_string());
                break;
            }
        }
        // The TypeScript gate is `overlayFocused || !fullscreen.viewportControls`
        // (packages/tui/src/tui.ts:1004), i.e. only a FOCUSED overlay blocks the
        // transcript; a visible non-capturing overlay (the editor's autocomplete
        // dropdown, packages/tui/src/components/editor.ts:2352) never takes focus
        // (tui.ts:439) and must not steal the viewport keys.
        viewport_input
            .set(ui.borrow().is_fullscreen() && !ui.borrow().is_fullscreen_overlay_focused() && command_dialog.is_none());
        ui.borrow_mut().drain_input();
        if let Some(received) = input_received.take() { ui_metrics.input(received); }
        if history_requested.replace(false) {
            history_runtime.request(&mode.borrow());
        }
        for data in std::mem::take(&mut *input.borrow_mut()) {
            let data = if let Some(bridge) = &local_extension_bridge {
                let Some(data) = bridge.filter_input(data) else { continue; }; data
            } else { data };
            if let Some(splash) = &onboarding_splash {
                splash.borrow_mut().handle_input(&data);
            } else if let Some((_, component, _, _)) = custom_extension.as_ref().filter(|(_, _, handle, _)| handle.is_focused()) {
                component.borrow_mut().handle_input(&data);
            } else if let Some(dialog) = &extension {
                dialog.component.borrow_mut().handle_input(&data);
            } else if let Some(dialog) = logins.dialog() {
                dialog.borrow_mut().handle_input(&data);
            } else if let Some((component, _)) = &command_dialog {
                if let Some(cancel) = &command_cancel {
                    if pi_tui::keybindings::get_keybindings().matches(&data, "tui.select.cancel") || pi_tui::keybindings::get_keybindings().matches(&data, "app.interrupt") { cancel.cancel(); }
                } else { component.borrow_mut().handle_input(&data); }
            } else if let Some(menu) = &configuration {
                menu.borrow_mut().handle_input(&data);
            } else if let Some(picker) = &model_selector {
                let mut picker = picker.borrow_mut();
                picker.handle_input(&data);
                if picker.cancelled {
                    let _ = selection_send.send(None);
                } else if let Some(model) = picker.selected_model.take() {
                    let _ = selection_send.send(Some(format!("{}/{}", model.provider, model.id)));
                }
            } else if let Some(manager) = &heartbeat_manager {
                manager.borrow_mut().handle_input(&data);
            } else if let Some(picker) = &settings_selector {
                picker.borrow_mut().handle_input(&data);
            } else if let Some(picker) = &thinking_selector {
                picker.borrow_mut().handle_input(&data);
            } else if let Some(selector) = &selector {
                selector.borrow_mut().handle_input(&data);
            } else if !side_pane.borrow().is_open() && subagents.borrow_mut().input(&data, &editor, &actions) {
            } else if !side_pane.borrow().is_open() && queue_runtime.handle_input(&data) {
            } else if pi_tui::keybindings::get_keybindings().matches(&data, "app.message.followUp")
            {
                actions.borrow_mut().push(InputAction::Submit(
                    editor.borrow().editor().get_expanded_text(),
                    true,
                ));
            } else {
                editor.borrow_mut().handle_input(&data);
            }
            ui.borrow_mut().request_render();
        }
        while let Ok(reply) = extension_receive.try_recv() {
            if extension.as_ref().map(|dialog| dialog.request.id.as_str()) == Some(reply.0.as_str())
            {
                if let Some(dialog) = extension.take() {
                    dialog.overlay.hide();
                }
                respond_extension(&connection, &send, reply, local_extension_bridge.as_ref());
            }
        }
        while let Ok(selected) = selection_receive.try_recv() {
            if let Some(handle) = overlay.take() {
                handle.hide();
            }
            selector = None;
            model_selector = None;
            thinking_selector = None;
            if let Some(selected) = selected {
                if let Some(target) = selected.strip_prefix("login:") {
                    if let Some((kind, provider)) = target.split_once(':') {
                        let _ = send.send(HostEvent::BeginLogin(provider.into(), kind == "oauth"));
                    }
                    continue;
                }
                if let Some(handle) = configuration_overlay.take() {
                    handle.hide();
                }
                configuration = None;
                if let Some(model) = models
                    .iter()
                    .find(|model| format!("{}/{}", model.provider, model.id) == selected)
                {
                    // `ensureModelProviderConfigured(model, authFlows, providerOptions)`
                    // (interactive-mode.ts:8015-8040) with
                    // `isModelProviderConfigured` (interactive-mode.ts:8042-8044).
                    // The provider list is the same one the Providers tab was built
                    // from, so a custom API-key provider offered there is accepted
                    // here too.
                    let registry_has_auth = match &options.runtime {
                        Some(runtime) => runtime
                            .services()
                            .model_registry
                            .lock()
                            .map_err(|e| e.to_string())?
                            .has_configured_auth(model),
                        None => false,
                    };
                    match native_configuration::model_selection_action(
                        &native_configuration::login_options_for(&models),
                        &model.provider,
                        configured_providers.contains(&model.provider),
                        registry_has_auth,
                    ) {
                        native_configuration::ModelSelectionAction::ExternallyConfigured => {
                            mode.borrow_mut().show_error(&format!(
                                "Authentication for {} must be configured externally.",
                                model.provider
                            ));
                            continue;
                        }
                        native_configuration::ModelSelectionAction::BeginLogin { oauth } => {
                            // `authFlows.loginProvider(provider)` routes to the
                            // OAuth dialog for an `oauth` entry and to the
                            // API-key dialog otherwise (auth-flows.ts:173-185).
                            pending_login_model = Some(selected);
                            let _ = send.send(HostEvent::BeginLogin(model.provider.clone(), oauth));
                            continue;
                        }
                        native_configuration::ModelSelectionAction::Switch => {}
                    }
                    mode.borrow_mut()
                        .show_status(&format!("Switching model: {}", model.id), "dim");
                    let (connection, send, model, session_id) = (
                        connection.clone(),
                        send.clone(),
                        model.clone(),
                        current_session_id.clone(),
                    );
                    tokio::spawn(async move {
                        let result = async {
                            connection.set_model(&model.provider, &model.id).await?;
                            connection.get_state().await
                        }
                        .await;
                        let _ = send.send(HostEvent::ModelSelected {
                            session_id,
                            model,
                            result,
                        });
                    });
                }
            } else {
                if let Some(handle) = configuration_overlay.take() {
                    handle.hide();
                }
                configuration = None;
            }
        }
        if let (Some(splash), Some(settled)) = (&onboarding_splash, &onboarding_settled) {
            match settled.get() {
                // `showOnboardingModelSelection` / `showConfigurationMenu("models")`
                // (interactive-mode.ts:1904-1921): the splash dismisses and the model
                // menu takes over.
                1 => {
                    splash.borrow_mut().dispose();
                    if let Some(handle) = onboarding_overlay.take() {
                        handle.hide();
                    }
                    onboarding_splash = None;
                    onboarding_settled = None;
                    submit(&connection, &send, "/model".into(), false, None);
                    ui.borrow_mut().request_render();
                }
                -1 => {
                    splash.borrow_mut().dispose();
                    if let Some(handle) = onboarding_overlay.take() {
                        handle.hide();
                    }
                    onboarding_splash = None;
                    onboarding_settled = None;
                    ui.borrow_mut().request_render();
                }
                _ => {}
            }
        }
        for action in std::mem::take(&mut *actions.borrow_mut()) {
            match action {
                InputAction::Submit(text, follow_up) => {
                    if side_pane.borrow().is_open() {
                        side_pane.borrow_mut().submit(
                            text, &mode, &editor, connection.clone(), send.clone(),
                        );
                        ui.borrow_mut().request_render();
                        continue;
                    }
                    if queue_runtime.submit(&text, follow_up) {
                        continue;
                    }
                    if text.trim().is_empty() {
                        continue;
                    }
                    if should_record_prompt_history(&text) {
                        editor.borrow_mut().editor_mut().add_to_history(&text);
                    }
                    editor.borrow_mut().editor_mut().set_text("");
                    if matches!(text.trim(), "/quit" | "/exit") {
                        mode.borrow_mut().shutdown_requested = true;
                        continue;
                    }
                    if text.trim() == "/hotkeys" {
                        // `/hotkeys` appends the full reference to the chat, unlike
                        // the ephemeral `?` guide (interactive-mode.ts:4931-4936).
                        mode.borrow_mut().handle_hotkeys_command();
                    } else if text.trim() == "/help" {
                        mode.borrow_mut().show_status("/model  select a model\n/login  provider setup\n/new  new session\n/context  context usage\n/compact [instructions]  compact session\n/refine  refine reusable knowledge\n/goal <objective>  pursue a goal\n!<command>  run shell command\n/quit  exit\nEsc interrupts; Ctrl+C twice exits; Ctrl+D exits an empty prompt; Alt+Enter queues follow-up; Ctrl+O expands tools.", "dim");
                    } else {
                        submit_with_metrics(&connection, &send, text, follow_up, None, Some(ui_metrics.recorder.clone()));
                    }
                }
                InputAction::Interrupt | InputAction::Escape => {
                    if side_pane.borrow().is_open() {
                        side_pane.borrow_mut().close(connection.clone());
                        ui.borrow_mut().request_render();
                        continue;
                    }
                    let second = matches!(action, InputAction::Interrupt)
                        && mode.borrow().is_ctrl_c_exit_hint_visible();
                    if second {
                        mode.borrow_mut().shutdown_requested = true;
                        continue;
                    }
                    // TS handleCtrlC clears the escape repeat (interactive-mode.ts).
                    if matches!(action, InputAction::Interrupt) {
                        mode.borrow_mut().clear_escape_repeat();
                    }
                    // TS handleEscape (interactive-mode.ts:6924-6953): a repeated
                    // Escape inside the window runs the armed action instead of
                    // the interrupt flow; a first press arms and interrupts.
                    if matches!(action, InputAction::Escape)
                        && escape_repeat_step(&mode, &editor, &ui, &connection, &send).await
                    {
                        ui.borrow_mut().request_render();
                        continue;
                    }
                    if mode.borrow().has_interruptible_work() {
                        let activity = InterruptActivity::from_mode(&mode.borrow());
                        mode.borrow_mut().show_status("Stopping current work; keeping queued input paused.", "dim");
                        let connection = connection.clone();
                        let send = send.clone();
                        tokio::spawn(async move {
                            let result = interrupt_active_work(&connection, activity).await;
                            if result.is_ok() && activity.abort_session {
                                let _ = send.send(HostEvent::Status("Stop requested. Queued work is paused; late child reports are saved without starting replies. In-flight work is cancelling. Send a new prompt to resume.".into()));
                            }
                            let _ = send.send(HostEvent::Completed(result));
                        });
                    } else {
                        let draft = mode.borrow_mut().queue_selection.reset();
                        editor.borrow_mut().editor_mut().set_text(&draft);
                    }
                    if matches!(action, InputAction::Interrupt) {
                        mode.borrow_mut().show_ctrl_c_exit_hint();
                    }
                }
                InputAction::Exit => {
                    mode.borrow_mut().shutdown_requested = true;
                }
                InputAction::Heartbeats => {
                    submit(&connection, &send, "/heartbeats".into(), false, None)
                }
                InputAction::ToggleTools => {
                    mode.borrow_mut().toggle_tool_output_expansion();
                    transcript.borrow_mut().apply_chat_detail();
                }
                InputAction::ToggleExecutionMode => {
                    let command = if connection.supports_node_execution_mode() {
                        "/mode toggle"
                    } else if mode.borrow().connection_state.as_ref().and_then(|s| s.execution_mode)
                        == Some(crate::core::execution_mode::ExecutionMode::Direct) {
                        "/mode ipython"
                    } else { "/mode direct" };
                    submit(&connection, &send, command.into(), true, None);
                }
                InputAction::ToggleThinking => {
                    let mut mode = mode.borrow_mut();
                    mode.hide_thinking_block = !mode.hide_thinking_block;
                    mode.settings_manager()
                        .lock()
                        .map_err(|e| e.to_string())?
                        .set_hide_thinking_block(mode.hide_thinking_block);
                    for assistant in transcript.borrow().all_assistants() {
                        assistant
                            .borrow_mut()
                            .set_hide_thinking_block(mode.hide_thinking_block);
                    }
                }
                InputAction::ToggleMessages => {
                    mode.borrow_mut().toggle_agent_message_expansion();
                    for message in transcript.borrow().all_agent_messages() {
                        message.borrow_mut().set_expanded(mode.borrow().agent_messages_expanded);
                    }
                    for tool in transcript.borrow().all_tools() {
                        tool.borrow_mut().set_agent_messages_expanded(mode.borrow().agent_messages_expanded);
                    }
                }
                InputAction::Subagents => {
                    stash_editor_draft_for_agents_view(&mode, &editor, &current_session_id);
                    mode.borrow_mut().return_to_agents_view(InteractiveModeRunResultType::ScopedAgentsView);
                }
                InputAction::AgentsBack => {
                    // `returnToAgentsView` stashes the live draft first
                    // (interactive-mode.ts:7122) so the handoff does not lose it.
                    stash_editor_draft_for_agents_view(&mode, &editor, &current_session_id);
                    mode.borrow_mut()
                        .return_to_agents_view(InteractiveModeRunResultType::AgentsView);
                }
                // Ctrl+S: `handlePromptStash` (interactive-mode.ts:4379-4394).
                InputAction::PromptStash => {
                    handle_prompt_stash_action(&mode, &editor, &current_session_id)
                }
                InputAction::Model => {
                    submit(&connection, &send, "/model".into(), false, None);
                }
                // `app.edits.expand` (interactive-mode.ts:4300 ->
                // toggleEditDiffExpansion): flip the mode flag and re-apply it to
                // the transcript's tool components.
                InputAction::ToggleEditDiffs => {
                    mode.borrow_mut().toggle_edit_diff_expansion();
                    let expanded = mode.borrow().edit_diffs_expanded;
                    for tool in transcript.borrow().all_tools() {
                        tool.borrow_mut().set_edit_diffs_expanded(expanded);
                    }
                }
                // `app.editor.external` (interactive-mode.ts:4306).
                InputAction::OpenExternalEditor => open_external_editor_for(&mode, &editor, &ui),
                // `app.session.new|tree|fork|resume` (interactive-mode.ts:4313-4322):
                // handleClearCommand / showTreeSelector / showUserMessageSelector /
                // requestAgentsView. The port routes each through its existing
                // built-in command flow.
                InputAction::SessionNew => submit(&connection, &send, "/new".into(), false, None),
                InputAction::SessionTree => submit(&connection, &send, "/tree".into(), false, None),
                InputAction::SessionFork => submit(&connection, &send, "/fork".into(), false, None),
                InputAction::SessionResume => {
                    stash_editor_draft_for_agents_view(&mode, &editor, &current_session_id);
                    mode.borrow_mut()
                        .return_to_agents_view(InteractiveModeRunResultType::AgentsView);
                }
                // `applyThinkingLevel` (interactive-mode.ts:8309-8321): the picker
                // already closed itself, so only the level is applied.
                InputAction::ThinkingLevel(level) => {
                    if let Some(handle) = overlay.take() {
                        handle.hide();
                    }
                    thinking_selector = None;
                    let (connection, send, mode) = (connection.clone(), send.clone(), mode.clone());
                    tokio::spawn(async move {
                        let result = match connection.set_thinking_level(level).await {
                            Ok(()) => {
                                let _ = send.send(HostEvent::Status(format!(
                                    "Thinking level: {}",
                                    level.as_str()
                                )));
                                Ok(())
                            }
                            Err(error) => Err(error),
                        };
                        let _ = send.send(HostEvent::Completed(result));
                    });
                    let _ = &mode;
                }
                InputAction::DismissSelector => {
                    if let Some(handle) = overlay.take() {
                        handle.hide();
                    }
                    thinking_selector = None;
                    selector = None;
                    model_selector = None;
                }
                // `?` opens an ephemeral guide; it never reaches the chat history
                // (interactive-mode.ts:4289, :10197-10204).
                InputAction::Shortcuts => mode.borrow_mut().show_shortcut_guide(),
                InputAction::Suspend => mode.borrow_mut().handle_ctrl_z(),
            }
            ui.borrow_mut().request_render();
        }
        // Preserve event order, but yield back to input/rendering under a continuous stream.
        let mut event_budget = HostEventBudget::new();
        while let Some(event) = event_budget.next(&receive) {
            if matches!(&event,
                HostEvent::Connection(
                    wire::AgentConnectionEvent::SessionEvent { .. }
                    | wire::AgentConnectionEvent::SessionResynced { .. }
                    | wire::AgentConnectionEvent::SessionReplaced { .. }
                    | wire::AgentConnectionEvent::Closed { .. }
                ) | HostEvent::RefreshSnapshot(_)
                    | HostEvent::ModelSelected { .. } | HostEvent::SettingAccepted(_)
                    | HostEvent::ScopeChanged(..)
            ) {
                state_refresh.invalidate();
            }
            if matches!(&event, HostEvent::RefreshSnapshot(_)
                | HostEvent::Connection(wire::AgentConnectionEvent::SessionResynced { .. }
                    | wire::AgentConnectionEvent::SessionReplaced { .. })) {
                heartbeat_status_refresh.cancel();
                mode.borrow_mut().heartbeat_catalog.clear();
                mode.borrow_mut().heartbeat_catalog_authoritative = false;
                heartbeat_catalog.borrow_mut().clear();
            }
            match event {
                HostEvent::MenuTiming(started) => ui_metrics.menu(started),
                HostEvent::Shutdown => mode.borrow_mut().shutdown_requested = true,
                HostEvent::Extension(event) => {
                    use native_extension_bridge::Event;
                    use crate::core::extensions::types::ExtensionUiContext;
                    match event {
                        event @ (Event::Reset | Event::RuntimeRebound) => {
                            // TWO different reset contracts, kept distinct:
                            // * `Event::Reset` (/reload, extension re-init)
                            //   resets the SAME session: the session and its
                            //   settings are unchanged, so the host-published
                            //   Jev segments are retained and refreshed by key
                            //   on the next publish.
                            // * `Event::RuntimeRebound` (fork, /new, in-chat
                            //   /resume rebuilt the runtime) changed the
                            //   SESSION: blanket reset, because old-session
                            //   values must never survive into the new
                            //   session's first frame. The new session's
                            //   authoritative footer arrives right after: fork
                            //   and branch send RefreshSnapshot from the
                            //   command, and /new and /resume now do too.
                            if matches!(&event, Event::RuntimeRebound) {
                                extension_surfaces.borrow_mut().reset();
                            } else {
                                extension_surfaces.borrow_mut().reset_keeping_jev();
                            }
                            if let Some(bridge) = &local_extension_bridge { bridge.reset(); }
                            if let Some((_, _, handle, reply)) = custom_extension.take() { handle.hide(); let _ = reply.send(None); }
                            early_custom_results.clear();
                            if let Some(dialog) = extension.take() {
                                dialog.overlay.hide();
                                respond_extension(&connection, &send, (dialog.request.id, cancel_extension_response(&dialog.request.method)), local_extension_bridge.as_ref());
                            }
                            for request in extension_queue.drain(..) { respond_extension(&connection, &send, (request.id, cancel_extension_response(&request.method)), local_extension_bridge.as_ref()); }
                            let mut mode = mode.borrow_mut();
                            mode.working_message = None; mode.set_working_visible(true); mode.set_working_indicator(None);
                            mode.update_terminal_title();
                            ui.borrow_mut().terminal.set_title(&mode.ui.terminal.title);
                        }
                        Event::Paste(text) => editor.borrow_mut().handle_input(&format!("\x1b[200~{text}\x1b[201~")),
                        Event::ToolsExpanded(expanded) => {
                            mode.borrow_mut().set_tools_expanded(expanded);
                            transcript.borrow_mut().apply_chat_detail();
                        }
                        Event::Widget(key, factory, options) => {
                            if let Some(bridge) = &local_extension_bridge {
                                let component = factory.map(|factory| Box::new(native_extension_bridge::ComponentAdapter(factory(bridge.tui(), bridge.theme()), false)) as Box<dyn TuiComponent>);
                                let below = options.and_then(|o| o.placement) == Some(crate::core::extensions::types::WidgetPlacement::BelowEditor);
                                extension_surfaces.borrow_mut().set_widget_component(key, component, below);
                            }
                        }
                        Event::Header(factory) => {
                            if let Some(bridge) = &local_extension_bridge {
                                extension_surfaces.borrow_mut().header = factory.map(|f| Box::new(native_extension_bridge::ComponentAdapter(f(bridge.tui(), bridge.theme()), false)) as Box<dyn TuiComponent>);
                            }
                        }
                        Event::Footer(factory) => {
                            if let Some(bridge) = &local_extension_bridge {
                                extension_surfaces.borrow_mut().footer = factory.map(|f| Box::new(native_extension_bridge::ComponentAdapter(f(bridge.tui(), bridge.theme(), bridge.footer_data.clone()), false)) as Box<dyn TuiComponent>);
                            }
                        }
                        Event::Custom(id, component, options, reply) => {
                            if let Some(result) = early_custom_results.remove(&id) { component.dispose(); let _ = reply.send(Some(result)); continue; }
                            if let Some((_, _, handle, previous)) = custom_extension.take() { handle.hide(); let _ = previous.send(None); }
                            let component: Rc<RefCell<dyn TuiComponent>> = Rc::new(RefCell::new(native_extension_bridge::ComponentAdapter(component, false)));
                            let handle = ui.borrow_mut().show_overlay(component.clone(), native_extension_bridge::overlay_options(options.as_ref()));
                            custom_extension = Some((id, component, handle, reply));
                        }
                        Event::Done(id, value) => {
                            if custom_extension.as_ref().is_some_and(|(active,_,_,_)| *active == id) {
                                if let Some((_, _, handle, reply)) = custom_extension.take() { handle.hide(); let _ = reply.send(Some(value)); }
                            } else { early_custom_results.insert(id, value); }
                        }
                    }
                }
                HostEvent::CommandDialog(dialog) => {
                    if let Some((_, Some(handle))) = command_dialog.take() { handle.hide(); }
                    transcript.borrow().stats_panel.clear();
                    if let Some(cancel) = command_cancel.take() { cancel.cancel(); }
                    let docked = matches!(&dialog, native_commands::Dialog::Stats);
                    let component = native_commands::mount(dialog, ui.clone(), &mode.borrow(), connection.clone(), send.clone());
                    let handle = if docked {
                        transcript.borrow().stats_panel.show(component.clone());
                        None
                    } else { Some(ui.borrow_mut().show_overlay(component.clone(), pi_tui::tui::OverlayOptions {
                        width: Some(pi_tui::tui::SizeValue::Percent("100%".into())),
                        max_height: Some(pi_tui::tui::SizeValue::Percent("100%".into())),
                        row: Some(pi_tui::tui::SizeValue::Number(0.0)), col: Some(pi_tui::tui::SizeValue::Number(0.0)),
                        ..Default::default()
                    })) };
                    command_dialog = Some((component, handle));
                }
                HostEvent::CommandBusy(message, cancel) => {
                    if let Some((_, Some(handle))) = command_dialog.take() { handle.hide(); }
                    transcript.borrow().stats_panel.clear();
                    if let Some(cancel) = command_cancel.replace(cancel) { cancel.cancel(); }
                    let component: Rc<RefCell<dyn TuiComponent>> = Rc::new(RefCell::new(TuiText::new(format!("{message}\nEsc to cancel"), 1, 1, None)));
                    let handle = ui.borrow_mut().show_overlay(component.clone(), Default::default());
                    command_dialog = Some((component, Some(handle)));
                }
                HostEvent::CloseCommandDialog => {
                    if let Some((_, Some(handle))) = command_dialog.take() { handle.hide(); }
                    transcript.borrow().stats_panel.clear();
                    command_cancel = None;
                }
                HostEvent::EditorText(text) => editor.borrow_mut().editor_mut().set_text(&text),
                HostEvent::ReconfigureAutocomplete => {
                    // TS: refreshConnectionCatalog + setupAutocompleteProvider
                    // (interactive-mode.ts:9185-9186). `configure` refetches
                    // `get_commands` off-thread and rebuilds the provider; the
                    // reply applies on the next keystroke.
                    let cwd = mode.borrow().get_current_cwd();
                    native_autocomplete::configure(&mut editor.borrow_mut(), mode.clone(), &cwd);
                }
                HostEvent::PromptSession { text } => {
                    handle_prompt_session(&mode, &editor, &connection, &send, &text);
                }
                HostEvent::ReloadSettings => {
                    mode.borrow().settings_manager().lock().map_err(|e| e.to_string())?.reload().await;
                }
                HostEvent::ScopeChanged(session, scoped) => {
                    if current_session_id == session {
                        if let Some(state) = mode.borrow_mut().connection_state.as_mut() {
                            state.scoped_models = scoped.into_iter().map(|s| local::AgentConnectionScopedModel { model: s.model }).collect();
                        }
                    }
                }
                HostEvent::AuthChanged => {
                    if let Some(runtime) = &options.runtime { runtime.services().model_registry.lock().map_err(|e| e.to_string())?.refresh(); }
                    let current_model = mode.borrow().get_current_model().cloned();
                    let scoped = mode.borrow().get_scoped_model_state().into_iter().map(|s| s.model).collect::<Vec<_>>();
                    match native_configuration::refresh_after_login(connection.as_ref(), configuration.as_ref(), current_model.as_ref(), &scoped).await {
                        Ok(catalog) => { models = catalog.models; configured_providers = catalog.configured_providers; }
                        Err(error) => mode.borrow_mut().show_error(&error),
                    }
                }
                HostEvent::RunUpdate(args) => {
                    match native_commands::update(&args, &options, &mode, &ui, &connection).await {
                        Ok(Some(args)) => { pending_relaunch = Some(args); mode.borrow_mut().shutdown_requested = true; }
                        Ok(None) => {
                            let fullscreen = mode.borrow().fullscreen_enabled;
                            native_settings::fullscreen(fullscreen, &mode, &editor, &ui, &transcript);
                            // Same-session reload: keep the Jev segments (see
                            // the Event::Reset arm).
                            extension_surfaces.borrow_mut().reset_keeping_jev(); submit(&connection, &send, "/reload".into(), false, None);
                        }
                        Err(error) => {
                            let fullscreen = mode.borrow().fullscreen_enabled;
                            native_settings::fullscreen(fullscreen, &mode, &editor, &ui, &transcript);
                            mode.borrow_mut().show_error(&error);
                        }
                    }
                }
                HostEvent::SideQuestion(question) => {
                    let padding = mode.borrow().settings_manager().lock()
                        .map(|settings| settings.get_editor_padding_x() as usize).unwrap_or(2);
                    side_pane.borrow_mut().start(question, padding, connection.clone(), send.clone());
                    ui.borrow_mut().request_render();
                }
                HostEvent::SideCommands(id, commands) => side_pane.borrow_mut().set_commands(&id, commands),
                HostEvent::SideBashFailed(id, error) => {
                    if side_pane.borrow_mut().bash_failed(&id, &error) {
                        state_refresh.request(connection.clone(), current_session_id.clone());
                    }
                    ui.borrow_mut().request_render();
                }
                HostEvent::Debug => {
                    if let Err(error) = native_commands::debug(&ui, &connection, &send).await { mode.borrow_mut().show_error(&error); }
                }
                HostEvent::Connection(wire::AgentConnectionEvent::SideQuestionEvent { event }) => {
                    side_pane.borrow_mut().update(event);
                    ui.borrow_mut().request_render();
                }
                HostEvent::Connection(wire::AgentConnectionEvent::Closed { error }) => {
                    native_state::stop_activity(&mut mode.borrow_mut());
                    exit_error = error;
                    mode.borrow_mut().shutdown_requested = true;
                }
                HostEvent::Connection(wire::AgentConnectionEvent::SessionEvent { event }) => {
                    if side_pane.borrow_mut().handle_bash_event(&event, &ui, &connection) {
                        let mut mode = mode.borrow_mut();
                        native_state::track_activity(&mut mode, &event);
                        match event {
                            wire::AgentConnectionSessionEvent::BashStart { .. } => {
                                mode.patch_connection_state(|state| state.is_bash_running = true);
                            }
                            wire::AgentConnectionSessionEvent::BashEnd { .. } => {
                                mode.patch_connection_state(|state| state.is_bash_running = false);
                            }
                            _ => {}
                        }
                        ui.borrow_mut().request_render();
                        continue;
                    }
                    if event.type_name() == "session_action_update" {
                        queue_runtime.observe_queue_change();
                    }
                    let finished = matches!(event.type_name(), "agent_end" | "session_action_update");
                    let renamed = matches!(
                        &event,
                        wire::AgentConnectionSessionEvent::SessionInfoChanged { .. }
                    );
                    apply_event(&mode, &transcript, event);
                    if renamed {
                        refresh_terminal_title(&mode, &ui);
                    }
                    if finished {
                        state_refresh.request(connection.clone(), current_session_id.clone());
                    }
                }
                HostEvent::Connection(wire::AgentConnectionEvent::SessionReplaced {
                    state,
                    messages,
                }) => {
                    if current_session_id != state.session_id {
                        ui.borrow_mut().scroll_to_bottom();
                    }
                    extension_surfaces.borrow_mut().reset();
                    side_pane.borrow_mut().close(connection.clone());
                    if let Some(dialog) = extension.take() {
                        dialog.overlay.hide();
                        respond_extension(
                            &connection,
                            &send,
                            (
                                dialog.request.id,
                                wire::AgentConnectionExtensionUiResponse::Cancelled {
                                    cancelled: true,
                                },
                            ),
                        local_extension_bridge.as_ref(),
                        );
                    }
                    for request in extension_queue.drain(..) {
                        respond_extension(
                            &connection,
                            &send,
                            (
                                request.id,
                                wire::AgentConnectionExtensionUiResponse::Cancelled {
                                    cancelled: true,
                                },
                            ),
                        local_extension_bridge.as_ref(),
                        );
                    }
                    mode.borrow_mut().reset_subagent_summary();
                    current_session_id = state.session_id.clone();
                    queue_runtime.reset_session(current_session_id.clone());
                    mode.borrow_mut()
                        .apply_connection_state_snapshot(project_state(state));
                    heartbeat_status_refresh.request(connection.clone(), &current_session_id, false);
                    refresh_terminal_title(&mode, &ui);
                    if let Some(error) = apply_history_snapshot(
                        None,
                        messages,
                        None,
                        &transcript,
                        &editor,
                        &mut history_runtime,
                        Some(native_history::ViewportFill::from_tui(&ui.borrow())),
                    ) {
                        mode.borrow_mut().show_error(&error);
                    }
                }
                HostEvent::RefreshSnapshot(snapshot) | HostEvent::Connection(wire::AgentConnectionEvent::SessionResynced { snapshot }) => {
                    native_subagents::seed(&mode, &snapshot);
                    if current_session_id != snapshot.state.session_id {
                        ui.borrow_mut().scroll_to_bottom();
                        extension_surfaces.borrow_mut().reset();
                        side_pane.borrow_mut().close(connection.clone());
                        // The reset cleared the Jev segments; republish the new
                        // session's effective settings so the row is never
                        // stale across a session switch (in-process only: an
                        // attached daemon pushes its own footer per attach).
                        if in_process_connection.is_some() {
                            native_commands::jev_host::publish_session_footer(
                                &send,
                                &snapshot.state.session_id,
                            );
                        }
                    }
                    current_session_id = snapshot.state.session_id.clone();
                    queue_runtime.reset_session(current_session_id.clone());
                    mode.borrow_mut()
                        .apply_connection_state_snapshot(project_state(snapshot.state));
                    heartbeat_status_refresh.request(connection.clone(), &current_session_id, false);
                    refresh_terminal_title(&mode, &ui);
                    if let Some(error) = apply_history_snapshot(
                        snapshot.history,
                        snapshot.messages,
                        snapshot.streaming_message,
                        &transcript,
                        &editor,
                        &mut history_runtime,
                        Some(native_history::ViewportFill::from_tui(&ui.borrow())),
                    ) {
                        mode.borrow_mut().show_error(&error);
                    }
                    // TS: refreshCommandCatalogForCurrentSession on session
                    // resync (interactive-mode.ts:2951, :3048-3055).
                    let _ = send.send(HostEvent::ReconfigureAutocomplete);
                }
                HostEvent::Connection(wire::AgentConnectionEvent::ExtensionError {
                    error, ..
                }) => mode.borrow_mut().show_error(&error),
                HostEvent::Connection(wire::AgentConnectionEvent::ExtensionUiRequest {
                    request,
                }) => match request.method.as_str() {
                    "setStatus" => {
                        if let Some(key) = optional_string(&request.payload, "statusKey") {
                            // `statusCompactText` is an optional backward-compatible
                            // field: senders without it (old daemons, generic
                            // extensions) degrade to left-truncation instead of
                            // segment compaction on narrow rows.
                            extension_surfaces.borrow_mut().set_status(
                                key,
                                optional_string(&request.payload, "statusText"),
                                optional_string(&request.payload, "statusCompactText"),
                            );
                        }
                    }
                    "setWidget" => {
                        if let Some(key) = optional_string(&request.payload, "widgetKey") {
                            extension_surfaces.borrow_mut().set_widget(key, get_payload_string_array(&request.payload, "widgetLines"), optional_string(&request.payload, "widgetPlacement").as_deref() == Some("belowEditor"));
                        }
                    }
                    "setWorkingIndicator" => { mode.borrow_mut().set_working_indicator(get_payload_working_indicator_options(&request.payload, "options")); }
                    "select" | "confirm" | "input" | "editor" => extension_queue.push_back(request),
                    "notify" => {
                        let message = string(&request.payload, "message");
                        match string(&request.payload, "notifyType").as_str() {
                            "error" => mode.borrow_mut().show_error(&message),
                            "warning" => mode.borrow_mut().show_warning(&message),
                            _ => mode.borrow_mut().show_status(&message, "dim"),
                        }
                    }
                    "setEditorText" => {
                        if let Some(text) = optional_string(&request.payload, "text") {
                            editor.borrow_mut().editor_mut().set_text(&text);
                        }
                    }
                    "setTitle" => {
                        if let Some(title) = optional_string(&request.payload, "title") {
                            ui.borrow_mut().terminal.set_title(&title);
                        }
                    }
                    "setWorkingMessage" => {
                        mode.borrow_mut().working_message =
                            optional_string(&request.payload, "message")
                    }
                    "setWorkingVisible" => {
                        if let Some(visible) = request
                            .payload
                            .get("visible")
                            .and_then(|value| value.as_bool())
                        {
                            mode.borrow_mut().set_working_visible(visible);
                        }
                    }
                    "setHiddenThinkingLabel" => {
                        mode.borrow_mut()
                            .set_hidden_thinking_label(optional_string(&request.payload, "label"));
                        for assistant in transcript.borrow().all_assistants() {
                            assistant
                                .borrow_mut()
                                .set_hidden_thinking_label(&mode.borrow().hidden_thinking_label);
                        }
                    }
                    _ => mode.borrow_mut().show_status(
                        &format!("Unsupported extension UI request: {}", request.method),
                        "dim",
                    ),
                },
                HostEvent::Connection(wire::AgentConnectionEvent::ConnectionStatus {
                    status,
                    error,
                }) => {
                    transcript.borrow_mut().connection_status = status.clone();
                    mode.borrow_mut().show_status(&error.unwrap_or(status), "dim");
                }
                HostEvent::Connection(wire::AgentConnectionEvent::SessionStatus { recap }) => {
                    mode.borrow_mut().session_recap = recap;
                    mode.borrow_mut().render_recap();
                }
                HostEvent::Connection(wire::AgentConnectionEvent::HeartbeatsChanged) => {
                    heartbeat_status_refresh.request(connection.clone(), &current_session_id, true);
                }
                HostEvent::ModelSelected {
                    session_id,
                    model,
                    result,
                } => match result {
                    Ok(mut state)
                        if current_session_id == session_id && state.session_id == session_id =>
                    {
                        let mut controller = mode.borrow_mut();
                        controller
                            .settings_manager()
                            .lock()
                            .map_err(|error| error.to_string())?
                            .set_default_model_and_provider(&model.provider, &model.id);
                        // Model selection can settle after a newer turn has started.
                        state.is_streaming = controller.is_agent_streaming();
                        state.is_compacting = controller.is_agent_compacting();
                        state.is_bash_running = controller.is_bash_running();
                        native_state::apply_refresh(&mut controller, state);
                        controller.show_status(&format!("Model: {}", model.id), "success");
                    }
                    Ok(_) => {}
                    Err(error) => mode.borrow_mut().show_error(&error),
                },
                HostEvent::Completed(result) => {
                    if let Err(error) = result {
                        mode.borrow_mut().show_error(&error);
                    }
                    state_refresh.request(connection.clone(), current_session_id.clone());
                }
                HostEvent::Render => {}
                HostEvent::Heartbeats(catalog, open) => {
                    heartbeat_status_refresh.cancel();
                    native_status::apply_catalog_result(
                        &mut mode.borrow_mut(), &catalog,
                        connection.heartbeat_catalog_supported() == Some(true),
                    );
                    *heartbeat_catalog.borrow_mut() = catalog;
                    if open {
                        if heartbeat_manager.is_some() {
                            if let Some(handle) = &overlay {
                                handle.focus();
                            }
                        } else {
                            let manager = Rc::new(RefCell::new(native_heartbeats::create(
                                mode.clone(),
                                heartbeat_catalog.clone(),
                                model_rows.clone(),
                                send.clone(),
                                connection.clone(),
                            )));
                            if let Some(handle) = overlay.take() {
                                handle.hide();
                            }
                            overlay = Some(ui.borrow_mut().show_overlay(
                                manager.clone(),
                                pi_tui::tui::OverlayOptions {
                                    width: Some(pi_tui::tui::SizeValue::Percent("100%".into())),
                                    max_height: Some(pi_tui::tui::SizeValue::Percent(
                                        "100%".into(),
                                    )),
                                    suspend_fullscreen_mouse: true,
                                    ..Default::default()
                                },
                            ));
                            heartbeat_manager = Some(manager);
                        }
                    }
                    if heartbeat_manager.is_some() {
                        if let Some(delay) = native_heartbeats::refresh_delay(
                            &native_heartbeats::scoped(&mode.borrow(), &heartbeat_catalog.borrow()),
                        ) {
                            let next = Instant::now() + delay;
                            heartbeat_refresh_at =
                                Some(heartbeat_refresh_at.map_or(next, |old| old.min(next)));
                        } else {
                            heartbeat_refresh_at = None;
                        }
                    }
                }
                HostEvent::HeartbeatUpdated(heartbeat, job) => {
                    let id = job["id"].as_str().unwrap_or_default();
                    let mut catalog = heartbeat_catalog.borrow_mut();
                    catalog.retain(|h| h.job["id"].as_str() != Some(id));
                    if job["status"] == "active" || job["status"] == "paused" {
                        catalog.push(wire::AgentConnectionHeartbeat {
                            job: job.clone(),
                            ..heartbeat
                        });
                    }
                    let mut controller = mode.borrow_mut();
                    if job["source"] == "heartbeat"
                        && controller
                            .connection_state
                            .as_ref()
                            .and_then(|s| s.active_session_id.as_deref())
                            == job["activeSessionId"].as_str()
                    {
                        let heartbeat = if job["status"] == "active" || job["status"] == "paused" {
                            project_heartbeat(job)
                        } else {
                            None
                        };
                        controller.patch_connection_state(|s| s.heartbeat = heartbeat);
                    }
                    native_heartbeats::apply_catalog(&mut controller, &catalog);
                }
                HostEvent::CloseHeartbeats => {
                    heartbeat_manager = None;
                    heartbeat_refresh_at = None;
                    if let Some(handle) = overlay.take() {
                        handle.hide();
                    }
                    ui.borrow_mut().set_focus(Some(editor.clone()));
                }
                HostEvent::Settings(state) => {
                    match native_settings::create(&mode.borrow(), &state, &send) {
                        Ok(picker) => {
                            if let Some(handle) = overlay.take() {
                                handle.hide();
                            }
                            let picker = Rc::new(RefCell::new(picker));
                            overlay = Some(
                                ui.borrow_mut()
                                    .show_overlay(picker.clone(), Default::default()),
                            );
                            settings_selector = Some(picker);
                        }
                        Err(error) => {
                            let _ = send.send(HostEvent::Warning(error));
                        }
                    }
                }
                HostEvent::Setting(native_settings::Change::Close) => {
                    if let Some(handle) = overlay.take() {
                        handle.hide();
                    }
                    settings_selector = None;
                    ui.borrow_mut().set_focus(Some(editor.clone()));
                }
                HostEvent::SettingAccepted(change) => {
                    native_settings::apply_remote_applied(&change, &mode, &ui);
                }
                HostEvent::Setting(change) => {
                    // DEFECT A: the daemon calls are fire-and-forget, so the
                    // owner loop never waits on a daemon round-trip. TypeScript
                    // fires `void agentConnection.set*.catch(showError)`
                    // (interactive-mode.ts:7804, :7836, :7842, :7847, :7852)
                    // AFTER patching the connection state locally (:7803,
                    // :7835, :7840), so the local state and its persistence
                    // survive a failed remote call.
                    native_settings::apply_change(
                        change,
                        &mode,
                        &editor,
                        &ui,
                        &transcript,
                        &connection,
                        &send,
                    );
                }
                HostEvent::Status(status) => mode.borrow_mut().show_status(&status, "dim"),
                HostEvent::ClipboardNotice(message) => {
                    transcript.borrow_mut().clipboard_notice.show(message, Instant::now());
                }
                HostEvent::Warning(warning) => mode.borrow_mut().show_warning(&warning),
                HostEvent::Panel(panel) => {
                    transcript.borrow_mut().panel(&panel);
                }
                HostEvent::EchoLocal(text) => {
                    transcript.borrow_mut().echo_local(&text);
                }
                HostEvent::ContextTree(tree) => {
                    let width = context_tree_width(ui.borrow().terminal.columns());
                    match serde_json::from_value::<crate::core::context_tree::ContextTreeNode>(tree)
                    {
                        Ok(root) => transcript.borrow_mut().panel(
                            &crate::modes::interactive::components::context_tree_format::format_context_tree(
                                &root, width,
                            ),
                        ),
                        // `handleContextCommand`'s `catch` reports a formatting
                        // failure through `showError` (interactive-mode.ts:9802-9804).
                        Err(error) => mode.borrow_mut().show_error(&error.to_string()),
                    }
                }
                // `showThinkingSelector` (interactive-mode.ts:8285-8307) mounts
                // `ThinkingSelectorComponent`; selecting a level calls
                // `applyThinkingLevel`, cancelling just closes.
                HostEvent::ThinkingLevels { current, levels } => {
                    let levels_for_select = levels.clone();
                    let actions_for_select = actions.clone();
                    let levels_for_cancel = levels.clone();
                    let actions_for_cancel = actions.clone();
                    let picker = Rc::new(RefCell::new(ThinkingSelectorComponent::new(
                        current,
                        &levels,
                        Box::new(move |level| {
                            let _ = &levels_for_select;
                            actions_for_select
                                .borrow_mut()
                                .push(InputAction::ThinkingLevel(level));
                        }),
                        Box::new(move || {
                            let _ = &levels_for_cancel;
                            actions_for_cancel
                                .borrow_mut()
                                .push(InputAction::DismissSelector);
                        }),
                    )));
                    if let Some(handle) = overlay.take() {
                        handle.hide();
                    }
                    selector = None;
                    overlay = Some(ui.borrow_mut().show_overlay(
                        picker.clone() as Rc<RefCell<dyn TuiComponent>>,
                        Default::default(),
                    ));
                    thinking_selector = Some(picker);
                }
                HostEvent::Configuration(catalog, tab, search) => {
                    models = catalog.models;
                    configured_providers = catalog.configured_providers.into_iter().collect();
                    let menu = native_configuration::create(
                        &mode.borrow(),
                        ui.clone(),
                        tab,
                        &models,
                        &configured_providers,
                        search,
                        model_rows.clone(),
                        selection_send.clone(),
                        send.clone(),
                    )?;
                    if let Some(handle) = configuration_overlay.take() {
                        handle.hide();
                    }
                    let menu = Rc::new(RefCell::new(menu));
                    configuration_overlay = Some(ui.borrow_mut().show_overlay(
                        menu.clone(),
                        pi_tui::tui::OverlayOptions {
                            width: Some(pi_tui::tui::SizeValue::Number(96.0)),
                            max_height: Some(pi_tui::tui::SizeValue::Percent("100%".into())),
                            suspend_fullscreen_mouse: true,
                            ..Default::default()
                        },
                    ));
                    configuration = Some(menu);
                }
                HostEvent::BeginLogin(provider, oauth) => {
                    login_provider = provider.clone();
                    // `new LoginDialogComponent(...)` gives this login its own
                    // `abortController` (login-dialog.ts:77). `begin()` retires
                    // the previous login first, exactly like the previous
                    // dialog's `cancel()` aborting only its own controller
                    // (login-dialog.ts:139-146).
                    let generation = logins.begin();
                    let tx = send.clone();
                    let mut dialog = LoginDialogComponent::new(
                        ui.clone(),
                        &provider,
                        Box::new(move |success, error| {
                            if !success {
                                let _ = tx.send(HostEvent::LoginFinished(
                                    generation,
                                    Err(error.unwrap_or_else(|| "Login cancelled".into())),
                                ));
                            }
                        }),
                        None,
                        None,
                    );
                    let token = tokio_util::sync::CancellationToken::new();
                    if oauth {
                        dialog.show_progress("Starting sign-in...");
                    }
                    let dialog = Rc::new(RefCell::new(dialog));
                    let handle = ui
                        .borrow_mut()
                        .show_overlay(dialog.clone(), Default::default());
                    logins.install(LoginSlot {
                        generation,
                        token: token.clone(),
                        overlay: handle,
                        dialog: dialog.clone(),
                    });
                    start_login(generation, provider, oauth, send.clone(), token);
                }
                HostEvent::LoginAuth(url, instructions) => {
                    if let Some(dialog) = logins.dialog() {
                        dialog.borrow_mut().show_auth(&url, instructions.as_deref());
                    }
                }
                HostEvent::LoginProgress(message) => {
                    if let Some(dialog) = logins.dialog() {
                        dialog.borrow_mut().show_progress(&message);
                    }
                }
                HostEvent::LoginPrompt(message, placeholder, sender) => {
                    if let Some(dialog) = logins.dialog() {
                        let receiver = dialog
                            .borrow_mut()
                            .show_prompt(&message, placeholder.as_deref());
                        tokio::spawn(async move {
                            if let Ok(value) = receiver.await {
                                let _ = sender.send(value);
                            }
                        });
                    }
                }
                HostEvent::LoginFinished(generation, result) => {
                    // Only the login still in flight may cancel, hide, or touch
                    // host state. A superseded generation is ignored: its dialog
                    // was already retired by the newer `BeginLogin`, and its
                    // completion must not disturb the newer dialog
                    // (login-dialog.ts:77 keeps the two controllers independent).
                    if !apply_login_finished(&mut logins, generation) {
                        continue;
                    }
                    match result {
                        Ok(()) => {
                            if login_provider.starts_with("mcp:") {
                                mode.borrow_mut().show_status("MCP credentials saved. Reloading connections...", "success");
                                let _ = send.send(HostEvent::AuthChanged);
                                if !mode.borrow().is_agent_streaming() { submit(&connection, &send, "/reload".into(), false, None); }
                                else { mode.borrow_mut().show_status("Run /reload after the current turn to activate the connection.", "dim"); }
                                continue;
                            }
                            mode.borrow_mut().show_status(
                                "Credentials saved. Choose a model with /model.",
                                "success",
                            );
                            if let Some(runtime) = &options.runtime {
                                runtime
                                    .services()
                                    .model_registry
                                    .lock()
                                    .map_err(|e| e.to_string())?
                                    .refresh();
                            }
                            // `invalidateConnectionModels(); await
                            // this.getConnectionAvailableModels()` via
                            // `onAuthChanged` (interactive-mode.ts:8123-8126, :8855-8858),
                            // then the menu refresh at :8384-8401. The daemon
                            // re-derives its catalog on this call
                            // (`daemon_mode.rs` "get_model_catalog" ->
                            // `refresh_model_catalog`), which is what makes the
                            // newly authenticated provider visible to the
                            // selection loop below.
                            let current_model = mode.borrow().get_current_model().cloned();
                            let scoped: Vec<wire::AgentConnectionModel> = mode
                                .borrow()
                                .get_scoped_model_state()
                                .into_iter()
                                .map(|scoped| scoped.model)
                                .collect();
                            match native_configuration::refresh_after_login(
                                connection.as_ref(),
                                configuration.as_ref(),
                                current_model.as_ref(),
                                &scoped,
                            )
                            .await
                            {
                                Ok(refreshed) => {
                                    // The outer state the model-selection loop reads
                                    // (`this.connectionModelCatalog` /
                                    // `this.connectionConfiguredProviders`,
                                    // interactive-mode.ts:8042-8044).
                                    models = refreshed.models;
                                    configured_providers = refreshed.configured_providers;
                                }
                                Err(error) => mode.borrow_mut().show_error(&error),
                            }
                            if let Some(key) = pending_login_model.take() {
                                submit(&connection, &send, format!("/model {key}"), false, None);
                            } else if let Some(handle) = &configuration_overlay {
                                handle.focus();
                            } else {
                                submit(&connection, &send, "/model".into(), false, None);
                            }
                        }
                        Err(error) => {
                            pending_login_model = None;
                            // `menu.refreshAuthentication()` runs before the
                            // status check in `authenticate`
                            // (interactive-mode.ts:8379-8382), so a cancelled or
                            // failed login still re-reads the credential store.
                            if let Some(menu) = &configuration {
                                menu.borrow_mut().refresh_authentication();
                            }
                            if error != "Login cancelled" {
                                mode.borrow_mut().show_error(&error);
                            }
                        }
                    }
                }
                HostEvent::AgentsView => {
                    // `requestAgentsView` (interactive-mode.ts:3509-3519): resident
                    // sessions return to the agents view, ephemeral ones report why
                    // they cannot.
                    mode.borrow_mut().request_agents_view();
                }
                HostEvent::Error(error) => mode.borrow_mut().show_error(&error),
                HostEvent::Fullscreen(requested) => {
                    apply_fullscreen_request(requested, &mode, &editor, &ui, &transcript)
                }
                // `showModelsSelector` (interactive-mode.ts:8457-8533).
                HostEvent::Models(catalog, search, started) => {
                    models = catalog.models;
                    configured_providers = catalog.configured_providers.into_iter().collect();
                    if let Some(model) = search.as_deref().and_then(|query| {
                        crate::core::model_resolver::find_exact_model_reference_match(
                            query, &models,
                        )
                    }) {
                        let _ =
                            selection_send.send(Some(format!("{}/{}", model.provider, model.id)));
                    } else {
                        let _ = send.send(HostEvent::Configuration(
                            wire::AgentConnectionModelCatalog {
                                models: models.clone(),
                                configured_providers: configured_providers
                                    .iter()
                                    .cloned()
                                    .collect(),
                                ..Default::default()
                            },
                            "models",
                            search,
                        ));
                        let _ = send.send(HostEvent::MenuTiming(started));
                    }
                }
            }
            ui.borrow_mut().request_render();
        }
        if extension
            .as_ref()
            .and_then(|dialog| dialog.deadline)
            .is_some_and(|deadline| Instant::now() >= deadline)
        {
            if let Some(dialog) = extension.take() {
                dialog.overlay.hide();
                let response = cancel_extension_response(&dialog.request.method);
                respond_extension(&connection, &send, (dialog.request.id, response), local_extension_bridge.as_ref());
            }
        }
        if extension.is_none()
            && custom_extension.is_none()
            && command_dialog.is_none()
            && !logins.is_active()
            && heartbeat_manager.is_none()
            && settings_selector.is_none()
            && thinking_selector.is_none()
            && selector.is_none()
            && configuration.is_none()
            && model_selector.is_none()
        {
            if let Some(request) = extension_queue.pop_front() {
                extension = extension_dialog(request.clone(), &ui, &extension_send);
                if extension.is_none() {
                    respond_extension(
                        &connection,
                        &send,
                        (
                            request.id,
                            wire::AgentConnectionExtensionUiResponse::Cancelled { cancelled: true },
                        ),
                    local_extension_bridge.as_ref(),
                        );
                }
                ui.borrow_mut().request_render();
            }
        }
        if let Some(manager) = &heartbeat_manager {
            manager.borrow_mut().poll_action();
        }
        if let Some((catalog, authoritative)) = heartbeat_status_refresh.poll(connection.clone(), &current_session_id).await {
            native_status::apply_catalog_result(&mut mode.borrow_mut(), &catalog, authoritative);
            *heartbeat_catalog.borrow_mut() = catalog;
            ui.borrow_mut().request_render();
        }
        if heartbeat_refresh_at.is_some_and(|at| Instant::now() >= at) {
            heartbeat_refresh_at = Some(Instant::now() + Duration::from_secs(5));
            let connection = connection.clone();
            let send = send.clone();
            tokio::spawn(async move {
                if let Ok(catalog) = connection.list_heartbeats().await {
                    let _ = send.send(HostEvent::Heartbeats(catalog, false));
                }
            });
        }
        if mode.borrow().shutdown_requested {
            break;
        }
        if last_tick.elapsed() >= Duration::from_millis(250) {
            let progress = mode.borrow().should_show_working_loader()
                && mode
                    .borrow()
                    .settings_manager()
                    .lock()
                    .map(|s| s.get_show_terminal_progress())
                    .unwrap_or(false);
            if progress != terminal_progress {
                ui.borrow_mut().terminal.set_progress(progress);
                terminal_progress = progress;
            }
            mode.borrow_mut().tick_working_pulse();
            ui.borrow_mut().request_render();
            last_tick = Instant::now();
        }
        // The pi-tui `Loader` repaints on an 80 ms interval; the host tick is
        // slower, so drive the extra frames while the working loader is mounted.
        if last_loader_tick.elapsed() >= Duration::from_millis(80) {
            if mode.borrow().should_show_working_loader() {
                ui.borrow_mut().request_render();
            }
            last_loader_tick = Instant::now();
        }
        // `showCtrlCExitHint`'s 2 s timer has no Rust counterpart, so the host
        // expires the hint on its own 16 ms cadence (interactive-mode.ts:7018-7025).
        mode.borrow_mut().expire_ctrl_c_exit_hint();
        if transcript.borrow_mut().clipboard_notice.expire(Instant::now()) {
            ui.borrow_mut().request_render();
        }
        history_runtime.poll(&mode, &transcript, &ui);
        if let Some(bridge) = &local_extension_bridge {
            *bridge.editor_text.lock().unwrap_or_else(|e| e.into_inner()) = editor.borrow().editor().get_text();
            bridge.tools_expanded.store(mode.borrow().tool_output_expanded, std::sync::atomic::Ordering::Relaxed);
            bridge.footer_data.set_cwd(&mode.borrow().get_current_cwd());
            bridge.footer_data.set_available_provider_count(models.iter().map(|m| &m.provider).collect::<std::collections::HashSet<_>>().len());
        }
        queue_runtime.poll();
        if state_refresh.poll(&mode, &current_session_id) {
            ui.borrow_mut().request_render();
        }
        state_refresh.reconcile_if_due(connection.clone(), current_session_id.clone(), &mode.borrow());
        let rows = ui.borrow().terminal_rows();
        model_rows.set(rows as f64);
        editor.borrow_mut().editor_mut().set_terminal_rows(rows);
        editor.borrow_mut().editor_mut().poll_autocomplete();
        ui_metrics.session(&current_session_id);
        let render_requested = ui.borrow().render_requested();
        let render_started = Instant::now();
        ui.borrow_mut().run_pending_render(now_ms());
        if render_requested { ui_metrics.rendered(render_started.elapsed()); }
        tokio::time::sleep(Duration::from_millis(16)).await;
    }
    unsubscribe();
    ui_metrics.flush_render();
    if let Some(bridge) = &local_extension_bridge { bridge.close(); }
    if let Some((_, _, handle, reply)) = custom_extension { handle.hide(); let _ = reply.send(None); }
    extension_surfaces.borrow_mut().reset();
    if let Some(cancel) = command_cancel { cancel.cancel(); }
    if let Some((_, Some(handle))) = command_dialog { handle.hide(); }
    transcript.borrow().stats_panel.clear();
    side_pane.borrow_mut().close(connection.clone());
    let mut cancelled_dialogs = Vec::new();
    if let Some(dialog) = extension {
        dialog.overlay.hide();
        cancelled_dialogs.push(dialog.request.id);
    }
    for request in extension_queue {
        cancelled_dialogs.push(request.id);
    }
    // The login in flight owns its own token (login-dialog.ts:77); cancelling it
    // here aborts a pending sign-in before the terminal is torn down.
    logins.cancel_current();
    mode.borrow_mut().shutdown().await;
    drop(guard);
    let returning_to_browser = mode.borrow().agents_view_request.is_some();
    let cleanup = close_session_view(connection.clone(), cancelled_dialogs);
    if returning_to_browser {
        // Detach acknowledges are not a prerequisite for drawing the browser.
        // This only disposes the UI's connection, never aborts the agent.
        tokio::spawn(async move {
            if let Err(error) = cleanup.await {
                crate::modes::agents_view::agents_view_mode::log_client_error("Failed to detach session view", &error);
            }
        });
    } else {
        cleanup.await?;
    }
    if let Some(args) = pending_relaunch {
        let status = std::process::Command::new(std::env::current_exe().map_err(|e| e.to_string())?)
            .args(args).current_dir(mode.borrow().get_current_cwd()).status().map_err(|e| format!("Failed to relaunch optimus-rust: {e}"))?;
        if !status.success() { return Err(format!("Relaunched optimus-rust exited with {status}")); }
        return Ok(None);
    }
    let result = if mode.borrow().agents_view_request.is_some() {
        Some(mode.borrow_mut().run().await)
    } else {
        None
    };
    match exit_error {
        Some(error) => Err(error),
        None => Ok(result),
    }
}

async fn close_session_view(connection: Arc<dyn wire::AgentConnection>, cancelled_dialogs: Vec<String>) -> Result<(), String> {
    let cancel_dialogs = async {
        for id in cancelled_dialogs {
            let _ = connection.respond_to_extension_ui_request(
                &id, wire::AgentConnectionExtensionUiResponse::Cancelled { cancelled: true },
            ).await;
        }
    };
    let _ = tokio::time::timeout(Duration::from_secs(2), cancel_dialogs).await;
    connection.dispose().await
}

#[derive(Clone, Copy)]
struct InterruptActivity {
    compacting: bool,
    bash_running: bool,
    retrying: bool,
    abort_session: bool,
}

impl InterruptActivity {
    fn from_mode(mode: &InteractiveMode) -> Self {
        Self {
            compacting: mode.is_agent_compacting(),
            bash_running: mode.is_bash_running(),
            retrying: mode.get_retry_attempt() > 0.0,
            abort_session: mode.is_agent_streaming()
                || mode.connection_state.as_ref().is_some_and(|state| {
                    state.session_actions.active.is_some() || state.session_actions.queued_count > 0
                }),
        }
    }
}

async fn interrupt_active_work(
    connection: &Arc<dyn wire::AgentConnection>,
    activity: InterruptActivity,
) -> Result<(), String> {
    let mut requests = Vec::new();
    if activity.abort_session {
        requests.push(connection.abort());
    }
    if activity.retrying {
        requests.push(connection.abort_retry());
    }
    if activity.compacting {
        requests.push(connection.abort_compaction());
        requests.push(connection.abort_branch_summary());
    }
    if activity.bash_running {
        requests.push(connection.abort_bash());
    }
    // Issue independent cancellation requests together; an unresponsive activity
    // must not prevent another owner from receiving its cancellation.
    let results = futures::future::join_all(requests).await;
    results.into_iter().collect::<Result<Vec<_>, _>>().map(|_| ())
}

/// Dispatches one submitted line.
///
/// Port of the submission path at interactive-mode.ts:4784-5039: the line is
/// parsed through the shared built-in registry, built-ins run as local commands,
/// and every other line - free text and extension commands - reaches the model
/// exactly as typed.
fn submit(
    connection: &Arc<dyn wire::AgentConnection>,
    send: &mpsc::Sender<HostEvent>,
    text: String,
    follow_up: bool,
    images: Option<Vec<ImageContent>>,
) {
    submit_with_metrics(connection, send, text, follow_up, images, None);
}

fn submit_with_metrics(
    connection: &Arc<dyn wire::AgentConnection>,
    send: &mpsc::Sender<HostEvent>,
    text: String,
    follow_up: bool,
    images: Option<Vec<ImageContent>>,
    recorder: Option<Arc<dyn pi_agent_core::performance_metrics::PerformanceMetricRecorder>>,
) {
    let (connection, send) = (connection.clone(), send.clone());
    tokio::spawn(async move {
        let is_prompt = match classify_submission(&text) {
            SlashDispatch::Model(line) => !line.starts_with('!'),
            SlashDispatch::SessionCommand(_) => true,
            SlashDispatch::Builtin { .. } => false,
        };
        let submission = dispatch_submission(&connection, &send, &text, follow_up, images);
        let result = match recorder.filter(|_| is_prompt) {
            Some(recorder) => native_metrics::acknowledged(recorder, submission).await,
            None => submission.await,
        };
        let _ = send.send(HostEvent::Completed(result));
    });
}

/// The awaited body of [`submit`].
///
/// Split out of the `tokio::spawn` so the dispatch chain - the part that picks
/// between a local built-in handler, a session slash command, and a model
/// prompt - is directly awaitable and can be asserted against a recording
/// connection (`interactive-mode.ts:4784-5039`).
async fn dispatch_submission(
    connection: &Arc<dyn wire::AgentConnection>,
    send: &mpsc::Sender<HostEvent>,
    text: &str,
    follow_up: bool,
    images: Option<Vec<ImageContent>>,
) -> Result<(), String> {
    let (connection, send) = (connection.clone(), send.clone());
    let text = text.to_string();
    {
        // `const slashCommand = parseSlashCommand(text)` then
        // `resolveBuiltinSlashCommandName` (interactive-mode.ts:4784-4785).
        let dispatch = classify_submission(&text);
        // `/login` keeps its dedicated provider picker (interactive-mode.ts:4952-4956).
        let result = match dispatch {
            SlashDispatch::Builtin { name, args, raw } if name == "login" && args.is_empty() => {
                let started = Instant::now();
                match connection.get_model_catalog().await {
                    Ok(catalog) => {
                        let _ = send.send(HostEvent::Configuration(catalog, "providers", None));
                        let _ = send.send(HostEvent::MenuTiming(started));
                    }
                    Err(error) => {
                        let _ = send.send(HostEvent::Warning(error));
                    }
                }
                Ok(())
            }
            SlashDispatch::Builtin { name, args, raw } => {
                // `if (commandName === "login")` with an argument logs that
                // provider in directly (interactive-mode.ts:4952-4956).
                if name == "login" {
                    let provider = args.trim();
                    let oauth = crate::core::auth_storage::get_oauth_provider(provider).is_some();
                    let _ = send.send(HostEvent::BeginLogin(provider.into(), oauth));
                    Ok(())
                } else {
                    run_builtin_command(&connection, &send, &raw, &name, &args)
                        .await
                        .map(|output| {
                            for event in output.into_events() {
                                let _ = send.send(event);
                            }
                        })
                }
            }
            // A session slash command reaches the session exactly like free
            // text: TypeScript has no arm for `compact`/`refine`/`goal`/
            // `autonomous` in its local chain (interactive-mode.ts:4821-5030),
            // so they fall through to `agentConnection.prompt(...)`
            // (interactive-mode.ts:5177-5181) and `AgentSession` turns the text
            // into a session command action (`agent-session.ts:5065-5068`,
            // `_executeQueuedSessionCommand` :6760-6824).
            SlashDispatch::SessionCommand(line) => {
                if crate::core::slash_commands::parse_session_slash_command(&line)
                    .is_some_and(|command| command.name == "mode") && !connection.supports_execution_mode() {
                    return Err("This session host does not support execution mode switching. Update and restart the daemon.".into());
                }
                if crate::core::slash_commands::parse_session_slash_command(&line)
                    .is_some_and(|command| command.name == "mode" && matches!(command.args.trim(), "node" | "toggle"))
                    && !connection.supports_node_execution_mode() {
                    return Err("This session host does not support Node mode. Update and restart the daemon, or use /mode ipython or /mode direct.".into());
                }
                prompt_model(&connection, &line, follow_up, images).await
            }
            // `!command` runs shell (interactive-mode.ts:5043-5070).
            SlashDispatch::Model(line) => match line.strip_prefix('!') {
                Some(command) => connection.execute_bash(command, None).await,
                // Anything else - free text and extension commands - prompts the
                // model with the original text (interactive-mode.ts:5130-5145).
                None => prompt_model(&connection, &line, follow_up, images).await,
            },
        };
        result
    }
}

/// `agentConnection.prompt(text, { streamingBehavior, queueIfBusy, images })`
/// (interactive-mode.ts:5177-5181).
///
/// Session slash commands and free text share this call, which is why the port
/// routes them through the same helper instead of a parallel execution path.
async fn prompt_model(
    connection: &Arc<dyn wire::AgentConnection>,
    line: &str,
    follow_up: bool,
    images: Option<Vec<ImageContent>>,
) -> Result<(), String> {
    connection
        .prompt(
            line,
            Some(wire::AgentConnectionPromptOptions {
                images,
                streaming_behavior: Some(if follow_up { "followUp" } else { "steer" }.into()),
                ..Default::default()
            }),
        )
        .await
}

/// The `/new` prompt handoff (interactive-mode.ts:10243-10248): collect pasted
/// images for the prompt, record it in the up-arrow history, then prompt the
/// model with the text verbatim — never through the slash dispatcher.
fn handle_prompt_session(
    mode: &Rc<RefCell<InteractiveMode>>,
    editor: &Rc<RefCell<CustomEditor>>,
    connection: &Arc<dyn wire::AgentConnection>,
    send: &mpsc::Sender<HostEvent>,
    text: &str,
) {
    let images = mode.borrow_mut().collect_images_for(text);
    editor.borrow_mut().editor_mut().add_to_history(text);
    let (connection, send, text) = (connection.clone(), send.clone(), text.to_string());
    tokio::spawn(async move {
        let result =
            prompt_model(&connection, &text, false, (!images.is_empty()).then_some(images)).await;
        let _ = send.send(HostEvent::Completed(result));
    });
}

/// `getAvailableThinkingLevels` (interactive-mode.ts:8189-8193).
///
/// The dispatch task holds only the connection, so it applies the same rule to
/// `AgentConnectionState` that `InteractiveMode::get_available_thinking_levels`
/// applies to its snapshot.
fn available_thinking_levels(
    state: &wire::AgentConnectionState,
) -> Vec<pi_agent_core::types::ThinkingLevel> {
    let levels = state.available_thinking_levels.clone();
    let supports_thinking = !levels.is_empty()
        && !(levels.len() == 1 && levels[0] == pi_agent_core::types::ThinkingLevel::Off);
    if supports_thinking {
        levels
    } else {
        Vec::new()
    }
}

/// Built-ins whose TypeScript arm requires `!commandArgs`
/// (interactive-mode.ts:4826-5030) and which no later arm accepts.
///
/// `/clear` is deliberately absent: the alias reports `Usage: /clear` instead of
/// prompting, and `/new` takes arguments (:4968-4989). `/login` and `/model` are
/// absent too - both have argument-taking arms (:4836-4840, :4953-4956).
fn no_argument_builtin_slash_command(name: &str) -> bool {
    matches!(
        name,
        "settings"
            | "scoped-models"
            | "share"
            | "copy"
            | "session"
            | "system-prompt"
            | "context"
            | "logs"
            | "changelog"
            | "hotkeys"
            | "fork"
            | "clone"
            | "tree"
            | "logout"
            | "reload"
            | "debug"
    )
}

/// The decision the host makes for one submitted line.
///
/// Port of interactive-mode.ts:4784-4786 (`parseSlashCommand` +
/// `resolveBuiltinSlashCommandName`).
#[derive(Debug, Clone, PartialEq, Eq)]
enum SlashDispatch {
    /// A built-in command or one of its aliases: `name` is canonical, `args` are
    /// trimmed, `raw` is the command as the user typed it.
    Builtin {
        name: String,
        args: String,
        raw: String,
    },
    /// A session slash command (`compact`, `refine`, `goal`, `autonomous`):
    /// the line is submitted verbatim, exactly as free text is, so the session
    /// recognises it (`slash-commands.ts:273-281`).
    SessionCommand(String),
    /// Free text, an extension command, or a bare `/` - it goes to the model
    /// exactly as typed (interactive-mode.ts:5130-5145).
    Model(String),
}

/// Resolves a submitted line through the shared built-in registry.
///
/// The registry (`core/slash_commands.rs`) is the single source of truth for
/// names and aliases, which is what routes `/clear`, `/usage`, `/thinking`,
/// `/rename`, and `/side` to their canonical commands.
fn classify_submission(text: &str) -> SlashDispatch {
    // Session commands are checked BEFORE the local built-in chain: TypeScript
    // turns them into session command actions inside `AgentSession`
    // (`agent-session.ts:5065-5068`) and has no local arm for them
    // (interactive-mode.ts:4821-5030), so the host must not run them locally.
    if crate::core::slash_commands::parse_session_slash_command(text).is_some() {
        return SlashDispatch::SessionCommand(text.trim().to_string());
    }

    if let Some(command) = crate::core::slash_commands::resolve_leading_builtin_slash_command(text)
    {
        // TypeScript gates these arms on `!commandArgs` and has no later arm for
        // them (interactive-mode.ts:4826-5030), so a command WITH arguments falls
        // through the whole local chain and reaches
        // `agentConnection.prompt(text, ...)` like free text (:5177-5181).
        if !command.args.is_empty() && no_argument_builtin_slash_command(&command.name) {
            return SlashDispatch::Model(text.to_string());
        }
        return SlashDispatch::Builtin {
            name: command.name,
            args: command.args,
            raw: text.trim().to_string(),
        };
    }

    // TypeScript dispatches a few commands straight off `parseSlashCommand`'s name
    // (`interactive-mode.ts:5025`) without them being registry entries, because
    // `resolveBuiltinSlashCommandName` only rewrites aliases and otherwise passes the
    // name through (slash-commands.ts:249-251). `/debug` is the remaining one: TS
    // handles it locally (`handleDebugCommand`, interactive-mode.ts:10259) instead of
    // sending it to the model, so the port must not forward it as chat text. The
    // registry gains no invented entry; the dispatcher reports it as a known command
    // the host does not implement yet.
    if let Some(command) = crate::core::slash_commands::parse_slash_command(text) {
        if command.name == "debug" {
            return SlashDispatch::Builtin {
                name: command.name,
                args: command.args,
                raw: text.trim().to_string(),
            };
        }
    }

    SlashDispatch::Model(text.to_string())
}

/// Whether the submitted line belongs in the up-arrow history. TypeScript
/// records prompt history only in the paths that prompt or run bash
/// (interactive-mode.ts:5079, :5150); the local built-in arms record nothing,
/// while session/extension commands fall through to the prompt and are
/// recorded.
fn should_record_prompt_history(text: &str) -> bool {
    !matches!(classify_submission(text), SlashDispatch::Builtin { .. })
}

/// The text a local command leaves in the chat.
///
/// Each arm maps to one of the TypeScript's local render paths
/// (`showStatus`, `showWarning`, `echoLocalCommand`), so a dispatched command
/// never reaches the model as chat text.
#[derive(Debug)]
enum CommandOutput {
    Nothing,
    Status(String),
    ClipboardNotice(String),
    Warning(String),
    EchoLocal(String),
    /// `echoLocalCommand(text)` followed by the command's own panel.
    ///
    /// The TypeScript pairs the two in ONE arm for every command that renders a
    /// panel: `/session` interactive-mode.ts:4887+9499-9501, `/system-prompt`
    /// :4893+9541-9544, `/context` :4904+9807-9808, `/logs` :4910+9532-9533, and
    /// `/changelog` :4926+10016-10021. `echoLocalCommand` (6405-6413) mounts the
    /// TYPED command as the user's own message, so the panel output stays anchored
    /// to a visible command instead of floating, and the command itself is never
    /// lost. `command` is the raw text as typed (`/clear` stays `/clear`), not the
    /// canonical name.
    ///
    /// `/rlm-max-depth` (:4881-4884) and `/heartbeat` (:4914-4918) call no
    /// `echoLocalCommand`, so they keep the bare [`CommandOutput::Panel`].
    EchoPanel {
        command: String,
        panel: String,
    },
    /// `chatContainer.addChild(new Spacer(1)); addChild(new Text(info, 1, 0))`.
    Panel(String),
    /// `handleContextCommand` (interactive-mode.ts:9796-9809): the raw
    /// `getContextTree()` reply, echoed first like every other panel command.
    ///
    /// The tree is formatted on the owner loop because the reference width is the
    /// LIVE terminal width (`Math.max(60, Math.min(this.ui.terminal.columns - 2,
    /// 120))`, :9800) and the dispatch task holds no UI handle.
    ContextTree {
        command: String,
        tree: serde_json::Value,
    },
    /// `showError` (interactive-mode.ts:7672-7676) - `/fullscreen bogus`.
    Error(String),
    /// `setFullscreenMode(enable)` (interactive-mode.ts:5014-5023, :7522-7537).
    /// `Some(enabled)` is an explicit `on`/`off`; `None` is the bare toggle,
    /// which resolves against the LIVE state on the main loop.
    Fullscreen(Option<bool>),
}

impl CommandOutput {
    /// Applies the output to the host, then reports any failure.
    fn report(self, mode: &Rc<RefCell<InteractiveMode>>, send: &mpsc::Sender<HostEvent>) {
        match self {
            CommandOutput::Nothing => {}
            CommandOutput::Status(status) => {
                mode.borrow_mut().show_status(&status, "dim");
            }
            CommandOutput::ClipboardNotice(message) => {
                let _ = send.send(HostEvent::ClipboardNotice(message));
            }
            CommandOutput::Warning(warning) => {
                mode.borrow_mut().show_warning(&warning);
            }
            CommandOutput::EchoLocal(text) => {
                let _ = send.send(HostEvent::EchoLocal(text));
            }
            CommandOutput::EchoPanel { command, panel } => {
                let _ = send.send(HostEvent::EchoLocal(command));
                let _ = send.send(HostEvent::Panel(panel));
            }
            CommandOutput::Panel(panel) => {
                let _ = send.send(HostEvent::Panel(panel));
            }
            CommandOutput::ContextTree { command, tree } => {
                let _ = send.send(HostEvent::EchoLocal(command));
                let _ = send.send(HostEvent::ContextTree(tree));
            }
            CommandOutput::Error(error) => {
                let _ = send.send(HostEvent::Error(error));
            }
            CommandOutput::Fullscreen(requested) => {
                let _ = send.send(HostEvent::Fullscreen(requested));
            }
        }
    }
}

impl CommandOutput {
    /// Turns the local reply into the host events that render it.
    ///
    /// The mode handle is not `Send`, so the spawned dispatch task reports what
    /// to render instead of touching the UI itself.
    fn into_events(self) -> Vec<HostEvent> {
        match self {
            CommandOutput::Nothing => Vec::new(),
            CommandOutput::Status(status) => vec![HostEvent::Status(status)],
            CommandOutput::ClipboardNotice(message) => vec![HostEvent::ClipboardNotice(message)],
            CommandOutput::Warning(warning) => vec![HostEvent::Warning(warning)],
            CommandOutput::EchoLocal(text) => vec![HostEvent::EchoLocal(text)],
            CommandOutput::EchoPanel { command, panel } => {
                vec![HostEvent::EchoLocal(command), HostEvent::Panel(panel)]
            }
            CommandOutput::Panel(panel) => vec![HostEvent::Panel(panel)],
            CommandOutput::ContextTree { command, tree } => {
                vec![HostEvent::EchoLocal(command), HostEvent::ContextTree(tree)]
            }
            CommandOutput::Error(error) => vec![HostEvent::Error(error)],
            CommandOutput::Fullscreen(requested) => vec![HostEvent::Fullscreen(requested)],
        }
    }
}

/// `Math.max(60, Math.min(this.ui.terminal.columns - 2, 120))`
/// (interactive-mode.ts:9800): the live width `/context` formats its tree at.
fn context_tree_width(columns: usize) -> f64 {
    60.0f64.max((columns as f64 - 2.0).min(120.0))
}

/// `handleSessionCommand` (interactive-mode.ts:9481-9502).
async fn session_panel(
    connection: &Arc<dyn wire::AgentConnection>,
    session_name: Option<String>,
    command: &str,
) -> Result<CommandOutput, String> {
    let stats = connection.get_session_stats().await?;
    let text = |key: &str| {
        stats
            .get(key)
            .and_then(serde_json::Value::as_f64)
            .map(|value| (value as i64).to_string())
            .unwrap_or_else(|| "0".to_string())
    };
    let mut info = String::from("Session Info\n\n");
    if let Some(name) = session_name {
        info.push_str(&format!("Name: {name}\n"));
    }
    info.push_str(&format!(
        "File: {}\nID: {}\n\nMessages\n",
        stats
            .get("sessionFile")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("In-memory"),
        stats
            .get("sessionId")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default(),
    ));
    info.push_str(&format!("User: {}\n", text("userMessages")));
    info.push_str(&format!("Assistant: {}\n", text("assistantMessages")));
    info.push_str(&format!("Tool Calls: {}\n", text("toolCalls")));
    info.push_str(&format!("Tool Results: {}\n", text("toolResults")));
    info.push_str(&format!("Total: {}\n\n", text("totalMessages")));
    info.push_str("Use /context for token, cost, and context usage.");
    Ok(CommandOutput::EchoPanel {
        command: command.to_string(),
        panel: info,
    })
}

/// `handleLogsCommand` (interactive-mode.ts:9504-9536).
fn logs_panel(command: &str) -> CommandOutput {
    let logs_dir = crate::config::get_logs_dir();
    let mut info = format!("Logs\n\nDirectory: {logs_dir}\n\n");
    let mut files: Vec<String> = std::fs::read_dir(&logs_dir)
        .map(|entries| {
            entries
                .filter_map(|entry| entry.ok())
                .filter_map(|entry| entry.file_name().into_string().ok())
                .filter(|name| !name.starts_with('.'))
                .collect()
        })
        .unwrap_or_default();
    files.sort();
    if files.is_empty() {
        info.push_str("No logs written yet.\n");
    } else {
        for name in files {
            let size = std::fs::metadata(format!("{logs_dir}/{name}"))
                .map(|metadata| format!(" ({:.1} KB)", metadata.len() as f64 / 1024.0))
                .unwrap_or_default();
            info.push_str(&format!("\u{2022} {name}{size}\n"));
        }
    }
    info.push_str(
        "\nDaemon crashes log to <socket>.log; agent-open failures log to client-errors.log.",
    );
    CommandOutput::EchoPanel {
        command: command.to_string(),
        panel: info,
    }
}

/// `handleChangelogCommand` (interactive-mode.ts:10004-10021).
fn changelog_panel(command: &str) -> CommandOutput {
    let entries = crate::utils::changelog::parse_changelog(&crate::config::get_changelog_path());
    let markdown = if entries.is_empty() {
        "No changelog entries found.".to_string()
    } else {
        entries
            .iter()
            .rev()
            .map(|entry| entry.content.clone())
            .collect::<Vec<String>>()
            .join("\n\n")
    };
    CommandOutput::EchoPanel {
        command: command.to_string(),
        panel: format!("What's New\n\n{markdown}"),
    }
}

/// `handleRlmMaxDepthCommand` (interactive-mode.ts:9430-9479).
async fn rlm_max_depth_command(
    connection: &Arc<dyn wire::AgentConnection>,
    args: &str,
) -> Result<CommandOutput, String> {
    let tokens: Vec<&str> = if args.is_empty() {
        Vec::new()
    } else {
        args.split_whitespace().collect()
    };
    if tokens.is_empty() {
        let status = connection.get_rlm_max_depth_status().await?;
        let max_depth = status
            .get("maxDepth")
            .map(|value| value.to_string())
            .unwrap_or_default();
        let source = status
            .get("source")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("default");
        return Ok(CommandOutput::Panel(format!(
            "RLM max depth: {max_depth} ({source})"
        )));
    }
    let global = tokens.get(1) == Some(&"--global");
    let valid_depth = tokens
        .first()
        .map(|token| !token.is_empty() && token.chars().all(|ch| ch.is_ascii_digit()))
        .unwrap_or(false);
    if tokens.len() > if global { 2 } else { 1 } || !valid_depth {
        return Ok(CommandOutput::Warning(
            "Usage: /rlm-max-depth [<non-negative integer> [--global]]".to_string(),
        ));
    }
    let max_depth: f64 = tokens[0]
        .parse()
        .map_err(|_| "RLM max depth must be a non-negative integer.".to_string())?;
    let result = connection
        .set_rlm_max_depth(max_depth, Some(serde_json::json!({ "global": global })))
        .await?;
    let depth = result
        .get("maxDepth")
        .map(|value| value.to_string())
        .unwrap_or_else(|| max_depth.to_string());
    let saved = result
        .get("globalSaved")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    Ok(CommandOutput::Panel(format!(
        "RLM max depth set: {depth}{}",
        if saved {
            " and saved as global default"
        } else {
            ""
        }
    )))
}

/// `handleHeartbeatCommand` (interactive-mode.ts:9812-9873).
async fn heartbeat_command(
    connection: &Arc<dyn wire::AgentConnection>,
    command_text: &str,
) -> Result<CommandOutput, String> {
    match crate::core::cron_jobs::parse_heartbeat_command(command_text)? {
        crate::core::cron_jobs::ParsedHeartbeatCommand::Status => {
            Ok(match connection.get_heartbeat().await? {
                Some(job) => CommandOutput::Panel(format!("Heartbeat\n\n{job}")),
                None => CommandOutput::Panel("No active heartbeat".to_string()),
            })
        }
        crate::core::cron_jobs::ParsedHeartbeatCommand::Set {
            schedule,
            instruction,
            delivery_mode,
        } => {
            let job = connection
                .set_heartbeat(&schedule, &instruction, delivery_mode.as_deref())
                .await?;
            let delivery = job
                .get("deliveryMode")
                .and_then(serde_json::Value::as_str)
                .unwrap_or(crate::core::cron_jobs::DEFAULT_HEARTBEAT_DELIVERY_MODE);
            let next_run = job
                .get("nextRunAt")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("-");
            Ok(CommandOutput::Status(format!(
                "Heartbeat set\nDelivery: {delivery}\nNext run: {next_run}"
            )))
        }
        command => {
            let action = match command {
                crate::core::cron_jobs::ParsedHeartbeatCommand::Pause => "pause",
                crate::core::cron_jobs::ParsedHeartbeatCommand::Resume => "resume",
                _ => "clear",
            };
            let Some(job) = connection
                .update_heartbeat(serde_json::Value::String(action.to_string()))
                .await?
            else {
                return Ok(CommandOutput::Status("No active heartbeat".to_string()));
            };
            if action == "resume" {
                let next_run = job
                    .get("nextRunAt")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("-");
                return Ok(CommandOutput::Status(format!(
                    "Heartbeat resumed\nNext run: {next_run}"
                )));
            }
            Ok(CommandOutput::Status(
                if action == "clear" {
                    "Heartbeat cleared"
                } else {
                    "Heartbeat paused"
                }
                .to_string(),
            ))
        }
    }
}

/// Runs one built-in slash command.
///
/// Rust counterpart of the `if (commandName === "<name>")` chain at
/// interactive-mode.ts:4821-5030. `name` is always canonical because the
/// registry already resolved any alias.
async fn run_builtin_command(
    connection: &Arc<dyn wire::AgentConnection>,
    send: &mpsc::Sender<HostEvent>,
    text: &str,
    name: &str,
    args: &str,
) -> Result<CommandOutput, String> {
    match name {
        "mcp" if args.trim().is_empty() => {
            let started = Instant::now();
            let _ = send.send(HostEvent::Configuration(
                connection.get_model_catalog().await?,
                "mcp-connections",
                None,
            ));
            let _ = send.send(HostEvent::MenuTiming(started));
            Ok(CommandOutput::Nothing)
        }
        "btw" | "side" | "fork" | "logout" | "scoped-models" | "share" | "traces" | "monitor" | "tree" | "update" | "debug" | "mcp" | "jev" | "stats" => native_commands::run(connection, send, if name == "side" { "btw" } else { name }, args).await,
        "settings" => {
            let started = Instant::now();
            let _ = send.send(HostEvent::Settings(connection.get_state().await?));
            let _ = send.send(HostEvent::MenuTiming(started));
            Ok(CommandOutput::Nothing)
        }
        // `commandName === "fullscreen"` (interactive-mode.ts:5014-5023).
        //
        // The TypeScript parses `on`/`off` and resolves the bare toggle against
        // the LIVE `this.fullscreenEnabled` (:5021), then `setFullscreenMode`
        // persists and applies it (:7522-7537). The previous port built a fresh
        // `SettingsManager` and negated the SAVED value, so it ignored its
        // arguments and diverged from the live terminal state.
        "fullscreen" => match parse_fullscreen_argument(args) {
            Ok(requested) => Ok(CommandOutput::Fullscreen(requested)),
            Err(()) => Ok(CommandOutput::Error(
                "Usage: /fullscreen [on|off]".to_string(),
            )),
        },
        // `/model` keeps its existing search behaviour exactly.
        "model" => {
            let started = Instant::now();
            let search = (!args.is_empty()).then(|| args.to_string());
            connection.get_model_catalog().await.map(|catalog| {
                let _ = send.send(HostEvent::Models(catalog, search, started));
            })?;
            Ok(CommandOutput::Nothing)
        }
        // `commandName === "effort"` (interactive-mode.ts:4843-4847): with no
        // argument the selector opens, with one the level is set directly.
        "effort" => {
            let state = connection.get_state().await?;
            let levels = available_thinking_levels(&state);
            if levels.is_empty() {
                return Ok(CommandOutput::Status(
                    "Current model does not support thinking".to_string(),
                ));
            }
            let requested = args.trim().to_lowercase();
            if requested.is_empty() {
                let _ = send.send(HostEvent::ThinkingLevels {
                    current: state.thinking_level,
                    levels,
                });
                return Ok(CommandOutput::Nothing);
            }
            let Some(level) = levels
                .iter()
                .find(|level| level.as_str() == requested.as_str())
                .copied()
            else {
                let available: Vec<&str> = levels.iter().map(|level| level.as_str()).collect();
                // TypeScript uses showError for an unknown level
                // (interactive-mode.ts:8279), so the port reports it as an error.
                return Ok(CommandOutput::Error(format!(
                    "Unknown thinking level '{requested}'. Available: {}",
                    available.join(", ")
                )));
            };
            connection.set_thinking_level(level).await?;
            Ok(CommandOutput::Status(format!(
                "Thinking level: {}",
                level.as_str()
            )))
        }
        // `commandName === "fast"` (interactive-mode.ts:4848-4853).
        "fast" => {
            if !args.is_empty() {
                return Ok(CommandOutput::Error("Usage: /fast".to_string()));
            }
            let unavailable = "Fast mode requires GPT-5.4, GPT-5.5, or GPT-5.6 with ChatGPT or OpenAI API key authentication";
            let state = connection.get_state().await?;
            let supports = state
                .model
                .as_ref()
                .map(pi_ai::models::supports_fast_mode)
                .unwrap_or(false);
            if !supports {
                return Ok(CommandOutput::Status(unavailable.to_string()));
            }
            let enabled =
                state.service_tier.as_ref().and_then(|tier| tier.as_deref()) == Some("priority");
            connection
                .set_service_tier(Some(Some(
                    if enabled { "default" } else { "priority" }.to_string(),
                )))
                .await?;
            let next = connection.get_state().await?;
            let on =
                next.service_tier.as_ref().and_then(|tier| tier.as_deref()) == Some("priority");
            Ok(CommandOutput::Status(format!(
                "Fast mode: {}",
                if on { "on" } else { "off" }
            )))
        }
        // `commandName === "name"` (interactive-mode.ts:9410-9428).
        "name" => {
            let requested = text
                .trim()
                .strip_prefix("/name")
                .or_else(|| text.trim().strip_prefix("/rename"))
                .unwrap_or("")
                .trim()
                .to_string();
            if requested.is_empty() {
                let state = connection.get_state().await?;
                return Ok(match state.session_name {
                    Some(current) => CommandOutput::Panel(format!("Session name: {current}")),
                    None => CommandOutput::Warning("Usage: /name <name>".to_string()),
                });
            }
            connection.set_session_name(&requested).await?;
            Ok(CommandOutput::Panel(format!(
                "Session name set: {requested}"
            )))
        }
        // `commandName === "session"` (interactive-mode.ts:4885-4889).
        "session" => {
            let state = connection.get_state().await?;
            session_panel(connection, state.session_name, text.trim()).await
        }
        // `commandName === "system-prompt"` (interactive-mode.ts:4891-4895,
        // panel at :9541-9544).
        "system-prompt" => {
            let prompt = connection.get_system_prompt().await?;
            Ok(CommandOutput::EchoPanel {
                command: text.trim().to_string(),
                panel: format!(
                    "System Prompt ({} chars)\n\n{prompt}",
                    prompt.chars().count()
                ),
            })
        }
        // `commandName === "context"` (interactive-mode.ts:4903-4907, handler at
        // :9796-9809). The tree is returned raw; the owner loop formats it at the
        // live terminal width.
        "context" => connection
            .get_context_tree()
            .await
            .map(|tree| CommandOutput::ContextTree {
                command: text.trim().to_string(),
                tree,
            }),
        // `commandName === "logs"` (interactive-mode.ts:4909-4913).
        "logs" => Ok(logs_panel(text.trim())),
        // `commandName === "changelog"` (interactive-mode.ts:4925-4929).
        "changelog" => Ok(changelog_panel(text.trim())),
        // `commandName === "rlm-max-depth"` (interactive-mode.ts:4881-4884).
        "rlm-max-depth" => rlm_max_depth_command(connection, args).await,
        // `commandName === "heartbeat"` (interactive-mode.ts:4914-4918).
        "heartbeat" => heartbeat_command(connection, text).await,
        // `commandName === "heartbeats"` (interactive-mode.ts:4919-4923).
        "heartbeats" => {
            let _ = send.send(HostEvent::Heartbeats(
                connection.list_heartbeats().await?,
                true,
            ));
            Ok(CommandOutput::Nothing)
        }
        // `commandName === "export"` (interactive-mode.ts:4854-4858, 9210-9225).
        "export" => {
            let output_path = path_command_argument(text.trim(), "/export");
            let as_jsonl = output_path
                .as_deref()
                .map(|path| path.ends_with(".jsonl"))
                .unwrap_or(false);
            let exported = if as_jsonl {
                connection.export_to_jsonl(output_path.as_deref()).await
            } else {
                connection.export_to_html(output_path.as_deref()).await
            };
            exported
                .map(|file_path| CommandOutput::Status(format!("Session exported to: {file_path}")))
        }
        // `commandName === "import"` (interactive-mode.ts:4859-4863, 9255-9270).
        "import" => {
            let Some(input_path) = path_command_argument(text.trim(), "/import") else {
                return Ok(CommandOutput::Error(
                    "Usage: /import <path.jsonl>".to_string(),
                ));
            };
            let cancelled = connection.import_from_jsonl(&input_path, None).await?;
            if cancelled {
                return Ok(CommandOutput::Status("Import cancelled".to_string()));
            }
            Ok(CommandOutput::Status(format!(
                "Session imported from: {input_path}"
            )))
        }
        // `commandName === "copy"` (interactive-mode.ts:4868-4872, 9395-9408).
        "copy" => {
            let Some(last) = connection.get_last_assistant_text().await? else {
                return Ok(CommandOutput::ClipboardNotice(
                    "No agent messages to copy yet.".to_string(),
                ));
            };
            let outcome = crate::utils::clipboard::copy_to_clipboard(&last)
                .await
                .map_err(|error| error.to_string())?;
            Ok(CommandOutput::ClipboardNotice(outcome.status().to_string()))
        }
        // `commandName === "clone"` (interactive-mode.ts:4942-4945, 8582-8602).
        "clone" => {
            let tree = connection.get_session_tree().await?;
            let Some(leaf_id) = tree.leaf_id.clone() else {
                return Ok(CommandOutput::Status("Nothing to clone yet".to_string()));
            };
            let result = connection
                .fork(
                    &leaf_id,
                    Some(wire::AgentConnectionForkOptions {
                        position: Some("at".into()),
                        ..Default::default()
                    }),
                )
                .await?;
            if result
                .get("cancelled")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false)
            {
                return Ok(CommandOutput::Nothing);
            }
            Ok(CommandOutput::Status("Cloned to new session".to_string()))
        }
        // `commandName === "resume"` (interactive-mode.ts:4991-4994, 8734-8750).
        "resume" => {
            if args.trim().is_empty() {
                let _ = send.send(HostEvent::AgentsView);
                return Ok(CommandOutput::Nothing);
            }
            let state = connection.get_state().await?;
            let resolved = crate::core::session_resolver::resolve_session_path(
                args.trim(),
                &state.cwd,
                state.session_dir.as_deref(),
            )
            .await
            .map_err(|error| error.to_string())?;
            let session_path = match resolved {
                crate::core::session_resolver::ResolvedSession::Path { path }
                | crate::core::session_resolver::ResolvedSession::Local { path }
                | crate::core::session_resolver::ResolvedSession::Global { path, .. } => path,
            };
            if connection.switch_session(&session_path, None).await? {
                return Ok(CommandOutput::Status("Resume cancelled".to_string()));
            }
            // Runtime rebind handoff (same as fork): the rebind's blanket reset
            // lands first, then this snapshot republishes the resumed session's
            // authoritative state, including the Jev footer.
            let _ = send.send(HostEvent::RefreshSnapshot(
                connection.get_initial_snapshot().await?,
            ));
            Ok(CommandOutput::Status("Resumed session".to_string()))
        }
        // `commandName === "reload"` (interactive-mode.ts:4996-4999, 9124-9133).
        "reload" => {
            let state = connection.get_state().await?;
            if state.is_streaming {
                return Ok(CommandOutput::Warning(
                    "Wait for the current response to finish before reloading.".to_string(),
                ));
            }
            if state.is_compacting {
                return Ok(CommandOutput::Warning(
                    "Wait for compaction to finish before reloading.".to_string(),
                ));
            }
            let _ = send.send(HostEvent::Extension(native_extension_bridge::Event::Reset));
            connection.reload().await?;
            // TS handleReloadCommand: refreshConnectionCatalog +
            // setupAutocompleteProvider (interactive-mode.ts:9185-9186). The
            // owner loop owns the editor, so it reconfigures the provider and
            // refetches the command catalogue on this event.
            let _ = send.send(HostEvent::ReconfigureAutocomplete);
            Ok(CommandOutput::Status(
                "Reloaded keybindings, extensions, skills, prompts, themes".to_string(),
            ))
        }
        // `if (slashCommand?.name === "clear")` / `"new"` (interactive-mode.ts:4968-4989).
        // `/clear` is the alias, so the registry reports the canonical `new` with
        // raw `/clear`; both TypeScript arms therefore key on the name AS TYPED,
        // not on the canonical one.
        "new" => {
            // `clear` is the no-argument compatibility alias
            // (`builtin_slash_command_takes_argument` is false for it,
            // core/slash_commands.rs:530-533), and TypeScript answers any argument
            // with `showError("Usage: /clear")` (:4969-4971) INSTEAD of starting a
            // session. Reading the contract from the registry keeps the alias
            // resolution in one place.
            let typed = crate::core::slash_commands::parse_slash_command(text)
                .map(|command| command.name)
                .filter(|name| crate::core::slash_commands::is_builtin_slash_command_name(name));
            let alias_without_arguments = typed.as_deref().is_some_and(|name| {
                name != "new"
                    && !crate::core::slash_commands::builtin_slash_command_takes_argument(name)
            });
            if alias_without_arguments && !args.trim().is_empty() {
                return Ok(CommandOutput::Error("Usage: /clear".to_string()));
            }
            let parsed = crate::core::new_session_command::parse_new_session_command(
                text.trim().strip_prefix("/new").unwrap_or(""),
            );
            let parsed = match parsed {
                Ok(parsed) => parsed,
                Err(error) => return Ok(CommandOutput::Error(error)),
            };
            if connection.new_session(None).await? {
                return Ok(CommandOutput::Nothing);
            }
            // Runtime rebind handoff (same as fork): the rebind's blanket reset
            // lands first, then this snapshot resets and republishes the NEW
            // session's authoritative state, including the Jev footer. Without
            // it the new session would start with no footer until a /jev write.
            let _ = send.send(HostEvent::RefreshSnapshot(
                connection.get_initial_snapshot().await?,
            ));
            if let Some(name) = parsed.name {
                connection.set_session_name(&name).await?;
            }
            if let Some(prompt) = parsed.prompt {
                // interactive-mode.ts:10243-10248: the prompt reaches the model
                // verbatim; the owner loop collects images, records history and
                // prompts. Re-classifying it would turn a leading slash into a
                // local command.
                let _ = send.send(HostEvent::PromptSession { text: prompt });
            }
            Ok(CommandOutput::Status("New session started".to_string()))
        }
        // `commandName === "quit"` (interactive-mode.ts:5035-5039) is handled in
        // the input loop, which owns `shutdown_requested`.
        "quit" => Ok(CommandOutput::Nothing),
        other => Ok(CommandOutput::Status(format!(
            "/{other} is recognised but the native host has no handler for it yet."
        ))),
    }
}

/// `commandArgs?.trim().toLowerCase()` for `/fullscreen`
/// (interactive-mode.ts:5016-5020).
///
/// `Ok(Some(on))` is an explicit `on`/`off`; `Ok(None)` is the bare toggle;
/// `Err(())` is an unrecognised argument, which the TypeScript answers with
/// `showError("Usage: /fullscreen [on|off]")`.
fn parse_fullscreen_argument(args: &str) -> Result<Option<bool>, ()> {
    let arg = args.trim().to_lowercase();
    if arg.is_empty() {
        return Ok(None);
    }
    match arg.as_str() {
        "on" => Ok(Some(true)),
        "off" => Ok(Some(false)),
        _ => Err(()),
    }
}

/// `/fullscreen`'s argument resolution (interactive-mode.ts:5016-5022).
///
/// `arg === "on" ? true : arg === "off" ? false : !this.fullscreenEnabled` - the
/// bare toggle reads the LIVE field, never the persisted setting.
fn apply_fullscreen_request(
    requested: Option<bool>,
    mode: &Rc<RefCell<InteractiveMode>>,
    editor: &Rc<RefCell<CustomEditor>>,
    ui: &Rc<RefCell<TUI>>,
    transcript: &Rc<RefCell<Transcript>>,
) {
    let enable = requested.unwrap_or_else(|| !mode.borrow().fullscreen_enabled);
    set_fullscreen_mode(enable, mode, editor, ui, transcript);
}

/// `setFullscreenMode(enabled)` (interactive-mode.ts:7522-7537).
///
/// Persists the choice, refuses fullscreen on a non-interactive stdout with the
/// exact status line, then applies the live state and reports it.
fn set_fullscreen_mode(
    enable: bool,
    mode: &Rc<RefCell<InteractiveMode>>,
    editor: &Rc<RefCell<CustomEditor>>,
    ui: &Rc<RefCell<TUI>>,
    transcript: &Rc<RefCell<Transcript>>,
) {
    if let Ok(mut settings) = mode.borrow().settings_manager().lock() {
        settings.set_fullscreen(enable);
    }
    // `enabled && !process.stdout.isTTY` (interactive-mode.ts:7524-7528).
    if enable && !std::io::stdout().is_terminal() {
        mode.borrow_mut().fullscreen_enabled = false;
        mode.borrow_mut().show_status(
            "Fullscreen rendering requires an interactive terminal",
            "dim",
        );
        return;
    }
    mode.borrow_mut().fullscreen_enabled = enable;
    native_settings::fullscreen(enable, mode, editor, ui, transcript);
    let follow_key = mode.borrow().get_editor_key_display("tui.viewport.follow");
    mode.borrow_mut().show_status(
        &if enable {
            format!("Fullscreen rendering on — wheel/pageUp scroll, {follow_key} follows output")
        } else {
            "Fullscreen rendering off".to_string()
        },
        "dim",
    );
}

fn model_command_search(text: &str) -> Option<Option<String>> {
    let rest = text.trim().strip_prefix("/model")?;
    if !rest.is_empty() && !rest.starts_with(char::is_whitespace) {
        return None;
    }
    let search = rest.trim();
    Some((!search.is_empty()).then(|| search.to_string()))
}

fn model_item(model: &wire::AgentConnectionModel) -> ModelItemModel {
    ModelItemModel {
        provider: model.provider.clone(),
        id: model.id.clone(),
        name: model.name.clone(),
        featured: model.featured.unwrap_or(false),
        raw: serde_json::to_value(model).unwrap_or_default(),
    }
}

fn make_model_selector(
    current: Option<&wire::AgentConnectionModel>,
    scoped: Vec<ScopedModelItem>,
    models: &[wire::AgentConnectionModel],
    configured: Vec<String>,
    recent: Vec<String>,
    search: Option<String>,
    rows: Rc<Cell<f64>>,
) -> ModelSelectorComponent {
    ModelSelectorComponent::new(
        current.map(model_item),
        scoped,
        ModelSelectorOptions {
            available_models: Some(models.iter().map(model_item).collect()),
            configured_providers: Some(configured),
            recent_models: Some(recent),
            initial_search_input: search,
            get_rows: Some(Rc::new(move || rows.get())),
            ..Default::default()
        },
    )
}

fn string(value: &serde_json::Value, key: &str) -> String {
    optional_string(value, key).unwrap_or_default()
}
fn optional_string(value: &serde_json::Value, key: &str) -> Option<String> {
    value
        .get(key)
        .and_then(|value| value.as_str())
        .map(str::to_string)
}
fn number(value: &serde_json::Value, key: &str) -> Option<f64> {
    value.get(key).and_then(|value| value.as_f64())
}
fn project_heartbeat(value: serde_json::Value) -> Option<local::AgentCronJob> {
    let job: crate::core::cron_jobs::AgentCronJob = serde_json::from_value(value).ok()?;
    Some(local::AgentCronJob {
        id: job.id,
        session_id: job.session_id,
        active_session_id: job.active_session_id,
        prompt: job.prompt,
        status: job.status,
        delivery_mode: job.delivery_mode,
        last_run_at: job.last_run_at,
        next_run_at: job.next_run_at,
        run_count: job.run_count,
        last_error: job.last_error,
        source: job.source,
        schedule_expression: job.schedule.expression,
        schedule_interval_ms: job.schedule.interval_ms,
    })
}

fn project_state(state: wire::AgentConnectionState) -> local::AgentConnectionState {
    let actions = &state.session_actions;
    let texts = |key| {
        actions
            .get(key)
            .and_then(|v| v.as_array())
            .into_iter()
            .flatten()
            .filter_map(|v| {
                v.as_str()
                    .map(str::to_string)
                    .or_else(|| optional_string(v, "text"))
            })
            .collect::<Vec<_>>()
    };
    local::AgentConnectionState {
        active_session_id: state.active_session_id,
        cwd: state.cwd,
        model: state.model,
        thinking_level: state.thinking_level,
        service_tier: state.service_tier,
        available_thinking_levels: state.available_thinking_levels,
        is_streaming: state.is_streaming,
        is_compacting: state.is_compacting,
        is_bash_running: state.is_bash_running,
        retry_attempt: state.retry_attempt,
        steering_mode: state.steering_mode,
        follow_up_mode: state.follow_up_mode,
        session_file: state.session_file,
        session_id: state.session_id,
        session_name: state.session_name,
        session_dir: state.session_dir,
        leaf_id: state.leaf_id,
        auto_compaction_enabled: state.auto_compaction_enabled,
        message_count: state.message_count,
        session_actions: local::SessionActionSnapshot {
            steering: texts("steering"),
            follow_ups: texts("followUps"),
            queued_count: number(actions, "queuedCount").unwrap_or(0.0) as usize,
            active: optional_string(actions, "active"),
        },
        compaction_count: state.compaction_count,
        goal: serde_json::from_value(state.goal).unwrap_or_else(|_| empty_goal_state()),
        heartbeat: state.heartbeat.flatten().and_then(project_heartbeat),
        scoped_models: state
            .scoped_models
            .into_iter()
            .map(|model| local::AgentConnectionScopedModel { model: model.model })
            .collect(),
        execution_mode: crate::core::execution_mode::ExecutionMode::from_tools(&state.active_tool_names),
        active_tool_names: state.active_tool_names,
        context_usage: local::ContextUsage {
            tokens: number(&state.context_usage, "tokens"),
            context_window: number(&state.context_usage, "contextWindow").unwrap_or(0.0),
            percent: number(&state.context_usage, "percent"),
        },
        recap: state.recap,
    }
}

/// Re-apply the branded terminal title after session state changes, then sync
/// the recorded title to the real terminal like the extension Reset path does.
fn refresh_terminal_title(mode: &Rc<RefCell<InteractiveMode>>, ui: &Rc<RefCell<TUI>>) {
    mode.borrow_mut().update_terminal_title();
    let title = mode.borrow().ui.terminal.title.clone();
    ui.borrow_mut().terminal.set_title(&title);
}

fn apply_event(
    mode: &Rc<RefCell<InteractiveMode>>,
    transcript: &Rc<RefCell<Transcript>>,
    event: wire::AgentConnectionSessionEvent,
) {
    native_state::track_activity(&mut mode.borrow_mut(), &event);
    let value = match &event {
        wire::AgentConnectionSessionEvent::Agent(event) => serde_json::to_value(event),
        event => serde_json::to_value(event),
    }
    .unwrap_or_default();
    match event.type_name() {
        "refinement_update" => {
            if let wire::AgentConnectionSessionEvent::RefinementUpdate { active, reason } = &event {
                transcript.borrow_mut().refinement_progress = active.then(|| {
                    reason.as_deref().filter(|text| !text.trim().is_empty())
                        .unwrap_or("Refining saved knowledge")
                        .chars().map(|ch| if ch.is_control() { ' ' } else { ch }).collect()
                });
            }
        }
        "refine_complete" | "refine_failed" => {
            transcript.borrow_mut().refinement_progress = None;
            if let wire::AgentConnectionSessionEvent::RefineFailed { error } = &event {
                mode.borrow_mut().show_error(error);
            }
        }
        "rlm_child_update" => {
            if let wire::AgentConnectionSessionEvent::RlmChildUpdate { child } = &event {
                mode.borrow_mut().update_subagent_summary(native_subagents::project_child(child));
            }
        }
        "ipython_sent_agent_message" => {
            if let wire::AgentConnectionSessionEvent::IpythonSentAgentMessage { tool_call_id, message } = &event {
                transcript.borrow_mut().sent_agent_message(tool_call_id, message.clone());
            }
        }
        "session_action_update" => {
            if let Some(actions) = value.get("actions") {
                let strings = |key| {
                    actions
                        .get(key)
                        .and_then(serde_json::Value::as_array)
                        .into_iter()
                        .flatten()
                        .filter_map(|value| {
                            value
                                .as_str()
                                .map(str::to_string)
                                .or_else(|| optional_string(value, "text"))
                        })
                        .collect::<Vec<_>>()
                };
                mode.borrow_mut().patch_connection_queue(|state| {
                    state.session_actions.steering = strings("steering");
                    state.session_actions.follow_ups = strings("followUps");
                    state.session_actions.queued_count =
                        number(actions, "queuedCount").unwrap_or(0.0) as usize;
                    state.session_actions.active = optional_string(actions, "active");
                });
            }
        }
        "agent_start" => {
            let mut mode = mode.borrow_mut();
            mode.patch_connection_state(|s| s.is_streaming = true);
            mode.working_started_at = Some(now_ms());
        }
        "agent_end" => {
            let mut mode = mode.borrow_mut();
            mode.patch_connection_state(|s| {
                s.is_streaming = false;
                s.active_tool_names.clear();
            });
            mode.stop_working_loader();
        }
        "message_start" | "message_update" | "message_end" => {
            if let Some(message) = value
                .get("message")
                .and_then(|value| serde_json::from_value::<AgentMessage>(value.clone()).ok())
            {
                let kind = event.type_name();
                // Durable refinement outcomes emit MessageStart only; the old
                // end-only custom path hid successful automatic learning live.
                let refinement_outcome = matches!(&message, AgentMessage::Custom(pi_agent_core::types::CustomAgentMessage::Custom { custom_type, .. }) if custom_type == crate::core::messages::REFINEMENT_OUTCOME_CUSTOM_TYPE);
                if message.role() == "assistant"
                    || (kind == "message_start" && refinement_outcome)
                    || (kind == "message_end" && !refinement_outcome && message.role() != "assistant")
                {
                    transcript
                        .borrow_mut()
                        .message(message, kind != "message_end");
                }
                if (kind == "message_end" && !refinement_outcome) || (kind == "message_start" && refinement_outcome) {
                    mode.borrow_mut()
                        .patch_connection_state(|s| s.message_count += 1.0);
                }
            }
        }
        "tool_execution_start" => {
            let id = string(&value, "toolCallId");
            let mut transcript = transcript.borrow_mut();
            transcript.tool_start(&id, &string(&value, "toolName"), value.get("args").cloned().unwrap_or_default());
            if let Some(meta) = transcript.row_metadata.get_mut(format!("tool:{id}").as_str()) {
                meta.started.get_or_insert_with(Instant::now);
                meta.timestamp = Some(now_ms() as i64);
            }
        },
        "tool_execution_end" | "tool_execution_update" => transcript.borrow_mut().tool_result(
            &string(&value, "toolCallId"),
            value
                .get("result")
                .or_else(|| value.get("partialResult"))
                .unwrap_or(&serde_json::Value::Null),
            value
                .get("isError")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            event.type_name() == "tool_execution_update",
        ),
        "bash_start" => {
            mode.borrow_mut()
                .patch_connection_state(|s| s.is_bash_running = true);
            mode.borrow_mut()
                .show_status(&format!("$ {}", string(&value, "command")), "dim");
        }
        "bash_output" => mode
            .borrow_mut()
            .show_status(&string(&value, "chunk"), "text"),
        "bash_end" => {
            mode.borrow_mut()
                .patch_connection_state(|s| s.is_bash_running = false);
            if let Some(error) = optional_string(&value, "errorMessage") {
                mode.borrow_mut().show_error(&error);
            }
        }
        "goal_update" => {
            if let wire::AgentConnectionSessionEvent::GoalUpdate { goal } = event {
                mode.borrow_mut().handle_goal_update(&goal, 120.0);
                mode.borrow_mut().patch_connection_state(|s| s.goal = goal);
            }
        }
        "compaction_start" => {
            if let wire::AgentConnectionSessionEvent::CompactionStart {
                reason,
                custom_instructions,
            } = event
            {
                let mut mode = mode.borrow_mut();
                mode.patch_connection_state(|s| s.is_compacting = true);
                mode.start_compaction_loader(&reason, custom_instructions.as_deref());
            }
        }
        "compaction_end" => {
            let mut mode = mode.borrow_mut();
            mode.patch_connection_state(|s| s.is_compacting = false);
            mode.stop_compaction_loader();
            if let wire::AgentConnectionSessionEvent::CompactionEnd { result, aborted, error_message, .. } = event {
                if let Some(error) = error_message {
                    mode.show_compaction_error(&error);
                } else if !aborted && result.is_some() {
                    // An idle snapshot or cancelled/skipped attempt is not recovery.
                    mode.clear_compaction_notices();
                }
            }
        }
        "recap_update" => {
            mode.borrow_mut().session_recap = optional_string(&value, "recap");
            mode.borrow_mut().render_recap();
        }
        "auto_retry_start" => {
            let mut mode = mode.borrow_mut();
            mode.patch_connection_state(|s| s.retry_attempt = number(&value, "attempt").unwrap_or(0.0));
            mode.show_warning(&string(&value, "errorMessage"));
        }
        "auto_retry_end" => mode.borrow_mut().patch_connection_state(|s| s.retry_attempt = 0.0),
        "session_info_changed" => {
            if let wire::AgentConnectionSessionEvent::SessionInfoChanged { name } = event {
                mode.borrow_mut().patch_connection_state(|s| s.session_name = name.clone());
            }
        }
        _ => {}
    }
}

fn make_selector(
    items: Vec<pi_tui::components::select_list::SelectItem>,
    send: mpsc::Sender<Option<String>>,
) -> Rc<RefCell<pi_tui::components::select_list::SelectList>> {
    let mut list = pi_tui::components::select_list::SelectList::new(
        items,
        12,
        select_theme(),
        Default::default(),
    );
    let selected = send.clone();
    list.on_select = Some(Box::new(move |item| {
        let _ = selected.send(Some(item.value.clone()));
    }));
    list.on_cancel = Some(Box::new(move || {
        let _ = send.send(None);
    }));
    Rc::new(RefCell::new(list))
}

fn start_login(
    generation: LoginGeneration,
    provider: String,
    oauth: bool,
    send: mpsc::Sender<HostEvent>,
    cancel: tokio_util::sync::CancellationToken,
) {
    let runtime = tokio::runtime::Handle::current();
    tokio::task::spawn_blocking(move || {
        runtime.block_on(async move {
        let mut auth = crate::core::auth_storage::AuthStorage::create(None, None);
        let result = if oauth {
            use pi_ai::utils::oauth::types::OAuthLoginCallbacks;
            let auth_send = send.clone();
            let prompt_send = send.clone();
            let progress_send = send.clone();
            let manual_send = send.clone();
            let callbacks = OAuthLoginCallbacks {
                on_auth: Some(Arc::new(move |info| { let _ = auth_send.send(HostEvent::LoginAuth(info.url, info.instructions)); })),
                on_progress: Some(Arc::new(move |message| { let _ = progress_send.send(HostEvent::LoginProgress(message)); })),
                on_prompt: Some(Arc::new(move |prompt| {
                    let (sender, receiver) = tokio::sync::oneshot::channel();
                    let _ = prompt_send.send(HostEvent::LoginPrompt(prompt.message, prompt.placeholder, sender));
                    Box::pin(async move { receiver.await.unwrap_or_default() })
                })),
                on_manual_code_input: Some(Arc::new(move || {
                    let (sender, receiver) = tokio::sync::oneshot::channel();
                    let _ = manual_send.send(HostEvent::LoginPrompt("Paste the authorization code or redirect URL:".into(), None, sender));
                    Box::pin(async move { receiver.await.map_err(|_| "Login cancelled".into()) })
                })),
                signal: Some(cancel.clone()),
                ..Default::default()
            };
            tokio::select! {
                _ = cancel.cancelled() => Err("Login cancelled".into()),
                result = auth.login(&provider, callbacks) => result,
            }
        } else {
            let (sender, receiver) = tokio::sync::oneshot::channel();
            let _ = send.send(HostEvent::LoginPrompt("Enter API key:".into(), None, sender));
            let key = tokio::select! { _ = cancel.cancelled() => None, value = receiver => value.ok() };
            match key {
                Some(key) if !key.trim().is_empty() => {
                    auth.set(&provider, crate::core::auth_storage::AuthCredential::ApiKey { key: key.trim().into(), prime_team: None });
                    Ok(())
                }
                Some(_) => Err("API key cannot be empty.".into()),
                None => Err("Login cancelled".into()),
            }
        };
        let errors = auth.drain_errors();
        let result = if result.is_ok() && !errors.is_empty() { Err(errors.join("\n")) } else { result };
        let _ = send.send(HostEvent::LoginFinished(generation, result));
    })
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `AgentConnectionInputPause` stand-in for the recording connection.
    struct NoopInputPause;

    impl wire::AgentConnectionInputPause for NoopInputPause {
        fn release(&self) -> pi_ai::types::BoxFuture<Result<(), String>> {
            Box::pin(async { Ok(()) })
        }
    }

    /// Recording `AgentConnection`.
    ///
    /// Every method records its own name; the methods the local command handlers
    /// call also record their arguments, so a behaviour test asserts the ACTUAL
    /// connection call and its arguments rather than only the command name.
    #[derive(Default)]
    pub(super) struct RecordingConnection {
        pub(super) heartbeat_catalog_support: std::sync::Mutex<Option<bool>>,
        execution_mode_support: bool,
        calls: std::sync::Mutex<Vec<(String, Vec<String>)>>,
        state: std::sync::Mutex<wire::AgentConnectionState>,
        user_messages: std::sync::Mutex<Vec<wire::AgentConnectionUserMessage>>,
        last_assistant_text: std::sync::Mutex<Option<String>>,
        session_tree: std::sync::Mutex<wire::AgentConnectionWatchSessionTree>,
        export_path: std::sync::Mutex<Option<String>>,
        fork_result: std::sync::Mutex<serde_json::Value>,
        /// The `get_context_tree` reply, so a `/context` test drives a REAL tree
        /// through the real `format_context_tree` instead of an empty object.
        context_tree: std::sync::Mutex<serde_json::Value>,
    }

    impl RecordingConnection {
        fn new() -> Self {
            Self::default()
        }

        fn record(&self, name: &str) {
            self.record_with(name, &[]);
        }

        fn record_with(&self, name: &str, args: &[String]) {
            self.calls
                .lock()
                .unwrap()
                .push((name.to_string(), args.to_vec()));
        }

        /// `(method, args)` pairs in call order.
        pub(super) fn calls(&self) -> Vec<(String, Vec<String>)> {
            self.calls.lock().unwrap().clone()
        }

        /// The argument list of the single `name` call. Panics when the call is
        /// absent or repeated, so a test can never pass on the wrong call.
        fn only_call(&self, name: &str) -> Vec<String> {
            let matches: Vec<Vec<String>> = self
                .calls()
                .into_iter()
                .filter(|(method, _)| method == name)
                .map(|(_, args)| args)
                .collect();
            assert_eq!(
                matches.len(),
                1,
                "expected exactly one {name} call, got {matches:?}"
            );
            matches.into_iter().next().unwrap()
        }
    }

    impl wire::AgentConnection for RecordingConnection {
        fn supports_execution_mode(&self) -> bool { self.execution_mode_support }
        fn supports_node_execution_mode(&self) -> bool { self.execution_mode_support }
        fn get_state(&self) -> pi_ai::types::BoxFuture<Result<wire::AgentConnectionState, String>> {
            self.record("get_state");
            let state = self.state.lock().unwrap().clone();
            Box::pin(async move { Ok(state) })
        }
        fn get_initial_snapshot(
            &self,
        ) -> pi_ai::types::BoxFuture<Result<wire::AgentConnectionSnapshot, String>> {
            self.record("get_initial_snapshot");
            Box::pin(async move { Ok(wire::AgentConnectionSnapshot::default()) })
        }
        fn get_rlm_child_snapshots(
            &self,
        ) -> pi_ai::types::BoxFuture<Result<Vec<wire::AgentConnectionRlmChildAgentSnapshot>, String>>
        {
            self.record("get_rlm_child_snapshots");
            Box::pin(async move { Ok(Vec::new()) })
        }
        fn get_messages(
            &self,
        ) -> pi_ai::types::BoxFuture<Result<Vec<pi_agent_core::types::AgentMessage>, String>>
        {
            self.record("get_messages");
            Box::pin(async move { Ok(Vec::new()) })
        }
        fn get_session_header(
            &self,
        ) -> pi_ai::types::BoxFuture<Result<Option<wire::AgentConnectionSessionHeader>, String>>
        {
            self.record("get_session_header");
            Box::pin(async move { Ok(None) })
        }
        fn get_commands(
            &self,
        ) -> pi_ai::types::BoxFuture<Result<Vec<wire::AgentConnectionSlashCommand>, String>>
        {
            self.record("get_commands");
            Box::pin(async move { Ok(Vec::new()) })
        }
        fn get_resource_snapshot(
            &self,
        ) -> pi_ai::types::BoxFuture<Result<wire::AgentConnectionResourceSnapshot, String>>
        {
            self.record("get_resource_snapshot");
            Box::pin(async move { Ok(wire::AgentConnectionResourceSnapshot::default()) })
        }
        fn get_model_catalog(
            &self,
        ) -> pi_ai::types::BoxFuture<Result<wire::AgentConnectionModelCatalog, String>> {
            self.record("get_model_catalog");
            Box::pin(async move { Ok(wire::AgentConnectionModelCatalog::default()) })
        }
        fn get_available_models(
            &self,
        ) -> pi_ai::types::BoxFuture<Result<Vec<wire::AgentConnectionModel>, String>> {
            self.record("get_available_models");
            Box::pin(async move { Ok(Vec::new()) })
        }
        fn get_session_stats(&self) -> pi_ai::types::BoxFuture<Result<serde_json::Value, String>> {
            self.record("get_session_stats");
            Box::pin(async move { Ok(serde_json::json!({})) })
        }
        fn get_context_tree(&self) -> pi_ai::types::BoxFuture<Result<serde_json::Value, String>> {
            self.record("get_context_tree");
            let tree = self.context_tree.lock().unwrap().clone();
            Box::pin(async move { Ok(tree) })
        }
        fn get_session_context(
            &self,
        ) -> pi_ai::types::BoxFuture<Result<wire::AgentConnectionSessionContext, String>> {
            self.record("get_session_context");
            Box::pin(async move { Ok(wire::AgentConnectionSessionContext::default()) })
        }
        fn get_session_tree(
            &self,
        ) -> pi_ai::types::BoxFuture<Result<wire::AgentConnectionWatchSessionTree, String>>
        {
            self.record("get_session_tree");
            let tree = self.session_tree.lock().unwrap().clone();
            Box::pin(async move { Ok(tree) })
        }
        fn list_saved_sessions(
            &self,
            scope: &str,
        ) -> pi_ai::types::BoxFuture<Result<Vec<wire::AgentConnectionSavedSessionInfo>, String>>
        {
            self.record("list_saved_sessions");
            Box::pin(async move { Ok(Vec::new()) })
        }
        fn get_queue(
            &self,
        ) -> pi_ai::types::BoxFuture<Result<wire::AgentConnectionQueueState, String>> {
            self.record("get_queue");
            Box::pin(async move { Ok(wire::AgentConnectionQueueState::default()) })
        }
        fn mutate_queued_message(
            &self,
            lane: &str,
            index: i64,
            expected_text: &str,
            mutation: serde_json::Value,
        ) -> pi_ai::types::BoxFuture<Result<String, String>> {
            self.record("mutate_queued_message");
            Box::pin(async move { Ok(String::new()) })
        }
        fn clear_queue(
            &self,
        ) -> pi_ai::types::BoxFuture<Result<wire::AgentConnectionQueueState, String>> {
            self.record("clear_queue");
            Box::pin(async move { Ok(wire::AgentConnectionQueueState::default()) })
        }
        fn abort_and_clear_queue(
            &self,
        ) -> pi_ai::types::BoxFuture<Result<wire::AgentConnectionQueueState, String>> {
            self.record("abort_and_clear_queue");
            Box::pin(async move { Ok(wire::AgentConnectionQueueState::default()) })
        }
        fn acquire_session_input_pause(
            &self,
            lease_key: &str,
        ) -> pi_ai::types::BoxFuture<Result<wire::AgentConnectionSessionInputPause, String>>
        {
            self.record("acquire_session_input_pause");
            Box::pin(async move {
                Ok(std::sync::Arc::new(NoopInputPause) as wire::AgentConnectionSessionInputPause)
            })
        }
        fn list_cron_jobs(
            &self,
            include_inactive: bool,
        ) -> pi_ai::types::BoxFuture<Result<Vec<serde_json::Value>, String>> {
            self.record("list_cron_jobs");
            Box::pin(async move { Ok(Vec::new()) })
        }
        fn heartbeat_catalog_supported(&self) -> Option<bool> {
            *self.heartbeat_catalog_support.lock().unwrap()
        }

        fn list_heartbeats(
            &self,
        ) -> pi_ai::types::BoxFuture<Result<Vec<wire::AgentConnectionHeartbeat>, String>> {
            self.record("list_heartbeats");
            Box::pin(async move { Ok(Vec::new()) })
        }
        fn manage_heartbeat(
            &self,
            active_session_id: &str,
            job_id: &str,
            action: serde_json::Value,
        ) -> pi_ai::types::BoxFuture<Result<serde_json::Value, String>> {
            self.record("manage_heartbeat");
            Box::pin(async move { Ok(serde_json::json!({})) })
        }
        fn add_cron_job(
            &self,
            schedule: &str,
            prompt: &str,
        ) -> pi_ai::types::BoxFuture<Result<serde_json::Value, String>> {
            self.record("add_cron_job");
            Box::pin(async move { Ok(serde_json::json!({})) })
        }
        fn cancel_cron_job(
            &self,
            job_id: &str,
        ) -> pi_ai::types::BoxFuture<Result<serde_json::Value, String>> {
            self.record("cancel_cron_job");
            Box::pin(async move { Ok(serde_json::json!({})) })
        }
        fn get_heartbeat(
            &self,
        ) -> pi_ai::types::BoxFuture<Result<Option<serde_json::Value>, String>> {
            self.record("get_heartbeat");
            Box::pin(async move { Ok(None) })
        }
        fn set_heartbeat(
            &self,
            schedule: &str,
            instruction: &str,
            delivery_mode: Option<&str>,
        ) -> pi_ai::types::BoxFuture<Result<serde_json::Value, String>> {
            self.record("set_heartbeat");
            Box::pin(async move { Ok(serde_json::json!({})) })
        }
        fn update_heartbeat(
            &self,
            action: serde_json::Value,
        ) -> pi_ai::types::BoxFuture<Result<Option<serde_json::Value>, String>> {
            self.record("update_heartbeat");
            Box::pin(async move { Ok(None) })
        }
        fn send_agent_message(
            &self,
            target_active_session_id: &str,
            message: &str,
        ) -> pi_ai::types::BoxFuture<Result<serde_json::Value, String>> {
            self.record("send_agent_message");
            Box::pin(async move { Ok(serde_json::json!({})) })
        }
        fn get_agent_message_status(
            &self,
        ) -> pi_ai::types::BoxFuture<Result<serde_json::Value, String>> {
            self.record("get_agent_message_status");
            Box::pin(async move { Ok(serde_json::json!({})) })
        }
        fn pause_agent_messages(
            &self,
        ) -> pi_ai::types::BoxFuture<Result<serde_json::Value, String>> {
            self.record("pause_agent_messages");
            Box::pin(async move { Ok(serde_json::json!({})) })
        }
        fn resume_agent_messages(
            &self,
        ) -> pi_ai::types::BoxFuture<Result<serde_json::Value, String>> {
            self.record("resume_agent_messages");
            Box::pin(async move { Ok(serde_json::json!({})) })
        }
        fn clear_agent_messages(&self) -> pi_ai::types::BoxFuture<Result<f64, String>> {
            self.record("clear_agent_messages");
            Box::pin(async move { Ok(0.0) })
        }
        fn get_user_messages_for_forking(
            &self,
        ) -> pi_ai::types::BoxFuture<Result<Vec<wire::AgentConnectionUserMessage>, String>>
        {
            self.record("get_user_messages_for_forking");
            let messages = self.user_messages.lock().unwrap().clone();
            Box::pin(async move { Ok(messages) })
        }
        fn get_last_assistant_text(
            &self,
        ) -> pi_ai::types::BoxFuture<Result<Option<String>, String>> {
            self.record("get_last_assistant_text");
            let text = self.last_assistant_text.lock().unwrap().clone();
            Box::pin(async move { Ok(text) })
        }
        fn get_system_prompt(&self) -> pi_ai::types::BoxFuture<Result<String, String>> {
            self.record("get_system_prompt");
            Box::pin(async move { Ok(String::new()) })
        }
        fn get_tool_definition(
            &self,
            name: &str,
        ) -> pi_ai::types::BoxFuture<Result<Option<wire::AgentConnectionToolDefinition>, String>>
        {
            self.record("get_tool_definition");
            Box::pin(async move { Ok(None) })
        }
        fn set_session_entry_label(
            &self,
            entry_id: &str,
            label: Option<&str>,
        ) -> pi_ai::types::BoxFuture<Result<(), String>> {
            self.record_with(
                "set_session_entry_label",
                &[format!("{entry_id}:{label:?}")],
            );
            Box::pin(async { Ok(()) })
        }
        fn respond_to_extension_ui_request(
            &self,
            request_id: &str,
            response: wire::AgentConnectionExtensionUiResponse,
        ) -> pi_ai::types::BoxFuture<Result<(), String>> {
            self.record("respond_to_extension_ui_request");
            Box::pin(async move { Ok(()) })
        }
        fn prompt(
            &self,
            message: &str,
            options: Option<wire::AgentConnectionPromptOptions>,
        ) -> pi_ai::types::BoxFuture<Result<(), String>> {
            let images = options
                .as_ref()
                .and_then(|options| options.images.as_ref())
                .map(|images| images.len());
            let behavior = options
                .as_ref()
                .and_then(|options| options.streaming_behavior.clone())
                .unwrap_or_else(|| "none".to_string());
            self.record_with(
                "prompt",
                &[message.to_string(), behavior, format!("{images:?}")],
            );
            Box::pin(async { Ok(()) })
        }
        fn prompt_and_wait(
            &self,
            message: &str,
            options: Option<wire::AgentConnectionPromptOptions>,
        ) -> pi_ai::types::BoxFuture<Result<(), String>> {
            self.record("prompt_and_wait");
            Box::pin(async move { Ok(()) })
        }
        fn start_side_question(
            &self,
            id: &str,
            question: &str,
            previous_turns: Option<Vec<wire::AgentConnectionSideQuestionTurn>>,
        ) -> pi_ai::types::BoxFuture<Result<(), String>> {
            self.record_with(
                "start_side_question",
                &[question.to_string(), format!("{previous_turns:?}")],
            );
            let _ = id;
            Box::pin(async { Ok(()) })
        }
        fn abort_side_question(&self, id: &str) -> pi_ai::types::BoxFuture<Result<bool, String>> {
            self.record("abort_side_question");
            Box::pin(async move { Ok(false) })
        }
        fn steer(
            &self,
            message: &str,
            images: Option<Vec<ImageContent>>,
        ) -> pi_ai::types::BoxFuture<Result<(), String>> {
            self.record("steer");
            Box::pin(async move { Ok(()) })
        }
        fn follow_up(
            &self,
            message: &str,
            images: Option<Vec<ImageContent>>,
        ) -> pi_ai::types::BoxFuture<Result<(), String>> {
            self.record("follow_up");
            Box::pin(async move { Ok(()) })
        }
        fn abort(&self) -> pi_ai::types::BoxFuture<Result<(), String>> {
            self.record("abort");
            Box::pin(async move { Ok(()) })
        }
        fn cancel_rlm_child(
            &self,
            child_id: &str,
        ) -> pi_ai::types::BoxFuture<Result<bool, String>> {
            self.record("cancel_rlm_child");
            Box::pin(async move { Ok(false) })
        }
        fn wait_for_idle(&self) -> pi_ai::types::BoxFuture<Result<(), String>> {
            self.record("wait_for_idle");
            Box::pin(async move { Ok(()) })
        }
        fn wait_for_headless_completion(
            &self,
            options: Option<wire::AgentConnectionHeadlessCompletionOptions>,
        ) -> pi_ai::types::BoxFuture<Result<crate::core::autonomous::AgentAutonomousStatus, String>>
        {
            self.record("wait_for_headless_completion");
            Box::pin(async move {
                Ok(crate::core::autonomous::AgentAutonomousStatus {
                    enabled: false,
                    continuations_used: 0.0,
                    turns_used: 0.0,
                    tokens_used: 0.0,
                    started_at: None,
                    limits: crate::core::autonomous::default_autonomous_limits(),
                    gates: crate::core::autonomous::default_autonomous_gates(),
                    gate_attempts: std::collections::BTreeMap::new(),
                    last_gate_failure: None,
                })
            })
        }
        fn execute_bash(
            &self,
            command: &str,
            options: Option<wire::AgentConnectionExecuteBashOptions>,
        ) -> pi_ai::types::BoxFuture<Result<(), String>> {
            self.record_with("execute_bash", &[format!("{command}|{options:?}")]);
            Box::pin(async { Ok(()) })
        }
        fn execute_bash_and_wait(
            &self,
            command: &str,
        ) -> pi_ai::types::BoxFuture<Result<serde_json::Value, String>> {
            self.record("execute_bash_and_wait");
            Box::pin(async move { Ok(serde_json::json!({})) })
        }
        fn abort_bash(&self) -> pi_ai::types::BoxFuture<Result<(), String>> {
            self.record("abort_bash");
            Box::pin(async move { Ok(()) })
        }
        fn set_model(
            &self,
            provider: &str,
            model_id: &str,
        ) -> pi_ai::types::BoxFuture<Result<wire::AgentConnectionModel, String>> {
            self.record("set_model");
            Box::pin(async move { Ok(wire::AgentConnectionModel::default()) })
        }
        fn cycle_model(
            &self,
            direction: Option<&str>,
        ) -> pi_ai::types::BoxFuture<Result<Option<wire::AgentConnectionModelCycleResult>, String>>
        {
            self.record("cycle_model");
            Box::pin(async move { Ok(None) })
        }
        fn set_scoped_models(
            &self,
            scoped_models: Vec<wire::AgentConnectionScopedModel>,
        ) -> pi_ai::types::BoxFuture<Result<(), String>> {
            self.record_with(
                "set_scoped_models",
                &[scoped_models
                    .iter()
                    .map(|scoped| format!("{}/{}", scoped.model.provider, scoped.model.id))
                    .collect::<Vec<_>>()
                    .join(",")],
            );
            Box::pin(async { Ok(()) })
        }
        fn set_thinking_level(
            &self,
            level: pi_agent_core::types::ThinkingLevel,
        ) -> pi_ai::types::BoxFuture<Result<(), String>> {
            self.record_with("set_thinking_level", &[level.as_str().to_string()]);
            Box::pin(async { Ok(()) })
        }
        fn set_service_tier(
            &self,
            service_tier: ServiceTier,
        ) -> pi_ai::types::BoxFuture<Result<(), String>> {
            self.record("set_service_tier");
            Box::pin(async move { Ok(()) })
        }
        fn cycle_thinking_level(
            &self,
        ) -> pi_ai::types::BoxFuture<Result<Option<pi_agent_core::types::ThinkingLevel>, String>>
        {
            self.record("cycle_thinking_level");
            Box::pin(async move { Ok(None) })
        }
        fn set_transport(
            &self,
            transport: pi_ai::types::Transport,
        ) -> pi_ai::types::BoxFuture<Result<(), String>> {
            self.record("set_transport");
            Box::pin(async move { Ok(()) })
        }
        fn set_steering_mode(&self, mode: &str) -> pi_ai::types::BoxFuture<Result<(), String>> {
            self.record("set_steering_mode");
            Box::pin(async move { Ok(()) })
        }
        fn set_follow_up_mode(&self, mode: &str) -> pi_ai::types::BoxFuture<Result<(), String>> {
            self.record("set_follow_up_mode");
            Box::pin(async move { Ok(()) })
        }
        fn set_auto_compaction_enabled(
            &self,
            enabled: bool,
        ) -> pi_ai::types::BoxFuture<Result<(), String>> {
            self.record("set_auto_compaction_enabled");
            Box::pin(async move { Ok(()) })
        }
        fn set_auto_retry_enabled(
            &self,
            enabled: bool,
        ) -> pi_ai::types::BoxFuture<Result<(), String>> {
            self.record("set_auto_retry_enabled");
            Box::pin(async move { Ok(()) })
        }
        fn compact(
            &self,
            custom_instructions: Option<&str>,
        ) -> pi_ai::types::BoxFuture<Result<serde_json::Value, String>> {
            self.record("compact");
            Box::pin(async move { Ok(serde_json::json!({})) })
        }
        fn refine(
            &self,
            options: serde_json::Value,
        ) -> pi_ai::types::BoxFuture<Result<serde_json::Value, String>> {
            self.record("refine");
            Box::pin(async move { Ok(serde_json::json!({})) })
        }
        fn abort_compaction(&self) -> pi_ai::types::BoxFuture<Result<(), String>> {
            self.record("abort_compaction");
            Box::pin(async move { Ok(()) })
        }
        fn abort_branch_summary(&self) -> pi_ai::types::BoxFuture<Result<(), String>> {
            self.record("abort_branch_summary");
            Box::pin(async move { Ok(()) })
        }
        fn abort_retry(&self) -> pi_ai::types::BoxFuture<Result<(), String>> {
            self.record("abort_retry");
            Box::pin(async move { Ok(()) })
        }
        fn reload(&self) -> pi_ai::types::BoxFuture<Result<(), String>> {
            self.record("reload");
            Box::pin(async move { Ok(()) })
        }
        fn new_session(
            &self,
            options: Option<wire::AgentConnectionNewSessionOptions>,
        ) -> pi_ai::types::BoxFuture<Result<bool, String>> {
            self.record("new_session");
            Box::pin(async move { Ok(false) })
        }
        fn switch_session(
            &self,
            session_path: &str,
            options: Option<wire::AgentConnectionSwitchSessionOptions>,
        ) -> pi_ai::types::BoxFuture<Result<bool, String>> {
            self.record("switch_session");
            Box::pin(async move { Ok(false) })
        }
        fn fork(
            &self,
            entry_id: &str,
            options: Option<wire::AgentConnectionForkOptions>,
        ) -> pi_ai::types::BoxFuture<Result<serde_json::Value, String>> {
            self.record_with("fork", &[format!("{entry_id}|{options:?}")]);
            let result = self.fork_result.lock().unwrap().clone();
            Box::pin(async move { Ok(result) })
        }
        fn navigate_tree(
            &self,
            target_id: &str,
            options: Option<wire::AgentConnectionNavigateTreeOptions>,
        ) -> pi_ai::types::BoxFuture<Result<wire::AgentConnectionNavigateTreeResult, String>>
        {
            self.record("navigate_tree");
            Box::pin(async move { Ok(wire::AgentConnectionNavigateTreeResult::default()) })
        }
        fn import_from_jsonl(
            &self,
            input_path: &str,
            cwd_override: Option<&str>,
        ) -> pi_ai::types::BoxFuture<Result<bool, String>> {
            self.record("import_from_jsonl");
            Box::pin(async move { Ok(false) })
        }
        fn export_to_html(
            &self,
            output_path: Option<&str>,
        ) -> pi_ai::types::BoxFuture<Result<String, String>> {
            self.record_with("export_to_html", &[format!("{output_path:?}")]);
            let path = self
                .export_path
                .lock()
                .unwrap()
                .clone()
                .unwrap_or_else(|| "/tmp/session.html".to_string());
            Box::pin(async move { Ok(path) })
        }
        fn export_to_jsonl(
            &self,
            output_path: Option<&str>,
        ) -> pi_ai::types::BoxFuture<Result<String, String>> {
            self.record_with("export_to_jsonl", &[format!("{output_path:?}")]);
            let path = self
                .export_path
                .lock()
                .unwrap()
                .clone()
                .unwrap_or_else(|| "/tmp/session.jsonl".to_string());
            Box::pin(async move { Ok(path) })
        }
        fn set_session_name(&self, name: &str) -> pi_ai::types::BoxFuture<Result<(), String>> {
            self.record_with("set_session_name", &[name.to_string()]);
            Box::pin(async { Ok(()) })
        }
        fn get_rlm_max_depth_status(
            &self,
        ) -> pi_ai::types::BoxFuture<Result<serde_json::Value, String>> {
            self.record("get_rlm_max_depth_status");
            Box::pin(async move { Ok(serde_json::json!({})) })
        }
        fn set_rlm_max_depth(
            &self,
            max_depth: f64,
            options: Option<serde_json::Value>,
        ) -> pi_ai::types::BoxFuture<Result<serde_json::Value, String>> {
            self.record("set_rlm_max_depth");
            Box::pin(async move { Ok(serde_json::json!({})) })
        }
        fn rename_saved_session(
            &self,
            session_path: &str,
            name: &str,
        ) -> pi_ai::types::BoxFuture<Result<(), String>> {
            self.record("rename_saved_session");
            Box::pin(async move { Ok(()) })
        }
        fn delete_saved_session(
            &self,
            session_path: &str,
        ) -> pi_ai::types::BoxFuture<Result<serde_json::Value, String>> {
            self.record("delete_saved_session");
            Box::pin(async move { Ok(serde_json::json!({})) })
        }
        fn watch_session(
            &self,
            active_session_id: &str,
        ) -> pi_ai::types::BoxFuture<
            Result<Option<Box<dyn wire::AgentConnectionSessionWatcher>>, String>,
        > {
            self.record("watch_session");
            Box::pin(async move { Ok(None) })
        }
        fn dispose(&self) -> pi_ai::types::BoxFuture<Result<(), String>> {
            self.record("dispose");
            Box::pin(async move { Ok(()) })
        }
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
    }

    #[test]
    fn recovery_notice_refinement_start_event_is_visible_once_and_reopens_as_a_card() {
        use pi_agent_core::types::{CustomAgentMessage, CustomMessageContent};
        let mode = Rc::new(RefCell::new(stash_mode("refinement-display-test")));
        let transcript = Rc::new(RefCell::new(Transcript::new(mode.clone())));
        let message = AgentMessage::Custom(CustomAgentMessage::Custom {
            custom_type: crate::core::messages::REFINEMENT_OUTCOME_CUSTOM_TYPE.into(),
            content: CustomMessageContent::Text("Refinement complete: Saved a useful lesson".into()),
            display: true,
            details: Some(serde_json::json!({"refinementId":"test", "summary":"Saved a useful lesson", "scope":"local", "edits":[]})), timestamp: 1,
        });
        apply_event(&mode, &transcript, wire::AgentConnectionSessionEvent::MessageStart { message: message.clone() });
        let display = transcript.borrow_mut().render(100.0).join("\n");
        assert!(display.contains("[refinement]"));
        assert!(display.contains("Saved a useful lesson"));
        apply_event(&mode, &transcript, wire::AgentConnectionSessionEvent::MessageEnd { message: message.clone() });
        assert_eq!(transcript.borrow().refinement_outcomes.len(), 1);
        transcript.borrow_mut().replace(vec![message.clone()]);
        assert_eq!(transcript.borrow().refinement_outcomes.len(), 1);
        transcript.borrow_mut().set_recovery_notices_expanded(true);
        assert!(transcript.borrow().refinement_outcomes[0].borrow().expanded());
        transcript.borrow_mut().replace_history(vec![message], 1.0);
        transcript.borrow_mut().set_recovery_notices_expanded(false);
        assert!(!transcript.borrow().history.as_ref().unwrap().refinement_outcomes[0].borrow().expanded());
    }

    #[test]
    fn recovery_notice_is_compact_live_reopened_and_paged_without_changing_context() {
        use pi_agent_core::types::{CustomAgentMessage, CustomMessageContent};
        let mode = Rc::new(RefCell::new(stash_mode("recovery-test")));
        let text = format!("<ipython_state_restored>\nYour Python kernel state was revived from your previous session. These names are available again: {}.\n</ipython_state_restored>", (0..4000).map(|n| format!("saved_variable_{n}")).collect::<Vec<_>>().join(", "));
        let message = AgentMessage::Custom(CustomAgentMessage::Custom {
            custom_type: crate::core::messages::IPYTHON_STATE_RESTORED_CUSTOM_TYPE.into(),
            content: CustomMessageContent::Text(text.clone()), display: true,
            details: Some(serde_json::json!({"restored": true})), timestamp: 1,
        });
        let original = serde_json::to_string(&message).unwrap();
        let mut transcript = Transcript::new(mode);
        for route in 0..3 {
            match route {
                0 => transcript.message(message.clone(), true),
                1 => transcript.replace(vec![message.clone()]),
                _ => { transcript.replace(vec![]); transcript.replace_history(vec![message.clone()], 1.0); }
            }
            for width in [40.0, 80.0, 150.0] {
                let lines = transcript.render(width);
                assert!(lines.len() <= 4, "route={route}, width={width}: {} lines", lines.len());
                assert!(lines.join("\n").contains("4000 variables"));
                assert!(!lines.join("\n").contains("saved_variable_3999"));
            }
            transcript.set_recovery_notices_expanded(true);
            assert!(transcript.render(150.0).join("\n").contains("saved_variable_3999"));
            transcript.set_recovery_notices_expanded(false);
            assert!(!transcript.render(150.0).join("\n").contains("saved_variable_3999"));
        }
        assert_eq!(serde_json::to_string(&message).unwrap(), original);
    }

    #[test]
    fn recovery_notice_failures_remain_visible_when_collapsed() {
        let _mode = stash_mode("recovery-warning-test");
        for text in [
            "These names are available again: alpha, beta.\nThese could not be restored and must be recreated if needed: missing_dataframe, missing_module.",
            "Your previous Python kernel state could not be revived; the kernel is starting fresh, so re-create any variables, imports, or loaded data you need.",
            "Unknown future recovery format",
        ] {
            let mut notice = native_recovery_notice::RecoveryNotice::new(text.into(), false);
            let display = notice.render(40.0).join("\n");
            assert!(display.contains("not restored") || display.contains("check details"));
            notice.set_expanded(true);
            assert!(notice.render(200.0).join("\n").contains(text.lines().next().unwrap()));
        }
    }

    #[test]
    fn hidden_custom_context_is_not_rendered_live_or_after_reopen() {
        use pi_agent_core::types::{CustomAgentMessage, CustomMessageContent};
        let mode = Rc::new(RefCell::new(stash_mode("visibility-test")));
        let hidden = AgentMessage::Custom(CustomAgentMessage::Custom {
            custom_type: "harness_state".into(),
            content: CustomMessageContent::Text("private model context".into()),
            display: false, details: None, timestamp: 1,
        });
        let visible = AgentMessage::Custom(CustomAgentMessage::Custom {
            custom_type: "notice".into(),
            content: CustomMessageContent::Text("visible notice".into()),
            display: true, details: None, timestamp: 2,
        });
        let mut transcript = Transcript::new(mode);
        transcript.message(hidden.clone(), true);
        assert!(transcript.rows.is_empty());
        transcript.message(visible.clone(), false);
        assert_eq!(transcript.rows.len(), 1);
        let messages = vec![hidden.clone(), visible.clone()];
        transcript.replace(messages.clone());
        assert_eq!(transcript.rows.len(), 1);
        transcript.replace_history(messages.clone(), 2.0);
        assert_eq!(transcript.history.as_ref().unwrap().rows.len(), 1);
        assert_eq!(messages, vec![hidden, visible], "rendering must not modify model context");
        let rendered = transcript.render(100.0).join("\n");
        assert!(rendered.contains("visible notice"));
        assert!(!rendered.contains("private model context"));
    }

    #[test]
    fn event_budget_yields_without_dropping_or_reordering_events() {
        let (send, receive) = mpsc::channel();
        for index in 0..MAX_HOST_EVENTS_PER_FRAME + 3 { send.send(index).unwrap(); }
        let mut budget = HostEventBudget::new();
        let frame_start = budget.started;
        let mut drained = Vec::new();
        while let Some(event) = budget.next_at(&receive, frame_start) { drained.push(event); }
        assert_eq!(drained.len(), MAX_HOST_EVENTS_PER_FRAME);
        let mut expired = HostEventBudget { started: frame_start, remaining: 1 };
        assert_eq!(expired.next_at(&receive, frame_start + MAX_HOST_EVENT_TIME_PER_FRAME), None);
        let rest: Vec<_> = receive.try_iter().collect();
        drained.extend(rest);
        assert_eq!(drained, (0..MAX_HOST_EVENTS_PER_FRAME + 3).collect::<Vec<_>>());
    }

    #[tokio::test]
    async fn session_view_cleanup_only_detaches_and_never_aborts_work() {
        let recorder = Arc::new(RecordingConnection::new());
        close_session_view(recorder.clone(), Vec::new()).await.unwrap();
        assert_eq!(recorder.calls().iter().map(|(name, _)| name.as_str()).collect::<Vec<_>>(), vec!["dispose"]);
    }

    #[test]
    fn extension_dialogs_return_real_component_selection_and_text() {
        crate::modes::interactive::theme::theme::init_theme(Some("prime"), false);
        crate::core::keybindings::KeybindingsManager::new(Default::default(), None).install();
        let ui = Rc::new(RefCell::new(TUI::new(
            Box::new(pi_tui::terminal::ProcessTerminal::new()),
            None,
        )));
        let (send, receive) = mpsc::channel();
        let confirm = wire::AgentConnectionExtensionUiRequest {
            id: "confirm-1".into(),
            method: "confirm".into(),
            payload: serde_json::json!({"title":"Run tool?","message":"Review the request","timeout":5000}),
        };
        let dialog = extension_dialog(confirm, &ui, &send).expect("confirm dialog");
        assert!(dialog.deadline.is_some());
        dialog.component.borrow_mut().handle_input("\r");
        assert_eq!(
            receive.try_recv().unwrap(),
            (
                "confirm-1".into(),
                wire::AgentConnectionExtensionUiResponse::Confirmed { confirmed: true }
            )
        );
        dialog.overlay.hide();

        let input = wire::AgentConnectionExtensionUiRequest {
            id: "input-1".into(),
            method: "input".into(),
            payload: serde_json::json!({"title":"Project name"}),
        };
        let dialog = extension_dialog(input, &ui, &send).expect("input dialog");
        dialog.component.borrow_mut().handle_input("demo");
        dialog.component.borrow_mut().handle_input("\r");
        assert_eq!(
            receive.try_recv().unwrap(),
            (
                "input-1".into(),
                wire::AgentConnectionExtensionUiResponse::Value {
                    value: "demo".into()
                }
            )
        );
        dialog.overlay.hide();

        assert!(extension_dialog(
            wire::AgentConnectionExtensionUiRequest {
                id: "invalid".into(),
                method: "select".into(),
                payload: serde_json::json!({"title":"Select","options":[false]}),
            },
            &ui,
            &send
        )
        .is_none());
    }

    /// The registry resolves names and aliases; everything else reaches the model.
    /// This is the executable form of interactive-mode.ts:4784-4786.
    #[test]
    fn submission_classification_resolves_through_the_builtin_registry() {
        // Canonical names keep their arguments.
        assert_eq!(
            classify_submission("/effort high"),
            SlashDispatch::Builtin {
                name: "effort".into(),
                args: "high".into(),
                raw: "/effort high".into(),
            }
        );
        assert_eq!(
            classify_submission("  /effort   xhigh  ".trim()),
            SlashDispatch::Builtin {
                name: "effort".into(),
                args: "xhigh".into(),
                raw: "/effort   xhigh".into(),
            }
        );
        // Aliases resolve to their canonical target
        // (`builtin_slash_command_aliases`, core/slash_commands.rs).
        for (alias, canonical) in [
            ("/clear", "new"),
            ("/usage", "context"),
            ("/thinking", "effort"),
            ("/rename", "name"),
            ("/side", "btw"),
        ] {
            match classify_submission(alias) {
                SlashDispatch::Builtin { name, .. } => assert_eq!(name, canonical, "{alias}"),
                other => panic!("{alias} resolved to {other:?}"),
            }
        }
        // Aliases with arguments too.
        match classify_submission("/thinking minimal") {
            SlashDispatch::Builtin { name, args, .. } => {
                assert_eq!(name, "effort");
                assert_eq!(args, "minimal");
            }
            other => panic!("{other:?}"),
        }
        // Free text, extension commands, and a bare slash are not built-ins.
        for text in [
            "hello there",
            "tell me about /model",
            "/",
            "/not-a-builtin",
            "/effortx",
            "!ls -la",
        ] {
            assert_eq!(
                classify_submission(text),
                SlashDispatch::Model(text.to_string()),
                "{text}"
            );
        }
    }

    /// Every built-in the TypeScript dispatches is recognised here, and none of
    /// them fall through to the model by accident.
    ///
    /// `/debug` is listed because TypeScript dispatches it from
    /// `parseSlashCommand`'s name (`interactive-mode.ts:5025`) even though it is NOT a
    /// registry entry: `resolveBuiltinSlashCommandName` only maps aliases and passes
    /// unknown names through, so `commandName === "debug"` matches without the name
    /// ever being a `builtinSlashCommand`. The registry stays the authority for names
    /// and gains no invented entry; the dispatcher recognises `debug` as a known
    /// command the host does not implement yet.
    #[test]
    fn every_dispatched_builtin_is_a_registry_name() {
        for name in [
            "btw",
            "changelog",
            "clone",
            "copy",
            "context",
            "debug",
            "effort",
            "export",
            "fast",
            "fork",
            "fullscreen",
            "heartbeat",
            "heartbeats",
            "hotkeys",
            "import",
            "login",
            "logout",
            "logs",
            "mcp",
            "model",
            "name",
            "new",
            "reload",
            "resume",
            "rlm-max-depth",
            "scoped-models",
            "session",
            "settings",
            "share",
            "system-prompt",
            "traces",
            "tree",
            "update",
        ] {
            match classify_submission(&format!("/{name}")) {
                SlashDispatch::Builtin { name: resolved, .. } => assert_eq!(resolved, name),
                other => panic!("/{name} resolved to {other:?}"),
            }
        }
    }

    /// `availableThinkingLevels` (interactive-mode.ts:8189-8193): a model that
    /// only reports `off` supports no thinking, so `/effort` says so.
    #[test]
    fn available_thinking_levels_drops_off_only_models() {
        let mut state = wire::AgentConnectionState::default();
        state.available_thinking_levels = vec![pi_agent_core::types::ThinkingLevel::Off];
        assert!(available_thinking_levels(&state).is_empty());

        state.available_thinking_levels = vec![
            pi_agent_core::types::ThinkingLevel::Off,
            pi_agent_core::types::ThinkingLevel::High,
        ];
        assert_eq!(available_thinking_levels(&state).len(), 2);

        state.available_thinking_levels = Vec::new();
        assert!(available_thinking_levels(&state).is_empty());
    }

    #[test]
    fn model_command_supports_prefilled_search_without_consuming_other_commands() {
        assert_eq!(model_command_search(" /model "), Some(None));
        assert_eq!(
            model_command_search("/model signed/model-2"),
            Some(Some("signed/model-2".into()))
        );
        assert_eq!(
            model_command_search("/model\t model 2 "),
            Some(Some("model 2".into()))
        );
        assert_eq!(model_command_search("/models"), None);
        assert_eq!(model_command_search("tell me about /model"), None);
    }

    #[test]
    fn settings_search_reaches_the_real_change_and_cancel_callbacks() {
        let mode = stash_mode("settings-host");
        let (send, receive) = mpsc::channel();
        let mut picker =
            native_settings::create(&mode, &wire::AgentConnectionState::default(), &send).unwrap();
        assert!(picker.render(80.0).join("\n").contains("Auto-compact"));
        for key in "padding".chars() {
            picker.handle_input(&key.to_string());
        }
        assert!(picker.render(80.0).join("\n").contains("Editor padding"));
        picker.handle_input("\r");
        assert!(matches!(
            receive.try_recv().unwrap(),
            HostEvent::Setting(native_settings::Change::EditorPaddingX(1.0))
        ));
        picker.handle_input("\u{1b}");
        assert!(matches!(
            receive.try_recv().unwrap(),
            HostEvent::Setting(native_settings::Change::Close)
        ));
    }

    #[test]
    fn heartbeat_catalog_scopes_the_wire_jobs_and_keeps_display_details() {
        let mut mode = stash_mode("heartbeat-host");
        mode.apply_connection_state_snapshot(local::AgentConnectionState {
            session_id: "session".into(),
            active_session_id: Some("active".into()),
            ..Default::default()
        });
        let catalog: Vec<wire::AgentConnectionHeartbeat> = ["active", "unrelated"]
            .iter()
            .enumerate()
            .map(|(i, active)| {
                let job = crate::core::cron_jobs::AgentCronJob {
                    id: format!("job-{i}"),
                    active_session_id: (*active).into(),
                    prompt: "scheduled work".into(),
                    ..Default::default()
                };
                wire::AgentConnectionHeartbeat {
                    job: serde_json::to_value(job).unwrap(),
                    session_name: Some("named session".into()),
                    ..Default::default()
                }
            })
            .collect();
        native_heartbeats::apply_catalog(&mut mode, &catalog);
        let scoped = native_heartbeats::scoped(&mode, &catalog);
        assert_eq!(scoped, vec![catalog[0].clone()]);
    }

    /// A mode with no connection, enough for the Ctrl+S dispatch chain and for the
    /// history-runtime regression tests in the sibling `native_history` module.
    pub(super) fn stash_mode(session_id: &str) -> InteractiveMode {
        crate::modes::interactive::theme::theme::init_theme(Some("prime"), false);
        crate::core::keybindings::KeybindingsManager::new(Default::default(), None).install();
        let services = local::InteractiveModeUiServices {
            settings_manager: Arc::new(std::sync::Mutex::new(local::SettingsManager::in_memory(
                serde_json::Map::new(),
            ))),
            model_registry: Arc::new(std::sync::Mutex::new(local::ModelRegistry::in_memory())),
            get_initial_cwd: Box::new(|| "/initial".to_string()),
            get_initial_session_name: Box::new(|| Some("initial".to_string())),
            get_themes: Box::new(Vec::new),
            refresh_mcp_providers: None,
        };
        InteractiveMode::new(InteractiveModeOptions {
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
            return_to_agents_view: true,
            force_fullscreen: false,
            agents_view_owns_startup_notices: false,
            session_depth: None,
            session_has_children: false,
            prompt_stash_store: Some(
                crate::modes::interactive::prompt_stash_state::shared_prompt_stash_store(),
            ),
            prompt_stash_session_id: Some(session_id.to_string()),
        })
        .expect("mode")
    }

    #[test]
    fn transcript_wrap_cache_preserves_live_updates_resize_and_unicode() {
        let mut cache = TranscriptWrapCache::default();
        let mut lines = vec!["\x1b[32mhello 世界 👩‍💻\x1b[0m".into(), "long output ".repeat(20)];
        let expected = |lines: &Vec<String>, width| lines.iter()
            .flat_map(|line| pi_tui::utils::wrap_text_with_ansi(line, width)).collect::<Vec<_>>();
        assert_eq!(cache.render(lines.clone(), 20), expected(&lines, 20));
        let stable_buffer = cache.lines[0].1.as_ptr();
        assert_eq!(cache.render(lines.clone(), 20), expected(&lines, 20));
        assert_eq!(cache.lines[0].1.as_ptr(), stable_buffer, "unchanged history was rewrapped");
        lines[1].push_str("stream delta");
        assert_eq!(cache.render(lines.clone(), 20), expected(&lines, 20));
        assert_eq!(cache.lines[0].1.as_ptr(), stable_buffer);
        assert_eq!(cache.render(lines.clone(), 8), expected(&lines, 8));
        lines[0] = "\x1b[33mchanged theme\x1b[0m".into();
        assert_eq!(cache.render(lines.clone(), 8), expected(&lines, 8));
        lines.truncate(1);
        assert_eq!(cache.render(lines.clone(), 8), expected(&lines, 8));
        assert_eq!(cache.lines.len(), 1);
        assert!(cache.render(Vec::new(), 8).is_empty());
        assert!(cache.lines.is_empty());
    }

    #[test]
    fn transcript_wrap_cache_large_repaint_measurement() {
        let lines: Vec<String> = (0..2000).map(|i| format!("\x1b[32m{i}: café 世界 output text {}\x1b[0m", "abcd ".repeat(12))).collect();
        let mut cache = TranscriptWrapCache::default();
        let started = Instant::now();
        let expected = cache.render(lines.clone(), 150);
        let cold = started.elapsed();
        let started = Instant::now();
        for _ in 0..10 { assert_eq!(cache.render(lines.clone(), 150), expected); }
        eprintln!("transcript wrap 2000 lines: cold={cold:?}, cached mean={:?}", started.elapsed() / 10);
    }

    /// DEFECT 6: `/compact`, `/refine`, `/goal` and `/autonomous` must reach the
    /// connection's `prompt`, not a local handler (interactive-mode.ts:4821-5030
    /// has no local arm for them; they fall through to
    /// `agentConnection.prompt(text, ...)` at :5177-5181, which
    /// `AgentSession._normalizeSubmission` turns into the session command action
    /// at agent-session.ts:5065-5068).
    #[tokio::test]
    async fn session_slash_commands_are_prompted_not_handled_locally() {
        for text in [
            "/compact",
            "/compact keep the plan",
            "/refine",
            "/goal ship the port",
            "/autonomous",
        ] {
            let recorder = Arc::new(RecordingConnection::new());
            let connection: Arc<dyn wire::AgentConnection> = recorder.clone();
            let (send, _receive) = mpsc::channel();
            assert_eq!(
                classify_submission(text),
                SlashDispatch::SessionCommand(text.trim().to_string()),
                "{text} must classify as a session command"
            );
            // The session-command route is the prompt path, so it must run through
            // `dispatch_submission`, not the local built-in chain.
            let result = dispatch_submission(&connection, &send, text, false, None).await;
            let recorded = recorder.calls();
            let prompt = recorded
                .iter()
                .find(|(name, _)| name == "prompt")
                .unwrap_or_else(|| panic!("{text} must reach connection.prompt, got {recorded:?}"));
            assert_eq!(prompt.1[0], text.trim(), "{text} must be prompted verbatim");
            assert_eq!(
                prompt.1[1], "steer",
                "{text} must keep the default steer queue mode"
            );
            assert!(
                result.is_ok(),
                "{text} must prompt successfully: {result:?}"
            );
        }
    }

    /// The teeth of the DEFECT 6 assertion: a session command must never be
    /// answered by the local `/status`-style chain. `/compact` has no local arm,
    /// so a dispatcher that lost the `SessionCommand` branch would send it to the
    /// model as chat text and leave `prompt` uncalled.
    #[tokio::test]
    async fn a_session_command_does_not_take_the_local_builtin_path() {
        let text = "/compact keep the plan";
        assert!(
            crate::core::slash_commands::parse_session_slash_command(text).is_some(),
            "the registry must own the session-command classification"
        );
        // `compact` IS a registry entry, but it is marked `execution: "session"`
        // (slash-commands.ts:... / slash_commands.rs:438-440), which is exactly
        // why the local chain must not execute it.
        let command = crate::core::slash_commands::builtin_slash_commands()
            .iter()
            .find(|command| command.name == "compact")
            .expect("`compact` must be a registry entry");
        assert_eq!(
            command.execution.as_deref(),
            Some("session"),
            "`compact` must be a session-executed command, not a local one"
        );
        assert_eq!(
            crate::core::slash_commands::parse_session_slash_command("/compact keep the plan")
                .map(|command| command.name),
            Some("compact".to_string())
        );
        let recorder = Arc::new(RecordingConnection::new());
        let connection: Arc<dyn wire::AgentConnection> = recorder.clone();
        let (send, _receive) = mpsc::channel();
        // `run_builtin_command` is the LOCAL chain: no session command may appear
        // there, so it must report the name as unhandled rather than act on it.
        let local = run_builtin_command(&connection, &send, text, "compact", "keep the plan").await;
        assert!(
            matches!(local, Ok(CommandOutput::Status(ref message))
                if message.starts_with("/compact is recognised but the native host has no handler")),
            "the local chain must not silently handle a session command, got {local:?}"
        );
        assert_eq!(
            recorder.calls(),
            Vec::new(),
            "the local chain must not touch the connection"
        );
    }

    /// DEFECT 1 (DESTRUCTIVE): `/clear <anything>` must NOT start a new session.
    ///
    /// TypeScript keys the two arms on the name AS TYPED
    /// (interactive-mode.ts:4968 `slashCommand?.name === "clear"`), and answers any
    /// argument with `showError("Usage: /clear")` at :4969-4971 - it never reaches
    /// `handleClearCommand` (:10222). `clear` is the no-argument compatibility alias
    /// (`builtin_slash_command_takes_argument` returns false for it,
    /// core/slash_commands.rs:530-533). The assertion is on the RECORDED CONNECTION,
    /// so a handler that starts a session fails here even if it also prints a usage
    /// line.
    #[tokio::test]
    async fn clear_with_arguments_reports_usage_and_never_starts_a_session() {
        for text in ["/clear foo", "/clear extra words", "/clear --name x"] {
            let recorder = Arc::new(RecordingConnection::new());
            let connection: Arc<dyn wire::AgentConnection> = recorder.clone();
            let (send, _receive) = mpsc::channel();
            let command =
                crate::core::slash_commands::parse_slash_command(text).expect("a slash command");
            let resolved = crate::core::slash_commands::resolve_slash_command(&command);
            assert_eq!(resolved.name, "new", "{text} must resolve to the new arm");
            assert_eq!(command.name, "clear", "{text} must be the /clear alias");
            assert!(
                !crate::core::slash_commands::builtin_slash_command_takes_argument(&command.name),
                "{text} names the no-argument compatibility alias"
            );
            assert!(
                !resolved.args.is_empty(),
                "{text} must carry a real argument (TypeScript trims it, slash-commands.ts:246)"
            );

            let output =
                run_builtin_command(&connection, &send, text, &resolved.name, &resolved.args)
                    .await
                    .expect("the usage error is a local reply, not an Err");

            assert_eq!(
                recorder.calls(),
                Vec::new(),
                "{text} must NOT touch the connection at all, got {:?}",
                recorder.calls()
            );
            match output {
                CommandOutput::Error(message) => assert_eq!(message, "Usage: /clear"),
                other => panic!("{text} must report Usage: /clear via showError, got {other:?}"),
            }
        }
    }

    /// GUARD for DEFECT 1: the BARE `/clear` still clears, and a whitespace-only
    /// suffix is not an argument (`slashCommand.args` is trimmed,
    /// slash-commands.ts:246, so `if (commandArgs)` at interactive-mode.ts:4969 is
    /// false). Both must reach `handleClearCommand` (:4972-4974 -> :10232).
    #[tokio::test]
    async fn bare_clear_still_starts_a_new_session() {
        for text in ["/clear", "/clear   ", "/new"] {
            let recorder = Arc::new(RecordingConnection::new());
            let connection: Arc<dyn wire::AgentConnection> = recorder.clone();
            let (send, _receive) = mpsc::channel();
            let command = crate::core::slash_commands::parse_slash_command(text).unwrap();
            let resolved = crate::core::slash_commands::resolve_slash_command(&command);
            assert_eq!(resolved.name, "new");
            assert!(resolved.args.is_empty(), "{text} carries no argument");

            let output =
                run_builtin_command(&connection, &send, text, &resolved.name, &resolved.args)
                    .await
                    .expect("bare /clear clears");
            assert_eq!(
                recorder.only_call("new_session"),
                Vec::<String>::new(),
                "{text} must create a session"
            );
            assert!(
                matches!(output, CommandOutput::Status(ref message) if message == "New session started"),
                "{text} must report the new session, got {output:?}"
            );
        }
    }

    /// GUARD for DEFECT 1: the `/new` argument form still creates the session and
    /// applies both `--name` and the prompt-free default
    /// (interactive-mode.ts:4978-4989 -> :10232-10248).
    #[tokio::test]
    async fn new_with_a_name_still_starts_a_session_and_sets_the_name() {
        let recorder = Arc::new(RecordingConnection::new());
        let connection: Arc<dyn wire::AgentConnection> = recorder.clone();
        let (send, _receive) = mpsc::channel();
        let text = "/new --name scratch";
        let command = crate::core::slash_commands::parse_slash_command(text).unwrap();
        let resolved = crate::core::slash_commands::resolve_slash_command(&command);

        let output = run_builtin_command(&connection, &send, text, &resolved.name, &resolved.args)
            .await
            .expect("a /new with a name succeeds");

        assert_eq!(
            recorder.only_call("new_session"),
            Vec::<String>::new(),
            "/new must create the session"
        );
        assert_eq!(
            recorder.only_call("set_session_name"),
            vec!["scratch".to_string()],
            "/new --name scratch must apply the parsed name"
        );
        assert!(
            matches!(output, CommandOutput::Status(ref message) if message == "New session started"),
            "got {output:?}"
        );
    }

    /// DEFECT 2: `/context` renders the context tree, not the session-stats JSON.
    ///
    /// TypeScript calls `getContextTree()` and renders it with
    /// `formatContextTree(tree, width)` (interactive-mode.ts:9799-9803). The old
    /// port called `get_session_stats` and pretty-printed the raw JSON.
    #[tokio::test]
    async fn context_asks_for_the_context_tree_not_the_session_stats() {
        let recorder = Arc::new(RecordingConnection::new());
        *recorder.context_tree.lock().unwrap() = serde_json::json!({
            "id": "root",
            "label": "Session",
            "status": "running",
            "ownUsage": { "input": 5, "output": 5, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 10,
                          "cost": { "input": 0.0, "output": 0.0, "cacheRead": 0.0, "cacheWrite": 0.0, "total": 0.0 } },
            "totalUsage": { "input": 5, "output": 5, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 10,
                            "cost": { "input": 0.0, "output": 0.0, "cacheRead": 0.0, "cacheWrite": 0.0, "total": 0.0 } },
            "children": []
        });
        let connection: Arc<dyn wire::AgentConnection> = recorder.clone();
        let (send, _receive) = mpsc::channel();

        let output = run_builtin_command(&connection, &send, "/context", "context", "")
            .await
            .expect("the context tree reply succeeds");

        let methods: Vec<String> = recorder.calls().into_iter().map(|(name, _)| name).collect();
        assert_eq!(
            methods,
            vec!["get_context_tree".to_string()],
            "/context must call get_context_tree and nothing else"
        );
        let tree = match output {
            CommandOutput::ContextTree { ref command, .. } => {
                assert_eq!(command, "/context");
                true
            }
            _ => false,
        };
        assert!(tree, "got {output:?}");
        let events = output.into_events();
        let rendered = events
            .iter()
            .find_map(|event| match event {
                HostEvent::ContextTree(value) => Some(value.clone()),
                _ => None,
            })
            .expect("a ContextTree event must be emitted");
        assert!(
            !serde_json::to_string(&rendered)
                .unwrap()
                .contains("\"get_session_stats\""),
            "the raw reply must not be pretty-printed at the user: {rendered}"
        );
    }

    /// The `/context` tree is formatted at `Math.max(60, Math.min(columns - 2, 120))`
    /// (interactive-mode.ts:9800), and the formatter emits the human-readable tree
    /// rather than JSON.
    #[test]
    fn context_tree_width_matches_the_typescript_clamp() {
        crate::modes::interactive::theme::theme::init_theme(Some("prime"), false);
        assert_eq!(context_tree_width(80), 78.0);
        assert_eq!(context_tree_width(200), 120.0);
        assert_eq!(context_tree_width(30), 60.0);
        assert_eq!(context_tree_width(0), 60.0);

        let root = crate::core::context_tree::ContextTreeNode {
            id: "root".into(),
            label: "Session".into(),
            status: "running".into(),
            model: None,
            own_usage: pi_ai::types::Usage::default(),
            total_usage: pi_ai::types::Usage::default(),
            context_usage: None,
            children: Vec::new(),
        };
        let text = crate::modes::interactive::components::context_tree_format::format_context_tree(
            &root,
            context_tree_width(80),
        );
        assert!(
            text.contains("Context") && text.contains("tokens") && text.contains("agent"),
            "the tree formatter must produce the header rows, got {text:?}"
        );
        assert!(
            !text.trim_start().starts_with('{'),
            "the /context output must not be JSON, got {text:?}"
        );
    }

    /// The variant names of a command reply, so an assertion can name what it saw
    /// without a `Debug` impl on the whole `HostEvent` union.
    fn event_names(events: &[HostEvent]) -> Vec<&'static str> {
        events
            .iter()
            .map(|event| match event {
                HostEvent::Connection(_) => "Connection",
                HostEvent::Completed(_) => "Completed",
                HostEvent::Status(_) => "Status",
                HostEvent::ClipboardNotice(_) => "ClipboardNotice",
                HostEvent::Panel(_) => "Panel",
                HostEvent::EchoLocal(_) => "EchoLocal",
                HostEvent::ContextTree(_) => "ContextTree",
                HostEvent::Warning(_) => "Warning",
                HostEvent::ThinkingLevels { .. } => "ThinkingLevels",
                HostEvent::Render => "Render",
                HostEvent::Heartbeats(_, _) => "Heartbeats",
                HostEvent::HeartbeatUpdated(_, _) => "HeartbeatUpdated",
                HostEvent::CloseHeartbeats => "CloseHeartbeats",
                HostEvent::Settings(_) => "Settings",
                HostEvent::Setting(_) => "Setting",
                HostEvent::SettingAccepted(_) => "SettingAccepted",
                HostEvent::Models(_, _, _) => "Models",
                HostEvent::ModelSelected { .. } => "ModelSelected",
                HostEvent::Configuration(_, _, _) => "Configuration",
                HostEvent::BeginLogin(_, _) => "BeginLogin",
                HostEvent::LoginAuth(_, _) => "LoginAuth",
                HostEvent::LoginProgress(_) => "LoginProgress",
                HostEvent::LoginPrompt(_, _, _) => "LoginPrompt",
                HostEvent::LoginFinished(_, _) => "LoginFinished",
                HostEvent::AgentsView => "AgentsView",
                HostEvent::Error(_) => "Error",
                HostEvent::Fullscreen(_) => "Fullscreen",
                _ => "Other",
            })
            .collect()
    }

    /// DEFECT 3: every local command that renders a panel echoes the TYPED command
    /// first (`echoLocalCommand`, interactive-mode.ts:6405-6413), because the panel
    /// then anchors to a visible command instead of floating
    /// (`/session` :4887+9499-9501, `/logs` :4910+9532-9533, `/changelog`
    /// :4926+10016-10021).
    #[tokio::test]
    async fn panel_commands_echo_the_typed_command_before_the_panel() {
        let cases: [(&str, &str); 4] = [
            ("/session", "session"),
            ("/context", "context"),
            ("/logs", "logs"),
            ("/changelog", "changelog"),
        ];
        for (text, name) in cases {
            let recorder = Arc::new(RecordingConnection::new());
            *recorder.context_tree.lock().unwrap() = serde_json::json!({
                "id": "root", "label": "Session", "status": "running",
                "ownUsage": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0,
                              "cost": { "input": 0.0, "output": 0.0, "cacheRead": 0.0, "cacheWrite": 0.0, "total": 0.0 } },
                "totalUsage": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0,
                                "cost": { "input": 0.0, "output": 0.0, "cacheRead": 0.0, "cacheWrite": 0.0, "total": 0.0 } },
                "children": []
            });
            let connection: Arc<dyn wire::AgentConnection> = recorder.clone();
            let (send, _receive) = mpsc::channel();
            let output = run_builtin_command(&connection, &send, text, name, "")
                .await
                .unwrap_or_else(|error| panic!("{text} must succeed: {error}"));

            let events = output.into_events();
            let echoed = events
                .iter()
                .find_map(|event| match event {
                    HostEvent::EchoLocal(value) => Some(value.clone()),
                    _ => None,
                })
                .unwrap_or_else(|| {
                    panic!(
                        "{text} must echo the typed command, got {} events",
                        events.len()
                    )
                });
            assert_eq!(
                echoed, text,
                "{text} must echo the command AS TYPED (interactive-mode.ts:6405-6413)"
            );
            assert!(
                matches!(events.first(), Some(HostEvent::EchoLocal(_))),
                "{text} must echo BEFORE the panel content, got [{:?}]",
                event_names(&events)
            );
            assert!(
                events.len() >= 2,
                "{text} must also emit its panel content, got [{:?}]",
                event_names(&events)
            );
        }
    }

    /// The echo path itself renders the submitted text as the user's own message
    /// (`echoLocalCommand` -> `UserMessageComponent`, interactive-mode.ts:6409-6413).
    #[test]
    fn echo_local_renders_the_typed_command_as_a_user_message() {
        let mode = Rc::new(RefCell::new(stash_mode("host-echo-session")));
        let mut transcript = Transcript::new(mode);
        transcript.echo_local("/logs");
        assert_eq!(transcript.rows.len(), 2, "a spacer then the user message");
        let rendered = transcript.render(80.0).join("\n");
        assert!(
            rendered.contains("/logs"),
            "the typed command must be visible in the transcript, got {rendered:?}"
        );
    }

    /// DEFECT 4: `/fullscreen on|off` sets that exact state, an invalid argument is
    /// a usage error, and the bare form toggles the LIVE value
    /// (interactive-mode.ts:5014-5023, :7522-7537).
    #[test]
    fn fullscreen_arguments_select_the_requested_state() {
        assert_eq!(parse_fullscreen_argument(""), Ok(None));
        assert_eq!(parse_fullscreen_argument("  "), Ok(None));
        assert_eq!(parse_fullscreen_argument("on"), Ok(Some(true)));
        assert_eq!(parse_fullscreen_argument(" OFF "), Ok(Some(false)));
        assert_eq!(parse_fullscreen_argument("yes"), Err(()));
        assert_eq!(parse_fullscreen_argument("tru"), Err(()));
    }

    /// `/fullscreen off` while the live state is already `false` must not flip it
    /// to `true`, and the bare toggle must invert the LIVE field rather than the
    /// persisted setting. A handler that ignored its argument fails both rows.
    #[test]
    fn fullscreen_explicit_argument_beats_the_live_toggle() {
        let mode = Rc::new(RefCell::new(stash_mode("host-fullscreen-session")));
        let tui = Rc::new(RefCell::new(TUI::new(
            Box::new(pi_tui::terminal::ProcessTerminal::new()),
            None,
        )));
        let editor = Rc::new(RefCell::new(CustomEditor::new(
            tui.clone(),
            editor_theme(),
            CustomEditorOptions::default(),
        )));
        let transcript = Rc::new(RefCell::new(Transcript::new(mode.clone())));

        // The persisted value is written before the TTY guard
        // (interactive-mode.ts:7523), so it distinguishes "honour the argument"
        // from "toggle", which the guard would otherwise mask.
        let persisted = |mode: &Rc<RefCell<InteractiveMode>>| {
            mode.borrow()
                .with_settings(|settings| settings.get_fullscreen())
        };

        // `/fullscreen off` with the live state already off must stay off. A
        // handler that ignores its argument and toggles would persist `true`.
        mode.borrow_mut()
            .with_settings_mut(|settings| settings.set_fullscreen(false));
        mode.borrow_mut().fullscreen_enabled = false;
        apply_fullscreen_request(Some(false), &mode, &editor, &tui, &transcript);
        assert!(
            !mode.borrow().fullscreen_enabled,
            "`/fullscreen off` must keep the live state off"
        );
        assert!(
            !persisted(&mode),
            "`/fullscreen off` must persist `false`, not a toggled value"
        );

        // `/fullscreen on` with the live state already on must request `true`. A
        // toggle would request `false` and persist `false`.
        mode.borrow_mut()
            .with_settings_mut(|settings| settings.set_fullscreen(true));
        mode.borrow_mut().fullscreen_enabled = true;
        apply_fullscreen_request(Some(true), &mode, &editor, &tui, &transcript);
        assert!(
            persisted(&mode),
            "`/fullscreen on` must persist `true`, not a toggled value"
        );
        // The test harness has no TTY, so the refusal at
        // interactive-mode.ts:7524-7528 keeps the live state off. That is the
        // specified behaviour, not a failure of the argument handling.
        assert!(
            !mode.borrow().fullscreen_enabled,
            "the non-TTY guard must refuse to enable fullscreen (interactive-mode.ts:7525)"
        );

        // The bare toggle reads the LIVE field. With the setting `true` and the
        // live field `false`, only a live read requests `true`.
        mode.borrow_mut()
            .with_settings_mut(|settings| settings.set_fullscreen(true));
        mode.borrow_mut().fullscreen_enabled = false;
        apply_fullscreen_request(None, &mode, &editor, &tui, &transcript);
        assert!(
            persisted(&mode),
            "the bare toggle must invert the LIVE state, not the persisted setting"
        );
    }

    #[tokio::test]
    async fn execution_mode_refresh_retries_after_later_idle_events() {
        use crate::core::execution_mode::ExecutionMode;
        let mode = Rc::new(RefCell::new(stash_mode("mode-refresh-fixture")));
        mode.borrow_mut().apply_connection_state_snapshot(local::AgentConnectionState {
            session_id: "mode-refresh-fixture".into(), execution_mode: Some(ExecutionMode::Ipython), ..Default::default()
        });
        let recorder = Arc::new(RecordingConnection::new());
        *recorder.state.lock().unwrap() = wire::AgentConnectionState {
            session_id: "mode-refresh-fixture".into(), active_tool_names: vec!["bash".into(), "edit".into()], ..Default::default()
        };
        let connection: Arc<dyn wire::AgentConnection> = recorder;
        let mut refresh = native_state::StateRefresh::new();
        refresh.request(connection.clone(), "mode-refresh-fixture".into());
        refresh.invalidate(); // A later command-result event invalidates the first read.
        tokio::time::sleep(Duration::from_millis(120)).await;
        assert!(!refresh.poll(&mode, "mode-refresh-fixture"));
        assert_eq!(mode.borrow().connection_state.as_ref().unwrap().execution_mode, Some(ExecutionMode::Ipython));
        refresh.reconcile_if_due(connection, "mode-refresh-fixture".into(), &mode.borrow());
        tokio::time::timeout(Duration::from_secs(1), async {
            while !refresh.poll(&mode, "mode-refresh-fixture") { tokio::task::yield_now().await; }
        }).await.unwrap();
        assert_eq!(mode.borrow().connection_state.as_ref().unwrap().execution_mode, Some(ExecutionMode::Direct));
    }

    #[test]
    fn execution_mode_function_key_is_configurable_and_keeps_editor_draft() {
        use crate::core::keybindings::{KeybindingsConfig, KeybindingsManager, KeybindingSetting};
        let _mode = stash_mode("mode-key-fixture");
        let tui = Rc::new(RefCell::new(TUI::new(Box::new(pi_tui::terminal::ProcessTerminal::new()), None)));
        for (setting, key, fires) in [
            (None, "\x1b[17~", true),
            (Some(KeybindingSetting::Single("f8".into())), "\x1b[17~", false),
            (Some(KeybindingSetting::Single("f8".into())), "\x1b[19~", true),
            (Some(KeybindingSetting::List(vec![])), "\x1b[17~", false),
        ] {
            let mut bindings = KeybindingsConfig::new();
            if let Some(setting) = setting { bindings.insert("app.executionMode.toggle".into(), setting); }
            KeybindingsManager::new(bindings, None).install();
            let editor = Rc::new(RefCell::new(CustomEditor::new(tui.clone(), editor_theme(), CustomEditorOptions::default())));
            let actions = Rc::new(RefCell::new(Vec::<InputAction>::new()));
            bind_editor_actions(&editor, &actions);
            editor.borrow_mut().editor_mut().set_text("unfinished draft");
            editor.borrow_mut().handle_input(key);
            assert_eq!(matches!(actions.borrow().as_slice(), [InputAction::ToggleExecutionMode]), fires);
            assert_eq!(editor.borrow().editor().get_text(), "unfinished draft");
        }
        KeybindingsManager::new(Default::default(), None).install();
    }

    #[tokio::test]
    async fn execution_mode_dispatch_requires_host_support() {
        for supported in [false, true] {
            let recorder = Arc::new(RecordingConnection { execution_mode_support: supported, ..Default::default() });
            let connection: Arc<dyn wire::AgentConnection> = recorder.clone();
            let (send, _receive) = mpsc::channel();
            let result = dispatch_submission(&connection, &send, "/mode toggle", true, None).await;
            assert_eq!(result.is_ok(), supported);
            let calls = recorder.calls();
            if supported {
                assert!(calls.iter().any(|(name, args)| name == "prompt" && args[0] == "/mode toggle" && args[1] == "followUp"));
            } else {
                assert!(calls.is_empty());
                assert!(result.unwrap_err().contains("restart the daemon"));
            }
        }
    }

    /// The Ctrl+S byte reaches a real handler: the editor is cleared and a stash is
    /// stored. This is the row-2 path (`interactive-mode.ts:4307`).
    #[test]
    fn ctrl_s_dispatches_to_the_prompt_stash_handler() {
        let session_id = "host-ctrl-s-session";
        let mode = Rc::new(RefCell::new(stash_mode(session_id)));
        let tui = Rc::new(RefCell::new(TUI::new(
            Box::new(pi_tui::terminal::ProcessTerminal::new()),
            None,
        )));
        let editor = Rc::new(RefCell::new(CustomEditor::new(
            tui,
            editor_theme(),
            CustomEditorOptions::default(),
        )));
        let actions = Rc::new(RefCell::new(Vec::<InputAction>::new()));
        bind_editor_actions(&editor, &actions);
        editor
            .borrow_mut()
            .editor_mut()
            .set_text("half-written draft");

        // Ctrl+S decoded by pi-tui is 0x13.
        let stash_key = pi_tui::keybindings::get_keybindings()
            .get_keys("app.prompt.stash")
            .first()
            .cloned()
            .expect("ctrl+s binding");
        assert_eq!(stash_key, "ctrl+s");
        editor.borrow_mut().handle_input("\u{13}");

        let queued = actions.borrow();
        assert!(
            matches!(queued.as_slice(), [InputAction::PromptStash]),
            "Ctrl+S must queue the stash action, got {queued:?}"
        );
        drop(queued);
        actions.borrow_mut().clear();

        handle_prompt_stash_action(&mode, &editor, session_id);

        assert_eq!(editor.borrow().editor().get_text(), "");
        let stored = crate::modes::interactive::prompt_stash_state::PromptStashSession::open(
            crate::modes::interactive::prompt_stash_state::shared_prompt_stash_store(),
            session_id,
        )
        .state();
        assert_eq!(
            stored.stash.map(|stash| stash.text),
            Some("half-written draft".to_string())
        );
    }

    /// A real Ctrl+S round trip preserves the collapsed marker text and the paste
    /// table, so the restore is faithful (`interactive-mode.ts:4343-4356`, :4405-4410;
    /// test `interactive-mode-prompt-stash.test.ts:433-458`).
    #[test]
    fn ctrl_s_round_trip_keeps_the_collapsed_markers_and_paste_table() {
        let session_id = "host-paste-round-trip";
        let mode = Rc::new(RefCell::new(stash_mode(session_id)));
        let tui = Rc::new(RefCell::new(TUI::new(
            Box::new(pi_tui::terminal::ProcessTerminal::new()),
            None,
        )));
        let editor = Rc::new(RefCell::new(CustomEditor::new(
            tui,
            editor_theme(),
            CustomEditorOptions::default(),
        )));

        // A bracketed paste above the 10-line threshold collapses to a marker.
        let pasted = (1..=12)
            .map(|line| format!("line {line}"))
            .collect::<Vec<_>>()
            .join("\n");
        pi_tui::tui::Component::handle_input(
            &mut *editor.borrow_mut().editor_mut(),
            &format!("\u{1b}[200~{pasted}\u{1b}[201~"),
        );
        assert_eq!(
            editor.borrow().editor().get_text(),
            "[paste #1 +12 lines]",
            "the draft keeps the collapsed marker"
        );

        handle_prompt_stash_action(&mode, &editor, session_id);
        assert_eq!(
            editor.borrow().editor().get_text(),
            "",
            "stash clears the editor"
        );

        let session = crate::modes::interactive::prompt_stash_state::PromptStashSession::open(
            crate::modes::interactive::prompt_stash_state::shared_prompt_stash_store(),
            session_id,
        );
        let stored = session.state().stash.expect("stash");
        assert_eq!(stored.text, "[paste #1 +12 lines]");
        assert_eq!(stored.expanded_text.as_deref(), Some(pasted.as_str()));
        assert_eq!(
            stored
                .paste_snapshot
                .as_ref()
                .map(|snapshot| snapshot.paste_counter),
            Some(1)
        );

        let restored = session
            .restore_prompt_stash_if_editor_empty(None, "", true)
            .expect("restore");
        apply_prompt_stash_outcome(&mode, &editor, &restored);

        assert_eq!(editor.borrow().editor().get_text(), "[paste #1 +12 lines]");
        assert_eq!(
            editor.borrow().editor().get_expanded_text(),
            pasted,
            "the paste table is restored, so the marker expands again"
        );
    }

    fn compaction_notice_end(error: Option<&str>, committed: bool, aborted: bool) -> wire::AgentConnectionSessionEvent {
        wire::AgentConnectionSessionEvent::CompactionEnd {
            reason: "threshold".into(),
            result: committed.then(|| wire::CompactionResultSummary::default()),
            aborted,
            will_retry: false,
            error_message: error.map(str::to_string),
            error_severity: None,
            custom_instructions: None,
        }
    }

    #[test]
    fn compaction_notice_success_removes_only_owned_failure_and_preserves_draft_feedback() {
        let mode = Rc::new(RefCell::new(stash_mode("compaction-notice-recovered")));
        mode.borrow_mut().apply_connection_state_snapshot(local::AgentConnectionState {
            session_id: "one".into(), ..Default::default()
        });
        let transcript = Rc::new(RefCell::new(Transcript::new(mode.clone())));
        let failure = "Auto-compaction failed: temporary provider error";
        apply_event(&mode, &transcript, compaction_notice_end(Some(failure), false, false));
        mode.borrow_mut().show_error(failure); // Same text, but not a compaction-owned notice.
        mode.borrow_mut().show_error("New unrelated error");
        mode.borrow_mut().show_warning("Keep this warning");
        mode.borrow_mut().show_status("Keep this status", "dim");
        *mode.borrow().restored_draft_notice.borrow_mut() = Some("draft".into());
        assert_eq!(transcript.borrow_mut().render(100.0).join("\n").matches(failure).count(), 2);

        apply_event(&mode, &transcript, compaction_notice_end(None, true, false));
        let rendered = transcript.borrow_mut().render(100.0).join("\n");
        assert_eq!(rendered.matches(failure).count(), 1);
        for preserved in ["New unrelated error", "Keep this warning", "Keep this status", "Draft restored"] {
            assert!(rendered.contains(preserved), "missing {preserved}: {rendered}");
        }
        mode.borrow_mut().show_status("Status after recovery", "dim");
        assert!(transcript.borrow_mut().render(100.0).join("\n").contains("Status after recovery"));
    }

    #[test]
    fn compaction_notice_requires_committed_success_not_start_idle_or_abort() {
        let mode = Rc::new(RefCell::new(stash_mode("compaction-notice-attempts")));
        mode.borrow_mut().apply_connection_state_snapshot(local::AgentConnectionState {
            session_id: "one".into(), ..Default::default()
        });
        let transcript = Rc::new(RefCell::new(Transcript::new(mode.clone())));
        let failure = "Auto-compaction failed: preserve until recovered";
        apply_event(&mode, &transcript, compaction_notice_end(Some(failure), false, false));
        apply_event(&mode, &transcript, wire::AgentConnectionSessionEvent::CompactionStart {
            reason: "manual".into(), custom_instructions: None,
        });
        assert!(transcript.borrow_mut().render(100.0).join("\n").contains(failure));
        for (committed, aborted) in [(false, true), (false, false), (true, true)] {
            apply_event(&mode, &transcript, compaction_notice_end(None, committed, aborted));
            assert!(transcript.borrow_mut().render(100.0).join("\n").contains(failure));
        }
        mode.borrow_mut().apply_connection_state_snapshot(local::AgentConnectionState {
            session_id: "one".into(), is_compacting: false, ..Default::default()
        });
        assert!(transcript.borrow_mut().render(100.0).join("\n").contains(failure));
        let mut recovered = compaction_notice_end(None, true, false);
        if let wire::AgentConnectionSessionEvent::CompactionEnd { reason, .. } = &mut recovered {
            *reason = "manual".into();
        }
        apply_event(&mode, &transcript, recovered);
        assert!(!transcript.borrow_mut().render(100.0).join("\n").contains(failure));
        apply_event(&mode, &transcript, compaction_notice_end(Some("New compaction failure"), false, false));
        assert!(transcript.borrow_mut().render(100.0).join("\n").contains("New compaction failure"));
    }

    #[test]
    fn compaction_notice_is_session_scoped_and_does_not_remove_other_errors_on_switch() {
        let mode = Rc::new(RefCell::new(stash_mode("compaction-notice-scope")));
        mode.borrow_mut().apply_connection_state_snapshot(local::AgentConnectionState {
            session_id: "one".into(), ..Default::default()
        });
        let transcript = Rc::new(RefCell::new(Transcript::new(mode.clone())));
        apply_event(&mode, &transcript, compaction_notice_end(Some("Old session compaction failed"), false, false));
        mode.borrow_mut().show_error("Unrelated retained error");
        mode.borrow_mut().apply_connection_state_snapshot(local::AgentConnectionState {
            session_id: "two".into(), ..Default::default()
        });
        let rendered = transcript.borrow_mut().render(100.0).join("\n");
        assert!(!rendered.contains("Old session compaction failed"));
        assert!(rendered.contains("Unrelated retained error"));
        apply_event(&mode, &transcript, compaction_notice_end(Some("New session compaction failed"), false, false));
        mode.borrow_mut().chat_container.add_child(Box::new(CompactionNotice {
            session_id: Some("foreign".into()), text: Text::new("Foreign session notice", 1, 0),
        }));
        apply_event(&mode, &transcript, compaction_notice_end(None, true, false));
        let rendered = transcript.borrow_mut().render(100.0).join("\n");
        assert!(!rendered.contains("New session compaction failed"));
        assert!(rendered.contains("Foreign session notice"));
        assert!(rendered.contains("Unrelated retained error"));
    }

    #[test]
    fn draft_restore_notice_is_transient_and_does_not_clear_other_statuses() {
        let mode = Rc::new(RefCell::new(stash_mode("host-on-open-notice")));
        mode.borrow_mut().show_warning("Keep this warning");
        mode.borrow_mut().show_status("Compaction finished", "dim");
        let editor = Rc::new(RefCell::new(CustomEditor::new(
            Rc::new(RefCell::new(TUI::new(Box::new(pi_tui::terminal::ProcessTerminal::new()), None))),
            editor_theme(), CustomEditorOptions::default(),
        )));
        bind_draft_restore_notice(&mode.borrow(), &mut editor.borrow_mut());
        let outcome = PromptStashOutcome {
            editor: PromptStashEditorEffect::SetText { text: "restored draft".into(), paste_snapshot: None },
            status: Some("Restored stashed prompt"),
        };
        let mut transcript = Transcript::new(mode.clone());
        for replacement in ["", "replacement draft"] {
            apply_prompt_stash_outcome(&mode, &editor, &outcome);
            let rendered = transcript.render(80.0).join("\n");
            assert_eq!(rendered.matches("Draft restored").count(), 1);
            assert!(rendered.contains("Keep this warning"));
            assert!(rendered.contains("Compaction finished"));
            editor.borrow_mut().editor_mut().set_text(replacement);
            let rendered = transcript.render(80.0).join("\n");
            assert!(!rendered.contains("Draft restored"));
            assert!(rendered.contains("Keep this warning"));
            assert!(rendered.contains("Compaction finished"));
        }
        apply_prompt_stash_outcome(&mode, &editor, &outcome);
        editor.borrow_mut().handle_input("x");
        assert!(!transcript.render(80.0).join("\n").contains("Draft restored"));
        assert_eq!(editor.borrow().editor().get_text(), "restored draftx");
    }

    #[test]
    fn draft_restore_notice_clears_on_session_change_without_discarding_the_draft() {
        let mode = Rc::new(RefCell::new(stash_mode("host-notice-session")));
        mode.borrow_mut().apply_connection_state_snapshot(local::AgentConnectionState {
            session_id: "one".into(), ..Default::default()
        });
        *mode.borrow().restored_draft_notice.borrow_mut() = Some("keep draft".into());
        mode.borrow_mut().apply_connection_state_snapshot(local::AgentConnectionState {
            session_id: "one".into(), ..Default::default()
        });
        assert!(mode.borrow().restored_draft_notice.borrow().is_some());
        mode.borrow_mut().apply_connection_state_snapshot(local::AgentConnectionState {
            session_id: "two".into(), ..Default::default()
        });
        assert!(mode.borrow().restored_draft_notice.borrow().is_none());
    }

    #[test]
    fn session_info_changed_updates_the_session_name_state() {
        let mode = Rc::new(RefCell::new(stash_mode("rename-state")));
        mode.borrow_mut().apply_connection_state_snapshot(local::AgentConnectionState {
            session_id: "one".into(), ..Default::default()
        });
        let transcript = Rc::new(RefCell::new(Transcript::new(mode.clone())));
        apply_event(
            &mode,
            &transcript,
            wire::AgentConnectionSessionEvent::SessionInfoChanged { name: Some("renamed".into()) },
        );
        assert_eq!(
            mode.borrow().connection_state.as_ref().and_then(|s| s.session_name.clone()).as_deref(),
            Some("renamed")
        );
        apply_event(
            &mode,
            &transcript,
            wire::AgentConnectionSessionEvent::SessionInfoChanged { name: None },
        );
        assert_eq!(
            mode.borrow().connection_state.as_ref().and_then(|s| s.session_name.clone()).as_deref(),
            None
        );
    }

    /// An auto-stash for the agents view survives a reopen: the handoff stashes the
    /// draft (`interactive-mode.ts:7122`), the reopen restores it on open
    /// (`interactive-mode.ts:1628-1630`).
    #[test]
    fn agents_view_handoff_stashes_and_the_reopen_restores() {
        let session_id = "host-handoff-session";
        let mode = Rc::new(RefCell::new(stash_mode(session_id)));
        let tui = Rc::new(RefCell::new(TUI::new(
            Box::new(pi_tui::terminal::ProcessTerminal::new()),
            None,
        )));
        let editor = Rc::new(RefCell::new(CustomEditor::new(
            tui,
            editor_theme(),
            CustomEditorOptions::default(),
        )));
        editor
            .borrow_mut()
            .editor_mut()
            .set_text("draft typed before leaving");

        stash_editor_draft_for_agents_view(&mode, &editor, session_id);

        let store = crate::modes::interactive::prompt_stash_state::shared_prompt_stash_store();
        let session = crate::modes::interactive::prompt_stash_state::PromptStashSession::open(
            store, session_id,
        );
        assert!(
            session.restore_on_open_pending(),
            "handoff marks restoreOnOpen"
        );

        // The reopened view restores the draft into its own empty editor.
        let reopened_editor = Rc::new(RefCell::new(CustomEditor::new(
            Rc::new(RefCell::new(TUI::new(
                Box::new(pi_tui::terminal::ProcessTerminal::new()),
                None,
            ))),
            editor_theme(),
            CustomEditorOptions::default(),
        )));
        let restored = session
            .restore_prompt_stash_if_editor_empty(None, "", true)
            .expect("the auto-stash restores");
        apply_prompt_stash_outcome(&mode, &reopened_editor, &restored);
        assert_eq!(
            reopened_editor.borrow().editor().get_text(),
            "draft typed before leaving"
        );
        assert!(!session.restore_on_open_pending(), "the stash is consumed");
    }

    /// One login attempt's slot, shown in its own overlay.
    ///
    /// `new LoginDialogComponent(...)` + `showFullPaneOverlay(...)` is what
    /// starts an attempt (auth-flows.ts:512-522), and each dialog owns an
    /// independent `abortController` (login-dialog.ts:77).
    fn login_slot(
        ui: &Rc<RefCell<TUI>>,
        generation: LoginGeneration,
        token: tokio_util::sync::CancellationToken,
    ) -> LoginSlot {
        let dialog = Rc::new(RefCell::new(LoginDialogComponent::new(
            ui.clone(),
            "anthropic",
            Box::new(|_, _| {}),
            None,
            None,
        )));
        let overlay = ui
            .borrow_mut()
            .show_overlay(dialog.clone(), Default::default());
        LoginSlot {
            generation,
            token,
            overlay,
            dialog,
        }
    }

    fn login_test_ui() -> Rc<RefCell<TUI>> {
        crate::modes::interactive::theme::theme::init_theme(Some("prime"), false);
        crate::core::keybindings::KeybindingsManager::new(Default::default(), None).install();
        Rc::new(RefCell::new(TUI::new(
            Box::new(pi_tui::terminal::ProcessTerminal::new()),
            None,
        )))
    }

    /// A superseded login that finishes late must not cancel or hide the dialog
    /// that replaced it.
    ///
    /// TypeScript scopes cancellation to one `LoginDialogComponent`: its
    /// `abortController` is per-instance (login-dialog.ts:77) and `cancel()`
    /// aborts only that controller (:139-146), while the overlay hidden on
    /// completion is the handle this dialog was shown in (auth-flows.ts:524-527).
    /// The port's old shared `login_cancel`/`overlay` slots let a stale
    /// completion take the newer dialog down with it; this is that regression.
    #[test]
    fn a_stale_login_does_not_cancel_or_hide_the_newer_login() {
        let ui = login_test_ui();
        let mut logins = LoginCoordinator::default();

        // Login A starts and owns the host's login state.
        let a_generation = logins.begin();
        let a_token = tokio_util::sync::CancellationToken::new();
        logins.install(login_slot(&ui, a_generation, a_token.clone()));
        assert_eq!(logins.current_generation(), Some(a_generation));

        // Login B starts: `begin()` retires A exactly as A's own `cancel()` would
        // (login-dialog.ts:140), so A's controller and overlay go away HERE.
        let b_generation = logins.begin();
        assert_ne!(
            a_generation, b_generation,
            "each attempt gets its own identity"
        );
        let b_token = tokio_util::sync::CancellationToken::new();
        logins.install(login_slot(&ui, b_generation, b_token.clone()));
        assert!(
            a_token.is_cancelled(),
            "the retired dialog aborts its own sign-in"
        );
        ui.borrow_mut().sync_overlays();
        assert!(
            ui.borrow().has_overlay(),
            "only B's overlay is on screen, and it is"
        );

        // A completes late. It no longer owns the login state.
        let applied = apply_login_finished(&mut logins, a_generation);

        // The whole point: B keeps its own controller and its own overlay.
        assert!(
            !b_token.is_cancelled(),
            "the stale login stole the newer dialog's cancellation"
        );
        ui.borrow_mut().sync_overlays();
        assert!(
            ui.borrow().has_overlay(),
            "the stale login hid the newer dialog's overlay"
        );
        assert!(
            !applied,
            "a superseded generation must report that it was ignored"
        );
        assert_eq!(
            logins.current_generation(),
            Some(b_generation),
            "B still owns the login state"
        );
    }

    /// Over-correction guard: the login that still owns the state must cancel its
    /// own token and hide its own overlay, and report that it was applied.
    ///
    /// `cancel()` aborts this dialog's controller (login-dialog.ts:139-146) and
    /// completion hides this dialog's overlay (auth-flows.ts:524-527).
    #[test]
    fn the_current_login_still_cancels_and_hides_its_own_dialog() {
        let ui = login_test_ui();
        let mut logins = LoginCoordinator::default();

        let generation = logins.begin();
        let token = tokio_util::sync::CancellationToken::new();
        logins.install(login_slot(&ui, generation, token.clone()));
        ui.borrow_mut().sync_overlays();
        assert!(ui.borrow().has_overlay(), "its overlay is on screen");

        let applied = apply_login_finished(&mut logins, generation);
        assert!(
            token.is_cancelled(),
            "the current dialog aborts its own sign-in"
        );
        ui.borrow_mut().sync_overlays();
        assert!(
            !ui.borrow().has_overlay(),
            "the current dialog hides its own overlay"
        );
        assert!(applied, "the current generation is applied");
        assert_eq!(logins.current_generation(), None, "the slot is consumed");
        assert!(!logins.is_active(), "no login is in flight");
    }
    // ==================== T14 terminal-parity suite ====================

    /// Minimal recording `Terminal` so command paths can be observed without a
    /// real console. The shared event list survives the `Box` move into the TUI.
    struct RecordingTerminal {
        events: Rc<RefCell<Vec<&'static str>>>,
    }

    impl RecordingTerminal {
        fn new() -> (Self, Rc<RefCell<Vec<&'static str>>>) {
            let events = Rc::new(RefCell::new(Vec::<&'static str>::new()));
            (Self { events: events.clone() }, events)
        }
    }

    impl pi_tui::terminal::Terminal for RecordingTerminal {
        fn start(&mut self, _on_input: Box<dyn Fn(String)>, _on_resize: Box<dyn Fn()>) {
            self.events.borrow_mut().push("start");
        }
        fn stop(&mut self, _options: pi_tui::terminal::TerminalStopOptions) {
            self.events.borrow_mut().push("stop");
        }
        fn drain_input(&mut self, _max_ms: u64, _idle_ms: u64) {
            self.events.borrow_mut().push("drain");
        }
        fn write(&mut self, _data: &str) {}
        fn columns(&self) -> usize { 80 }
        fn rows(&self) -> usize { 24 }
        fn kitty_protocol_active(&self) -> bool { false }
        fn move_by(&mut self, _lines: i64) {}
        fn hide_cursor(&mut self) {}
        fn show_cursor(&mut self) {}
        fn clear_line(&mut self) {}
        fn clear_from_cursor(&mut self) {}
        fn clear_screen(&mut self) {}
        fn enter_alt_screen(&mut self) {}
        fn leave_alt_screen(&mut self) {}
        fn alt_screen_active(&self) -> bool { false }
        fn set_mouse_tracking(&mut self, _enabled: bool) {}
        fn mouse_tracking_active(&self) -> bool { false }
        fn set_title(&mut self, _title: &str) {}
        fn set_progress(&mut self, _active: bool) {}
    }

    /// A-01: a non-self `/update` during an active turn must be refused with the
    /// TypeScript warning (interactive-mode.ts:5001-5012) instead of draining,
    /// stopping the terminal and launching the updater mid-turn.
    #[tokio::test]
    async fn t14_a01_nonself_refresh_command_blocked_while_busy() {
        let recorder = Arc::new(RecordingConnection::new());
        let connection: Arc<dyn wire::AgentConnection> = recorder.clone();
        let mode = Rc::new(RefCell::new(stash_mode("t14-a01-gate")));
        mode.borrow_mut().apply_connection_state_snapshot(local::AgentConnectionState {
            is_streaming: true,
            cwd: ".".to_string(),
            ..Default::default()
        });
        let (terminal, terminal_events) = RecordingTerminal::new();
        let ui = Rc::new(RefCell::new(TUI::new(Box::new(terminal), None)));
        let seam = crate::main_entry::InteractiveModeSeamOptions {
            daemon_socket_path: None,
            migrated_providers: Vec::new(),
            model_fallback_message: None,
            initial_message: None,
            initial_images: None,
            initial_messages: Vec::new(),
            verbose: false,
            return_to_agents_view: false,
            session_depth: None,
            session_has_children: false,
            connection: None,
            runtime: None,
        };
        // Not self-update arguments, so the TypeScript gate applies.
        let args = vec!["--extensions".to_string()];
        let result = native_commands::update(&args, &seam, &mode, &ui, &connection).await;
        assert!(result.is_ok(), "the busy gate must refuse cleanly: {result:?}");
        let rendered = {
            let mode = mode.borrow();
            mode.get_main_view_containers()
                .into_iter()
                .flat_map(|container| local::Component::render(container, 80))
                .collect::<Vec<String>>()
                .join("\n")
        };
        assert!(
            rendered.contains("Wait for the current work to finish before updating."),
            "expected the TypeScript busy warning on the transcript, got:\n{rendered}"
        );
        assert!(
            terminal_events.borrow().is_empty(),
            "the busy gate must not drain/stop/restart the terminal: {:?}",
            terminal_events.borrow()
        );
    }

    /// A-12: an unknown `/effort` level is ERROR-styled like the TypeScript
    /// `showError` (interactive-mode.ts:8279), not warning-styled.
    #[tokio::test]
    async fn t14_a12_effort_unknown_level_reports_error_style() {
        use pi_agent_core::types::ThinkingLevel;
        let recorder = Arc::new(RecordingConnection::new());
        {
            let mut state = recorder.state.lock().unwrap();
            state.thinking_level = ThinkingLevel::Low;
            state.available_thinking_levels =
                vec![ThinkingLevel::Off, ThinkingLevel::Minimal, ThinkingLevel::Low, ThinkingLevel::Medium, ThinkingLevel::High];
        }
        let connection: Arc<dyn wire::AgentConnection> = recorder.clone();
        let (send, _receive) = mpsc::channel();
        let output = run_builtin_command(&connection, &send, "/effort bogus", "effort", "bogus")
            .await
            .expect("effort command must succeed");
        match output {
            CommandOutput::Error(message) => {
                assert!(message.contains("Unknown thinking level 'bogus'"), "{message}");
            }
            other => panic!("expected error-styled output (TypeScript showError), got {other:?}"),
        }
    }


    /// A-02 fix: a second Escape inside the window clears the editor
    /// (interactive-mode.ts:6924-6953).
    #[tokio::test]
    async fn t14_a02_escape_repeat_second_press_clears_the_editor() {
        let mode = Rc::new(RefCell::new(stash_mode("t14-a02-clear")));
        let (terminal, _events) = RecordingTerminal::new();
        let ui = Rc::new(RefCell::new(TUI::new(Box::new(terminal), None)));
        let editor = Rc::new(RefCell::new(CustomEditor::new(
            ui.clone(),
            editor_theme(),
            CustomEditorOptions::default(),
        )));
        editor.borrow_mut().editor_mut().set_text("draft text");
        let connection: Arc<dyn wire::AgentConnection> = Arc::new(RecordingConnection::new());
        let (send, _receive) = mpsc::channel();
        assert!(!escape_repeat_step(&mode, &editor, &ui, &connection, &send).await);
        assert!(escape_repeat_step(&mode, &editor, &ui, &connection, &send).await);
        assert_eq!(editor.borrow().editor().get_text(), "");
    }

    #[tokio::test]
    async fn emergency_escape_suspends_queued_only_work() {
        let mut mode = stash_mode("emergency-queued-only");
        mode.apply_connection_state_snapshot(local::AgentConnectionState {
            session_actions: local::SessionActionSnapshot {
                steering: vec!["accepted steer".into()], queued_count: 1,
                ..Default::default()
            }, ..Default::default()
        });
        assert!(mode.has_interruptible_work());
        let recorder = Arc::new(RecordingConnection::new());
        let connection: Arc<dyn wire::AgentConnection> = recorder.clone();
        interrupt_active_work(&connection, InterruptActivity::from_mode(&mode)).await.unwrap();
        assert_eq!(recorder.calls(), vec![("abort".into(), Vec::new())]);
    }

    #[tokio::test]
    async fn emergency_escape_aborts_preparing_turn_without_clearing_queue() {
        let mut mode = stash_mode("emergency-escape-preparing");
        mode.apply_connection_state_snapshot(local::AgentConnectionState {
            session_actions: local::SessionActionSnapshot {
                active: Some("preparing".into()),
                steering: vec!["keep my queued message".into()],
                queued_count: 1,
                ..Default::default()
            },
            ..Default::default()
        });
        assert!(mode.has_interruptible_work());
        assert!(!mode.is_agent_streaming());
        let recorder = Arc::new(RecordingConnection::new());
        let connection: Arc<dyn wire::AgentConnection> = recorder.clone();
        interrupt_active_work(&connection, InterruptActivity::from_mode(&mode)).await.unwrap();
        assert_eq!(recorder.calls(), vec![("abort".into(), Vec::new())]);
    }

    #[tokio::test]
    async fn emergency_escape_targets_each_busy_owner_but_not_idle_sessions() {
        let recorder = Arc::new(RecordingConnection::new());
        let connection: Arc<dyn wire::AgentConnection> = recorder.clone();
        let mut mode = stash_mode("emergency-escape-owners");
        interrupt_active_work(&connection, InterruptActivity::from_mode(&mode)).await.unwrap();
        assert!(recorder.calls().is_empty());
        mode.apply_connection_state_snapshot(local::AgentConnectionState {
            is_streaming: true, is_compacting: true, is_bash_running: true, retry_attempt: 1.0,
            ..Default::default()
        });
        interrupt_active_work(&connection, InterruptActivity::from_mode(&mode)).await.unwrap();
        let names: Vec<_> = recorder.calls().into_iter().map(|(name, _)| name).collect();
        assert_eq!(names, ["abort", "abort_retry", "abort_compaction", "abort_branch_summary", "abort_bash"]);
    }

    /// A-02 fix: outside the 500ms window the second press re-arms.
    #[tokio::test]
    async fn t14_a02_escape_repeat_outside_the_window_re_arms() {
        let mode = Rc::new(RefCell::new(stash_mode("t14-a02-window")));
        let (terminal, _events) = RecordingTerminal::new();
        let ui = Rc::new(RefCell::new(TUI::new(Box::new(terminal), None)));
        let editor = Rc::new(RefCell::new(CustomEditor::new(
            ui.clone(),
            editor_theme(),
            CustomEditorOptions::default(),
        )));
        editor.borrow_mut().editor_mut().set_text("kept draft");
        let connection: Arc<dyn wire::AgentConnection> = Arc::new(RecordingConnection::new());
        let (send, _receive) = mpsc::channel();
        assert!(!escape_repeat_step(&mode, &editor, &ui, &connection, &send).await);
        tokio::time::sleep(std::time::Duration::from_millis(550)).await;
        assert!(!escape_repeat_step(&mode, &editor, &ui, &connection, &send).await);
        assert_eq!(editor.borrow().editor().get_text(), "kept draft");
    }

    /// A-02 fix: with a streaming turn and an empty editor the repeat opens the
    /// tree flow.
    #[tokio::test]
    async fn t14_a02_escape_repeat_second_press_opens_the_tree() {
        let mode = Rc::new(RefCell::new(stash_mode("t14-a02-tree")));
        mode.borrow_mut().apply_connection_state_snapshot(local::AgentConnectionState {
            is_streaming: true,
            ..Default::default()
        });
        let (terminal, _events) = RecordingTerminal::new();
        let ui = Rc::new(RefCell::new(TUI::new(Box::new(terminal), None)));
        let editor = Rc::new(RefCell::new(CustomEditor::new(
            ui.clone(),
            editor_theme(),
            CustomEditorOptions::default(),
        )));
        let recorder = Arc::new(RecordingConnection::new());
        let connection: Arc<dyn wire::AgentConnection> = recorder.clone();
        let (send, receive) = mpsc::channel();
        assert!(!escape_repeat_step(&mode, &editor, &ui, &connection, &send).await);
        assert!(escape_repeat_step(&mode, &editor, &ui, &connection, &send).await);
        let status = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if let Ok(event) = receive.try_recv() { break event; }
                tokio::task::yield_now().await;
            }
        }).await.expect("the tree dispatch must answer");
        match status {
            HostEvent::Status(text) => assert_eq!(text, "No entries in session"),
            other => panic!(
                "expected the tree flow status, got {:?}",
                event_names(std::slice::from_ref(&other))
            ),
        }
    }

    #[tokio::test]
    async fn escape_repeat_nonempty_tree_does_not_block_the_input_owner() {
        let mode = Rc::new(RefCell::new(stash_mode("escape-tree-owner")));
        let (terminal, _) = RecordingTerminal::new();
        let ui = Rc::new(RefCell::new(TUI::new(Box::new(terminal), None)));
        let editor = Rc::new(RefCell::new(CustomEditor::new(
            ui.clone(), editor_theme(), CustomEditorOptions::default(),
        )));
        let recorder = Arc::new(RecordingConnection::new());
        *recorder.session_tree.lock().unwrap() = serde_json::from_value(serde_json::json!({
            "tree": [{"entry": {"type": "session_info", "id": "entry", "parentId": null,
                "timestamp": "2026-09-17T00:00:00Z", "name": "large chat"}, "children": []}],
            "leafId": "entry"
        })).unwrap();
        let connection: Arc<dyn wire::AgentConnection> = recorder.clone();
        let (send, receive) = mpsc::channel();
        assert!(!escape_repeat_step(&mode, &editor, &ui, &connection, &send).await);
        assert!(tokio::time::timeout(Duration::from_millis(500),
            escape_repeat_step(&mode, &editor, &ui, &connection, &send),
        ).await.expect("input owner must return before the tree dialog is answered"));
        // The owner can still edit input, then display and cancel the dialog.
        editor.borrow_mut().editor_mut().set_text("still responsive");
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                match receive.try_recv() {
                    Ok(HostEvent::CommandDialog(native_commands::Dialog::Tree(_, _, reply))) => {
                        assert_eq!(editor.borrow().editor().get_text(), "still responsive");
                        let _ = reply.send(None);
                    }
                    Ok(HostEvent::Completed(result)) => { result.unwrap(); break; }
                    _ => tokio::task::yield_now().await,
                }
            }
        }).await.expect("cancelled tree must settle");
        recorder.only_call("get_session_tree");
    }

    /// A-06 fix: only lines that reach the model enter the up-arrow history.
    #[test]
    fn t14_a06_history_records_only_lines_that_reach_the_model() {
        for (text, expected) in [
            ("fix the login flow", true),
            ("!ls -la", true),
            ("/compact keep the plan", true),
            ("/compact", true),
            ("/telegram status", true),
            ("/no-such-command hi", true),
            ("/model", false),
            ("/btw side question", false),
            ("/quit", false),
            ("/reload", false),
            ("/new --name x -- hi", false),
            ("/effort high", false),
        ] {
            assert_eq!(should_record_prompt_history(text), expected, "{text}");
        }
    }

    /// A-08 fix: the reload arm asks the owner loop to reconfigure the
    /// autocomplete provider and refetch the catalogue.
    #[tokio::test]
    async fn t14_a08_reload_refreshes_the_command_catalogue() {
        let recorder = Arc::new(RecordingConnection::new());
        let connection: Arc<dyn wire::AgentConnection> = recorder.clone();
        let (send, receive) = mpsc::channel();
        let output = run_builtin_command(&connection, &send, "/reload", "reload", "")
            .await
            .expect("reload must succeed");
        let _events = output.into_events();
        // The reload arm first resets extensions, then asks the owner loop to
        // reconfigure the autocomplete provider.
        let deadline8 = std::time::Instant::now() + std::time::Duration::from_secs(5);
        let reconfigured = loop {
            let remaining = deadline8.saturating_duration_since(std::time::Instant::now());
            match receive.recv_timeout(remaining) {
                Ok(event) if matches!(event, HostEvent::ReconfigureAutocomplete) => break true,
                Ok(_) => continue,
                Err(_) => break false,
            }
        };
        assert!(
            reconfigured,
            "reload must ask the owner loop to reconfigure the autocomplete provider"
        );
        assert!(recorder.calls().iter().any(|(name, _)| name == "reload"));
    }

    /// A-13 fix: the new-session prompt reaches the owner loop verbatim.
    #[tokio::test]
    async fn t14_a13_new_with_prompt_prompts_verbatim() {
        let recorder = Arc::new(RecordingConnection::new());
        let connection: Arc<dyn wire::AgentConnection> = recorder.clone();
        let (send, receive) = mpsc::channel();
        let output = run_builtin_command(
            &connection,
            &send,
            "/new --name t14 -- /model",
            "new",
            "--name t14 -- /model",
        )
        .await
        .expect("new command must succeed");
        let events = output.into_events();
        // A new session first publishes its authoritative snapshot (including
        // footer state), then sends the verbatim prompt to the owner loop.
        let snapshot_event = receive
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("the new-session snapshot must precede its prompt");
        assert!(matches!(snapshot_event, HostEvent::RefreshSnapshot(_)));
        let prompt_event = receive
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("the new-session prompt must be handed to the owner loop");
        match prompt_event {
            HostEvent::PromptSession { text } => assert_eq!(text, "/model"),
            other => panic!(
                "the new-session prompt must reach the owner loop verbatim, got {:?}",
                event_names(std::slice::from_ref(&other))
            ),
        }
        assert!(
            !events.iter().any(|event| matches!(event, HostEvent::Models(_, _, _))),
            "the prompt text must never re-enter the dispatcher, got {:?}",
            event_names(&events)
        );
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        let drained: Vec<HostEvent> = receive.try_iter().collect();
        assert!(
            !drained.iter().any(|event| matches!(event, HostEvent::Models(_, _, _))),
            "no model picker may open for the prompt text: {:?}",
            event_names(&drained)
        );
        assert!(recorder.calls().iter().any(|(name, _)| name == "new_session"));
        assert!(
            recorder
                .calls()
                .iter()
                .any(|(name, args)| name == "set_session_name" && *args == ["t14"]),
            "the session name must be applied: {:?}",
            recorder.calls()
        );
    }

    /// A-13 fix: the owner loop prompts the text verbatim and records history.
    #[tokio::test]
    async fn t14_a13_prompt_session_prompts_verbatim_without_rereading() {
        let recorder = Arc::new(RecordingConnection::new());
        let connection: Arc<dyn wire::AgentConnection> = recorder.clone();
        let mode = Rc::new(RefCell::new(stash_mode("t14-a13-loop")));
        let (terminal, _events) = RecordingTerminal::new();
        let ui = Rc::new(RefCell::new(TUI::new(Box::new(terminal), None)));
        let editor = Rc::new(RefCell::new(CustomEditor::new(
            ui.clone(),
            editor_theme(),
            CustomEditorOptions::default(),
        )));
        let (send, _receive) = mpsc::channel();
        handle_prompt_session(&mode, &editor, &connection, &send, "/model");
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        let prompt = recorder.only_call("prompt");
        assert_eq!(prompt[0], "/model", "the prompt must reach the model verbatim");
        assert_eq!(prompt[1], "steer");
        assert!(
            editor.borrow().editor().get_history().iter().any(|entry| entry == "/model"),
            "the prompt must be recorded in history: {:?}",
            editor.borrow().editor().get_history()
        );
    }

    /// Painted regression for the status-surface merge: each Jev label appears
    /// exactly ONCE (on the tray row, next to the right-aligned counter), other
    /// extension statuses keep their own line below, and no Jev label leaks
    /// into that line.
    #[test]
    fn the_painted_status_surface_shows_each_jev_label_once_and_others_below() {
        let mut mode = stash_mode("footer-paint");
        mode.apply_connection_state_snapshot(local::AgentConnectionState {
            session_id: "footer-paint".into(),
            active_session_id: Some("active-footer-paint".into()),
            context_usage: local::ContextUsage {
                tokens: Some(146_000.0),
                context_window: 1_000_000.0,
                percent: Some(14.0),
            },
            ..Default::default()
        });
        let mode = Rc::new(RefCell::new(mode));
        let ui = Rc::new(RefCell::new(TUI::new(
            Box::new(pi_tui::terminal::ProcessTerminal::new()),
            None,
        )));
        let editor = Rc::new(RefCell::new(CustomEditor::new(
            ui,
            editor_theme(),
            CustomEditorOptions::default(),
        )));
        let mut surfaces = native_extensions::Surfaces::default();
        surfaces.set_status(
            native_commands::jev_menu::JEV_STATUS_KEY.to_string(),
            Some("\u{25cf} Jev On (Compare)".to_string()),
            Some("\u{25cf} Jev C On".to_string()),
        );
        surfaces.set_status(
            native_commands::jev_menu::JEV_COMPACT_STATUS_KEY.to_string(),
            Some("\u{25cf} Jev compact on".to_string()),
            Some("\u{25cf} Jev Cmp on".to_string()),
        );
        surfaces.set_status(
            "tools".to_string(),
            Some("\u{25cf} tools ready".to_string()),
            None,
        );
        let mut statuses =
            native_extensions::Statuses(Rc::new(RefCell::new(surfaces)), Tray(mode, editor));
        let lines = TuiComponent::render(&mut statuses, 120.0);
        let plain: Vec<String> = lines.iter().map(|line| plain_line(line)).collect();
        let all = plain.join("\n");
        // Each FULL Jev label exactly once in the wide painted output; the
        // compact forms ride the same payload but paint only when the row
        // narrows below the full-fit width (the tray ladder's second rung).
        assert_eq!(all.matches("Jev On (Compare)").count(), 1, "{all:?}");
        assert_eq!(all.matches("Jev compact on").count(), 1, "{all:?}");
        assert_eq!(all.matches("Jev C On").count(), 0, "{all:?}");
        assert_eq!(all.matches("Jev Cmp on").count(), 0, "{all:?}");
        // The tray row carries both segments and the right-aligned counter.
        let row = plain
            .iter()
            .find(|line| line.contains("Jev On (Compare)"))
            .expect("tray row line");
        assert!(row.contains("Jev compact on"), "{row:?}");
        assert!(row.ends_with("146k (14%)"), "{row:?}");
        assert!(!row.contains("tools ready"), "{row:?}");
        // At the verified compact rung each compact label paints exactly
        // once, the full-only forms are gone, and the counter and the other
        // status keep their places.
        let narrow = TuiComponent::render(&mut statuses, 60.0);
        let narrow_all = narrow
            .iter()
            .map(|line| plain_line(line))
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(narrow_all.matches("Jev C On").count(), 1, "{narrow_all:?}");
        assert_eq!(narrow_all.matches("Jev Cmp on").count(), 1, "{narrow_all:?}");
        assert_eq!(narrow_all.matches("Jev On (Compare)").count(), 0, "{narrow_all:?}");
        assert_eq!(narrow_all.matches("Jev compact on").count(), 0, "{narrow_all:?}");
        let narrow_row = narrow
            .iter()
            .map(|line| plain_line(line))
            .find(|line| line.contains("Jev C On"))
            .expect("narrow tray row line");
        assert!(narrow_row.ends_with("146k (14%)"), "{narrow_row:?}");
        assert!(narrow_all.contains("tools ready"), "{narrow_all:?}");
        let narrow_tools = narrow
            .iter()
            .map(|line| plain_line(line))
            .find(|line| line.contains("tools ready"))
            .expect("other status keeps its own line at every width");
        assert!(!narrow_tools.contains("Jev"), "{narrow_tools:?}");
        // The other status keeps its own line below, with no Jev label on it.
        let tools_line = plain
            .iter()
            .find(|line| line.contains("tools ready"))
            .expect("other status line");
        assert!(!tools_line.contains("Jev"), "{tools_line:?}");
    }

    /// ANSI-stripped copy of one painted line, for plain-text assertions.
    fn plain_line(line: &str) -> String {
        let mut out = String::new();
        let mut chars = line.chars().peekable();
        while let Some(ch) = chars.next() {
            if ch == '\u{1b}' {
                for escape in chars.by_ref() {
                    if escape.is_ascii_alphabetic() {
                        break;
                    }
                }
            } else {
                out.push(ch);
            }
        }
        out
    }

    /// A SAME-SESSION extension reset (`/reload`, an extension Reset event, the
    /// settings-change reload path) keeps the two host-published Jev segments:
    /// the session and its settings are unchanged, so the segments stay
    /// truthful and are refreshed by key on the next publish. Extension-owned
    /// statuses are cleared. A session switch uses the blanket
    /// `Surfaces::reset`, which clears them too: old-session labels never
    /// persist, and an attached daemon pushes the new session's footer itself.
    #[test]
    fn the_same_session_extension_reset_keeps_the_jev_segments_and_clears_others() {
        let mut mode = stash_mode("footer-reset");
        mode.apply_connection_state_snapshot(local::AgentConnectionState {
            session_id: "footer-reset".into(),
            active_session_id: Some("active-footer-reset".into()),
            context_usage: local::ContextUsage {
                tokens: Some(146_000.0),
                context_window: 1_000_000.0,
                percent: Some(14.0),
            },
            ..Default::default()
        });
        let mode = Rc::new(RefCell::new(mode));
        let ui = Rc::new(RefCell::new(TUI::new(
            Box::new(pi_tui::terminal::ProcessTerminal::new()),
            None,
        )));
        let editor = Rc::new(RefCell::new(CustomEditor::new(
            ui,
            editor_theme(),
            CustomEditorOptions::default(),
        )));
        let surfaces = Rc::new(RefCell::new(native_extensions::Surfaces::default()));
        {
            let mut surfaces = surfaces.borrow_mut();
            surfaces.set_status(
                native_commands::jev_menu::JEV_STATUS_KEY.to_string(),
                Some("\u{25cf} Jev On (Compare)".to_string()),
                Some("\u{25cf} Jev C On".to_string()),
            );
            surfaces.set_status(
                native_commands::jev_menu::JEV_COMPACT_STATUS_KEY.to_string(),
                Some("\u{25cf} Jev compact on".to_string()),
                Some("\u{25cf} Jev Cmp on".to_string()),
            );
            surfaces.set_status(
                "tools".to_string(),
                Some("\u{25cf} tools ready".to_string()),
                None,
            );
        }
        let mut statuses =
            native_extensions::Statuses(Rc::clone(&surfaces), Tray(mode, editor));
        let before = TuiComponent::render(&mut statuses, 120.0);
        assert!(
            before
                .iter()
                .any(|line| plain_line(line).contains("tools ready")),
            "the extension status is painted before the reset"
        );
        surfaces.borrow_mut().reset_keeping_jev();
        let after = TuiComponent::render(&mut statuses, 120.0);
        let all = after
            .iter()
            .map(|line| plain_line(line))
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(all.matches("Jev On (Compare)").count(), 1, "{all:?}");
        assert_eq!(all.matches("Jev compact on").count(), 1, "{all:?}");
        assert_eq!(all.matches("Jev C On").count(), 0, "{all:?}");
        assert_eq!(all.matches("Jev Cmp on").count(), 0, "{all:?}");
        assert!(
            !all.contains("tools ready"),
            "extension-owned status cleared: {all:?}"
        );
        // The kept Jev payloads still carry their compact forms: at the
        // verified compact rung each paints exactly once, the full-only forms
        // are gone, and the cleared extension status stays cleared.
        let narrow = TuiComponent::render(&mut statuses, 60.0);
        let narrow_all = narrow
            .iter()
            .map(|line| plain_line(line))
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(narrow_all.matches("Jev C On").count(), 1, "{narrow_all:?}");
        assert_eq!(narrow_all.matches("Jev Cmp on").count(), 1, "{narrow_all:?}");
        assert_eq!(narrow_all.matches("Jev On (Compare)").count(), 0, "{narrow_all:?}");
        assert_eq!(narrow_all.matches("Jev compact on").count(), 0, "{narrow_all:?}");
        assert!(
            !narrow_all.contains("tools ready"),
            "the cleared extension status must stay cleared at every width: {narrow_all:?}"
        );
        // The blanket reset (session switch) clears the Jev segments as well:
        // old-session labels never persist across a session switch; an
        // attached daemon pushes the new session's authoritative footer.
        surfaces.borrow_mut().reset();
        let cleared = TuiComponent::render(&mut statuses, 120.0);
        let all = cleared
            .iter()
            .map(|line| plain_line(line))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            !all.contains("Jev"),
            "blanket reset clears the Jev segments: {all:?}"
        );
    }

    /// Runtime rebind regression (fork, /new, in-chat /resume): session A ran
    /// with operative Compare and compaction on; the rebind blanket-resets the
    /// surface and the NEW session B publishes its OWN authoritative settings
    /// (Off + compaction off here). The first painted frame of B must show
    /// ONLY B's labels - none of session A's may survive.
    #[test]
    fn a_runtime_rebind_never_carries_the_old_session_jev_labels_into_the_new_session() {
        let mut mode = stash_mode("footer-rebind");
        mode.apply_connection_state_snapshot(local::AgentConnectionState {
            session_id: "session-b".into(),
            active_session_id: Some("active-session-b".into()),
            context_usage: local::ContextUsage {
                tokens: Some(146_000.0),
                context_window: 1_000_000.0,
                percent: Some(14.0),
            },
            ..Default::default()
        });
        let mode = Rc::new(RefCell::new(mode));
        let ui = Rc::new(RefCell::new(TUI::new(
            Box::new(pi_tui::terminal::ProcessTerminal::new()),
            None,
        )));
        let editor = Rc::new(RefCell::new(CustomEditor::new(
            ui,
            editor_theme(),
            CustomEditorOptions::default(),
        )));
        let surfaces = Rc::new(RefCell::new(native_extensions::Surfaces::default()));
        {
            // Session A: Compare + compaction on (full and short forms).
            let mut surfaces = surfaces.borrow_mut();
            surfaces.set_status(
                native_commands::jev_menu::JEV_STATUS_KEY.to_string(),
                Some("\u{25cf} Jev On (Compare)".to_string()),
                Some("\u{25cf} Jev C On".to_string()),
            );
            surfaces.set_status(
                native_commands::jev_menu::JEV_COMPACT_STATUS_KEY.to_string(),
                Some("\u{25cf} Jev compact on".to_string()),
                Some("\u{25cf} Jev Cmp on".to_string()),
            );
        }
        let mut statuses =
            native_extensions::Statuses(Rc::clone(&surfaces), Tray(mode, editor));
        // The runtime rebind (Event::RuntimeRebound) blanket-resets: no
        // old-session label may survive into the new session's first frame.
        surfaces.borrow_mut().reset();
        // Session B publishes its OWN authoritative settings through the
        // normal setStatus path (the same payload the startup publisher sends).
        {
            let mut surfaces = surfaces.borrow_mut();
            surfaces.set_status(
                native_commands::jev_menu::JEV_STATUS_KEY.to_string(),
                Some("\u{25cf} Jev Off".to_string()),
                Some("\u{25cf} Jev Off".to_string()),
            );
            surfaces.set_status(
                native_commands::jev_menu::JEV_COMPACT_STATUS_KEY.to_string(),
                Some("\u{25cf} Jev compact off".to_string()),
                Some("\u{25cf} Jev Cmp off".to_string()),
            );
        }
        let lines = TuiComponent::render(&mut statuses, 120.0);
        let all = lines
            .iter()
            .map(|line| plain_line(line))
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(all.matches("Jev Off").count(), 1, "B's decision label only: {all:?}");
        assert_eq!(all.matches("Jev compact off").count(), 1, "{all:?}");
        assert_eq!(all.matches("Jev Cmp off").count(), 0, "{all:?}");
        assert!(
            !all.contains("Jev On (Compare)") && !all.contains("Jev C On"),
            "session A's decision label must not survive: {all:?}"
        );
        assert!(
            !all.contains("Jev compact on") && !all.contains("Jev Cmp on"),
            "session A's compaction label must not survive: {all:?}"
        );
        // At the verified compact rung B's compacts paint exactly once. B's
        // decision label is shared between its full and compact forms, so it
        // is asserted PRESENT, never absent; session A's labels (full and
        // compact) stay absent at every width.
        let narrow = TuiComponent::render(&mut statuses, 60.0);
        let narrow_all = narrow
            .iter()
            .map(|line| plain_line(line))
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(narrow_all.matches("Jev Off").count(), 1, "{narrow_all:?}");
        assert_eq!(narrow_all.matches("Jev Cmp off").count(), 1, "{narrow_all:?}");
        assert_eq!(narrow_all.matches("Jev compact off").count(), 0, "{narrow_all:?}");
        assert_eq!(narrow_all.matches("Jev On (Compare)").count(), 0, "{narrow_all:?}");
        assert_eq!(narrow_all.matches("Jev C On").count(), 0, "{narrow_all:?}");
        assert_eq!(narrow_all.matches("Jev compact on").count(), 0, "{narrow_all:?}");
        assert_eq!(narrow_all.matches("Jev Cmp on").count(), 0, "{narrow_all:?}");
    }

    // ----------------------------------------------------------------------
    // ROOT-CONTRACT v9 model controls (PATCH-v2-ADDENDUM): REAL host-dispatch
    // tests. These drive the production slash-command chain
    // (`run_builtin_command` -> native_commands::run -> jev_host::run ->
    // parse_jev_request -> the model arms -> the durable settings store), not
    // parser helpers. The agent dir is scoped to a per-test tempdir through
    // the same environment variable the TS tests and the agent-session tests
    // use (agent_session.rs `std::env::set_var(env_agent_dir(), ..)`), held
    // under a module-level lock and always restored on drop.
    static AGENT_DIR_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    struct ScopedAgentDir {
        previous: Option<String>,
        root: std::path::PathBuf,
        _guard: std::sync::MutexGuard<'static, ()>,
    }

    impl ScopedAgentDir {
        fn in_temp(label: &str) -> Self {
            let guard = AGENT_DIR_TEST_LOCK
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let previous = std::env::var(crate::config::env_agent_dir()).ok();
            let unique = format!(
                "jev-model-dispatch-{label}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            );
            let root = std::env::temp_dir().join(unique);
            std::fs::create_dir_all(root.join("agent")).expect("temp agent dir");
            std::env::set_var(crate::config::env_agent_dir(), root.join("agent"));
            Self { previous, root, _guard: guard }
        }
    }

    impl Drop for ScopedAgentDir {
        fn drop(&mut self) {
            match &self.previous {
                Some(value) => std::env::set_var(crate::config::env_agent_dir(), value),
                None => std::env::remove_var(crate::config::env_agent_dir()),
            }
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }

    fn scoped_settings_file(scope: &ScopedAgentDir) -> std::path::PathBuf {
        scope.root.join("agent").join("jev").join("jev-settings.json")
    }

    fn read_scoped_settings(scope: &ScopedAgentDir) -> Option<serde_json::Value> {
        let raw = std::fs::read_to_string(scoped_settings_file(scope)).ok()?;
        serde_json::from_str(&raw).ok()
    }

    struct TestCatalogTransportGuard;

    impl TestCatalogTransportGuard {
        fn install(transport: Arc<dyn pi_jev::types::Transport>) -> Self {
            crate::core::jev_bridge::set_test_catalog_transport(Some(transport));
            Self
        }
    }

    impl Drop for TestCatalogTransportGuard {
        fn drop(&mut self) {
            crate::core::jev_bridge::set_test_catalog_transport(None);
        }
    }

    #[tokio::test]
    async fn jev_model_set_persists_exactly_through_the_real_dispatch() {
        let scope = ScopedAgentDir::in_temp("set");
        let recorder = Arc::new(RecordingConnection::new());
        let connection: Arc<dyn wire::AgentConnection> = recorder.clone();
        let (send, _receive) = mpsc::channel();
        let raw = "/jev model set jev-two";
        let parsed = crate::core::slash_commands::parse_slash_command(raw)
            .expect("the real slash boundary parses /jev");
        assert_eq!(parsed.args, "model set jev-two");
        let output = run_builtin_command(
            &connection,
            &send,
            raw,
            &parsed.name,
            &parsed.args,
        )
        .await
        .expect("the real dispatch must not error for a safe id");
        let text = match output {
            CommandOutput::Status(text) => text,
            other => panic!("expected a status panel, got {other:?}"),
        };
        assert!(
            text.contains("Requested Jev model: jev-two"),
            "the exact id is echoed from the persisted outcome: {text}"
        );
        assert!(text.contains("No network call was made"), "{text}");
        let saved = read_scoped_settings(&scope).expect("the durable store exists");
        assert_eq!(
            saved["requested_model"],
            serde_json::json!("jev-two"),
            "the id persists byte-for-byte (no normalization)"
        );
        assert_eq!(saved["write_revision"], serde_json::json!(1));
        // Status reads the durable truth through the SAME real dispatch.
        let status = run_builtin_command(
            &connection,
            &send,
            "/jev model status",
            "jev",
            "model status",
        )
        .await
        .unwrap();
        let panel = match status {
            CommandOutput::Panel(panel) => panel,
            other => panic!("expected a panel, got {other:?}"),
        };
        assert!(
            panel.contains("Requested Jev model: jev-two (explicit /jev model set selection)"),
            "{panel}"
        );
        assert!(panel.contains("Durable settings write revision: 1"), "{panel}");
        // An identical set is a truthful no-op: nothing written, revision unmoved.
        let again = run_builtin_command(
            &connection,
            &send,
            "/jev model set jev-two",
            "jev",
            "model set jev-two",
        )
        .await
        .unwrap();
        match again {
            CommandOutput::Status(text) => assert!(
                text.contains("already jev-two")
                    && text.contains("nothing was written"),
                "{text}"
            ),
            other => panic!("expected a status panel, got {other:?}"),
        }
        assert_eq!(
            read_scoped_settings(&scope).unwrap()["write_revision"],
            serde_json::json!(1),
            "an identical set must not move the durable revision"
        );
    }

    #[tokio::test]
    async fn jev_model_set_refuses_hostile_ids_through_the_real_dispatch() {
        // Exact-ID rejection is exercised through the REAL chain: the parser
        // hands the id byte-for-byte to the validator, and every
        // whitespace-bearing id (tab-prefixed, plain-space interior) is
        // REFUSED - never trimmed, folded or normalized into acceptance -
        // with nothing persisted and no value echoed.
        let scope = ScopedAgentDir::in_temp("refuse");
        let recorder = Arc::new(RecordingConnection::new());
        let connection: Arc<dyn wire::AgentConnection> = recorder.clone();
        let (send, _receive) = mpsc::channel();
        for (label, args, raw_id) in [
            ("tab-prefixed", "model set \tjev-latest", "\tjev-latest"),
            ("leading space", "model set  jev-latest", " jev-latest"),
            ("interior space", "model set my model", "my model"),
            ("trailing space", "model set jev-latest ", "jev-latest "),
            ("trailing tab", "model set jev-latest\t", "jev-latest\t"),
        ] {
            let raw = format!("/jev {args}");
            let parsed = crate::core::slash_commands::parse_slash_command(&raw)
                .expect("the real slash boundary parses /jev");
            assert_eq!(parsed.name, "jev");
            let result = run_builtin_command(
                &connection,
                &send,
                &raw,
                &parsed.name,
                &parsed.args,
            )
            .await;
            match result {
                Ok(CommandOutput::Error(text)) => {
                    assert!(
                        text.contains("Jev model id refused"),
                        "{label}: {text}"
                    );
                    assert!(
                        !text.contains(raw_id),
                        "{label}: the refused id must never be echoed: {text}"
                    );
                }
                Err(error) => {
                    assert!(error.contains("Jev model id refused"), "{label}: {error}");
                    assert!(
                        !error.contains(raw_id),
                        "{label}: the refused id must never be echoed: {error}"
                    );
                }
                other => panic!("{label}: expected a refusal, got {other:?}"),
            }
            assert!(
                read_scoped_settings(&scope).is_none(),
                "{label}: a refused id must persist nothing"
            );
        }
        // The bare `model set` form stays a usage error through the real arm.
        let usage = run_builtin_command(&connection, &send, "/jev model set", "jev", "model set")
            .await
            .unwrap();
        match usage {
            CommandOutput::Error(text) => assert!(text.contains("Usage: /jev model set <id>"), "{text}"),
            other => panic!("expected the usage error, got {other:?}"),
        }
        assert!(read_scoped_settings(&scope).is_none(), "usage must write nothing");
    }

    #[tokio::test]
    async fn jev_models_success_routes_once_through_the_real_dispatch_without_selection() {
        let scope = ScopedAgentDir::in_temp("models-success");
        let recorder = Arc::new(RecordingConnection::new());
        let connection: Arc<dyn wire::AgentConnection> = recorder.clone();
        let (send, _receive) = mpsc::channel();
        let mock = Arc::new(pi_jev::mock::MockJevTransport::scripted(vec![
            pi_jev::mock::MockStep::ModelsBody(
                serde_json::json!({
                    "models": [{
                        "name": "jev-native-positive",
                        "description": "Offline native command fixture.",
                        "release_date": "2026-09-21"
                    }]
                })
                .to_string(),
            ),
        ]));
        let _catalog_guard = TestCatalogTransportGuard::install(
            mock.clone() as Arc<dyn pi_jev::types::Transport>,
        );
        let raw = "/jev models";
        let parsed = crate::core::slash_commands::parse_slash_command(raw)
            .expect("the real slash boundary parses /jev models");
        let output = run_builtin_command(
            &connection,
            &send,
            raw,
            &parsed.name,
            &parsed.args,
        )
        .await
        .expect("the injected offline catalog succeeds");
        let panel = match output {
            CommandOutput::Panel(panel) => panel,
            other => panic!("expected the catalog panel, got {other:?}"),
        };
        assert!(panel.contains("jev-native-positive"), "{panel}");
        assert_eq!(mock.models_call_count(), 1, "one explicit catalog GET");
        assert_eq!(mock.call_count(), 0, "the decision endpoint was untouched");
        assert!(
            read_scoped_settings(&scope).is_none(),
            "catalog success must not select a model or write settings"
        );
    }

    #[tokio::test]
    async fn jev_models_reports_honest_unavailability_without_any_fetch_through_the_real_dispatch() {
        // The only networked command, driven through the REAL arm with no
        // credential configured: an honest unavailable panel, no transport
        // construction, no fetch, and no settings write of any kind.
        let scope = ScopedAgentDir::in_temp("models");
        let recorder = Arc::new(RecordingConnection::new());
        let connection: Arc<dyn wire::AgentConnection> = recorder.clone();
        let (send, _receive) = mpsc::channel();
        let output = run_builtin_command(&connection, &send, "/jev models", "jev", "models")
            .await
            .unwrap();
        match output {
            CommandOutput::Error(text) => {
                assert!(
                    text.contains("Model catalog unavailable: no Jev credential is configured"),
                    "{text}"
                );
                assert!(text.contains("Nothing was fetched"), "{text}");
            }
            other => panic!("expected the honest unavailable panel, got {other:?}"),
        }
        assert!(
            read_scoped_settings(&scope).is_none(),
            "the catalog query must write nothing"
        );
    }

    #[tokio::test]
    async fn jev_model_reset_restores_the_native_default_through_the_real_dispatch() {
        let scope = ScopedAgentDir::in_temp("reset");
        let recorder = Arc::new(RecordingConnection::new());
        let connection: Arc<dyn wire::AgentConnection> = recorder.clone();
        let (send, _receive) = mpsc::channel();
        run_builtin_command(
            &connection,
            &send,
            "/jev model set jev-two",
            "jev",
            "model set jev-two",
        )
        .await
        .unwrap();
        let reset = run_builtin_command(&connection, &send, "/jev model reset", "jev", "model reset")
            .await
            .unwrap();
        match reset {
            CommandOutput::Status(text) => assert!(
                text.contains("Requested Jev model reset to the native default jev-latest"),
                "{text}"
            ),
            other => panic!("expected a status panel, got {other:?}"),
        }
        let saved = read_scoped_settings(&scope).unwrap();
        let cleared = saved
            .get("requested_model")
            .map(|value| value.is_null())
            .unwrap_or(true);
        assert!(cleared, "the tombstone removes the explicit selection: {saved}");
        assert_eq!(
            saved["write_revision"],
            serde_json::json!(2),
            "a real reset advances the durable revision"
        );
        // An already-default reset is a truthful no-op (no write, no revision move).
        let again = run_builtin_command(&connection, &send, "/jev model reset", "jev", "model reset")
            .await
            .unwrap();
        match again {
            CommandOutput::Status(text) => assert!(
                text.contains("already the native default") && text.contains("nothing was written"),
                "{text}"
            ),
            other => panic!("expected a status panel, got {other:?}"),
        }
        assert_eq!(
            read_scoped_settings(&scope).unwrap()["write_revision"],
            serde_json::json!(2),
        );
    }
}
