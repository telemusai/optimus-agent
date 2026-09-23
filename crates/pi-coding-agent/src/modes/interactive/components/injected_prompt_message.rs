//! Port of packages/coding-agent/src/modes/interactive/components/injected-prompt-message.ts

use std::rc::Rc;
use std::sync::Arc;

use pi_agent_core::types::{AgentMessage, CustomAgentMessage, CustomMessageContent};
use pi_tui::components::markdown::{Markdown, MarkdownOptions, MarkdownTheme as TuiMarkdownTheme};
use pi_tui::components::spacer::Spacer;
use pi_tui::components::text::Text;
use pi_tui::tui::Component;
use pi_tui::utils::{truncate_to_width, visible_width};

use crate::core::goals::{GoalContextDetails, GoalContextKind, GOAL_CONTEXT_CUSTOM_TYPE};
use crate::core::messages::{
    AsyncBashCompletionDetails, HeartbeatPromptDetails, IpythonStateRestoredDetails,
    ASYNC_BASH_COMPLETION_CUSTOM_TYPE, ASYNC_BASH_COMPLETION_PREVIEW_LABEL,
    HEARTBEAT_PROMPT_CUSTOM_TYPE, IPYTHON_STATE_RESTORED_CUSTOM_TYPE,
    RLM_CHILD_FAILURE_CUSTOM_TYPE, RLM_CHILD_TERMINAL_NOTICE_CUSTOM_TYPE,
};
use crate::modes::interactive::theme::theme::{get_markdown_theme, theme, MarkdownTheme};

use super::keybinding_hints::expand_collapse_hint;

/// `agentMessageSummaryLine` (components/agent-message.ts) belongs to another
/// slice, so this module keeps a private copy of the one-line summary it builds;
/// see evidence/status/ca-interactive-components-3.json -> blocked_on.
fn agent_message_summary_line(label: &str, participant: &str, preview: Option<&str>) -> String {
    let mut parts = vec![
        format!(
            "{} {}",
            theme().fg("accent", "\u{25c6}"),
            theme().fg("muted", label)
        ),
        theme().fg("muted", participant),
    ];
    if let Some(preview) = preview {
        if !preview.is_empty() {
            parts.push(theme().fg("muted", preview));
        }
    }
    parts.join(&theme().fg("dim", " \u{00b7} "))
}

/// Port of `isInjectedPromptMessage`.
pub fn is_injected_prompt_message(message: &AgentMessage) -> bool {
    match message {
        AgentMessage::Custom(CustomAgentMessage::Custom { custom_type, .. }) => {
            let custom_type = custom_type.as_str();
            custom_type == ASYNC_BASH_COMPLETION_CUSTOM_TYPE
                || custom_type == HEARTBEAT_PROMPT_CUSTOM_TYPE
                || custom_type == GOAL_CONTEXT_CUSTOM_TYPE
                || custom_type == IPYTHON_STATE_RESTORED_CUSTOM_TYPE
                || custom_type == RLM_CHILD_FAILURE_CUSTOM_TYPE
                || custom_type == RLM_CHILD_TERMINAL_NOTICE_CUSTOM_TYPE
        }
        _ => false,
    }
}

/// `readCustomText`.
fn read_custom_text(content: &CustomMessageContent) -> String {
    match content {
        CustomMessageContent::Text(text) => text.clone(),
        CustomMessageContent::Blocks(blocks) => blocks
            .iter()
            .map(|block| match block.as_text() {
                Some(text) => text.to_string(),
                None => "[image]".to_string(),
            })
            .collect::<Vec<String>>()
            .join("\n"),
    }
}

/// `collapseText`: `text.replace(/\s+/g, " ").trim()`.
fn collapse_text(text: &str) -> String {
    let mut collapsed = String::new();
    let mut last_was_space = false;
    for ch in text.chars() {
        if ch.is_whitespace() {
            if !last_was_space {
                collapsed.push(' ');
            }
            last_was_space = true;
        } else {
            last_was_space = false;
            collapsed.push(ch);
        }
    }
    collapsed.trim().to_string()
}

/// Port of `goalLabel`.
pub fn goal_label(details: Option<&GoalContextDetails>) -> &'static str {
    match details.map(|details| details.kind) {
        Some(GoalContextKind::Continuation) => "Goal continuation",
        Some(GoalContextKind::BudgetLimit) => "Goal budget limit",
        Some(GoalContextKind::ObjectiveUpdated) => "Goal updated",
        _ => "Goal context",
    }
}

/// Port of `compactHeartbeatSchedule`.
pub fn compact_heartbeat_schedule(schedule: Option<&str>) -> String {
    let trimmed = schedule.unwrap_or("").trim().to_string();
    if trimmed.is_empty() {
        return "prompt".to_string();
    }
    strip_every_prefix(&trimmed)
}

/// `/^every\s+/i` - drop a leading "every" plus following whitespace.
fn strip_every_prefix(text: &str) -> String {
    if text.len() >= 5 && text[..5].eq_ignore_ascii_case("every") {
        let rest = &text[5..];
        let trimmed = rest.trim_start();
        if trimmed.len() != rest.len() {
            return trimmed.to_string();
        }
    }
    text.to_string()
}

/// Port of `heartbeatPromptSchedule`.
pub fn heartbeat_prompt_schedule(schedule: Option<&str>) -> String {
    let compact = compact_heartbeat_schedule(schedule);
    if compact == "prompt" {
        "scheduled".to_string()
    } else {
        format!("every {compact}")
    }
}

/// `MarkdownTheme` of `theme.ts` carries `Arc` closures with `Send + Sync`; the
/// pi-tui component holds `Rc` closures.
fn to_tui_markdown_theme(theme: MarkdownTheme) -> TuiMarkdownTheme {
    fn rc(value: Arc<dyn Fn(&str) -> String + Send + Sync>) -> Rc<dyn Fn(&str) -> String> {
        Rc::new(move |text: &str| value(text))
    }

    TuiMarkdownTheme {
        heading: rc(theme.heading),
        link: rc(theme.link),
        link_url: rc(theme.link_url),
        code: rc(theme.code),
        code_block: rc(theme.code_block),
        code_block_border: rc(theme.code_block_border),
        quote: rc(theme.quote),
        quote_border: rc(theme.quote_border),
        hr: rc(theme.hr),
        list_bullet: rc(theme.list_bullet),
        bold: rc(theme.bold),
        italic: rc(theme.italic),
        strikethrough: rc(theme.strikethrough),
        underline: rc(theme.underline),
        highlight_code: Some(Rc::new(move |code: &str, lang: Option<&str>| {
            (theme.highlight_code)(code, lang)
        })),
        code_block_indent: theme.code_block_indent,
        math: Some(Rc::new(move |text: &str| (theme.math)(text))),
        math_block: Some(Rc::new(move |text: &str| (theme.math_block)(text))),
    }
}

/// Port of `InjectedPromptMessageComponent`.
pub struct InjectedPromptMessageComponent {
    message: InjectedPromptMessage,
    header: Text,
    expanded: bool,
    /// `this.content` container: the header, plus the markdown body when expanded.
    body: Option<Markdown>,
}

/// `type InjectedPromptMessage = CustomMessage<InjectedPromptDetails>`.
#[derive(Debug, Clone)]
pub struct InjectedPromptMessage {
    pub custom_type: String,
    pub content: CustomMessageContent,
    pub details: Option<serde_json::Value>,
}

impl InjectedPromptMessage {
    /// Reads a `CustomAgentMessage::Custom` into the narrowed union.
    pub fn from_agent_message(message: &AgentMessage) -> Option<Self> {
        match message {
            AgentMessage::Custom(CustomAgentMessage::Custom {
                custom_type,
                content,
                details,
                ..
            }) => Some(Self {
                custom_type: custom_type.clone(),
                content: content.clone(),
                details: details.clone(),
            }),
            _ => None,
        }
    }

    fn details_as<T: serde::de::DeserializeOwned>(&self) -> Option<T> {
        self.details
            .as_ref()
            .and_then(|value| serde_json::from_value::<T>(value.clone()).ok())
    }
}

impl InjectedPromptMessageComponent {
    pub fn new(message: InjectedPromptMessage, markdown_theme: Option<MarkdownTheme>) -> Self {
        let markdown_theme = markdown_theme.unwrap_or_else(get_markdown_theme);
        let mut component = Self {
            message,
            header: Text::new(String::new(), 1, 0, None),
            expanded: false,
            body: None,
        };
        let _ = markdown_theme;
        component.update_display();
        component
    }

    /// Port of `setExpanded`.
    pub fn set_expanded(&mut self, expanded: bool) {
        if self.expanded == expanded {
            return;
        }
        self.expanded = expanded;
        self.update_display();
    }

    pub fn expanded(&self) -> bool {
        self.expanded
    }

    /// Port of `updateDisplay`.
    fn update_display(&mut self) {
        self.header.set_text(self.header_text());
        self.body = None;
        if self.expanded && self.message.custom_type != IPYTHON_STATE_RESTORED_CUSTOM_TYPE {
            let color = Rc::new(|text: &str| theme().fg("customMessageText", text));
            self.body = Some(Markdown::new(
                read_custom_text(&self.message.content),
                1,
                0,
                to_tui_markdown_theme(get_markdown_theme()),
                Some(pi_tui::components::markdown::DefaultTextStyle {
                    color: Some(color),
                    ..Default::default()
                }),
                MarkdownOptions::default(),
            ));
        }
    }

    /// Port of `heartbeatHeaderText`.
    fn heartbeat_header_text(&self) -> String {
        let details: Option<HeartbeatPromptDetails> = self.message.details_as();
        let pulse = theme().fg("error", "\u{2665}");
        let schedule = theme().fg(
            "muted",
            &heartbeat_prompt_schedule(details.as_ref().map(|d| d.schedule.as_str())),
        );
        let hint = if self.expanded {
            String::new()
        } else {
            format!(" {}", expand_collapse_hint("app.tools.expand", false))
        };
        format!(
            "{pulse} {}{}{}{hint}",
            theme().fg("muted", "Heartbeat prompt"),
            theme().fg("dim", " \u{00b7} "),
            schedule
        )
    }

    /// Port of `metaText`.
    fn meta_text(&self) -> String {
        let goal: Option<GoalContextDetails> = self.message.details_as();
        let Some(objective) = goal.and_then(|goal| {
            if goal.objective.is_empty() {
                None
            } else {
                Some(goal.objective)
            }
        }) else {
            return String::new();
        };
        let prefix_width = visible_width("Goal continuation \u{00b7} ");
        theme().fg(
            "muted",
            &format!(
                " \u{00b7} {}",
                truncate_to_width(
                    &collapse_text(&objective),
                    (90.0 - prefix_width as f64).max(20.0),
                    "",
                    false
                )
            ),
        )
    }

    /// Port of `headerText`.
    fn header_text(&self) -> String {
        if self.message.custom_type == HEARTBEAT_PROMPT_CUSTOM_TYPE {
            return self.heartbeat_header_text();
        }
        if self.message.custom_type == ASYNC_BASH_COMPLETION_CUSTOM_TYPE {
            let details: Option<AsyncBashCompletionDetails> = self.message.details_as();
            let participant = match &details {
                Some(details) => format!("pid {}", details.pid),
                None => "bash".to_string(),
            };
            let status = details.map(|details| format!("exit {}", details.exit_code));
            let hint = if self.expanded {
                String::new()
            } else {
                format!(" {}", expand_collapse_hint("app.tools.expand", false))
            };
            return agent_message_summary_line(
                ASYNC_BASH_COMPLETION_PREVIEW_LABEL,
                &participant,
                status.as_deref(),
            ) + &theme().fg("dim", &hint);
        }
        if self.message.custom_type == IPYTHON_STATE_RESTORED_CUSTOM_TYPE {
            let details: Option<IpythonStateRestoredDetails> = self.message.details_as();
            let label = if details.map(|d| d.restored) == Some(false) {
                "Started fresh Python kernel"
            } else {
                "Restored Python kernel state"
            };
            return format!(
                "{} {}",
                theme().fg("accent", "\u{25c6}"),
                theme().fg("muted", label)
            );
        }
        if self.message.custom_type == RLM_CHILD_FAILURE_CUSTOM_TYPE
            || self.message.custom_type == RLM_CHILD_TERMINAL_NOTICE_CUSTOM_TYPE
        {
            let hint = if self.expanded {
                String::new()
            } else {
                format!(" {}", expand_collapse_hint("app.tools.expand", false))
            };
            return theme().fg("muted", "RLM child status") + &theme().fg("dim", &hint);
        }

        let goal: Option<GoalContextDetails> = self.message.details_as();
        let title = goal_label(goal.as_ref());
        let meta = self.meta_text();
        let hint = if self.expanded {
            String::new()
        } else {
            format!(" {}", expand_collapse_hint("app.tools.expand", false))
        };
        theme().fg("muted", title) + &meta + &theme().fg("dim", &hint)
    }
}

impl Component for InjectedPromptMessageComponent {
    fn render(&mut self, width: f64) -> Vec<String> {
        if self.message.custom_type == ASYNC_BASH_COMPLETION_CUSTOM_TYPE && !self.expanded {
            return super::shell_completion::ShellCompletion::from_details(self.message.details.as_ref()).render(width);
        }
        let mut spacer = Spacer::new(1);
        let mut lines = spacer.render(width);
        // `this.content`: the header, then the markdown body when expanded.
        lines.extend(self.header.render(width));
        if let Some(body) = self.body.as_mut() {
            lines.extend(<Markdown as Component>::render(body, width));
        }
        lines
    }

    fn invalidate(&mut self) {
        self.header.invalidate();
        if let Some(body) = self.body.as_mut() {
            body.invalidate();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::goals::GoalStatus;

    fn custom_message(
        custom_type: &str,
        content: &str,
        details: serde_json::Value,
    ) -> InjectedPromptMessage {
        InjectedPromptMessage {
            custom_type: custom_type.to_string(),
            content: CustomMessageContent::Text(content.to_string()),
            details: Some(details),
        }
    }

    fn component(message: InjectedPromptMessage) -> InjectedPromptMessageComponent {
        InjectedPromptMessageComponent::new(message, None)
    }

    fn goal_details(kind: &str) -> serde_json::Value {
        serde_json::json!({
            "kind": kind,
            "objective": "ship the port",
            "status": "active",
            "continuationsUsed": 1.0
        })
    }

    #[test]
    fn injected_prompt_types_are_recognized() {
        for custom_type in [
            ASYNC_BASH_COMPLETION_CUSTOM_TYPE,
            HEARTBEAT_PROMPT_CUSTOM_TYPE,
            GOAL_CONTEXT_CUSTOM_TYPE,
            IPYTHON_STATE_RESTORED_CUSTOM_TYPE,
            RLM_CHILD_FAILURE_CUSTOM_TYPE,
            RLM_CHILD_TERMINAL_NOTICE_CUSTOM_TYPE,
        ] {
            let message = AgentMessage::Custom(CustomAgentMessage::Custom {
                custom_type: custom_type.to_string(),
                content: CustomMessageContent::Text("x".to_string()),
                display: true,
                details: None,
                timestamp: 0,
            });
            assert!(is_injected_prompt_message(&message), "{custom_type}");
        }
        let other = AgentMessage::Custom(CustomAgentMessage::Custom {
            custom_type: "something_else".to_string(),
            content: CustomMessageContent::Text("x".to_string()),
            display: true,
            details: None,
            timestamp: 0,
        });
        assert!(!is_injected_prompt_message(&other));
    }

    #[test]
    fn custom_content_is_read_as_text_or_image_blocks() {
        assert_eq!(
            read_custom_text(&CustomMessageContent::Text("hi".to_string())),
            "hi"
        );
        let blocks = CustomMessageContent::Blocks(vec![
            pi_agent_core::types::ContentBlock::text("a"),
            pi_agent_core::types::ContentBlock::Image(
                serde_json::from_value(serde_json::json!({
                    "type": "image",
                    "data": "x",
                    "mimeType": "image/png"
                }))
                .unwrap(),
            ),
            pi_agent_core::types::ContentBlock::text("b"),
        ]);
        assert_eq!(read_custom_text(&blocks), "a\n[image]\nb");
    }

    #[test]
    fn collapse_text_squeezes_whitespace() {
        assert_eq!(collapse_text("  a \n\t b  "), "a b");
        assert_eq!(collapse_text(""), "");
    }

    #[test]
    fn goal_labels_match_the_kind() {
        assert_eq!(
            goal_label(Some(&GoalContextDetails {
                kind: GoalContextKind::Continuation,
                goal_id: None,
                objective: "x".to_string(),
                status: GoalStatus::Active,
                continuations_used: 0.0,
            })),
            "Goal continuation"
        );
        assert_eq!(
            goal_label(Some(&GoalContextDetails {
                kind: GoalContextKind::BudgetLimit,
                goal_id: None,
                objective: "x".to_string(),
                status: GoalStatus::Active,
                continuations_used: 0.0,
            })),
            "Goal budget limit"
        );
        assert_eq!(
            goal_label(Some(&GoalContextDetails {
                kind: GoalContextKind::ObjectiveUpdated,
                goal_id: None,
                objective: "x".to_string(),
                status: GoalStatus::Active,
                continuations_used: 0.0,
            })),
            "Goal updated"
        );
        assert_eq!(goal_label(None), "Goal context");
    }

    #[test]
    fn heartbeat_schedules_are_compacted() {
        assert_eq!(compact_heartbeat_schedule(None), "prompt");
        assert_eq!(compact_heartbeat_schedule(Some("   ")), "prompt");
        assert_eq!(compact_heartbeat_schedule(Some("every 5m")), "5m");
        assert_eq!(compact_heartbeat_schedule(Some("EVERY 5m")), "5m");
        assert_eq!(compact_heartbeat_schedule(Some("every")), "every");
        assert_eq!(compact_heartbeat_schedule(Some("5m")), "5m");
        assert_eq!(heartbeat_prompt_schedule(Some("5m")), "every 5m");
        assert_eq!(heartbeat_prompt_schedule(None), "scheduled");
    }

    #[test]
    fn headers_are_built_per_custom_type() {
        let async_bash = component(custom_message(
            ASYNC_BASH_COMPLETION_CUSTOM_TYPE,
            "body",
            serde_json::json!({"pid": 12, "command": "ls", "exitCode": 0}),
        ));
        let header = async_bash.header_text();
        assert!(header.contains(ASYNC_BASH_COMPLETION_PREVIEW_LABEL));
        assert!(header.contains("pid 12"));
        assert!(header.contains("exit 0"));

        let restored = component(custom_message(
            IPYTHON_STATE_RESTORED_CUSTOM_TYPE,
            "body",
            serde_json::json!({"restored": true}),
        ));
        assert!(restored
            .header_text()
            .contains("Restored Python kernel state"));
        let fresh = component(custom_message(
            IPYTHON_STATE_RESTORED_CUSTOM_TYPE,
            "body",
            serde_json::json!({"restored": false}),
        ));
        assert!(fresh.header_text().contains("Started fresh Python kernel"));

        let child = component(custom_message(
            RLM_CHILD_FAILURE_CUSTOM_TYPE,
            "body",
            serde_json::json!({}),
        ));
        assert!(child.header_text().contains("RLM child status"));

        let heartbeat = component(custom_message(
            HEARTBEAT_PROMPT_CUSTOM_TYPE,
            "body",
            serde_json::json!({
                "jobId": "j",
                "schedule": "every 10m",
                "status": "active",
                "runCount": 1.0
            }),
        ));
        let header = heartbeat.header_text();
        assert!(header.contains("Heartbeat prompt"));
        assert!(header.contains("every 10m"));

        let goal = component(custom_message(
            GOAL_CONTEXT_CUSTOM_TYPE,
            "body",
            goal_details("continuation"),
        ));
        let header = goal.header_text();
        assert!(header.contains("Goal continuation"));
        assert!(header.contains("ship the port"));
    }

    #[test]
    fn expanding_reveals_the_body_except_for_ipython_state() {
        let mut goal = component(custom_message(
            GOAL_CONTEXT_CUSTOM_TYPE,
            "body",
            goal_details("continuation"),
        ));
        let collapsed = goal.render(60.0).len();
        goal.set_expanded(true);
        assert!(goal.render(60.0).len() > collapsed);
        goal.set_expanded(true); // no-op when unchanged
        assert!(goal.expanded());

        let mut restored = component(custom_message(
            IPYTHON_STATE_RESTORED_CUSTOM_TYPE,
            "body",
            serde_json::json!({"restored": true}),
        ));
        let collapsed = restored.render(60.0).len();
        restored.set_expanded(true);
        assert_eq!(restored.render(60.0).len(), collapsed);
    }

    #[test]
    fn shell_completion_collapsed_is_one_line_and_explicit_inspection_keeps_details() {
        crate::modes::interactive::theme::theme::init_theme(Some("prime"), false);
        let mut shell = component(custom_message(
            ASYNC_BASH_COMPLETION_CUSTOM_TYPE,
            "full fixture command\nInspect the saved BashHandle",
            serde_json::json!({"pid": 33504, "command": "full fixture command", "exitCode": 0}),
        ));
        for width in [8, 20, 80] {
            let rows = shell.render(width as f64);
            assert_eq!(rows.len(), 1);
            assert!(visible_width(&rows[0]) <= width);
            assert!(!rows[0].contains("fixture command"));
        }
        assert!(shell.render(80.0)[0].contains("Shell finished — exit 0 (PID 33504)."));
        shell.set_expanded(true);
        assert!(shell.render(80.0).join("\n").contains("BashHandle"));
    }

    #[test]
    fn rendering_starts_with_the_leading_spacer() {
        let mut goal = component(custom_message(
            GOAL_CONTEXT_CUSTOM_TYPE,
            "body",
            goal_details("continuation"),
        ));
        let lines = goal.render(40.0);
        assert_eq!(lines[0], "");
        assert!(lines.iter().any(|line| line.contains("Goal continuation")));
    }
}
