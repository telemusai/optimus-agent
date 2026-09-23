//! The Neon presentation consumes existing session state; it owns no agent work.
use super::*;
use crate::modes::interactive::theme::theme::Theme;
use pi_tui::utils::{strip_ansi, visible_width};

pub(super) fn active() -> bool {
    theme().name.as_deref() == Some("neon")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    User,
    Assistant,
    Thinking,
    Tool,
    Success,
    Error,
    Notice,
}

#[derive(Clone, Debug)]
pub(super) struct RowMeta {
    pub kind: Kind,
    pub timestamp: Option<i64>,
    pub started: Option<Instant>,
    pub elapsed: Option<Duration>,
}

impl RowMeta {
    pub fn new(kind: Kind, timestamp: Option<i64>) -> Self {
        Self {
            kind,
            timestamp: timestamp.filter(|t| *t > 0),
            started: None,
            elapsed: None,
        }
    }
    pub fn message(message: &AgentMessage) -> Self {
        use pi_agent_core::types::CustomAgentMessage;
        use pi_ai::types::{ContentBlock, Message};
        let (kind, timestamp) = match message {
            AgentMessage::Message(Message::User(m)) => (Kind::User, m.timestamp),
            AgentMessage::Message(Message::Assistant(m)) => {
                let thinking_only =
                    m.content
                        .iter()
                        .any(|c| matches!(c, ContentBlock::Thinking(_)))
                        && !m.content.iter().any(
                            |c| matches!(c, ContentBlock::Text(t) if !t.text.trim().is_empty()),
                        );
                (
                    if m.error_message.is_some() || m.stop_reason == "error" {
                        Kind::Error
                    } else if thinking_only {
                        Kind::Thinking
                    } else {
                        Kind::Assistant
                    },
                    m.timestamp,
                )
            }
            AgentMessage::Message(Message::ToolResult(m)) => (
                if m.is_error {
                    Kind::Error
                } else {
                    Kind::Success
                },
                m.timestamp,
            ),
            AgentMessage::Custom(m) => (
                Kind::Notice,
                match m {
                    CustomAgentMessage::BashExecution { timestamp, .. }
                    | CustomAgentMessage::Custom { timestamp, .. }
                    | CustomAgentMessage::BranchSummary { timestamp, .. }
                    | CustomAgentMessage::CompactionSummary { timestamp, .. } => *timestamp,
                },
            ),
        };
        Self::new(kind, Some(timestamp))
    }
}

fn clean_label(text: &str) -> String {
    strip_ansi(text)
        .chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect()
}

fn fit(text: &str, width: usize) -> String {
    truncate_to_width(text, width as f64, "…", true)
}

/// Apply the dark canvas after component-specific ANSI resets as well.
fn surface_with(palette: &Theme, text: &str, width: usize) -> String {
    let bg = palette.get_bg_ansi("toolPanelBg");
    let fg = palette.get_fg_ansi("text");
    let text = fit(text, width)
        .replace("\x1b[0m", &format!("\x1b[0m{bg}{fg}"))
        .replace("\x1b[49m", &bg)
        .replace("\x1b[39m", &fg);
    format!("{bg}{fg}{text}\x1b[0m")
}

pub(super) fn surface(text: &str, width: usize) -> String {
    surface_with(&theme(), text, width)
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Timeline {
    pub width: usize,
    pub left: usize,
    pub right: usize,
}
impl Timeline {
    pub fn new(width: usize) -> Self {
        // Text wins over decoration on very narrow terminals.
        let (left, right) = if width >= 90 {
            (14, 2)
        } else if width >= 36 {
            (4, 2)
        } else {
            (0, 0)
        };
        Self { width, left, right }
    }
    pub fn content_width(self) -> usize {
        self.width.saturating_sub(self.left + self.right).max(1)
    }
    pub fn line(self, text: &str, meta: Option<&RowMeta>, first: bool) -> String {
        let palette = theme();
        if self.left == 0 {
            return surface_with(&palette, text, self.width);
        }
        let (color, marker) = match meta.map(|m| m.kind) {
            Some(Kind::User) => ("mdLink", "○"),
            Some(Kind::Thinking) => ("thinkingText", "◆"),
            Some(Kind::Assistant) => ("accent", "✦"),
            Some(Kind::Tool) => ("warning", "›"),
            Some(Kind::Success) => ("success", "✓"),
            Some(Kind::Error) => ("error", "×"),
            _ => ("borderMuted", "·"),
        };
        let timestamp = if first {
            meta.and_then(|m| m.timestamp)
                .and_then(chrono::DateTime::from_timestamp_millis)
                .map(|t| {
                    t.with_timezone(&chrono::Local)
                        .format("%H:%M:%S")
                        .to_string()
                })
        } else {
            None
        };
        let time = if self.left == 14 {
            format!(" {} ", timestamp.as_deref().unwrap_or("        "))
        } else {
            String::new()
        };
        let rail = palette.fg(
            if first { color } else { "borderMuted" },
            if first { marker } else { "│" },
        );
        let mut content = text.to_string();
        if first {
            if let Some(duration) = meta.and_then(|m| m.elapsed) {
                let label = format!(" {:.1}s", duration.as_secs_f64());
                let content_end = visible_width(strip_ansi(text).trim_end());
                let used = content_end + label.len() + 2;
                if used <= self.content_width() {
                    content = format!(
                        "{}{}",
                        fit(
                            &pi_tui::utils::slice_by_column(text, 0, content_end, false),
                            self.content_width() - label.len()
                        ),
                        palette.fg("dim", &label)
                    );
                }
            }
        }
        let row = format!(
            "{}{} {} {} {}",
            palette.fg("border", "│"),
            palette.fg("dim", &time),
            rail,
            fit(&content, self.content_width()),
            palette.fg("border", "│")
        );
        surface_with(&palette, &row, self.width)
    }
    pub fn padding(self) -> String {
        self.line("", None, false)
    }
}

pub(super) struct Header(
    pub Rc<RefCell<InteractiveMode>>,
    pub std::rc::Weak<RefCell<Transcript>>,
);

impl TuiComponent for Header {
    fn render(&mut self, width: f64) -> Vec<String> {
        self.render_with_height(width, 7)
    }
    fn render_with_height(&mut self, width: f64, height: usize) -> Vec<String> {
        if !active() {
            return Vec::new();
        }
        let mode = self.0.borrow();
        let transport = self
            .1
            .upgrade()
            .map(|t| t.borrow().connection_status.clone())
            .unwrap_or_default();
        let phase = if !transport.is_empty() && transport != "connected" {
            transport.as_str()
        } else if mode.is_agent_compacting() {
            "COMPACTING"
        } else if mode.get_retry_attempt() > 0.0 {
            "RETRYING"
        } else if mode.is_bash_running() {
            "SHELL RUNNING"
        } else if mode.is_agent_streaming() {
            "WORKING"
        } else {
            "READY"
        };
        let jev = self
            .1
            .upgrade()
            .and_then(|t| t.borrow().subagents.clone())
            .and_then(|bar| bar.borrow().compact_jev_status());
        render_header(
            width as usize,
            height,
            &HeaderData {
                cwd: &mode.get_current_cwd(),
                session: mode
                    .get_current_session_name()
                    .as_deref()
                    .unwrap_or("Untitled session"),
                model: mode
                    .get_current_model_id()
                    .as_deref()
                    .unwrap_or("No model selected"),
                phase,
                jev: jev.as_deref(),
                clock: &chrono::Local::now().format("%H:%M:%S").to_string(),
            },
        )
    }
    fn invalidate(&mut self) {}
}

struct HeaderData<'a> {
    cwd: &'a str,
    session: &'a str,
    model: &'a str,
    phase: &'a str,
    jev: Option<&'a str>,
    clock: &'a str,
}

fn paired(left: &str, right: &str, width: usize) -> String {
    let right_width = visible_width(right);
    if right_width + 4 >= width {
        return fit(left, width);
    }
    format!("{}  {}", fit(left, width - right_width - 2), right)
}

fn render_header(width: usize, budget: usize, data: &HeaderData<'_>) -> Vec<String> {
    if budget == 0 || width == 0 {
        return Vec::new();
    }
    let palette = theme();
    let status = palette.fg(
        if matches!(data.phase, "READY" | "WORKING") {
            "accent"
        } else {
            "warning"
        },
        &clean_label(data.phase).to_uppercase(),
    );
    let model = palette.fg("mdLink", &clean_label(data.model));
    let mut rows = Vec::new();
    if budget >= 7 && width >= 110 {
        const LOGO: [&str; 3] = [
            "█▀▀█ █▀▀█ ▀▀█▀▀ ▀█▀ █▀▄▀█ █  █ █▀▀",
            "█  █ █▀▀▀   █    █  █ ▀ █ █  █ ▀▀█",
            "▀▀▀▀ ▀      ▀   ▀▀▀ ▀   ▀ ▀▀▀▀ ▀▀▀",
        ];
        const HORIZON: [&str; 3] = [
            "        ▄▄▄▄▄▄▄        ",
            "   /\\  ▀▀▀▀▀▀▀  /\\   ",
            "__/__\\___/\\___/__\\__",
        ];
        for i in 0..3 {
            let brand = format!(
                " {}    {}",
                palette.fg("accent", LOGO[i]),
                palette.fg("thinkingText", HORIZON[i])
            );
            let right = match i {
                0 => palette.fg("mdLink", "telemus.ai"),
                1 => format!("SYSTEM // {status}"),
                _ => palette.fg("dim", "RUST NATIVE"),
            };
            rows.push(paired(&brand, &format!("{right} "), width));
        }
        rows.push(paired(
            &palette.fg("accent", " BUILT FOR WHAT'S NEXT  ///"),
            &palette.fg("dim", "AGENTS > CODE > REASON > TEST > SHIP "),
            width,
        ));
        rows.push(String::new());
    } else if budget >= 3 {
        rows.push(paired(
            &palette.fg("accent", " OPTIMUS  //  BUILT FOR WHAT'S NEXT"),
            &format!("{status} "),
            width,
        ));
    }
    let session = format!("{} · {}", clean_label(data.cwd), clean_label(data.session));
    let right = if width >= 100 {
        format!("{model} · {} · {}", data.jev.unwrap_or(&status), data.clock)
    } else if width >= 60 {
        format!("{model} · {status}")
    } else {
        model
    };
    if budget == 1 {
        rows.push(paired(&palette.fg("accent", "OPTIMUS"), &right, width));
    } else {
        let inside = paired(
            &palette.fg("muted", &session),
            &right,
            width.saturating_sub(4),
        );
        rows.push(format!(
            "{} {} {}",
            palette.fg("border", "┌"),
            inside,
            palette.fg("border", "┐")
        ));
        if rows.len() < budget {
            rows.push(palette.fg(
                "border",
                &format!("├{}┤", "─".repeat(width.saturating_sub(2))),
            ));
        }
    }
    rows.into_iter()
        .take(budget)
        .map(|r| surface_with(&palette, &r, width))
        .collect()
}

pub(super) struct Dock(pub Rc<RefCell<dyn TuiComponent>>);
impl TuiComponent for Dock {
    fn render(&mut self, width: f64) -> Vec<String> {
        let rows = self.0.borrow_mut().render(width);
        if !active() {
            return rows;
        }
        let mut output = vec![surface(
            &theme().fg(
                "border",
                &format!("└{}┘", "─".repeat((width as usize).saturating_sub(2))),
            ),
            width as usize,
        )];
        output.extend(rows.into_iter().map(|line| surface(&line, width as usize)));
        output
    }
    fn invalidate(&mut self) {
        self.0.borrow_mut().invalidate();
    }
}

pub(super) fn context_meter(
    usage: Option<&local::ContextUsage>,
    text: &str,
    width: usize,
) -> String {
    let palette = theme();
    let percent = usage
        .and_then(|u| u.percent)
        .filter(|p| p.is_finite())
        .map(|p| p.clamp(0.0, 100.0));
    if width < 100 || percent.is_none() {
        return palette.fg("thinkingText", text);
    }
    let percent = percent.unwrap();
    let filled = (percent / 10.0).round() as usize;
    let meter = format!("[{}{}]", "━".repeat(filled), "·".repeat(10 - filled));
    format!(
        "{} {}",
        palette.fg(
            if percent >= 90.0 {
                "warning"
            } else {
                "thinkingText"
            },
            &meter
        ),
        palette.fg("thinkingText", text)
    )
}

#[cfg(test)]
#[path = "native_host_neon_tests.rs"]
mod tests;
