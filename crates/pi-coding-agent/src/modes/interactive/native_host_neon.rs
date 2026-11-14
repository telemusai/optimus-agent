//! The Neon presentation consumes existing session state; it owns no agent work.
use super::*;
use crate::modes::interactive::theme::theme::Theme;
use pi_tui::utils::{strip_ansi, visible_width};

#[cfg(test)]
thread_local! {
    static LINE_FORMATS: Cell<usize> = const { Cell::new(0) };
}

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
        self.line_with_palette(&theme(), text, meta, first)
    }
    fn line_with_palette(self, palette: &Theme, text: &str, meta: Option<&RowMeta>, first: bool) -> String {
        #[cfg(test)]
        LINE_FORMATS.with(|count| count.set(count.get() + 1));
        if pi_tui::terminal_image::is_image_line(text) {
            // Paint the rail first, then graphics at the content column. Text
            // padding after a SIXEL would erase the image's bottom cell row.
            // Sidebar-safe placement: save the cursor BEFORE the blank rail
            // row, paint it, restore, then move RIGHT by `left` cells
            // relative to the CURRENT chat origin. An absolute CSI G would
            // land inside the sidebar once the transcript origin shifts;
            // the relative CSI C keeps the payload anchored to whatever
            // column the row started at. The outer save/restore finishes
            // before the payload's own save/restore begins.
            let mut placement = String::new();
            if self.left > 0 {
                placement.push_str(&format!("\x1b[{}C", self.left));
            }
            return format!("\x1b7{}\x1b8{}{}", self.line_with_palette(palette, "", meta, first), placement, text);
        }
        if self.left == 0 {
            return surface_with(palette, text, self.width);
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
        surface_with(palette, &row, self.width)
    }
    pub fn padding(self) -> String {
        self.line("", None, false)
    }
}

struct CachedTimelineLine {
    text: RenderedLine,
    metadata: Option<(Kind, Option<i64>, Option<Duration>)>,
    first: bool,
    rendered: RenderedLine,
    selection: (usize, usize),
}

/// Keep ANSI decoration and Unicode measurement off the editor repaint path.
/// Entries are positional and retained only for the current transcript lines.
#[derive(Default)]
pub(super) struct TimelineCache {
    timeline: Option<Timeline>,
    palette: Option<Arc<Theme>>,
    lines: Vec<CachedTimelineLine>,
}

impl TimelineCache {
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    pub fn render<'a>(
        &mut self,
        timeline: Timeline,
        rows: impl IntoIterator<Item = (&'a RenderedLine, Option<&'a RowMeta>, bool)>,
    ) -> (Vec<RenderedLine>, Vec<Option<(usize, usize)>>) {
        let palette = theme();
        if self.timeline != Some(timeline)
            || self.palette.as_ref().is_none_or(|previous| !Arc::ptr_eq(previous, &palette))
        {
            self.lines.clear();
            self.timeline = Some(timeline);
            self.palette = Some(palette.clone());
        }
        let mut output = Vec::new();
        let mut columns = Vec::new();
        for (index, (text, meta, first)) in rows.into_iter().enumerate() {
            let metadata = meta.map(|m| (m.kind, m.timestamp, m.elapsed));
            let unchanged = self.lines.get(index).is_some_and(|previous| {
                previous.text == *text && previous.metadata == metadata && previous.first == first
            });
            if !unchanged {
                let entry = CachedTimelineLine {
                    text: text.clone(),
                    metadata,
                    first,
                    rendered: timeline.line_with_palette(&palette, text, meta, first).into(),
                    selection: (timeline.left, timeline.left
                        + visible_width(strip_ansi(text).trim_end()).min(timeline.content_width())),
                };
                if let Some(previous) = self.lines.get_mut(index) {
                    *previous = entry;
                } else {
                    self.lines.push(entry);
                }
            }
            let entry = &self.lines[index];
            output.push(entry.rendered.clone());
            columns.push(Some(entry.selection));
        }
        self.lines.truncate(output.len());
        (output, columns)
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
        let mode = self.0.borrow();
        let transport = self
            .1
            .upgrade()
            .map(|t| t.borrow().connection_status.clone())
            .unwrap_or_default();
        let phase = if !transport.is_empty() && transport != "connected" {
            transport.as_str()
        } else if self.1.upgrade().is_some_and(|t| t.borrow().refinement_progress.is_some()) {
            "REFINING"
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
        let execution = execution_label(&mode);
        let (resource_width, resource_budget) = resource_space(width as usize, height);
        let resources = self.1.upgrade()
            .and_then(|t| t.borrow().subagents.clone())
            .map(|bar| bar.borrow().resource_rows(resource_width, resource_budget))
            .unwrap_or_default();
        let mut rows = render_header(
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
                execution: execution.as_deref(),
                jev: jev.as_deref(),
                clock: &chrono::Local::now().format("%H:%M:%S").to_string(),
                resources: &resources,
                graph: &[],
            },
        );
        let graph = self.1.upgrade()
            .and_then(|t| t.borrow().subagents.clone())
            .map(|bar| bar.borrow_mut().graph_rows(graph_space(width as usize, height),
                graph_height(width as usize, height, &rows)))
            .unwrap_or_default();
        overlay_graph(width as usize, height, &mut rows, &graph);
        rows
    }
    fn invalidate(&mut self) {}
}

pub(super) struct HeaderData<'a> {
    pub cwd: &'a str,
    pub session: &'a str,
    pub model: &'a str,
    pub execution: Option<&'a str>,
    pub phase: &'a str,
    pub jev: Option<&'a str>,
    pub clock: &'a str,
    pub resources: &'a [String],
    pub graph: &'a [String],
}

fn execution_label(mode: &InteractiveMode) -> Option<String> {
    use crate::core::execution_mode::ExecutionMode;
    use crate::modes::interactive::components::keybinding_hints::{key_text, KeyTextOptions};
    let state = mode.connection_state.as_ref()?;
    let current = state.execution_mode?;
    let mut pending = None;
    for text in state.session_actions.follow_ups.iter().chain(&state.session_actions.steering) {
        if let Some(command) = crate::core::slash_commands::parse_session_slash_command(text) {
            if command.name == "mode" && !command.args.is_empty() {
                pending = match command.args.as_str() {
                    "cycle" => Some(pending.unwrap_or(current).toggled()),
                    "toggle" => Some(pending.unwrap_or(current).legacy_toggled()),
                    value => ExecutionMode::parse(value).ok().or(pending),
                };
            }
        }
    }
    let label = match pending {
        Some(next) => format!("{} → {} pending", current.label(), next.label()),
        None => current.label().to_string(),
    };
    let key = key_text("app.executionMode.toggle", &KeyTextOptions::default());
    Some(if key.is_empty() { label } else { format!("{label} [{key}]") })
}

fn paired(left: &str, right: &str, width: usize) -> String {
    let right_width = visible_width(right);
    if right_width + 4 >= width {
        return fit(left, width);
    }
    format!("{}  {}", fit(left, width - right_width - 2), right)
}

pub(super) fn graph_space(width: usize, budget: usize) -> usize {
    if width >= 110 && budget >= 7 { 24 } else { 0 }
}

pub(super) fn graph_height(width: usize, budget: usize, header: &[String]) -> usize {
    if graph_space(width, budget) == 0 { return 0; }
    let column = width - resource_space(width, budget).0 - graph_space(width, budget) - 2;
    let metadata = header.len().saturating_sub(2);
    // Include the space beside session metadata, up to the horizontal border.
    // A long model/session label still takes priority over graph decoration.
    let metadata_is_clear = header.get(metadata).is_some_and(|row| {
        strip_ansi(&pi_tui::utils::slice_by_column(row, column, graph_space(width, budget), false))
            .trim().is_empty()
    });
    metadata + usize::from(metadata_is_clear)
}

fn overlay_graph(width: usize, budget: usize, rows: &mut [String], graph: &[String]) {
    let height = graph_height(width, budget, rows);
    let graph_width = graph_space(width, budget);
    if height == 0 || graph.is_empty() { return; }
    let column = width - resource_space(width, budget).0 - graph_width - 2;
    for (row, graph) in rows.iter_mut().take(height).zip(graph) {
        let original = fit(row, width);
        let text = pi_tui::utils::extract_segments(&original, column, column + graph_width,
            width - column - graph_width, false);
        *row = format!("{}{}{}", text.before, fit(graph, graph_width), text.after);
    }
}

fn resource_space(width: usize, budget: usize) -> (usize, usize) {
    if width >= 110 && budget >= 7 {
        (width.saturating_sub(39 + graph_space(width, budget)).min(60), 2)
    } else {
        (width.saturating_sub(2), budget.saturating_sub(2).min(2))
    }
}

pub(super) fn render_header(width: usize, budget: usize, data: &HeaderData<'_>) -> Vec<String> {
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
    let model_width = data.execution.map(|label| width.saturating_sub(visible_width(label) + 15));
    let model_name = clean_label(data.model);
    let model_name = model_width.map(|width| truncate_to_width(&model_name, width.max(1) as f64, "…", false)).unwrap_or(model_name);
    let model = palette.fg("accent", &format!("● {model_name}"));
    let model = match data.execution {
        Some(execution) => format!("{model} · {}", palette.fg("thinkingText", &clean_label(execution))),
        None => model,
    };
    let jev = data.jev.map(|label| {
        let lower = label.to_ascii_lowercase();
        let (color, dot) = if lower.contains("standard mode") || lower.contains("fallback") || lower.contains("error") {
            ("warning", "•")
        } else if lower.contains("off") || lower.contains("disabled") {
            ("dim", "◦")
        } else {
            ("accent", "•")
        };
        palette.fg(color, &format!("{dot} {}", clean_label(label)))
    });
    let mut rows = Vec::new();
    let has_resources = !data.resources.is_empty() && budget >= 3;
    if !has_resources && budget >= 4 {
        rows.push(String::new());
    }
    if budget >= 7 && width >= 110 {
        const LOGO: [&str; 3] = [
            "█▀▀█ █▀▀█ ▀▀█▀▀ ▀█▀ █▀▄▀█ █  █ █▀▀",
            "█  █ █▀▀▀   █    █  █ ▀ █ █  █ ▀▀█",
            "▀▀▀▀ ▀      ▀   ▀▀▀ ▀   ▀ ▀▀▀▀ ▀▀▀",
        ];
        let brand_width = 37;
        let graph_width = if data.graph.is_empty() { 0 } else { graph_space(width, budget) };
        let panel_width = resource_space(width, budget).0;
        let brand_space = width.saturating_sub(panel_width + graph_width + 2);
        let caption_width = if width >= 160 && (graph_width == 0 || brand_space >= 73) { 36 } else { 0 };
        for i in 0..3 {
            let caption = if caption_width > 0 {
                fit(
                    &match i {
                        0 => palette.fg("muted", "BUILT FOR WHAT'S NEXT"),
                        1 => palette.fg("thinkingText", "///"),
                        _ => palette.fg("accent", "AGENTS > CODE > REASON > TEST > SHIP"),
                    },
                    caption_width,
                )
            } else {
                String::new()
            };
            let brand = format!(
                "{}{}",
                fit(&format!(" {}", palette.fg("accent", LOGO[i])), brand_width),
                caption,
            );
            let brand_label = format!("{}  {}", palette.bold(&palette.fg("accent", "telemus.ai")), palette.fg("dim", &format!("v{}", crate::config::VERSION)));
            if has_resources {
                let right = if i == 0 {
                    format!("{brand_label}  {status}  {}", palette.fg("dim", "*shared CPU/RAM"))
                } else {
                    data.resources.get(i - 1).cloned().unwrap_or_default()
                };
                // Fixed column: labels and samples share the brand's left edge.
                let graph = if graph_width > 0 {
                    " ".repeat(graph_width + 1)
                } else { String::new() };
                rows.push(format!("{}{}{}", fit(&brand, width - panel_width - visible_width(&graph) - 1), graph, fit(&right, panel_width)));
            } else {
                let right = match i {
                    0 => brand_label,
                    1 => String::new(),
                    _ => format!("SYSTEM // {status}"),
                };
                rows.push(paired(&brand, &format!("{right} "), width));
            }
        }
        if caption_width == 0 {
            rows.push(paired(
                &palette.fg("muted", " BUILT FOR WHAT'S NEXT  ///"),
                &palette.fg("accent", "AGENTS > CODE > REASON > TEST > SHIP "),
                width,
            ));
        }
    } else if budget >= 3 {
        rows.push(paired(
            &format!("{} {}", palette.fg("accent", if has_resources { " telemus.ai" } else { " OPTIMUS  //  BUILT FOR WHAT'S NEXT" }), palette.fg("dim", &format!("v{}", crate::config::VERSION))),
            &format!("{status} "),
            width,
        ));
        if has_resources {
            rows.extend(data.resources.iter().take(resource_space(width, budget).1)
                .map(|row| format!(" {row}")));
        }
    }
    let session = if budget == 2 {
        format!("OPTIMUS v{} · {}", crate::config::VERSION, clean_label(data.session))
    } else {
        format!("{} · {}", clean_label(data.cwd), clean_label(data.session))
    };
    let right = if width >= 100 {
        format!(
            "{model}  │  {}  │  {}",
            jev.as_deref().unwrap_or(&status),
            data.clock
        )
    } else if width >= 60 {
        format!("{model} · {status}")
    } else {
        model.clone()
    };
    // Drop secondary metadata before hiding the model and execution mode.
    let right = if data.execution.is_some() && visible_width(&right) + 8 >= width {
        model
    } else { right };
    if budget == 1 {
        rows.push(paired(&palette.fg("accent", &format!("OPTIMUS v{}", crate::config::VERSION)), &right, width));
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
    overlay_graph(width, budget, &mut rows, data.graph);
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

#[cfg(test)]
#[path = "native_host_neon_cache_tests.rs"]
mod cache_tests;

#[cfg(test)]
mod image_tests {
    use super::*;

    #[test]
    fn neon_image_rows_preserve_graphics_and_reserve_the_content_column() {
        crate::modes::interactive::theme::theme::init_theme(Some("neon"), false);
        use pi_tui::terminal_image::{image_row_count, position_image};
        let graphic = position_image("\x1bP0;1;0q~\x1b\\", 3);
        let mut wrap = super::super::TranscriptWrapCache::default();
        for width in [24, 40, 120] {
            let timeline = Timeline::new(width);
            let wrapped = wrap.render(vec!["".into(), "".into(), graphic.clone().into()], timeline.content_width());
            assert_eq!(wrapped.len(), 3);
            let decorated = timeline.line(&wrapped[2], None, false);
            // Sidebar-safe relative placement: restore to the pre-rail
            // cursor, then CSI n C (omitted when the rail is empty).
            if timeline.left > 0 {
                assert!(decorated.ends_with(&format!("\x1b8\x1b[{}C{}", timeline.left, graphic)));
            } else {
                assert!(decorated.ends_with(&format!("\x1b8{}", graphic)));
            }
            assert_eq!(image_row_count(&decorated), Some(3));
        }
    }
}
