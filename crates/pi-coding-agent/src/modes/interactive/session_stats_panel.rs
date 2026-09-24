//! Portable, read-only ASCII dashboard mounted immediately above the chat input.
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

use pi_tui::tui::{Component, TUI};
use pi_tui::utils::{truncate_to_width, visible_width};
use tokio::sync::Notify;

use super::super::HostEvent;
use super::stats_data::{Loader, ModelUsage, Snapshot};
use crate::modes::agent_connection::types as wire;
use crate::modes::interactive::theme::theme::theme;

#[derive(Clone, Default)]
pub(crate) struct StatsDock {
    component: Rc<RefCell<Option<Rc<RefCell<dyn Component>>>>>,
}

impl StatsDock {
    pub fn show(&self, component: Rc<RefCell<dyn Component>>) {
        *self.component.borrow_mut() = Some(component);
    }
    pub fn clear(&self) {
        self.component.borrow_mut().take();
    }
}

impl Component for StatsDock {
    fn render(&mut self, width: f64) -> Vec<String> {
        self.component
            .borrow()
            .as_ref()
            .map(|component| component.borrow_mut().render(width))
            .unwrap_or_default()
    }
    fn invalidate(&mut self) {
        if let Some(component) = self.component.borrow().as_ref() {
            component.borrow_mut().invalidate();
        }
    }
}

#[derive(Default)]
struct LiveData {
    snapshot: Option<Snapshot>,
    error: Option<String>,
}

pub(super) struct StatsPanel {
    data: Arc<Mutex<LiveData>>,
    include_children: Arc<AtomicBool>,
    refresh: Arc<Notify>,
    task: Option<tokio::task::JoinHandle<()>>,
    send: mpsc::Sender<HostEvent>,
    page: usize,
    scroll: usize,
    rows: usize,
}

impl StatsPanel {
    pub fn new(
        connection: Arc<dyn wire::AgentConnection>,
        send: mpsc::Sender<HostEvent>,
        ui: Rc<RefCell<TUI>>,
    ) -> Self {
        let data = Arc::new(Mutex::new(LiveData::default()));
        let include_children = Arc::new(AtomicBool::new(false));
        let refresh = Arc::new(Notify::new());
        let (updates, scope, wake, events) = (
            data.clone(),
            include_children.clone(),
            refresh.clone(),
            send.clone(),
        );
        let task = tokio::spawn(async move {
            let mut loader = Loader::default();
            loop {
                let result = tokio::time::timeout(
                    Duration::from_secs(8),
                    loader.collect(&connection, scope.load(Ordering::Relaxed)),
                )
                .await;
                {
                    let mut data = updates.lock().unwrap_or_else(|p| p.into_inner());
                    match result {
                        Ok(Ok(snapshot)) => {
                            data.snapshot = Some(snapshot);
                            data.error = None;
                        }
                        Ok(Err(error)) => data.error = Some(error),
                        Err(_) => {
                            data.error = Some(
                                "Refresh timed out; displaying the last available snapshot".into(),
                            )
                        }
                    }
                }
                if events.send(HostEvent::Render).is_err() {
                    break;
                }
                tokio::select! {
                    _ = tokio::time::sleep(Duration::from_secs(2)) => {},
                    _ = wake.notified() => {},
                }
            }
        });
        let rows = ui.borrow().terminal_rows();
        Self {
            data,
            include_children,
            refresh,
            task: Some(task),
            send,
            page: 0,
            scroll: 0,
            rows,
        }
    }

    fn lines(&self, snapshot: &Snapshot, width: usize) -> Vec<String> {
        let mut lines = Vec::new();
        let title = |text: &str| theme().bold(&theme().fg("thinkingText", text));
        let dim = |text: &str| theme().fg("dim", text);
        let rows = snapshot.accounting.sorted_models();
        match self.page {
            0 => {
                let columns = width >= 96;
                let chart_width = if columns { (width - 3) / 2 } else { width };
                lines.push(title("TOKENS BY MODEL"));
                let peak = rows.first().map(|r| r.total()).unwrap_or(0.0).max(1.0);
                for row in rows.iter().take(8) {
                    lines.push(model_bar(row, peak, chart_width));
                }
                if rows.is_empty() {
                    lines.push(dim("No model usage recorded yet"));
                }
                if rows.len() > 8 {
                    lines.push(dim("More models in the Models section"));
                }
                lines.push(format!(
                    "Recorded model tokens: {}{}",
                    number(snapshot.accounting.total()),
                    if snapshot.accounting.missing() > 0 {
                        " + unreported usage"
                    } else {
                        ""
                    }
                ));
                if columns {
                    let mut savings = vec![title("JEV INPUT TOKENS REMOVED")];
                    savings.extend(savings_lines(snapshot, chart_width));
                    savings.push(dim("Applied estimates; not billed savings."));
                    let mut paired = Vec::new();
                    for index in 0..lines.len().max(savings.len()) {
                        let left = lines.get(index).map(String::as_str).unwrap_or("");
                        let right = savings.get(index).map(String::as_str).unwrap_or("");
                        paired.push(format!(
                            "{} {} {}",
                            truncate_to_width(left, chart_width as f64, "...", true),
                            theme().fg("borderMuted", "|"),
                            truncate_to_width(right, chart_width as f64, "...", false)
                        ));
                    }
                    lines = paired;
                }
                lines.push(title("CURRENT CHAT CONTEXT"));
                lines.extend(context_lines(snapshot, width));
                lines.push(title(&format!(
                    "JEV / {}   COMPACTION: {}",
                    jev_mode_label(snapshot),
                    on_off(snapshot.jev_compaction)
                )));
                lines.push(jev_activity(snapshot));
                if snapshot.jev.records > 0 {
                    lines.push(format!(
                        "Recorded requests: {}   Applied boundaries: {}   Fallbacks: {}",
                        snapshot.jev.requests, snapshot.jev.applied, snapshot.jev.fallbacks
                    ));
                    let overhead: f64 = snapshot.jev.models.values().map(ModelUsage::total).sum();
                    let unknown: usize = snapshot
                        .jev
                        .models
                        .values()
                        .map(|row| row.missing_usage)
                        .sum();
                    lines.push(format!(
                        "JEV token overhead: {}{}",
                        number(overhead),
                        if unknown > 0 {
                            " + unreported usage"
                        } else {
                            ""
                        }
                    ));
                } else {
                    lines.push(dim(
                        "JEV history: no retained records for this scope",
                    ));
                }
                if !columns {
                    lines.push(title("ESTIMATED INPUT TOKENS REMOVED"));
                    lines.extend(savings_lines(snapshot, width));
                    lines.push(dim(
                        "Applied candidate reductions; estimates, not billed savings.",
                    ));
                }
            }
            1 => {
                lines.push(title("MODEL USAGE / CURRENT BRANCH HISTORY"));
                lines.push(dim(
                    "In excludes cache. Calls count recorded responses, not retries.",
                ));
                for row in rows {
                    lines.push(title(&clean(&format!("{}/{}", row.provider, row.model))));
                    lines.push(format!(
                        "  Calls {:<5} In {:>8}  Out {:>8}  Total {:>8}{}",
                        row.calls,
                        number(row.usage.input),
                        number(row.usage.output),
                        number(row.total()),
                        if row.missing_usage > 0 { " + ?" } else { "" }
                    ));
                    lines.push(format!(
                        "  Cache read {:>8}  Cache write {:>8}  Est. cost {}",
                        number(row.usage.cache_read),
                        number(row.usage.cache_write),
                        cost(row)
                    ));
                    if row.missing_usage > 0 {
                        lines.push(dim(&format!(
                            "  Usage unavailable for {} response(s)",
                            row.missing_usage
                        )));
                    }
                    lines.push(String::new());
                }
                if snapshot.accounting.models.is_empty() {
                    lines.push(dim("No model responses recorded yet"));
                }
                lines.push(title("JEV OVERHEAD / RETAINED RECORDS"));
                for row in snapshot.jev.models.values() {
                    lines.push(format!(
                        "{}  requests {}  in {}  out {}{}",
                        clean(&row.model),
                        row.calls,
                        number(row.usage.input),
                        number(row.usage.output),
                        if row.missing_usage > 0 {
                            " + unreported usage"
                        } else {
                            ""
                        }
                    ));
                }
                lines.push(dim("JEV overhead is separate from the model totals above."));
                lines.push(dim(
                    "Unrecorded prices show unavailable, not a zero-cost claim.",
                ));
            }
            2 => {
                lines.push(title(&format!(
                    "JEV / {}",
                    jev_mode_label(snapshot)
                )));
                if snapshot.full_jev {
                    lines.push(dim("Full JEV includes Active; accepted decisions can apply."));
                }
                if snapshot.jev_mode == "compare-active" {
                    lines.push(dim("Comparison logging uses the same JEV request."));
                }
                lines.push(jev_activity(snapshot));
                lines.push(format!(
                    "Independent compaction: {}",
                    on_off(snapshot.jev_compaction)
                ));
                lines.push(String::new());
                lines.push(title("FEATURES"));
                for (name, enabled) in &snapshot.features {
                    lines.push(format!("[{}] {}", if *enabled { "x" } else { " " }, name));
                }
                lines.push(String::new());
                lines.push(title("RETAINED HISTORY / SELECTED SCOPE"));
                if snapshot.jev.records == 0 {
                    lines.push(dim("No retained records for this scope"));
                } else {
                    lines.push(format!(
                        "Requests {}  Applied boundaries {}  Fallbacks {}",
                        snapshot.jev.requests, snapshot.jev.applied, snapshot.jev.fallbacks
                    ));
                    lines.push(format!(
                        "Median request latency: {}",
                        snapshot
                            .jev
                            .median_ms
                            .map(|n| format!("{n}ms"))
                            .unwrap_or_else(|| "not recorded".into())
                    ));
                    let comparisons = snapshot.jev.agrees + snapshot.jev.disagrees;
                    lines.push(if comparisons > 0 {
                        format!(
                            "Compare agreement: {:.1}% ({}/{comparisons} comparable questions)",
                            100.0 * snapshot.jev.agrees as f64 / comparisons as f64,
                            snapshot.jev.agrees
                        )
                    } else {
                        "Compare agreement: no comparable decisions recorded".into()
                    });
                }
                lines.push(String::new());
                lines.extend(savings_lines(snapshot, width));
                let c = &snapshot.jev.compaction;
                lines.push(if c.samples > 0 {
                    format!(
                        "JEV compaction: {} -> {} estimated tokens across {} applications",
                        number(c.before as f64),
                        number(c.after as f64),
                        c.samples
                    )
                } else {
                    "JEV compaction: no measured reduction yet".into()
                });
                lines.push(dim("Savings need applied changes with recorded token estimates."));
                lines.push(dim("Enabled features still need eligible inputs and accepted decisions."));
                lines.push(dim(
                    "Compaction reduction is separate from candidate reductions.",
                ));
                lines.push(dim(
                    "Compare potential savings: no measured projection recorded.",
                ));
                lines.push(dim(
                    "Records may have rotated; this is not lifetime JEV accounting.",
                ));
            }
            _ => {
                lines.push(title("TOKENS PER USER TURN / CURRENT CHAT"));
                lines.push(dim(
                    "Recorded input + output + cache; subagent usage excluded.",
                ));
                lines.push(String::new());
                let turns = &snapshot.accounting.turns;
                let max_columns = width.saturating_sub(10).min(40).max(1);
                let start = turns.len().saturating_sub(max_columns);
                let visible = &turns[start..];
                let peak = visible.iter().copied().fold(0.0_f64, f64::max).max(1.0);
                let height = 7;
                for level in (1..=height).rev() {
                    let limit = peak * level as f64 / height as f64;
                    let cells: String = visible
                        .iter()
                        .map(|&value| {
                            if value > 0.0 && (value / peak * height as f64).ceil() >= level as f64
                            {
                                '#'
                            } else {
                                ' '
                            }
                        })
                        .collect();
                    lines.push(format!(
                        "{:>7} |{}",
                        number(limit),
                        theme().fg("accent", &cells)
                    ));
                }
                lines.push(format!("        +{}", "-".repeat(visible.len().max(1))));
                lines.push(format!(
                    "          Turns {}..{} (latest on the right)",
                    if turns.is_empty() { 0 } else { start + 1 },
                    turns.len()
                ));
                if turns.is_empty() {
                    lines.push(dim("No completed usage to plot yet"));
                }
                if snapshot.accounting.missing() > 0 {
                    lines.push(dim(
                        "Some responses have no usage; bars show reported tokens only.",
                    ));
                }
                lines.push(String::new());
                lines.extend(context_lines(snapshot, width));
            }
        }
        if !snapshot.notes.is_empty() {
            lines.push(String::new());
        }
        lines.extend(
            snapshot
                .notes
                .iter()
                .map(|note| theme().fg("warning", &clean(note))),
        );
        lines
    }

    fn render_at(&mut self, width: usize, height: usize) -> Vec<String> {
        let width = width.max(1);
        if width < 12 || height < 8 {
            return vec![truncate_to_width(
                "/stats - Esc closes",
                width as f64,
                "",
                false,
            )];
        }
        let outer = width.min(112);
        let inner = outer - 4;
        let inset = " ".repeat((width - outer) / 2);
        let live = self.data.lock().unwrap_or_else(|p| p.into_inner());
        let snapshot = live.snapshot.as_ref();
        let mut body = if let Some(snapshot) = snapshot {
            self.lines(snapshot, inner)
        } else {
            vec!["Loading local statistics...".into()]
        };
        if let Some(error) = &live.error {
            body.insert(
                0,
                theme().fg("warning", &format!("Stats refresh: {}", clean(error))),
            );
        }
        let scope = self.include_children.load(Ordering::Relaxed);
        let scope_text = if scope {
            format!(
                "Current chat + {} subagents",
                snapshot.map(|s| s.children).unwrap_or(0)
            )
        } else {
            "Current chat".into()
        };
        let updating = snapshot.is_some_and(|s| s.include_children != scope);
        if updating {
            body = vec!["Updating scope...".into()];
        }
        let visible_rows = height.saturating_sub(7).max(1);
        self.scroll = self.scroll.min(body.len().saturating_sub(visible_rows));
        let border = format!("+{}+", "-".repeat(outer - 2));
        let mut lines = vec![
            border.clone(),
            format!(
                "OPTIMUS // SESSION STATS    {}",
                ["OVERVIEW", "MODELS", "JEV", "TRENDS"][self.page]
            ),
            format!("Scope: {scope_text}   [Subagents: {}]", on_off(scope)),
            snapshot
                .map(|s| {
                    format!(
                        "{} [{}]   Refresh: 2s",
                        clean(&s.name),
                        s.session_id.chars().take(8).collect::<String>()
                    )
                })
                .unwrap_or_default(),
        ];
        lines.extend(body.iter().skip(self.scroll).take(visible_rows).cloned());
        let keys = pi_tui::keybindings::get_keybindings();
        let key = |action: &str| keys.get_keys(action).join("/ ");
        lines.push(format!(
            "[{}] Close  [{}] Section  [{}] Subagents",
            key("tui.select.cancel"),
            key("app.stats.nextSection"),
            key("app.stats.toggleSubagents")
        ));
        lines.push(format!(
            "[{} / {}] Scroll   {}/{}",
            key("tui.select.up"),
            key("tui.select.down"),
            self.scroll + 1,
            body.len().max(1)
        ));
        lines.push(border.clone());
        lines
            .into_iter()
            .enumerate()
            .map(|(index, line)| {
                if line == border {
                    return canvas(&format!("{inset}{}", theme().fg("accent", &line)), width);
                }
                let text = truncate_to_width(&line, inner as f64, "...", false);
                let padding = " ".repeat(inner.saturating_sub(visible_width(&text)));
                let text = if index == 1 {
                    theme().bold(&theme().fg("accent", &text))
                } else {
                    text
                };
                canvas(
                    &format!(
                        "{inset}{} {text}{padding} {}",
                        theme().fg("borderMuted", "|"),
                        theme().fg("borderMuted", "|")
                    ),
                    width,
                )
            })
            .collect()
    }
}

impl Drop for StatsPanel {
    fn drop(&mut self) {
        if let Some(task) = self.task.take() {
            task.abort();
        }
    }
}

impl Component for StatsPanel {
    fn render(&mut self, width: f64) -> Vec<String> {
        // Query through the TUI's portable terminal abstraction without starting it.
        let rows = pi_tui::terminal::Terminal::rows(&pi_tui::terminal::ProcessTerminal::new());
        self.render_at(
            width.max(1.0) as usize,
            dock_height(if rows > 0 { rows } else { self.rows }),
        )
    }
    fn render_with_height(&mut self, width: f64, height: usize) -> Vec<String> {
        self.render_at(width.max(1.0) as usize, height)
    }
    fn invalidate(&mut self) {}
    fn handle_input(&mut self, input: &str) {
        let keys = pi_tui::keybindings::get_keybindings();
        if keys.matches(input, "tui.select.cancel") || keys.matches(input, "app.interrupt") {
            if let Some(task) = self.task.take() {
                task.abort();
            }
            let _ = self.send.send(HostEvent::CloseCommandDialog);
        } else if keys.matches(input, "app.stats.toggleSubagents") {
            self.include_children.fetch_xor(true, Ordering::Relaxed);
            self.scroll = 0;
            self.refresh.notify_one();
        } else if keys.matches(input, "app.stats.nextSection") {
            self.page = (self.page + 1) % 4;
            self.scroll = 0;
        } else if keys.matches(input, "tui.select.down") {
            self.scroll = self.scroll.saturating_add(1);
        } else if keys.matches(input, "tui.select.up") {
            self.scroll = self.scroll.saturating_sub(1);
        } else if keys.matches(input, "tui.editor.pageDown") {
            self.scroll = self.scroll.saturating_add(10);
        } else if keys.matches(input, "tui.editor.pageUp") {
            self.scroll = self.scroll.saturating_sub(10);
        }
        let _ = self.send.send(HostEvent::Render);
    }
}

fn dock_height(rows: usize) -> usize {
    (rows / 2).clamp(1, 32)
}

fn canvas(line: &str, width: usize) -> String {
    // Restore the canvas after inline colour resets; the terminal's default
    // background may be grey even when the rest of the app uses Neon.
    let palette = theme();
    let bg = palette.get_bg_ansi("toolPanelBg");
    let fg = palette.get_fg_ansi("text");
    let line = truncate_to_width(line, width as f64, "", true)
        .replace("\x1b[0m", &format!("\x1b[0m{bg}{fg}"))
        .replace("\x1b[49m", &bg)
        .replace("\x1b[39m", &fg);
    format!("{bg}{fg}{line}\x1b[0m")
}

fn clean(value: &str) -> String {
    value
        .chars()
        .filter(|ch| !ch.is_control())
        .take(200)
        .collect()
}
fn on_off(on: bool) -> &'static str {
    if on {
        "ON"
    } else {
        "OFF"
    }
}
fn number(value: f64) -> String {
    if !value.is_finite() || value < 0.0 {
        "?".into()
    } else if value >= 1_000_000.0 {
        format!("{:.1}m", value / 1_000_000.0)
    } else if value >= 1_000.0 {
        format!("{:.1}k", value / 1_000.0)
    } else {
        format!("{value:.0}")
    }
}
fn cost(row: &ModelUsage) -> String {
    if row.missing_cost == row.calls {
        "unavailable".into()
    } else {
        format!(
            "${:.4}{}",
            row.usage.cost.total,
            if row.missing_cost > 0 {
                " + unknown"
            } else {
                ""
            }
        )
    }
}
fn bar(value: f64, peak: f64, width: usize) -> String {
    let filled = if value <= 0.0 || peak <= 0.0 {
        0
    } else {
        ((value / peak).clamp(0.0, 1.0) * width as f64)
            .round()
            .max(1.0) as usize
    };
    format!(
        "{}{}",
        theme().fg("accent", &"#".repeat(filled)),
        theme().fg("dim", &".".repeat(width.saturating_sub(filled)))
    )
}
fn model_bar(row: &ModelUsage, peak: f64, width: usize) -> String {
    let label_width = (width / 3).clamp(8, 30);
    let label = truncate_to_width(
        &clean(&if width < 72 {
            row.model.clone()
        } else {
            format!("{}/{}", row.provider, row.model)
        }),
        label_width as f64,
        "...",
        false,
    );
    let chart_width = width.saturating_sub(label_width + 19).clamp(1, 36);
    format!(
        "{}{} {} {:>7} {:>3} calls{}",
        label,
        " ".repeat(label_width.saturating_sub(visible_width(&label))),
        bar(row.total(), peak, chart_width),
        number(row.total()),
        row.calls,
        if row.missing_usage > 0 { "+?" } else { "" }
    )
}
fn context_lines(snapshot: &Snapshot, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let tokens = snapshot.context["tokens"].as_f64();
    let window = snapshot.context["contextWindow"].as_f64();
    if let Some((tokens, window)) = tokens.zip(window).filter(|(tokens, window)| {
        tokens.is_finite() && *tokens >= 0.0 && window.is_finite() && *window > 0.0
    }) {
        lines.push(format!(
            "[{}] {:.0}%  {} / {}  remaining {}",
            bar(tokens, window, width.saturating_sub(42).clamp(8, 32)),
            tokens / window * 100.0,
            number(tokens),
            number(window),
            number((window - tokens).max(0.0))
        ));
    } else {
        lines.push("Context usage: unavailable (may need a response after compaction)".into());
    }
    lines.push(format!(
        "Compactions: {} (current chat)",
        snapshot.accounting.compactions
    ));
    let cache: f64 = snapshot
        .accounting
        .models
        .values()
        .map(|m| m.usage.cache_read)
        .sum();
    let input: f64 = snapshot
        .accounting
        .models
        .values()
        .map(|m| m.usage.input + m.usage.cache_read + m.usage.cache_write)
        .sum();
    lines.push(if input > 0.0 && snapshot.accounting.missing() == 0 {
        format!(
            "Cache read share of selected-scope input: {:.1}%",
            cache / input * 100.0
        )
    } else {
        "Cache read share: unavailable".into()
    });
    lines
}
fn jev_mode_label(snapshot: &Snapshot) -> String {
    if snapshot.full_jev {
        "FULL JEV (ACTIVE + COMPARISON LOGGING)".into()
    } else {
        match snapshot.jev_mode.as_str() {
            "compare-active" => "ACTIVE + COMPARISON LOGGING".into(),
            "compare" => "COMPARE ONLY".into(),
            mode => clean(mode).to_uppercase(),
        }
    }
}

fn jev_activity(snapshot: &Snapshot) -> String {
    let usage = &snapshot.pipeline["usage"];
    if !usage.is_object() {
        return "Live JEV activity: no worker status available".into();
    }
    let n = |key: &str| {
        usage[key]
            .as_u64()
            .map(|n| n.to_string())
            .unwrap_or_else(|| "?".into())
    };
    format!(
        "Live/current chat: {}  in flight {}  failed {}  cancelled {}",
        clean(usage["activity"].as_str().unwrap_or("unknown")),
        n("in_flight"),
        n("failed"),
        n("cancelled")
    )
}
fn savings_lines(snapshot: &Snapshot, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let peak = snapshot
        .jev
        .savings
        .values()
        .map(|s| s.before.saturating_sub(s.after))
        .max()
        .unwrap_or(1)
        .max(1) as f64;
    for (category, label) in [
        ("code_search_relevance", "Search"),
        ("context_relevance", "Context"),
        ("memory_relevance", "Memory"),
        ("tool_candidates", "Tool menus"),
    ] {
        if let Some(savings) = snapshot.jev.savings.get(category) {
            let removed = savings.before.saturating_sub(savings.after) as f64;
            lines.push(format!(
                "{label:<10} {} {:>7}  kept {}/{}",
                bar(removed, peak, width.saturating_sub(42).clamp(1, 28)),
                number(removed),
                savings.retained,
                savings.candidates
            ));
        } else {
            lines.push(format!("{label:<10} No measured savings yet"));
        }
    }
    lines
}

#[cfg(test)]
#[path = "session_stats_panel_tests.rs"]
mod tests;
