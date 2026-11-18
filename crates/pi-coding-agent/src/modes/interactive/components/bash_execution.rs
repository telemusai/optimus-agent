//! Port of packages/coding-agent/src/modes/interactive/components/bash-execution.ts

use pi_tui::components::loader::{Loader, LoaderIndicatorOptions};
use pi_tui::components::spacer::Spacer;
use pi_tui::components::text::Text;
use pi_tui::tui::{Component, Container, TUI};
use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

use crate::core::tools::truncate::{
    TruncatedBy, TruncationResult, DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES,
};
use crate::modes::interactive::components::dynamic_border::DynamicBorder;
use crate::modes::interactive::components::keybinding_hints::{
    expand_collapse_hint, key_text, KeyTextOptions,
};
use crate::modes::interactive::components::visual_truncate::truncate_to_visual_lines;
use crate::modes::interactive::theme::theme::theme;

const PREVIEW_LINES: usize = 20;

/// `"running" | "complete" | "cancelled" | "error"`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BashExecutionStatus {
    Running,
    Complete,
    Cancelled,
    Error,
}

/// `{ suppressLeadingSpace?: boolean }`
#[derive(Debug, Clone, Copy, Default)]
pub struct BashExecutionOptions {
    pub suppress_leading_space: bool,
}

/// Port of `BashExecutionComponent`.
pub struct BashExecutionComponent {
    command: String,
    output_lines: VecDeque<String>,
    retained_bytes: usize,
    discarded_bytes: usize,
    discarded_lines: usize,
    // Bytes previously cut from the current first line still count when the
    // reference algorithm decides whether that original line fits a later tail.
    first_line_truncated_bytes: usize,
    retained_first_line_partial: bool,
    display_dirty: bool,
    // A terminal CR is already rendered as LF. Remember it so a leading LF in
    // the next chunk completes that CRLF pair instead of adding another line.
    pending_carriage_return: bool,
    #[cfg(test)]
    display_rebuild_count: usize,
    status: BashExecutionStatus,
    exit_code: Option<f64>,
    error_message: Option<String>,
    loader: Rc<RefCell<Loader>>,
    truncation_result: Option<TruncationResult>,
    full_output_path: Option<String>,
    expanded: bool,
    content_container: Rc<RefCell<Container>>,
    container: Container,
}

impl BashExecutionComponent {
    /// Port of the `BashExecutionComponent` constructor.
    pub fn new(
        command: &str,
        ui: Rc<RefCell<TUI>>,
        exclude_from_context: bool,
        options: BashExecutionOptions,
    ) -> Self {
        // Use dim border for excluded-from-context commands (!! prefix)
        let color_key: crate::modes::interactive::theme::theme::ThemeColor = if exclude_from_context
        {
            "dim"
        } else {
            "bashMode"
        };
        let border_color: crate::modes::interactive::components::dynamic_border::ColorFn =
            Box::new(move |text: &str| theme().fg(color_key, text));

        let mut container = Container::new();

        // Keep tool activity tight against a preceding agent-message notification.
        if !options.suppress_leading_space {
            container
                .add_child(Rc::new(RefCell::new(Spacer::new(1))) as Rc<RefCell<dyn Component>>);
        }

        container.add_child(Rc::new(RefCell::new(DynamicBorder::new(Box::new(
            move |text: &str| theme().fg(color_key, text),
        )))) as Rc<RefCell<dyn Component>>);

        let content_container = Rc::new(RefCell::new(Container::new()));
        container.add_child(Rc::clone(&content_container) as Rc<RefCell<dyn Component>>);

        let header = Text::new(
            theme().fg(color_key, &theme().bold(&format!("$ {command}"))),
            1,
            0,
            None,
        );
        content_container
            .borrow_mut()
            .add_child(Rc::new(RefCell::new(header)) as Rc<RefCell<dyn Component>>);

        let loader = Rc::new(RefCell::new(Loader::new(
            ui,
            Box::new(|spinner: &str| theme().fg("muted", spinner)),
            Box::new(|text: &str| theme().fg("muted", text)),
            format!(
                "Running... ({} to cancel)",
                key_text("tui.select.cancel", &KeyTextOptions::default())
            ),
            None::<LoaderIndicatorOptions>,
        )));
        // The TypeScript starts the loader animation through `Loader.setIndicator`.
        loader.borrow_mut().start();
        content_container
            .borrow_mut()
            .add_child(Rc::clone(&loader) as Rc<RefCell<dyn Component>>);

        container
            .add_child(Rc::new(RefCell::new(DynamicBorder::new(border_color)))
                as Rc<RefCell<dyn Component>>);

        Self {
            command: command.to_string(),
            output_lines: VecDeque::new(),
            retained_bytes: 0,
            discarded_bytes: 0,
            discarded_lines: 0,
            first_line_truncated_bytes: 0,
            retained_first_line_partial: false,
            display_dirty: false,
            pending_carriage_return: false,
            #[cfg(test)]
            display_rebuild_count: 0,
            status: BashExecutionStatus::Running,
            exit_code: None,
            error_message: None,
            loader,
            truncation_result: None,
            full_output_path: None,
            expanded: false,
            content_container,
            container,
        }
    }

    /// Set whether the output is expanded (shows full output) or collapsed
    /// (preview only).
    pub fn set_expanded(&mut self, expanded: bool) {
        self.expanded = expanded;
        self.display_dirty = true;
    }

    /// Port of `appendOutput`.
    pub fn append_output(&mut self, chunk: &str) {
        // Note: binary data is already sanitized in tui-renderer.ts executeBashCommand
        let clean = strip_ansi(chunk);
        let clean_input = if self.pending_carriage_return && !clean.is_empty() {
            self.pending_carriage_return = false;
            clean.strip_prefix('\n').unwrap_or(&clean)
        } else {
            &clean
        };
        if !clean.is_empty() {
            self.pending_carriage_return = clean_input.ends_with('\r');
        }
        let clean = clean_input.replace("\r\n", "\n").replace('\r', "\n");
        let mut new_lines = clean.split('\n');

        if let Some(last) = self.output_lines.back_mut() {
            if let Some(first) = new_lines.next() {
                self.retained_bytes += first.len();
                last.push_str(first);
            }
        }
        for line in new_lines {
            // Discard before growing at the line limit. In particular, one large
            // input chunk cannot make the retained deque allocate for all of its
            // lines before the old-head truncation pass runs.
            if self.output_lines.len() >= DEFAULT_MAX_LINES {
                self.discard_front_line();
            }
            if !self.output_lines.is_empty() {
                self.retained_bytes += 1;
            }
            self.retained_bytes += line.len();
            self.output_lines.push_back(line.to_string());
        }
        if self.output_lines.is_empty() {
            self.output_lines.push_back(String::new());
        }

        self.discard_excess_head();
        self.display_dirty = true;
    }

    fn discard_excess_head(&mut self) {
        while self.output_lines.len() > DEFAULT_MAX_LINES {
            self.discard_front_line();
        }

        while self
            .retained_bytes
            .saturating_add(self.first_line_truncated_bytes)
            > DEFAULT_MAX_BYTES
        {
            if self.output_lines.len() == 1 {
                self.truncate_front_line();
                break;
            }

            let trailing_lines_are_blank =
                self.output_lines.iter().skip(1).all(|line| line.is_empty());
            if trailing_lines_are_blank {
                self.truncate_front_line();
                break;
            }
            self.discard_front_line();
        }
    }

    fn discard_front_line(&mut self) {
        let Some(line) = self.output_lines.pop_front() else {
            return;
        };
        let discarded = line.len() + usize::from(!self.output_lines.is_empty());
        self.retained_bytes = self.retained_bytes.saturating_sub(discarded);
        self.discarded_bytes += discarded;
        self.discarded_lines += 1;
        self.first_line_truncated_bytes = 0;
        self.retained_first_line_partial = false;
    }

    fn truncate_front_line(&mut self) {
        let trailing_bytes = self
            .retained_bytes
            .saturating_sub(self.output_lines.front().map_or(0, String::len));
        let budget = DEFAULT_MAX_BYTES.saturating_sub(trailing_bytes);
        let Some(line) = self.output_lines.front_mut() else {
            return;
        };
        let mut start = line.len().saturating_sub(budget);
        while start < line.len() && !line.is_char_boundary(start) {
            start += 1;
        }
        self.retained_bytes = self.retained_bytes.saturating_sub(start);
        self.discarded_bytes += start;
        self.first_line_truncated_bytes += start;
        if start > 0 {
            self.retained_first_line_partial = true;
        }
        line.drain(..start);
        line.shrink_to_fit();
    }

    /// Port of `setComplete`.
    pub fn set_complete(
        &mut self,
        exit_code: Option<f64>,
        cancelled: bool,
        truncation_result: Option<TruncationResult>,
        full_output_path: Option<String>,
    ) {
        self.exit_code = exit_code;
        self.status = if cancelled {
            BashExecutionStatus::Cancelled
        } else if exit_code.is_some_and(|exit_code| exit_code != 0.0) {
            // `exitCode !== 0 && exitCode !== undefined && exitCode !== null`
            BashExecutionStatus::Error
        } else {
            BashExecutionStatus::Complete
        };
        self.truncation_result = truncation_result;
        self.full_output_path = full_output_path;

        self.loader.borrow_mut().stop();

        self.display_dirty = true;
    }

    /// Mark the execution as failed before producing a result (e.g. spawn failure).
    pub fn set_failed(&mut self, message: &str) {
        self.error_message = Some(message.to_string());
        self.status = BashExecutionStatus::Error;
        self.loader.borrow_mut().stop();
        self.display_dirty = true;
    }

    /// Port of `updateDisplay`.
    fn update_display(&mut self) {
        #[cfg(test)]
        {
            self.display_rebuild_count += 1;
        }
        self.display_dirty = false;
        let context_truncation = self.output_truncation();

        // Recompute wrapping from the render width so resizes and split panes
        // cannot use stale columns.
        let available_lines: Vec<String> = if context_truncation.content.is_empty() {
            Vec::new()
        } else {
            context_truncation
                .content
                .split('\n')
                .map(|line| line.to_string())
                .collect()
        };

        let preview_start = available_lines.len().saturating_sub(PREVIEW_LINES);
        let preview_logical_lines: Vec<String> = available_lines[preview_start..].to_vec();
        let hidden_line_count = available_lines.len() - preview_logical_lines.len();

        self.content_container.borrow_mut().clear();

        let header = Text::new(
            theme().fg("bashMode", &theme().bold(&format!("$ {}", self.command))),
            1,
            0,
            None,
        );
        self.content_container
            .borrow_mut()
            .add_child(Rc::new(RefCell::new(header)) as Rc<RefCell<dyn Component>>);

        if !available_lines.is_empty() {
            if self.expanded {
                let display_text = available_lines
                    .iter()
                    .map(|line| theme().fg("muted", line))
                    .collect::<Vec<String>>()
                    .join("\n");
                self.content_container
                    .borrow_mut()
                    .add_child(Rc::new(RefCell::new(Text::new(
                        format!("\n{display_text}"),
                        1,
                        0,
                        None,
                    ))) as Rc<RefCell<dyn Component>>);
            } else {
                // Use shared visual truncation utility with width-aware caching
                let styled_output = preview_logical_lines
                    .iter()
                    .map(|line| theme().fg("muted", line))
                    .collect::<Vec<String>>()
                    .join("\n");
                let styled_input = format!("\n{styled_output}");
                self.content_container
                    .borrow_mut()
                    .add_child(Rc::new(RefCell::new(PreviewText::new(styled_input)))
                        as Rc<RefCell<dyn Component>>);
            }
        }

        if self.status == BashExecutionStatus::Running {
            self.content_container
                .borrow_mut()
                .add_child(Rc::clone(&self.loader) as Rc<RefCell<dyn Component>>);
        } else {
            let mut status_parts: Vec<String> = Vec::new();

            if hidden_line_count > 0 {
                if self.expanded {
                    status_parts.push(expand_collapse_hint("app.tools.expand", true));
                } else {
                    status_parts.push(format!(
                        "{} {}",
                        theme().fg("muted", &format!("... {hidden_line_count} more lines")),
                        expand_collapse_hint("app.tools.expand", false)
                    ));
                }
            }

            if self.status == BashExecutionStatus::Cancelled {
                status_parts.push(theme().fg("warning", "(cancelled)"));
            } else if self.status == BashExecutionStatus::Error {
                let text = match &self.error_message {
                    Some(message) => format!("(failed: {message})"),
                    None => format!("(exit {})", js_number_display(self.exit_code)),
                };
                status_parts.push(theme().fg("error", &text));
            }

            // Add truncation warning (context truncation, not preview truncation)
            let was_truncated = self
                .truncation_result
                .as_ref()
                .map(|result| result.truncated)
                .unwrap_or(false)
                || context_truncation.truncated;
            if was_truncated {
                if let Some(full_output_path) = &self.full_output_path {
                    status_parts.push(theme().fg(
                        "warning",
                        &format!("Output truncated. Full output: {full_output_path}"),
                    ));
                }
            }

            if !status_parts.is_empty() {
                self.content_container
                    .borrow_mut()
                    .add_child(Rc::new(RefCell::new(Text::new(
                        format!("\n{}", status_parts.join("\n")),
                        1,
                        0,
                        None,
                    ))) as Rc<RefCell<dyn Component>>);
            }
        }
    }

    /// Return the retained output and exact totals for the original stream.
    pub fn output_truncation(&self) -> TruncationResult {
        let content = self.get_output();
        let truncated = self.discarded_bytes > 0;
        let total_lines = if self.output_lines.is_empty() {
            1
        } else {
            self.discarded_lines + self.output_lines.len()
        };
        let truncated_by = if !truncated {
            None
        } else if self.retained_first_line_partial {
            Some(TruncatedBy::Bytes)
        } else if self.output_lines.len() >= DEFAULT_MAX_LINES && total_lines > DEFAULT_MAX_LINES {
            Some(TruncatedBy::Lines)
        } else {
            Some(TruncatedBy::Bytes)
        };
        TruncationResult {
            content,
            truncated,
            truncated_by,
            total_lines,
            total_bytes: self.discarded_bytes + self.retained_bytes,
            output_lines: if self.output_lines.is_empty() {
                1
            } else {
                self.output_lines.len()
            },
            output_bytes: self.retained_bytes,
            last_line_partial: self.retained_first_line_partial,
            first_line_exceeds_limit: false,
            max_lines: DEFAULT_MAX_LINES,
            max_bytes: DEFAULT_MAX_BYTES,
        }
    }

    /// Get the retained output for creating BashExecutionMessage.
    pub fn get_output(&self) -> String {
        self.output_lines
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Get the command that was executed.
    pub fn get_command(&self) -> &str {
        &self.command
    }
}

/// JS template interpolation of an optional exit code (`${this.exitCode}` prints
/// `undefined` for a missing code).
fn js_number_display(value: Option<f64>) -> String {
    match value {
        None => "undefined".to_string(),
        Some(value) => {
            if value.fract() == 0.0 && value.abs() < 1e21 {
                format!("{}", value as i64)
            } else {
                format!("{value}")
            }
        }
    }
}

/// Port of the inline width-aware preview component in `updateDisplay`.
struct PreviewText {
    styled_input: String,
    cached_width: Option<usize>,
    cached_lines: Option<Vec<String>>,
}

impl PreviewText {
    fn new(styled_input: String) -> Self {
        Self {
            styled_input,
            cached_width: None,
            cached_lines: None,
        }
    }
}

impl Component for PreviewText {
    fn render(&mut self, width: f64) -> Vec<String> {
        let width = width.max(0.0).floor() as usize;
        if self.cached_lines.is_none() || self.cached_width != Some(width) {
            let result =
                truncate_to_visual_lines(&self.styled_input, PREVIEW_LINES, width as f64, 1);
            self.cached_lines = Some(result.visual_lines);
            self.cached_width = Some(width);
        }
        self.cached_lines.clone().unwrap_or_default()
    }

    fn invalidate(&mut self) {
        self.cached_width = None;
        self.cached_lines = None;
    }
}

impl Component for BashExecutionComponent {
    fn render(&mut self, width: f64) -> Vec<String> {
        if self.display_dirty {
            self.update_display();
        }
        self.container.render(width)
    }

    fn get_selection_regions(&self) -> Vec<pi_tui::selection_metadata::TableCellSelectionRegion> {
        self.container.get_selection_regions()
    }

    fn invalidate(&mut self) {
        self.container.invalidate();
        self.display_dirty = true;
    }
}

/// Port of `stripAnsi` for the output sanitised by the TUI renderer.
fn strip_ansi(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(character) = chars.next() {
        if character == '\u{1b}' {
            match chars.peek() {
                Some('[') => {
                    chars.next();
                    while let Some(&next) = chars.peek() {
                        chars.next();
                        if next.is_ascii_alphabetic() {
                            break;
                        }
                    }
                }
                Some(']') => {
                    chars.next();
                    while let Some(&next) = chars.peek() {
                        chars.next();
                        if next == '\u{7}' {
                            break;
                        }
                    }
                }
                _ => {}
            }
            continue;
        }
        result.push(character);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tools::truncate::{truncate_tail, TruncationOptions};
    use crate::modes::interactive::theme::theme::init_theme;
    use pi_tui::terminal::ProcessTerminal;
    use pi_tui::tui::TUI;

    fn init() {
        init_theme(Some("prime"), false);
    }

    fn ui() -> Rc<RefCell<TUI>> {
        Rc::new(RefCell::new(TUI::new(
            Box::new(ProcessTerminal::new()),
            None,
        )))
    }

    fn strip(text: &str) -> String {
        strip_ansi(text)
    }

    #[tokio::test]
    async fn running_component_renders_the_command_and_loader_message() {
        init();
        let mut component =
            BashExecutionComponent::new("echo hi", ui(), false, BashExecutionOptions::default());
        let lines = component.render(40.0);
        let text = lines
            .iter()
            .map(|line| strip(line))
            .collect::<Vec<String>>()
            .join("\n");
        assert!(text.contains("$ echo hi"));
        assert!(text.contains("Running..."));
    }

    #[tokio::test]
    async fn append_output_merges_partial_lines() {
        init();
        let mut component =
            BashExecutionComponent::new("cmd", ui(), false, BashExecutionOptions::default());
        component.append_output("a");
        component.append_output("b\nc");
        assert_eq!(component.get_output(), "ab\nc");
    }

    #[tokio::test]
    async fn append_output_normalises_newlines_and_strips_ansi() {
        init();
        let mut component =
            BashExecutionComponent::new("cmd", ui(), false, BashExecutionOptions::default());
        component.append_output("\u{1b}[31mred\u{1b}[0m\r\nnext\rthird");
        assert_eq!(component.get_output(), "red\nnext\nthird");
    }

    #[tokio::test]
    async fn split_crlf_chunks_match_one_shot_with_exact_counters() {
        init();
        let oversized = format!("{}\r", "é".repeat(DEFAULT_MAX_BYTES));
        let prefix_sensitive = format!("a{}\r", "😀".repeat(DEFAULT_MAX_BYTES));
        let limit_lines = format!("{}\r", "x\n".repeat(DEFAULT_MAX_LINES + 2));
        let cases = vec![
            vec!["a\r", "\nb"],
            vec!["a\r", "", "\nb"],
            vec!["a\r", "\u{1b}[31m", "\nb"],
            vec!["a\r", "\r", "\n", "\n", "b"],
            vec!["é\r", "\n😀"],
            vec![oversized.as_str(), "\n尾"],
            vec![prefix_sensitive.as_str(), "x\r"],
            vec![limit_lines.as_str(), "\nend"],
        ];

        for (case_index, chunks) in cases.iter().enumerate() {
            let full = chunks.concat();
            let mut component = BashExecutionComponent::new(
                "cmd",
                ui(),
                false,
                BashExecutionOptions::default(),
            );
            let mut prefix = String::new();
            for (chunk_index, chunk) in chunks.iter().enumerate() {
                prefix.push_str(chunk);
                component.append_output(chunk);
                let normalised = strip_ansi(&prefix).replace("\r\n", "\n").replace('\r', "\n");
                let expected = truncate_tail(&normalised, TruncationOptions::default());
                assert_eq!(
                    component.output_truncation(),
                    expected,
                    "case {case_index}, chunk {chunk_index}"
                );
                assert_eq!(component.get_output(), expected.content);
                assert_eq!(
                    component.discarded_bytes + component.retained_bytes,
                    normalised.len(),
                    "byte counters at case {case_index}, chunk {chunk_index}"
                );
                assert_eq!(
                    component.discarded_lines + component.output_lines.len(),
                    normalised.split('\n').count(),
                    "line counters at case {case_index}, chunk {chunk_index}"
                );
            }

            let mut one_shot = BashExecutionComponent::new(
                "cmd",
                ui(),
                false,
                BashExecutionOptions::default(),
            );
            one_shot.append_output(&full);
            component.set_complete(Some(0.0), false, None, None);
            one_shot.set_complete(Some(0.0), false, None, None);
            assert_eq!(component.render(60.0), one_shot.render(60.0));
        }
    }

    #[tokio::test]
    async fn one_large_chunk_does_not_set_retained_storage_capacity() {
        init();
        let full: String = (0..DEFAULT_MAX_LINES * 4)
            .map(|index| format!("line {index} café\n"))
            .collect();
        let mut component =
            BashExecutionComponent::new("cmd", ui(), false, BashExecutionOptions::default());
        component.append_output(&full);

        assert_eq!(
            component.output_truncation(),
            truncate_tail(&full, TruncationOptions::default())
        );
        assert_eq!(component.output_lines.len(), DEFAULT_MAX_LINES);
        assert!(
            component.output_lines.capacity() <= DEFAULT_MAX_LINES * 2,
            "retained capacity {} must not depend on the 8000-line input chunk",
            component.output_lines.capacity()
        );
        assert_eq!(component.discarded_lines, DEFAULT_MAX_LINES * 3 + 1);
        assert_eq!(component.discarded_bytes + component.retained_bytes, full.len());
        assert_eq!(component.display_rebuild_count, 0);
    }

    #[tokio::test]
    async fn complete_with_exit_code_zero_is_not_an_error() {
        init();
        let mut component =
            BashExecutionComponent::new("cmd", ui(), false, BashExecutionOptions::default());
        component.set_complete(Some(0.0), false, None, None);
        let text = component
            .render(40.0)
            .iter()
            .map(|line| strip(line))
            .collect::<Vec<String>>()
            .join("\n");
        assert!(!text.contains("(exit"));
    }

    #[tokio::test]
    async fn nonzero_exit_code_renders_the_error_status() {
        init();
        let mut component =
            BashExecutionComponent::new("cmd", ui(), false, BashExecutionOptions::default());
        component.set_complete(Some(2.0), false, None, None);
        let text = component
            .render(40.0)
            .iter()
            .map(|line| strip(line))
            .collect::<Vec<String>>()
            .join("\n");
        assert!(text.contains("(exit 2)"));
    }

    #[tokio::test]
    async fn cancelled_execution_renders_the_warning() {
        init();
        let mut component =
            BashExecutionComponent::new("cmd", ui(), false, BashExecutionOptions::default());
        component.set_complete(Some(130.0), true, None, None);
        let text = component
            .render(40.0)
            .iter()
            .map(|line| strip(line))
            .collect::<Vec<String>>()
            .join("\n");
        assert!(text.contains("(cancelled)"));
    }

    #[tokio::test]
    async fn set_failed_prefers_the_message_over_the_exit_code() {
        init();
        let mut component =
            BashExecutionComponent::new("cmd", ui(), false, BashExecutionOptions::default());
        component.set_failed("spawn ENOENT");
        let text = component
            .render(40.0)
            .iter()
            .map(|line| strip(line))
            .collect::<Vec<String>>()
            .join("\n");
        assert!(text.contains("(failed: spawn ENOENT)"));
    }

    #[tokio::test]
    async fn hidden_lines_are_counted_and_the_hint_renders() {
        init();
        let mut component =
            BashExecutionComponent::new("cmd", ui(), false, BashExecutionOptions::default());
        let output: String = (0..25).map(|index| format!("line {index}\n")).collect();
        component.append_output(&output);
        component.set_complete(Some(0.0), false, None, None);
        let text = component
            .render(60.0)
            .iter()
            .map(|line| strip(line))
            .collect::<Vec<String>>()
            .join("\n");
        assert!(text.contains("... 6 more lines"));
        assert!(text.contains("to expand"));
    }

    #[tokio::test]
    async fn retained_tail_matches_reference_across_utf8_and_chunk_boundaries() {
        init();
        let full = format!("a{}", "é".repeat(DEFAULT_MAX_BYTES));
        let mut component =
            BashExecutionComponent::new("cmd", ui(), false, BashExecutionOptions::default());
        let mut chunk = String::new();
        for character in full.chars() {
            chunk.push(character);
            if chunk.len() >= 997 {
                component.append_output(&chunk);
                chunk.clear();
            }
        }
        component.append_output(&chunk);

        let expected = truncate_tail(&full, TruncationOptions::default());
        assert_eq!(component.output_truncation(), expected);
        assert_eq!(component.get_output(), expected.content);
        assert!(component.retained_bytes <= DEFAULT_MAX_BYTES);
        assert!(component.output_lines.len() <= DEFAULT_MAX_LINES);
        assert_eq!(
            component.discarded_bytes + component.retained_bytes,
            full.len(),
            "discarded and retained bytes account for every input byte"
        );
        assert_eq!(component.display_rebuild_count, 0);

        component.render(60.0);
        component.render(60.0);
        assert_eq!(component.display_rebuild_count, 1);
        component.append_output("tail");
        component.render(60.0);
        assert_eq!(component.display_rebuild_count, 2);
    }

    #[tokio::test]
    async fn retained_line_tail_and_preview_counts_match_reference() {
        init();
        let full: String = (0..DEFAULT_MAX_LINES + 25)
            .map(|index| format!("line {index} café\n"))
            .collect();
        let mut component =
            BashExecutionComponent::new("cmd", ui(), false, BashExecutionOptions::default());
        let lines: Vec<&str> = full.split_inclusive('\n').collect();
        for chunk in lines.chunks(3) {
            component.append_output(&chunk.concat());
        }

        let expected = truncate_tail(&full, TruncationOptions::default());
        assert_eq!(component.output_truncation(), expected);
        assert_eq!(component.discarded_lines, 26);
        assert!(component.retained_bytes <= DEFAULT_MAX_BYTES);
        assert_eq!(component.display_rebuild_count, 0);

        component.set_complete(Some(0.0), false, None, Some("/tmp/full.txt".to_string()));
        let text = component
            .render(100.0)
            .iter()
            .map(|line| strip(line))
            .collect::<Vec<String>>()
            .join("\n");
        assert!(text.contains("... 1980 more lines"), "{text}");
        assert!(text.contains("Output truncated. Full output: /tmp/full.txt"));
        assert_eq!(component.display_rebuild_count, 1);
    }

    #[tokio::test]
    async fn oversized_line_with_trailing_blanks_matches_reference() {
        init();
        let full = format!("{}\n\n", "x".repeat(DEFAULT_MAX_BYTES + 20));
        let mut component =
            BashExecutionComponent::new("cmd", ui(), false, BashExecutionOptions::default());
        component.append_output(&full);
        assert_eq!(
            component.output_truncation(),
            truncate_tail(&full, TruncationOptions::default())
        );
    }

    #[tokio::test]
    async fn truncation_warning_names_the_full_output_path() {
        init();
        let mut component =
            BashExecutionComponent::new("cmd", ui(), false, BashExecutionOptions::default());
        component.append_output("short\n");
        component.set_complete(Some(0.0), false, None, Some("/tmp/full.txt".to_string()));
        // `truncationResult.truncated` is false and the output is short, so no warning.
        let text = component
            .render(60.0)
            .iter()
            .map(|line| strip(line))
            .collect::<Vec<String>>()
            .join("\n");
        assert!(!text.contains("Output truncated"));

        let truncated = crate::core::tools::truncate::truncate_tail(
            &"x\n".repeat(10),
            TruncationOptions {
                max_lines: Some(2),
                max_bytes: None,
            },
        );
        let mut component =
            BashExecutionComponent::new("cmd", ui(), false, BashExecutionOptions::default());
        component.append_output("x\n");
        component.set_complete(
            Some(0.0),
            false,
            Some(truncated),
            Some("/tmp/full.txt".to_string()),
        );
        let text = component
            .render(60.0)
            .iter()
            .map(|line| strip(line))
            .collect::<Vec<String>>()
            .join("\n");
        assert!(text.contains("Output truncated. Full output: /tmp/full.txt"));
    }
}
