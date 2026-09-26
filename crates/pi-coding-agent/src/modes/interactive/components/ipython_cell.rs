//! Port of packages/coding-agent/src/modes/interactive/components/ipython-cell.ts

use pi_tui::render_cache::VersionedRenderCache;
use pi_tui::tui::Component;
use pi_tui::utils::{truncate_to_width, visible_width, wrap_text_with_ansi};
use regex::Regex;
use serde_json::Value;

use crate::core::agent_messages::{
    format_agent_message_participant, AgentMessageDirection, AgentSessionMessageSender,
};
use crate::core::kernel::shared::{
    KernelDeliveryStatus, KernelReceiverRole, KernelSentAgentMessage,
};
use crate::core::tools::code_preview::preview_ipython_code;
use crate::core::tools::edit_diff::generate_diff_string;
use crate::core::tools::ipython_cell_code::parse_ipython_bash_cell;
use crate::modes::interactive::components::agent_message::{
    agent_message_body_lines, agent_message_preview, agent_message_summary_line,
};
use crate::modes::interactive::components::collapsible_error::{
    normalize_error_details, summarize_error_details,
};
use crate::modes::interactive::components::diff::{
    render_diff_separator, render_rich_diff, RichDiffOptions,
};
use crate::modes::interactive::components::edit_summary::{
    count_changed_lines, format_file_change_summary_line, FILE_CHANGE_DIFF_INDENT,
};
use crate::modes::interactive::components::keybinding_hints::expand_collapse_hint;
use crate::modes::interactive::theme::theme::theme;
use crate::modes::interactive::theme::theme::{get_language_from_path, highlight_code};
use crate::modes::interactive::theme::working_icon::{
    get_working_pulse_frame, working_icon_frame, WORKING_ICON_FRAMES,
};

/// `IPythonCellContentBlock`
#[derive(Debug, Clone, Default, PartialEq)]
pub struct IPythonCellContentBlock {
    pub r#type: String,
    pub text: Option<String>,
    pub data: Option<String>,
    pub mime_type: Option<String>,
}

/// `IPythonCellState`
#[derive(Debug, Clone, Default)]
pub struct IPythonCellState {
    pub code: String,
    pub content: Option<Vec<IPythonCellContentBlock>>,
    pub details: Option<Value>,
    pub is_partial: Option<bool>,
    pub is_error: Option<bool>,
    pub expanded: Option<bool>,
    pub agent_messages_expanded: Option<bool>,
    pub edit_diffs_expanded: Option<bool>,
    pub show_expand_hint: Option<bool>,
    pub execution_started: Option<bool>,
    pub args_complete: Option<bool>,
    pub show_images: Option<bool>,
    /// Session cwd - edit paths nested under it render relative, else absolute.
    pub cwd: Option<String>,
}

/// `DiffDisplay`
#[derive(Debug, Clone, PartialEq)]
struct DiffDisplay {
    path: String,
    old_str: String,
    new_str: String,
    start_line: Option<f64>,
}

/// `IpythonErrorDetails`
#[derive(Debug, Clone, PartialEq)]
struct IpythonErrorDetails {
    ename: String,
    evalue: String,
    traceback: Vec<String>,
}

/// `IpythonDetails`
#[derive(Debug, Clone, Default)]
struct IpythonDetails {
    duration_ms: Option<f64>,
    status: Option<String>,
    error_ename: Option<String>,
    stdout: Option<String>,
    stderr: Option<String>,
    result: Option<String>,
    background_output: Option<String>,
    diffs: Option<Vec<DiffDisplay>>,
    sent_agent_messages: Option<Vec<KernelSentAgentMessage>>,
    error: Option<IpythonErrorDetails>,
}

/// `TracebackParts`
struct TracebackParts {
    output: String,
    traceback: String,
    preview: String,
}

/// `MAGIC_LINE_PATTERN`
fn magic_line_pattern() -> &'static Regex {
    static PATTERN: std::sync::LazyLock<Regex> =
        std::sync::LazyLock::new(|| Regex::new(r"^\s*!").expect("valid magic line pattern"));
    &PATTERN
}

/// Two columns, matching the code body's "› "/"  " gutter so output aligns under it.
const OUTPUT_INDENT: &str = "  ";

/// `SGR_PATTERN`
fn sgr_pattern() -> &'static Regex {
    static PATTERN: std::sync::LazyLock<Regex> =
        std::sync::LazyLock::new(|| Regex::new(r"\x1b\[([0-9;]*)m").expect("valid sgr pattern"));
    &PATTERN
}

/// Append `ESC[0m` when `line` ends with a foreground or background color still
/// open, so a span that `wrapTextWithAnsi` split across lines cannot bleed into
/// the trailing padding or the next line.
fn close_open_sgr(line: &str) -> String {
    let mut fg_open = false;
    let mut bg_open = false;
    for captures in sgr_pattern().captures_iter(line) {
        let params_group = captures.get(1).map(|group| group.as_str()).unwrap_or("");
        let params: Vec<String> = if params_group.is_empty() {
            vec!["0".to_string()]
        } else {
            params_group
                .split(';')
                .map(|part| part.to_string())
                .collect()
        };
        let mut index = 0usize;
        while index < params.len() {
            let code = params[index].parse::<f64>().unwrap_or(0.0);
            if code == 0.0 {
                fg_open = false;
                bg_open = false;
            } else if code == 38.0 || code == 48.0 {
                // Skip the color data of `38;5;n` / `38;2;r;g;b` so a component
                // (e.g. 38) isn't read as a code.
                if code == 38.0 {
                    fg_open = true;
                } else {
                    bg_open = true;
                }
                let mode = params
                    .get(index + 1)
                    .and_then(|value| value.parse::<f64>().ok())
                    .unwrap_or(0.0);
                index += if mode == 2.0 {
                    4
                } else if mode == 5.0 {
                    2
                } else {
                    1
                };
            } else if code == 39.0 {
                fg_open = false;
            } else if code == 49.0 {
                bg_open = false;
            } else if (30.0..=37.0).contains(&code) || (90.0..=97.0).contains(&code) {
                fg_open = true;
            } else if (40.0..=47.0).contains(&code) || (100.0..=107.0).contains(&code) {
                bg_open = true;
            }
            index += 1;
        }
    }
    if fg_open || bg_open {
        format!("{line}\u{1b}[0m")
    } else {
        line.to_string()
    }
}

/// Port of `getIpythonCodeFromArgs`.
pub fn get_ipython_code_from_args(args: Option<&Value>) -> String {
    let Some(args) = args else {
        return String::new();
    };
    if !args.is_object() {
        return String::new();
    }
    args.get("code")
        .and_then(|code| code.as_str())
        .unwrap_or("")
        .to_string()
}

/// Port of `readDetails`.
fn read_details(details: Option<&Value>) -> IpythonDetails {
    let Some(details) = details.filter(|details| details.is_object()) else {
        return IpythonDetails::default();
    };
    let error = read_error_details(details.get("error"));
    IpythonDetails {
        duration_ms: details.get("durationMs").and_then(|value| value.as_f64()),
        status: details
            .get("status")
            .and_then(|value| value.as_str())
            .map(|value| value.to_string()),
        error_ename: error.as_ref().map(|error| error.ename.clone()).or_else(|| {
            details
                .get("errorEname")
                .and_then(|value| value.as_str())
                .map(|value| value.to_string())
        }),
        stdout: details
            .get("stdout")
            .and_then(|value| value.as_str())
            .map(|value| value.to_string()),
        stderr: details
            .get("stderr")
            .and_then(|value| value.as_str())
            .map(|value| value.to_string()),
        result: details
            .get("result")
            .and_then(|value| value.as_str())
            .map(|value| value.to_string()),
        background_output: details
            .get("backgroundOutput")
            .and_then(|value| value.as_str())
            .map(|value| value.to_string()),
        diffs: read_diff_displays(details.get("diffs")),
        sent_agent_messages: read_sent_agent_messages(details.get("sentAgentMessages")),
        error,
    }
}

/// Port of `readSentAgentMessages`.
fn read_sent_agent_messages(value: Option<&Value>) -> Option<Vec<KernelSentAgentMessage>> {
    let value = value?.as_array()?;
    let mut messages: Vec<KernelSentAgentMessage> = Vec::new();
    for entry in value {
        if !entry.is_object() {
            continue;
        }
        let Some(target) = entry.get("target").filter(|target| target.is_object()) else {
            continue;
        };
        let delivery_status = match entry.get("deliveryStatus").and_then(|value| value.as_str()) {
            Some("delivered") => KernelDeliveryStatus::Delivered,
            Some("queued") => KernelDeliveryStatus::Queued,
            _ => continue,
        };
        let (Some(id), Some(message)) = (
            entry.get("id").and_then(|value| value.as_str()),
            entry.get("message").and_then(|value| value.as_str()),
        ) else {
            continue;
        };
        let (Some(active_session_id), Some(session_id)) = (
            target
                .get("activeSessionId")
                .and_then(|value| value.as_str()),
            target.get("sessionId").and_then(|value| value.as_str()),
        ) else {
            continue;
        };
        let receiver_role = match entry.get("receiverRole").and_then(|value| value.as_str()) {
            Some("parent") => Some(KernelReceiverRole::Parent),
            Some("sibling") => Some(KernelReceiverRole::Sibling),
            Some("child") => Some(KernelReceiverRole::Child),
            _ => None,
        };
        messages.push(KernelSentAgentMessage {
            id: id.to_string(),
            message: message.to_string(),
            delivery_status,
            receiver_role,
            target: crate::core::kernel::shared::KernelSentAgentMessageTarget {
                active_session_id: active_session_id.to_string(),
                session_id: session_id.to_string(),
                session_name: target
                    .get("sessionName")
                    .and_then(|value| value.as_str())
                    .map(|value| value.to_string()),
            },
        });
    }
    if messages.is_empty() {
        None
    } else {
        Some(messages)
    }
}

/// Port of `readDiffDisplays`.
fn read_diff_displays(value: Option<&Value>) -> Option<Vec<DiffDisplay>> {
    let value = value?.as_array()?;
    let mut diffs: Vec<DiffDisplay> = Vec::new();
    for entry in value {
        if !entry.is_object() {
            continue;
        }
        let (Some(path), Some(old_str), Some(new_str)) = (
            entry.get("path").and_then(|value| value.as_str()),
            entry.get("oldStr").and_then(|value| value.as_str()),
            entry.get("newStr").and_then(|value| value.as_str()),
        ) else {
            continue;
        };
        diffs.push(DiffDisplay {
            path: path.to_string(),
            old_str: old_str.to_string(),
            new_str: new_str.to_string(),
            start_line: entry.get("startLine").and_then(|value| value.as_f64()),
        });
    }
    if diffs.is_empty() {
        None
    } else {
        Some(diffs)
    }
}

/// Port of `stripReprQuotes`.
fn strip_repr_quotes(text: &str) -> String {
    let trimmed = text.trim();
    let characters: Vec<char> = trimmed.chars().collect();
    if characters.len() >= 2 {
        let first = characters[0];
        let last = characters[characters.len() - 1];
        if (first == '\'' && last == '\'') || (first == '"' && last == '"') {
            return characters[1..characters.len() - 1].iter().collect();
        }
    }
    trimmed.to_string()
}

/// True when `text` is just the edit skill's "Edited <path>" confirmation for one
/// of `diffs`.
fn is_edit_confirmation(text: Option<&str>, diffs: &[DiffDisplay]) -> bool {
    let Some(text) = text else {
        return false;
    };
    if text.is_empty() {
        return false;
    }
    let stripped = strip_repr_quotes(text);
    diffs
        .iter()
        .any(|diff| stripped == format!("Edited {}", diff.path))
}

/// True when `text` is the `agent_message.send` receipt dict for one of the sent
/// messages already summarized above the output, so the raw receipt isn't shown.
fn is_agent_message_receipt(text: Option<&str>, messages: &[KernelSentAgentMessage]) -> bool {
    let Some(text) = text.filter(|text| !text.is_empty()) else {
        return false;
    };
    if messages.is_empty() {
        return false;
    }
    let stripped = strip_repr_quotes(text);
    messages.iter().any(|message| {
        stripped.starts_with(&format!("{{'id': '{}'", message.id))
            || stripped.starts_with(&format!("{{\"id\": \"{}\"", message.id))
    })
}

/// Port of `readErrorDetails`.
fn read_error_details(value: Option<&Value>) -> Option<IpythonErrorDetails> {
    let value = value.filter(|value| value.is_object())?;
    let ename = value
        .get("ename")
        .and_then(|value| value.as_str())?
        .to_string();
    Some(IpythonErrorDetails {
        ename,
        evalue: value
            .get("evalue")
            .and_then(|value| value.as_str())
            .unwrap_or("")
            .to_string(),
        traceback: value
            .get("traceback")
            .and_then(|value| value.as_array())
            .map(|lines| {
                lines
                    .iter()
                    .filter_map(|line| line.as_str().map(|line| line.to_string()))
                    .collect()
            })
            .unwrap_or_default(),
    })
}

/// Port of `formatDuration`.
fn format_duration(duration_ms: Option<f64>) -> Option<String> {
    let duration_ms = duration_ms?;
    if duration_ms < 1000.0 {
        return Some(format!("{}ms", js_round(duration_ms)));
    }
    Some(format!("{:.1}s", duration_ms / 1000.0))
}

/// `Math.round` for a finite JavaScript number.
fn js_round(value: f64) -> i64 {
    if !value.is_finite() {
        return 0;
    }
    (value + 0.5).floor() as i64
}

/// Port of `isImageBlock`.
fn is_image_block(block: &IPythonCellContentBlock) -> bool {
    block.r#type == "image" && block.data.is_some() && block.mime_type.is_some()
}

/// Port of `textFromBlocks`.
fn text_from_blocks(blocks: Option<&[IPythonCellContentBlock]>) -> String {
    let Some(blocks) = blocks else {
        return String::new();
    };
    blocks
        .iter()
        .filter(|block| block.r#type == "text" && block.text.is_some())
        .map(|block| block.text.clone().unwrap_or_default())
        .collect::<Vec<String>>()
        .join("\n")
}

/// Port of `splitTraceback`.
fn split_traceback(text: &str, error_name: Option<&str>) -> Option<TracebackParts> {
    let normalized = normalize_error_details(text);
    if normalized.trim().is_empty() {
        return None;
    }

    let lines: Vec<String> = normalized
        .split('\n')
        .map(|line| line.to_string())
        .collect();
    let mut traceback_index = lines
        .iter()
        .position(|line| line.contains("Traceback (most recent call last):"));
    if traceback_index.is_none() {
        if let Some(error_name) = error_name {
            traceback_index = lines
                .iter()
                .position(|line| line.trim().starts_with(&format!("{error_name}:")));
        }
    }
    let traceback_index = traceback_index?;

    let output = lines[..traceback_index].join("\n").trim_end().to_string();
    let traceback = lines[traceback_index..].join("\n").trim().to_string();
    let preview = summarize_error_details(&traceback);
    Some(TracebackParts {
        output,
        traceback,
        preview: if preview == "Error" {
            match error_name {
                Some(error_name) => error_name.to_string(),
                None => preview,
            }
        } else {
            preview
        },
    })
}

/// Port of `formatIpythonErrorSummary`.
fn format_ipython_error_summary(error: &IpythonErrorDetails) -> String {
    let normalized_value = normalize_error_details(&error.evalue);
    if normalized_value.trim().is_empty() {
        return error.ename.clone();
    }
    let value = summarize_error_details(&normalized_value);
    if value == "Error" {
        return error.ename.clone();
    }
    if visible_width(&value) <= 48 {
        format!("{}: {}", error.ename, value)
    } else {
        error.ename.clone()
    }
}

/// `"error" | "aborted" | "running" | "queued" | "done"`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StatusKind {
    Error,
    Aborted,
    Running,
    Queued,
    Done,
}

/// Port of `IPythonCellComponent`.
pub struct IPythonCellComponent {
    render_cache: VersionedRenderCache,
    state: IPythonCellState,
    state_version: u64,
}

impl IPythonCellComponent {
    /// Port of the `IPythonCellComponent` constructor.
    pub fn new(state: IPythonCellState) -> Self {
        Self {
            render_cache: VersionedRenderCache::new(),
            state,
            state_version: 0,
        }
    }

    /// Port of `update`.
    pub fn update(&mut self, state: IPythonCellState) {
        self.state = state;
        self.state_version += 1;
    }

    fn language(&self) -> &str {
        if self.state.details.as_ref().and_then(|v| v.get("language")).and_then(Value::as_str) == Some("javascript") { "javascript" } else { "python" }
    }

    /// Port of `collapsedLine`.
    fn collapsed_line(&self, details: &IpythonDetails) -> String {
        let code = self.state.code.trim_end().to_string();
        let is_bash_cell = self.language() == "python" && parse_ipython_bash_cell(&code).is_some();
        let mut preview = preview_ipython_code(&code);
        if self.language() == "javascript" {
            preview.text = code.lines().find(|line| !line.trim().is_empty()).unwrap_or("").trim().chars().take(100).collect();
        }
        let preview_language = if self.language() == "javascript" { "javascript" } else { preview.language.as_str() };
        let language_label = if is_bash_cell && preview_language != "bash" {
            format!("bash \u{b7} {preview_language}")
        } else {
            preview_language.to_string()
        };
        let mut parts: Vec<String> = vec![format!(
            "{} {}",
            self.marker(details),
            theme().fg("muted", &language_label)
        )];

        if !preview.text.is_empty() {
            parts.push(self.highlight_input_line(&preview.text, preview_language == "bash"));
        } else if self.state.execution_started != Some(true) {
            parts.push(theme().fg("muted", "waiting for code"));
        }

        if let Some(counts) = self.line_counts(details) {
            parts.push(theme().fg("muted", &counts));
        }

        if let Some(duration) = format_duration(details.duration_ms) {
            parts.push(theme().fg("muted", &duration));
        }

        let error_name = if self.state.is_partial != Some(true) {
            details
                .error
                .as_ref()
                .map(|error| error.ename.clone())
                .or_else(|| details.error_ename.clone())
        } else {
            None
        };
        if let Some(error_name) = error_name {
            parts.push(theme().fg("error", &error_name));
        }

        if self.state.show_expand_hint != Some(false) {
            parts.push(expand_collapse_hint(
                "app.tools.expand",
                self.state.expanded == Some(true),
            ));
        }
        parts.join(&theme().fg("dim", " \u{b7} "))
    }

    /// Status marker - color carries running/done/error; ✓/✗ once finished.
    fn marker(&self, details: &IpythonDetails) -> String {
        match self.status_kind(details) {
            StatusKind::Error => theme().fg("error", "\u{2717}"),
            StatusKind::Aborted => theme().fg("warning", "\u{2717}"),
            StatusKind::Done => theme().fg("success", "\u{2713}"),
            StatusKind::Running => {
                theme().fg("bashMode", working_icon_frame(get_working_pulse_frame()))
            }
            StatusKind::Queued => theme().fg("muted", "\u{25c7}"),
        }
    }

    /// `↑in ↓out lines` - the "lines" unit disambiguates from the token counts on
    /// the activity line. Output is omitted for edits (the diff shows on expand).
    fn line_counts(&self, details: &IpythonDetails) -> Option<String> {
        let bash_cell = parse_ipython_bash_cell(&self.state.code);
        let body = bash_cell
            .map(|parsed| parsed.body)
            .unwrap_or_else(|| self.state.code.clone());
        let input = split_lines_crlf(&body)
            .into_iter()
            .filter(|line| !line.trim().is_empty())
            .count();

        let has_diffs = details
            .diffs
            .as_ref()
            .map(|diffs| !diffs.is_empty())
            .unwrap_or(false);
        let sent_messages: &[KernelSentAgentMessage] =
            details.sent_agent_messages.as_deref().unwrap_or(&[]);
        let result = if is_agent_message_receipt(details.result.as_deref(), sent_messages) {
            None
        } else {
            details.result.clone()
        };
        let structured = [
            details.stdout.as_deref(),
            details.stderr.as_deref(),
            result.as_deref(),
            details.background_output.as_deref(),
        ]
        .into_iter()
        .flatten()
        .filter(|value| !value.trim().is_empty())
        .collect::<Vec<&str>>()
        .join("\n");
        let blocks_text = text_from_blocks(self.state.content.as_deref());
        let fallback = if is_agent_message_receipt(Some(&blocks_text), sent_messages) {
            String::new()
        } else {
            blocks_text
        };
        let output_text = if structured.is_empty() {
            fallback
        } else {
            structured
        };
        let output_text = output_text.trim().to_string();
        let output = if has_diffs || output_text.is_empty() {
            0
        } else {
            output_text.split('\n').count()
        };

        let mut segments: Vec<String> = Vec::new();
        if input > 0 {
            segments.push(format!("\u{2191} {input}"));
        }
        if output > 0 {
            segments.push(format!("\u{2193} {output}"));
        }
        if segments.is_empty() {
            None
        } else {
            Some(format!("{} lines", segments.join(" ")))
        }
    }

    /// Port of `statusKind`.
    fn status_kind(&self, details: &IpythonDetails) -> StatusKind {
        let status = details.status.as_deref();
        if self.state.is_error == Some(true) || status == Some("error") {
            return StatusKind::Error;
        }
        if status == Some("aborted") {
            return StatusKind::Aborted;
        }
        // Keyed off the result, not executionStarted, so calls rehydrated from a
        // past session (which never saw the live start) render done, not running.
        if self.state.is_partial != Some(true)
            && (status.is_some()
                || self.state.execution_started == Some(true)
                || self.has_result(details))
        {
            return StatusKind::Done;
        }
        if self.state.is_partial == Some(true) || self.state.execution_started == Some(true) {
            return StatusKind::Running;
        }
        StatusKind::Queued
    }

    /// Port of `hasResult`.
    fn has_result(&self, details: &IpythonDetails) -> bool {
        details.stdout.is_some()
            || details.stderr.is_some()
            || details.result.is_some()
            || details.error.is_some()
            || details
                .diffs
                .as_ref()
                .map(|diffs| !diffs.is_empty())
                .unwrap_or(false)
            || details
                .sent_agent_messages
                .as_ref()
                .map(|messages| !messages.is_empty())
                .unwrap_or(false)
            || self
                .state
                .content
                .as_ref()
                .map(|content| !content.is_empty())
                .unwrap_or(false)
    }

    /// Only runs when expanded - shows the full source below the fixed top line.
    fn render_code(&self, lines: &mut Vec<String>, width: f64) -> bool {
        let code = self.state.code.trim_end().to_string();
        if code.is_empty() {
            self.add_blank(lines, width);
            self.add_wrapped(
                lines,
                OUTPUT_INDENT,
                &theme().fg("muted", "waiting for code"),
                width,
            );
            return false;
        }

        self.add_blank(lines, width);
        let is_bash_cell = self.language() == "python" && parse_ipython_bash_cell(&code).is_some();
        let raw_lines: Vec<String> = code.split('\n').map(|line| line.to_string()).collect();
        // Highlight the whole cell at once so multi-line strings keep their color.
        let highlighted_lines = if is_bash_cell {
            Vec::new()
        } else {
            highlight_code(&code, Some(self.language()))
        };
        for (index, raw_line) in raw_lines.iter().enumerate() {
            let prefix = if index == 0 {
                theme().fg("dim", "\u{203a} ")
            } else {
                theme().fg("dim", "  ")
            };
            let highlighted = if is_bash_cell
                || magic_line_pattern().is_match(raw_line)
                || parse_ipython_bash_cell(raw_line).is_some()
            {
                theme().fg("bashMode", raw_line)
            } else {
                highlighted_lines
                    .get(index)
                    .cloned()
                    .unwrap_or_else(|| theme().fg("mdCodeBlock", raw_line))
            };
            self.add_wrapped(
                lines,
                &prefix,
                if highlighted.is_empty() {
                    " "
                } else {
                    &highlighted
                },
                width,
            );
        }

        true
    }

    /// Port of `highlightInputLine`.
    fn highlight_input_line(&self, line: &str, is_bash_cell: bool) -> String {
        if self.language() == "python" && (is_bash_cell
            || magic_line_pattern().is_match(line)
            || parse_ipython_bash_cell(line).is_some())
        {
            return theme().fg("bashMode", line);
        }
        let highlighted = highlight_code(line, Some(self.language()));
        highlighted
            .first()
            .cloned()
            .unwrap_or_else(|| theme().fg("mdCodeBlock", line))
    }

    /// Only runs when expanded - shows full output below the code, no previews.
    #[allow(unused_assignments)]
    fn render_output(
        &self,
        lines: &mut Vec<String>,
        width: f64,
        details: &IpythonDetails,
        has_code: bool,
    ) {
        let blocks: &[IPythonCellContentBlock] = self.state.content.as_deref().unwrap_or(&[]);
        let text = text_from_blocks(self.state.content.as_deref());
        let image_count = blocks.iter().filter(|block| is_image_block(block)).count();
        let has_structured_output = details.stdout.is_some()
            || details.stderr.is_some()
            || details.result.is_some()
            || details.error.is_some();
        let traceback = if !has_structured_output
            && (self.state.is_error == Some(true) || details.status.as_deref() == Some("error"))
        {
            split_traceback(&text, details.error_ename.as_deref())
        } else {
            None
        };
        let mut output_started = false;
        let mut rendered_text_output = false;

        let diffs: &[DiffDisplay] = details.diffs.as_deref().unwrap_or(&[]);
        let sent_messages: &[KernelSentAgentMessage] =
            details.sent_agent_messages.as_deref().unwrap_or(&[]);

        if has_structured_output {
            if details
                .stdout
                .as_deref()
                .map(|stdout| !stdout.trim().is_empty())
                .unwrap_or(false)
                && !is_edit_confirmation(details.stdout.as_deref(), diffs)
            {
                if !output_started {
                    output_started = true;
                    if has_code {
                        self.add_blank(lines, width);
                    }
                }
                rendered_text_output = true;
                self.render_output_text(
                    lines,
                    width,
                    &normalize_error_details(details.stdout.as_deref().unwrap_or("")),
                    "out",
                );
            }
            if details
                .stderr
                .as_deref()
                .map(|stderr| !stderr.trim().is_empty())
                .unwrap_or(false)
            {
                if !output_started {
                    output_started = true;
                    if has_code {
                        self.add_blank(lines, width);
                    }
                }
                rendered_text_output = true;
                self.render_output_text(
                    lines,
                    width,
                    &normalize_error_details(details.stderr.as_deref().unwrap_or("")),
                    "err",
                );
            }
            let result_text = details.result.as_deref();
            if result_text
                .map(|result| !result.trim().is_empty())
                .unwrap_or(false)
                && !is_edit_confirmation(result_text, diffs)
                && !is_agent_message_receipt(result_text, sent_messages)
            {
                if !output_started {
                    output_started = true;
                    if has_code {
                        self.add_blank(lines, width);
                    }
                }
                rendered_text_output = true;
                self.render_output_text(
                    lines,
                    width,
                    &normalize_error_details(result_text.unwrap_or("")),
                    "out",
                );
            }
        } else if let Some(traceback) = &traceback {
            if !traceback.output.is_empty() {
                output_started = true;
                if has_code {
                    self.add_blank(lines, width);
                }
                rendered_text_output = true;
                self.render_output_text(lines, width, &traceback.output, "out");
            }
        } else if !text.trim().is_empty() && !is_agent_message_receipt(Some(&text), sent_messages) {
            output_started = true;
            if has_code {
                self.add_blank(lines, width);
            }
            rendered_text_output = true;
            let label = if self.state.is_error == Some(true) {
                "err"
            } else {
                "out"
            };
            self.render_output_text(lines, width, &normalize_error_details(&text), label);
        }

        // Without structured fields the fallback content text above already
        // contains the appended background block.
        let background_output = if has_structured_output {
            details
                .background_output
                .as_deref()
                .filter(|background| !background.trim().is_empty())
        } else {
            None
        };
        if background_output.is_some() {
            // Suppresses the placeholders below when background output is the
            // cell's only output; rendered after the traceback to match the
            // model-facing order.
            rendered_text_output = true;
        }

        if !rendered_text_output && self.state.is_partial == Some(true) {
            if !output_started {
                output_started = true;
                if has_code {
                    self.add_blank(lines, width);
                }
            }
            self.add_wrapped(
                lines,
                OUTPUT_INDENT,
                &theme().fg("muted", "waiting for output..."),
                width,
            );
        } else if !rendered_text_output
            && self.state.execution_started == Some(true)
            && self.state.args_complete != Some(true)
        {
            if !output_started {
                output_started = true;
                if has_code {
                    self.add_blank(lines, width);
                }
            }
            self.add_wrapped(
                lines,
                OUTPUT_INDENT,
                &theme().fg("muted", "waiting for output..."),
                width,
            );
        } else if !rendered_text_output
            && traceback.is_none()
            && details.error.is_none()
            && diffs.is_empty()
            && details
                .sent_agent_messages
                .as_ref()
                .map(|messages| messages.is_empty())
                .unwrap_or(true)
            && self.state.execution_started == Some(true)
            && image_count == 0
        {
            if !output_started {
                output_started = true;
                if has_code {
                    self.add_blank(lines, width);
                }
            }
            self.add_wrapped(
                lines,
                OUTPUT_INDENT,
                &theme().fg("muted", "no output"),
                width,
            );
        }

        let traced = if let Some(error) = &details.error {
            let joined = error.traceback.join("\n");
            Some(if joined.is_empty() {
                format_ipython_error_summary(error)
            } else {
                joined
            })
        } else {
            traceback
                .as_ref()
                .map(|traceback| traceback.traceback.clone())
        };
        if let Some(traced) = traced {
            if !output_started {
                output_started = true;
                if has_code {
                    self.add_blank(lines, width);
                }
            }
            self.render_traceback(lines, width, &traced);
        }

        if let Some(background_output) = background_output {
            if !output_started {
                output_started = true;
                if has_code {
                    self.add_blank(lines, width);
                }
            }
            self.add_wrapped(
                lines,
                OUTPUT_INDENT,
                &theme().fg("muted", "background output (unattributed)"),
                width,
            );
            self.render_output_text(
                lines,
                width,
                &normalize_error_details(background_output),
                "err",
            );
        }

        if image_count > 0 {
            if !output_started {
                output_started = true;
                if has_code {
                    self.add_blank(lines, width);
                }
            }
            let text = if self.state.show_images == Some(true) {
                format!(
                    "{image_count} image{} rendered below",
                    if image_count == 1 { "" } else { "s" }
                )
            } else {
                format!(
                    "{image_count} image{} hidden",
                    if image_count == 1 { "" } else { "s" }
                )
            };
            self.add_wrapped(lines, OUTPUT_INDENT, &theme().fg("muted", &text), width);
        }
    }

    /// Summary line per message; expanding shows the message text in a `╰─`
    /// gutter instead of the collapsed preview, matching received agent-message UI.
    fn render_sent_agent_messages(
        &self,
        lines: &mut Vec<String>,
        width: f64,
        messages: &[KernelSentAgentMessage],
    ) {
        for message in messages {
            let label = if message.delivery_status == KernelDeliveryStatus::Delivered {
                "Agent message sent"
            } else {
                "Agent message queued"
            };
            let direction: AgentMessageDirection = "sent".to_string();
            let receiver_role = message.receiver_role.as_ref().map(|role| match role {
                KernelReceiverRole::Parent => "parent".to_string(),
                KernelReceiverRole::Sibling => "sibling".to_string(),
                KernelReceiverRole::Child => "child".to_string(),
            });
            let endpoint = AgentSessionMessageSender {
                active_session_id: Some(message.target.active_session_id.clone()),
                session_id: Some(message.target.session_id.clone()),
                session_name: message.target.session_name.clone(),
                runtime_kind: None,
                client_id: None,
            };
            let recipient = format_agent_message_participant(
                &direction,
                receiver_role.as_ref(),
                Some(&endpoint),
            );
            let hint = expand_collapse_hint(
                "app.messages.expand",
                self.state.agent_messages_expanded == Some(true),
            );
            if self.state.agent_messages_expanded == Some(true) {
                self.add_blank(lines, width);
                self.add_plain(
                    lines,
                    &truncate_to_width(
                        &format!(
                            "{} {hint}",
                            agent_message_summary_line(label, &recipient, None)
                        ),
                        f64_max(1.0, width - 1.0),
                        "\u{2026}",
                        false,
                    ),
                );
                for body_line in agent_message_body_lines(&message.message, width) {
                    lines.push(body_line);
                }
                continue;
            }
            let prefix_width =
                visible_width(&format!("\u{25c6} {label} \u{b7} {recipient} \u{b7} "));
            let preview = agent_message_preview(prefix_width, &message.message);
            self.add_plain(
                lines,
                &truncate_to_width(
                    &format!(
                        "{} {hint}",
                        agent_message_summary_line(label, &recipient, Some(&preview))
                    ),
                    f64_max(1.0, width - 1.0),
                    "\u{2026}",
                    false,
                ),
            );
        }
    }

    /// The `╰─ <path> +N -M` summary line renders in both states; ctrl+j only
    /// attaches or removes the indented diff rows underneath it.
    fn render_diffs(
        &self,
        lines: &mut Vec<String>,
        width: f64,
        diffs: &[DiffDisplay],
        has_code: bool,
    ) {
        let mut diffs_by_path: indexmap::IndexMap<String, Vec<DiffDisplay>> =
            indexmap::IndexMap::new();
        for diff in diffs {
            diffs_by_path
                .entry(diff.path.clone())
                .or_default()
                .push(diff.clone());
        }
        if has_code {
            self.add_plain(lines, "");
        }
        let total = diffs_by_path.len();
        for (index, (path, edits)) in diffs_by_path.iter().enumerate() {
            self.render_file_diff(lines, width, path, edits, index + 1 == total);
        }
    }

    /// Port of `renderFileDiff`.
    fn render_file_diff(
        &self,
        lines: &mut Vec<String>,
        width: f64,
        path: &str,
        edits: &[DiffDisplay],
        show_hint: bool,
    ) {
        let language = get_language_from_path(path);
        // Diff rows align with the summary line's text column (after the `╰─ ` gutter).
        let indent: String = FILE_CHANGE_DIFF_INDENT
            .chars()
            .take(std::cmp::max(0, width.max(0.0).floor() as i64 - 1) as usize)
            .collect();
        let content_width =
            std::cmp::max(1, width.max(0.0).floor() as usize - indent.chars().count());
        let mut added = 0i64;
        let mut removed = 0i64;
        let mut rows: Vec<String> = Vec::new();
        for (index, edit) in edits.iter().enumerate() {
            let start_line = edit.start_line.unwrap_or(1.0).max(0.0) as usize;
            let generated = generate_diff_string(&edit.old_str, &edit.new_str, 4, start_line);
            let diff_text = generated.0;
            let (edit_added, edit_removed) = count_changed_lines(&diff_text);
            added += edit_added;
            removed += edit_removed;
            if self.state.edit_diffs_expanded != Some(true) {
                continue;
            }
            if index > 0 {
                rows.push(format!(
                    "{indent}{}",
                    render_diff_separator(content_width as f64)
                ));
            }
            // Append, not spread: a huge edit's diff can exceed the JS arg-count limit.
            for row in render_rich_diff(
                &diff_text,
                content_width as f64,
                RichDiffOptions {
                    language: language.clone(),
                },
            ) {
                rows.push(format!("{indent}{row}"));
            }
        }

        // Unlike the ctrl+o hint (latest tool row only), the ctrl+j hint renders on
        // every tool row, matching the thinking and agent-message hints. Within a
        // row it renders once, on the last file's summary line (showHint).
        let hint = if show_hint {
            Some(self.state.edit_diffs_expanded == Some(true))
        } else {
            None
        };
        lines.push(format_file_change_summary_line(
            path,
            self.state.cwd.as_deref(),
            added,
            removed,
            hint,
            width,
        ));

        for row in rows {
            lines.push(row);
        }
    }

    /// Port of `renderOutputText`.
    fn render_output_text(&self, lines: &mut Vec<String>, width: f64, text: &str, label: &str) {
        let color = if label == "err" {
            "muted"
        } else {
            "toolOutput"
        };
        for line in text.split('\n') {
            let content = if line.is_empty() { " " } else { line };
            self.add_wrapped(lines, OUTPUT_INDENT, &theme().fg(color, content), width);
        }
    }

    /// Port of `renderTraceback`.
    fn render_traceback(&self, lines: &mut Vec<String>, width: f64, traceback: &str) {
        for line in traceback.split('\n') {
            let content = if line.is_empty() { " " } else { line };
            self.add_wrapped(lines, OUTPUT_INDENT, &theme().fg("muted", content), width);
        }
    }

    /// Backgroundless line, indented one space to align under the fixed top line.
    fn add_wrapped(&self, lines: &mut Vec<String>, prefix: &str, text: &str, width: f64) {
        let width = width.max(0.0).floor() as usize;
        let available = std::cmp::max(
            1,
            width
                .saturating_sub(1)
                .saturating_sub(visible_width(prefix)),
        );
        let wrapped = wrap_text_with_ansi(text, available);
        let wrapped = if wrapped.is_empty() {
            vec![String::new()]
        } else {
            wrapped
        };
        for (index, line) in wrapped.iter().enumerate() {
            let line_prefix = if index == 0 {
                prefix.to_string()
            } else {
                " ".repeat(visible_width(prefix))
            };
            // Truncate the composed line so a narrow pane can't exceed width
            // (fatal in the renderer).
            lines.push(truncate_to_width(
                &format!(" {line_prefix}{}", close_open_sgr(line)),
                width as f64,
                "",
                false,
            ));
        }
    }

    /// Port of `addBlank`.
    fn add_blank(&self, lines: &mut Vec<String>, _width: f64) {
        lines.push(String::new());
    }

    /// No-background line, indented one space to align with the summary line above.
    fn add_plain(&self, lines: &mut Vec<String>, text: &str) {
        lines.push(format!(" {text}"));
    }
}

/// `Math.max` for the widths this module passes.
fn f64_max(left: f64, right: f64) -> f64 {
    if left > right {
        left
    } else {
        right
    }
}

/// `text.split(/\r?\n/)`
fn split_lines_crlf(value: &str) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();
    let mut current = String::new();
    let characters: Vec<char> = value.chars().collect();
    let mut index = 0usize;
    while index < characters.len() {
        let character = characters[index];
        if character == '\r' && characters.get(index + 1) == Some(&'\n') {
            lines.push(std::mem::take(&mut current));
            index += 2;
            continue;
        }
        if character == '\n' {
            lines.push(std::mem::take(&mut current));
            index += 1;
            continue;
        }
        current.push(character);
        index += 1;
    }
    lines.push(current);
    lines
}

impl Component for IPythonCellComponent {
    /// Port of the `render` method.
    fn render(&mut self, width: f64) -> Vec<String> {
        let safe_width = std::cmp::max(1, width.max(0.0).floor() as usize);
        let details = read_details(self.state.details.as_ref());
        // Fold the animation frame into the cache key while running (offset within
        // a stateVersion slot so it never collides with another version).
        let frames = WORKING_ICON_FRAMES.len() as u64;
        let cache_version = if self.status_kind(&details) == StatusKind::Running {
            self.state_version * frames + (get_working_pulse_frame().max(0) as u64 % frames)
        } else {
            self.state_version * frames
        };
        if let Some(cached) = self.render_cache.get(safe_width, cache_version) {
            return cached;
        }

        // The top line is identical whether collapsed or expanded - same marker,
        // counts, duration, and expand hint - so toggling never shifts the layout
        // or indentation; expanding only attaches code and output below it.
        // Cached by state version so unrelated repaints don't re-render (flicker).
        let mut lines: Vec<String> = vec![truncate_to_width(
            &format!(" {}", self.collapsed_line(&details)),
            safe_width as f64,
            "",
            false,
        )];

        let has_code = if self.state.expanded == Some(true) {
            self.render_code(&mut lines, safe_width as f64)
        } else {
            false
        };
        if details
            .diffs
            .as_ref()
            .map(|diffs| !diffs.is_empty())
            .unwrap_or(false)
        {
            self.render_diffs(
                &mut lines,
                safe_width as f64,
                details.diffs.as_deref().unwrap_or(&[]),
                has_code,
            );
        }
        if details
            .sent_agent_messages
            .as_ref()
            .map(|messages| !messages.is_empty())
            .unwrap_or(false)
        {
            self.render_sent_agent_messages(
                &mut lines,
                safe_width as f64,
                details.sent_agent_messages.as_deref().unwrap_or(&[]),
            );
        }

        if self.state.expanded != Some(true) {
            return self.render_cache.set(safe_width, cache_version, lines);
        }

        self.render_output(&mut lines, safe_width as f64, &details, has_code);
        self.render_cache.set(safe_width, cache_version, lines)
    }

    fn invalidate(&mut self) {
        self.render_cache.invalidate();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modes::interactive::theme::theme::init_theme;
    use serde_json::json;

    fn init() {
        init_theme(Some("prime"), false);
    }

    fn strip_ansi(text: &str) -> String {
        let mut result = String::new();
        let mut chars = text.chars().peekable();
        while let Some(character) = chars.next() {
            if character == '\u{1b}' {
                if chars.peek() == Some(&'[') {
                    chars.next();
                    while let Some(&next) = chars.peek() {
                        chars.next();
                        if next.is_ascii_alphabetic() {
                            break;
                        }
                    }
                }
            } else {
                result.push(character);
            }
        }
        result
    }

    fn plain(lines: &[String]) -> String {
        lines
            .iter()
            .map(|line| strip_ansi(line))
            .collect::<Vec<String>>()
            .join("\n")
    }

    fn state(code: &str) -> IPythonCellState {
        IPythonCellState {
            code: code.to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn reads_code_from_args() {
        assert_eq!(
            get_ipython_code_from_args(Some(&json!({ "code": "print(1)" }))),
            "print(1)"
        );
        assert_eq!(get_ipython_code_from_args(Some(&json!({ "code": 5 }))), "");
        assert_eq!(get_ipython_code_from_args(Some(&json!({}))), "");
        assert_eq!(get_ipython_code_from_args(None), "");
    }

    #[test]
    fn close_open_sgr_appends_a_reset_only_for_open_colors() {
        assert_eq!(close_open_sgr("\u{1b}[31mred"), "\u{1b}[31mred\u{1b}[0m");
        assert_eq!(
            close_open_sgr("\u{1b}[31mred\u{1b}[39m"),
            "\u{1b}[31mred\u{1b}[39m"
        );
        assert_eq!(close_open_sgr("plain"), "plain");
        // 38;5;n colour data must not be read as a code.
        assert_eq!(
            close_open_sgr("\u{1b}[38;5;196mx\u{1b}[39m"),
            "\u{1b}[38;5;196mx\u{1b}[39m"
        );
    }

    #[test]
    fn duration_rounds_milliseconds_and_fixes_seconds() {
        assert_eq!(format_duration(Some(999.0)).as_deref(), Some("999ms"));
        assert_eq!(format_duration(Some(1500.0)).as_deref(), Some("1.5s"));
        assert_eq!(format_duration(None), None);
    }

    #[test]
    fn queues_and_runs_render_the_expected_marker() {
        init();
        let mut component = IPythonCellComponent::new(state("print(1)"));
        let queued = plain(&component.render(80.0));
        assert!(queued.contains("\u{25c7}"));
        assert!(queued.contains("print(1)"));
        assert!(!queued.contains("waiting for code"));
        let mut empty = IPythonCellComponent::new(state(""));
        assert!(plain(&empty.render(80.0)).contains("waiting for code"));

        let mut started = state("print(1)");
        started.execution_started = Some(true);
        started.is_partial = Some(true);
        let mut component = IPythonCellComponent::new(started);
        let running = plain(&component.render(80.0));
        assert!(running.contains("to expand"));
        assert!(!running.contains("\u{2713}"));
    }

    #[test]
    fn structured_result_marks_the_cell_done_and_counts_lines() {
        init();
        let mut cell = state("a = 1\nb = 2\n");
        cell.details = Some(json!({ "status": "ok", "stdout": "out\n", "durationMs": 1200.0 }));
        let mut component = IPythonCellComponent::new(cell);
        let text = plain(&component.render(120.0));
        assert!(text.contains("\u{2713}"));
        assert!(text.contains("\u{2191} 2"));
        assert!(text.contains("\u{2193} 1 lines"));
        assert!(text.contains("1.2s"));
    }

    #[test]
    fn bash_cell_reports_the_language_label() {
        init();
        let mut cell = state("%%bash\necho hi\n");
        cell.details = Some(json!({ "status": "ok" }));
        let mut component = IPythonCellComponent::new(cell);
        let text = plain(&component.render(120.0));
        assert!(text.contains("bash"));
        assert!(text.contains("echo hi"));
    }

    #[test]
    fn expanded_renders_the_code_with_the_gutter() {
        init();
        let mut cell = state("print(1)\nprint(2)");
        cell.expanded = Some(true);
        cell.execution_started = Some(true);
        let mut component = IPythonCellComponent::new(cell);
        let text = plain(&component.render(80.0));
        assert!(text.contains("\u{203a} print(1)"));
        assert!(text.contains("print(2)"));
    }

    #[test]
    fn expanded_error_renders_the_traceback() {
        init();
        let mut cell = state("1/0");
        cell.expanded = Some(true);
        cell.is_error = Some(true);
        cell.details = Some(json!({
            "status": "error",
            "errorEname": "ZeroDivisionError",
            "error": {
                "ename": "ZeroDivisionError",
                "evalue": "division by zero",
                "traceback": ["Traceback (most recent call last):", "ZeroDivisionError: division by zero"]
            }
        }));
        let mut component = IPythonCellComponent::new(cell);
        let text = plain(&component.render(100.0));
        assert!(text.contains("ZeroDivisionError"));
        assert!(text.contains("division by zero"));
    }

    #[test]
    fn expanded_without_output_renders_no_output() {
        init();
        let mut cell = state("x = 1");
        cell.expanded = Some(true);
        cell.execution_started = Some(true);
        cell.args_complete = Some(true);
        let mut component = IPythonCellComponent::new(cell);
        let text = plain(&component.render(80.0));
        assert!(text.contains("no output"));
    }

    #[test]
    fn partial_cell_shows_the_waiting_for_output_placeholder() {
        init();
        let mut cell = state("while True: pass");
        cell.expanded = Some(true);
        cell.is_partial = Some(true);
        cell.execution_started = Some(true);
        let mut component = IPythonCellComponent::new(cell);
        let text = plain(&component.render(80.0));
        assert!(text.contains("waiting for output..."));
    }

    #[test]
    fn hidden_images_are_reported() {
        init();
        let mut cell = state("display(img)");
        cell.expanded = Some(true);
        cell.execution_started = Some(true);
        cell.args_complete = Some(true);
        cell.content = Some(vec![IPythonCellContentBlock {
            r#type: "image".to_string(),
            data: Some("AAAA".to_string()),
            mime_type: Some("image/png".to_string()),
            text: None,
        }]);
        let mut component = IPythonCellComponent::new(cell);
        let text = plain(&component.render(80.0));
        assert!(text.contains("1 image hidden"));
    }

    #[test]
    fn diffs_summarise_the_changed_lines() {
        init();
        let mut cell = state("edit()");
        cell.execution_started = Some(true);
        cell.details = Some(json!({
            "status": "ok",
            "diffs": [{ "path": "src/a.py", "oldStr": "a\n", "newStr": "b\n", "startLine": 1 }]
        }));
        let mut component = IPythonCellComponent::new(cell);
        let text = plain(&component.render(80.0));
        assert!(text.contains("src/a.py"));
        assert!(text.contains("+1 -1"));
        // edits suppress the output line count
        assert!(!text.contains("\u{2193} 1"));
    }

    #[test]
    fn expanded_diffs_render_the_diff_rows() {
        init();
        let mut cell = state("edit()");
        cell.execution_started = Some(true);
        cell.edit_diffs_expanded = Some(true);
        cell.details = Some(json!({
            "status": "ok",
            "diffs": [{ "path": "src/a.py", "oldStr": "a\n", "newStr": "b\n", "startLine": 1 }]
        }));
        let mut component = IPythonCellComponent::new(cell);
        let text = plain(&component.render(80.0));
        assert!(text.contains("src/a.py"));
        // The rendered row is `gutter + content`, and the gutter puts the line
        // number BEFORE the marker: diff.ts:253
        //   `const gutter = ` ${lineNum} ${prefix === " " ? " " : prefix} `;`
        // so "+1 b" renders as " 1 + b". The TypeScript test asserts exactly that
        // shape: ipython-cell-diff.test.ts:62-63 expects `/11 - .*gamma/` and
        // `/11 \+ .*GAMMA/`, and its `changedRow` helper selects rows by
        // `stripAnsi(line).startsWith(` 1 ${prefix} `)` (:29-33). "+1 b" is the
        // unified-diff STRING form produced by generateDiffString
        // (edit-diff.ts:287, `output.push(`+${lineNum} ${line}`)`), which the cell
        // never shows raw; asserting it here was the wrong shape.
        assert!(text.contains(" 1 + b"), "expected the added row in:\n{text}");
        assert!(text.contains(" 1 - a"), "expected the removed row in:\n{text}");
    }

    #[test]
    fn sent_agent_messages_render_a_summary_line() {
        init();
        let mut cell = state("send()");
        cell.execution_started = Some(true);
        cell.details = Some(json!({
            "status": "ok",
            "sentAgentMessages": [{
                "id": "m1",
                "message": "hello",
                "deliveryStatus": "delivered",
                "receiverRole": "parent",
                "target": { "activeSessionId": "a1", "sessionId": "s1", "sessionName": "root" }
            }]
        }));
        let mut component = IPythonCellComponent::new(cell);
        let text = plain(&component.render(100.0));
        assert!(text.contains("Agent message sent"));
        assert!(text.contains("root"));
        assert!(text.contains("hello"));
    }

    #[test]
    fn agent_message_receipts_are_hidden_from_the_output() {
        init();
        let mut cell = state("send()");
        cell.expanded = Some(true);
        cell.execution_started = Some(true);
        cell.args_complete = Some(true);
        cell.details = Some(json!({
            "status": "ok",
            "result": "{'id': 'm1', 'source': 'x'}"
        }));
        let mut cell_without_messages = cell.clone();
        cell_without_messages.details = Some(json!({ "status": "ok", "result": "{'id': 'm1'}" }));
        let receipt = KernelSentAgentMessage {
            id: "m1".to_string(),
            message: "hello".to_string(),
            delivery_status: KernelDeliveryStatus::Delivered,
            receiver_role: None,
            target: crate::core::kernel::shared::KernelSentAgentMessageTarget {
                active_session_id: "a1".to_string(),
                session_id: "s1".to_string(),
                session_name: None,
            },
        };
        assert!(is_agent_message_receipt(
            Some("{'id': 'm1', 'source': 'x'}"),
            &[receipt]
        ));
        assert!(!is_agent_message_receipt(Some("{'id': 'm2'}"), &[]));
    }

    #[test]
    fn edit_confirmations_are_hidden_when_a_diff_exists() {
        let diffs = vec![DiffDisplay {
            path: "src/a.py".to_string(),
            old_str: "a".to_string(),
            new_str: "b".to_string(),
            start_line: None,
        }];
        assert!(is_edit_confirmation(Some("'Edited src/a.py'"), &diffs));
        assert!(!is_edit_confirmation(Some("other"), &diffs));
        assert!(!is_edit_confirmation(None, &diffs));
    }

    #[test]
    fn render_cache_key_changes_with_the_state_version() {
        init();
        let mut component = IPythonCellComponent::new(state("x = 1"));
        let first = component.render(40.0);
        let second = component.render(40.0);
        assert_eq!(first, second);
        component.update(state("y = 2"));
        let third = component.render(40.0);
        assert_ne!(first, third);
        assert!(plain(&third).contains("y = 2"));
    }

    #[test]
    fn status_kind_prefers_done_for_rehydrated_cells() {
        init();
        let mut cell = state("print(1)");
        cell.details = Some(json!({ "status": "ok", "stdout": "hi\n" }));
        let mut component = IPythonCellComponent::new(cell);
        assert_eq!(
            component.status_kind(&read_details(component.state.details.as_ref())),
            StatusKind::Done
        );
    }

    #[test]
    fn split_traceback_falls_back_to_the_error_name() {
        init();
        let parts = split_traceback(
            "Traceback (most recent call last):\nZeroDivisionError: x",
            Some("ZeroDivisionError"),
        )
        .expect("parts");
        assert_eq!(parts.output, "");
        assert!(parts.traceback.contains("ZeroDivisionError"));
        assert_eq!(parts.preview, "ZeroDivisionError: x");
    }

    #[test]
    fn text_from_blocks_joins_text_blocks_only() {
        let blocks = vec![
            IPythonCellContentBlock {
                r#type: "text".to_string(),
                text: Some("a".to_string()),
                ..Default::default()
            },
            IPythonCellContentBlock {
                r#type: "image".to_string(),
                data: Some("x".to_string()),
                mime_type: Some("image/png".to_string()),
                ..Default::default()
            },
            IPythonCellContentBlock {
                r#type: "text".to_string(),
                text: Some("b".to_string()),
                ..Default::default()
            },
        ];
        assert_eq!(text_from_blocks(Some(&blocks)), "a\nb");
    }
    /// G2-11: an expanded cell with stdout output keeps the blank separator
    /// between the code block and the output (TS startOutput addBlank,
    /// ipython-cell.ts:554-562; the port's stdout branch skipped add_blank).
    #[test]
    fn t14_g2_11_expanded_stdout_cell_keeps_the_blank_separator() {
        init();
        let mut cell = state("print(1)");
        cell.expanded = Some(true);
        cell.execution_started = Some(true);
        cell.details = Some(json!({ "status": "ok", "stdout": "hello\n" }));
        let mut component = IPythonCellComponent::new(cell);
        let rendered = plain(&component.render(80.0));
        let lines: Vec<&str> = rendered.split('\n').collect();
        // The collapsed header also mentions the code, so anchor on the LAST
        // code line (the end of the code block).
        let code_idx = lines
            .iter()
            .rposition(|line| line.contains("print(1)"))
            .expect("code line rendered");
        let out_idx = lines
            .iter()
            .position(|line| line.contains("hello"))
            .expect("stdout line rendered");
        assert!(
            out_idx > code_idx + 1
                && lines[code_idx + 1..out_idx].iter().any(|line| line.is_empty()),
            "expected a blank separator between the code block and the stdout output\n{rendered}"
        );
    }

}
