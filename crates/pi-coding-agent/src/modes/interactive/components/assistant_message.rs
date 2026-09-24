//! Port of packages/coding-agent/src/modes/interactive/components/assistant-message.ts

use std::collections::HashMap;

use pi_ai::types::{AssistantMessage, ContentBlock};
use pi_tui::components::markdown::{
    DefaultTextStyle, Markdown, MarkdownOptions, MarkdownTheme as TuiMarkdownTheme,
};
use pi_tui::components::spacer::Spacer;
use pi_tui::components::text::Text;
use pi_tui::tui::{Component, Container};
use pi_tui::utils::{truncate_to_width, visible_width};
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use crate::core::auth_guidance::LOGIN_RECOVERY_MESSAGE;
use crate::modes::interactive::components::collapsible_error::{
    normalize_error_details, should_collapse_error_details, summarize_error_details,
    CollapsibleErrorComponent, CollapsibleErrorOptions,
};
use crate::modes::interactive::components::keybinding_hints::expand_collapse_hint;
use crate::modes::interactive::components::mermaid::MermaidMarkdownTransform;
use crate::modes::interactive::theme::theme::{theme, MarkdownTheme};

/// Normal AI prose only; markdown code, semantic styles and activity keep their palette.
fn assistant_prose(text: &str) -> String {
    let ansi = match theme().color_mode() {
        pi_tui::terminal_colors::TerminalColorMode::Truecolor => "\x1b[38;2;19;161;14m",
        _ => "\x1b[38;5;34m",
    };
    format!("{ansi}{text}\x1b[39m")
}

const OSC133_ZONE_START: &str = "\u{1b}]133;A\u{7}";
const OSC133_ZONE_END: &str = "\u{1b}]133;B\u{7}";
const OSC133_ZONE_FINAL: &str = "\u{1b}]133;C\u{7}";

/// `\n\n${LOGIN_RECOVERY_MESSAGE}`
fn login_recovery_suffix() -> String {
    format!("\n\n{LOGIN_RECOVERY_MESSAGE}")
}

/// `AssistantMessageComponentOptions`
pub struct AssistantMessageComponentOptions {
    pub cwd: Option<String>,
    pub expanded: bool,
    pub preceded_by_tool_activity: bool,
    /// Replaces Mermaid code blocks in assistant text (never thinking) with
    /// Unicode diagrams.
    pub mermaid_transform: Option<Rc<MermaidMarkdownTransform>>,
}

impl Default for AssistantMessageComponentOptions {
    fn default() -> Self {
        Self {
            cwd: None,
            expanded: false,
            preceded_by_tool_activity: false,
            mermaid_transform: None,
        }
    }
}

/// Private port of `toTuiMarkdownTheme` (`MarkdownTheme` of theme.ts carries
/// `Arc` closures with `Send + Sync`; the pi-tui component holds `Rc` closures).
fn to_tui_markdown_theme(theme_source: MarkdownTheme) -> TuiMarkdownTheme {
    fn rc(value: Arc<dyn Fn(&str) -> String + Send + Sync>) -> Rc<dyn Fn(&str) -> String> {
        Rc::new(move |text: &str| value(text))
    }

    TuiMarkdownTheme {
        heading: rc(theme_source.heading),
        link: rc(theme_source.link),
        link_url: rc(theme_source.link_url),
        code: rc(theme_source.code),
        code_block: rc(theme_source.code_block),
        code_block_border: rc(theme_source.code_block_border),
        quote: rc(theme_source.quote),
        quote_border: rc(theme_source.quote_border),
        hr: rc(theme_source.hr),
        list_bullet: rc(theme_source.list_bullet),
        bold: rc(theme_source.bold),
        italic: rc(theme_source.italic),
        strikethrough: rc(theme_source.strikethrough),
        underline: rc(theme_source.underline),
        highlight_code: Some(Rc::new(move |code: &str, language: Option<&str>| {
            (theme_source.highlight_code)(code, language)
        })),
        code_block_indent: theme_source.code_block_indent,
        math: Some(rc(theme_source.math)),
        math_block: Some(rc(theme_source.math_block)),
    }
}

/// The theme closures are `Rc`, so a fresh owned theme can share them.
fn clone_tui_markdown_theme(theme_source: &TuiMarkdownTheme) -> TuiMarkdownTheme {
    TuiMarkdownTheme {
        heading: Rc::clone(&theme_source.heading),
        link: Rc::clone(&theme_source.link),
        link_url: Rc::clone(&theme_source.link_url),
        code: Rc::clone(&theme_source.code),
        code_block: Rc::clone(&theme_source.code_block),
        code_block_border: Rc::clone(&theme_source.code_block_border),
        quote: Rc::clone(&theme_source.quote),
        quote_border: Rc::clone(&theme_source.quote_border),
        hr: Rc::clone(&theme_source.hr),
        list_bullet: Rc::clone(&theme_source.list_bullet),
        bold: Rc::clone(&theme_source.bold),
        italic: Rc::clone(&theme_source.italic),
        strikethrough: Rc::clone(&theme_source.strikethrough),
        underline: Rc::clone(&theme_source.underline),
        highlight_code: theme_source.highlight_code.clone(),
        code_block_indent: theme_source.code_block_indent.clone(),
        math: theme_source.math.clone(),
        math_block: theme_source.math_block.clone(),
    }
}

/// Port of `getThinkingMarkdownTheme`.
fn get_thinking_markdown_theme(base_theme: &TuiMarkdownTheme) -> TuiMarkdownTheme {
    fn quiet(text: &str) -> String {
        theme().fg("thinkingText", text)
    }
    // The theme holds `Rc` closures, so each field gets its own `Rc` over the
    // shared function item (the same `quiet` behaviour for every field).
    let mut thinking_theme = clone_tui_markdown_theme(base_theme);
    thinking_theme.heading = Rc::new(quiet);
    thinking_theme.link = Rc::new(quiet);
    thinking_theme.link_url = Rc::new(quiet);
    thinking_theme.code = Rc::new(quiet);
    thinking_theme.code_block = Rc::new(quiet);
    thinking_theme.code_block_border = Rc::new(quiet);
    thinking_theme.quote = Rc::new(quiet);
    thinking_theme.quote_border = Rc::new(quiet);
    thinking_theme.hr = Rc::new(quiet);
    thinking_theme.list_bullet = Rc::new(quiet);
    thinking_theme.highlight_code = Some(Rc::new(|code: &str, _language: Option<&str>| {
        code.split('\n').map(quiet).collect()
    }));
    thinking_theme
}

/// Single collapsed-thinking row that truncates the recap to the render width
/// instead of wrapping.
struct CollapsedThinkingRow {
    label: String,
    recap: String,
    hint: String,
}

impl Component for CollapsedThinkingRow {
    fn render(&mut self, width: f64) -> Vec<String> {
        let safe_width = std::cmp::max(1, width.max(0.0).floor() as usize);
        let separator = theme().fg("dim", " \u{b7} ");
        let fixed_width = visible_width(&format!(" {}{separator} {}", self.label, self.hint));
        let recap_width = std::cmp::max(8, safe_width.saturating_sub(fixed_width));
        let recap = theme().fg(
            "thinkingText",
            &truncate_to_width(&self.recap, recap_width as f64, "", false),
        );
        vec![truncate_to_width(
            &format!(" {}{separator}{recap} {}", self.label, self.hint),
            safe_width as f64,
            "",
            false,
        )]
    }

    fn invalidate(&mut self) {}
}

/// One-line recap for a collapsed thinking block: the last bold section header
/// when the trace has one (reasoning summaries usually do), otherwise the first
/// non-empty line, stripped of markdown emphasis and truncated.
pub fn thinking_recap(thinking: &str, fallback: &str, max_width: Option<f64>) -> String {
    let lines: Vec<String> = thinking
        .split('\n')
        .map(|line| line.trim().to_string())
        .filter(|line| !line.is_empty())
        .collect();
    let last_header = lines
        .iter()
        .rev()
        .find(|line| is_bold_header(line) || is_markdown_header(line))
        .cloned();
    let source = last_header
        .or_else(|| lines.first().cloned())
        .unwrap_or_else(|| fallback.to_string());
    let plain = {
        let without_headers = strip_markdown_header(&source);
        let without_bold = strip_emphasis(&without_headers, "**");
        let without_italic = strip_emphasis(&without_bold, "*");
        let without_code = strip_emphasis(&without_italic, "`");
        let collapsed = collapse_whitespace(&without_code);
        let without_trailing_colon = collapsed
            .strip_suffix(':')
            .unwrap_or(&collapsed)
            .to_string();
        without_trailing_colon.trim().to_string()
    };
    let chosen = if plain.is_empty() {
        fallback.to_string()
    } else {
        plain
    };
    let max_width = max_width.unwrap_or(120.0);
    truncate_to_width(&chosen, f64_max(20.0, max_width), "", false)
}

fn f64_max(left: f64, right: f64) -> f64 {
    if left > right {
        left
    } else {
        right
    }
}

/// `^\*\*[^*]+\*\*:?$`
fn is_bold_header(line: &str) -> bool {
    let trimmed = line.strip_suffix(':').unwrap_or(line);
    trimmed.starts_with("**")
        && trimmed.ends_with("**")
        && trimmed.len() > 4
        && !trimmed[2..trimmed.len() - 2].contains('*')
}

/// `^#{1,6}\s+\S`
fn is_markdown_header(line: &str) -> bool {
    let hashes = line
        .chars()
        .take_while(|character| *character == '#')
        .count();
    if hashes == 0 || hashes > 6 {
        return false;
    }
    let rest = &line[hashes..];
    if let Some(rest) = rest.strip_prefix(' ') {
        return rest.chars().next().is_some();
    }
    rest.starts_with('\t') && rest.chars().nth(1).is_some()
}

/// `.replace(/^#{1,6}\s+/, "")`
fn strip_markdown_header(line: &str) -> String {
    let hashes = line
        .chars()
        .take_while(|character| *character == '#')
        .count();
    if hashes == 0 || hashes > 6 {
        return line.to_string();
    }
    let rest = &line[hashes..];
    if let Some(rest) = rest.strip_prefix(' ') {
        return rest.trim_start_matches(' ').to_string();
    }
    if let Some(rest) = rest.strip_prefix('\t') {
        return rest.to_string();
    }
    line.to_string()
}

/// `.replace(/\*\*([^*]+)\*\*/g, "$1")` and the single-character variants.
fn strip_emphasis(line: &str, marker: &str) -> String {
    let mut result = String::new();
    let mut rest = line;
    while let Some(start) = rest.find(marker) {
        let after_start = &rest[start + marker.len()..];
        let Some(end) = after_start.find(marker) else {
            break;
        };
        let inner = &after_start[..end];
        if marker == "**" && inner.contains('*') {
            result.push_str(&rest[..start + marker.len()]);
            rest = after_start;
            continue;
        }
        result.push_str(&rest[..start]);
        result.push_str(inner);
        rest = &after_start[end + marker.len()..];
    }
    result.push_str(rest);
    result
}

/// `\s+` -> single space (no trim; callers trim).
fn collapse_whitespace(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut in_whitespace = false;
    for character in text.chars() {
        if character.is_whitespace() {
            in_whitespace = true;
            continue;
        }
        if in_whitespace && !result.is_empty() {
            result.push(' ');
        }
        in_whitespace = false;
        result.push(character);
    }
    result
}

/// Port of `formatInlineLoginRecoveryMessage`.
fn format_inline_login_recovery_message(message: &str) -> Option<String> {
    let suffix = login_recovery_suffix();
    let normalized = normalize_error_details(message);
    if !normalized.ends_with(&suffix) {
        return None;
    }
    let base = normalized[..normalized.len() - suffix.len()]
        .trim_end()
        .to_string();
    if base.is_empty() || should_collapse_error_details(&base) {
        return None;
    }
    Some(format!("{base} \u{b7} {LOGIN_RECOVERY_MESSAGE}"))
}

/// Port of `AssistantMessageComponent`.
pub struct AssistantMessageComponent {
    content_container: Rc<RefCell<Container>>,
    hide_thinking_block: bool,
    markdown_theme: TuiMarkdownTheme,
    hidden_thinking_label: String,
    last_message: Option<AssistantMessage>,
    has_tool_calls: bool,
    expanded: bool,
    dirty: bool,
    last_signature: Option<String>,
    block_markdowns: HashMap<usize, Rc<RefCell<Markdown>>>,
    last_block_texts: HashMap<usize, String>,
    preceded_by_tool_activity: bool,
    mermaid_transform: Option<Rc<MermaidMarkdownTransform>>,
    base_url: Option<String>,
    is_streaming: bool,
    container: Container,
}

impl AssistantMessageComponent {
    /// Port of the `AssistantMessageComponent` constructor.
    pub fn new(
        message: Option<AssistantMessage>,
        hide_thinking_block: bool,
        markdown_theme: MarkdownTheme,
        hidden_thinking_label: &str,
        options: AssistantMessageComponentOptions,
    ) -> Self {
        let base_url = options
            .cwd
            .as_ref()
            .filter(|cwd| !cwd.is_empty())
            .map(|cwd| path_to_file_url(&format!("{cwd}{}", std::path::MAIN_SEPARATOR)));
        let tui_markdown_theme = to_tui_markdown_theme(markdown_theme);
        let content_container = Rc::new(RefCell::new(Container::new()));
        let mut container = Container::new();
        container.add_child(Rc::clone(&content_container) as Rc<RefCell<dyn Component>>);

        let mut component = Self {
            content_container,
            hide_thinking_block,
            markdown_theme: tui_markdown_theme,
            hidden_thinking_label: hidden_thinking_label.to_string(),
            last_message: None,
            has_tool_calls: false,
            expanded: options.expanded,
            dirty: false,
            last_signature: None,
            block_markdowns: HashMap::new(),
            last_block_texts: HashMap::new(),
            preceded_by_tool_activity: options.preceded_by_tool_activity,
            mermaid_transform: options.mermaid_transform,
            base_url,
            is_streaming: false,
            container,
        };

        if let Some(message) = message {
            component.update_content(message, component.is_streaming);
        }
        component
    }

    /// Port of `setHideThinkingBlock`.
    pub fn set_hide_thinking_block(&mut self, hide: bool) {
        self.hide_thinking_block = hide;
        self.dirty = true;
    }

    /// Port of `setHiddenThinkingLabel`.
    pub fn set_hidden_thinking_label(&mut self, label: &str) {
        self.hidden_thinking_label = label.to_string();
        self.dirty = true;
    }

    /// Port of `setExpanded`.
    pub fn set_expanded(&mut self, expanded: bool) {
        if self.expanded != expanded {
            self.expanded = expanded;
            self.dirty = true;
        }
    }

    /// Port of `updateContent`.
    pub fn update_content(&mut self, message: AssistantMessage, is_streaming: bool) {
        self.last_message = Some(message);
        self.is_streaming = is_streaming;
        self.dirty = true;
    }

    /// Port of `computeSignature`.
    ///
    /// The TypeScript JSON-encodes the recap so free text cannot forge part
    /// boundaries; the port uses `serde_json::to_string` for the same reason.
    fn compute_signature(&self, message: &AssistantMessage) -> String {
        let mut parts: Vec<String> = Vec::new();
        for (index, content) in message.content.iter().enumerate() {
            match content {
                ContentBlock::Text(text) => {
                    parts.push(format!(
                        "{index}:text:{}",
                        if !text.text.trim().is_empty() { 1 } else { 0 }
                    ));
                }
                ContentBlock::Thinking(thinking) => {
                    parts.push(format!(
                        "{index}:thinking:{}",
                        if !thinking.thinking.trim().is_empty() {
                            1
                        } else {
                            0
                        }
                    ));
                    if self.hide_thinking_block && !thinking.thinking.trim().is_empty() {
                        // The collapsed row bakes the recap into a static line, so a
                        // recap change must count as a structural change during streaming.
                        let recap =
                            thinking_recap(&thinking.thinking, &self.hidden_thinking_label, None);
                        parts.push(format!(
                            "{index}:recap:{}",
                            serde_json::to_string(&recap).unwrap_or_default()
                        ));
                    }
                }
                ContentBlock::ToolCall(_) => parts.push(format!("{index}:toolCall")),
            }
        }
        parts.push(format!("hide:{}", self.hide_thinking_block));
        parts.push(format!("label:{}", self.hidden_thinking_label));
        parts.push(format!("expanded:{}", self.expanded));
        // In the signature so the streaming->final transition rebuilds (mermaid
        // renders differently).
        parts.push(format!("streaming:{}", self.is_streaming));
        parts.push(format!("stop:{}", message.stop_reason));
        parts.push(format!(
            "error:{}",
            message.error_message.clone().unwrap_or_default()
        ));
        parts.join("|")
    }

    /// Port of `reconcile`.
    fn reconcile(&mut self, message: &AssistantMessage) {
        let signature = self.compute_signature(message);
        if Some(&signature) != self.last_signature.as_ref() {
            self.last_signature = Some(signature);
            self.rebuild(message);
            return;
        }

        // Structure unchanged: update only blocks whose text changed (during
        // streaming that is just the final block).
        for (index, content) in message.content.iter().enumerate() {
            let Some(markdown) = self.block_markdowns.get(&index).cloned() else {
                continue;
            };
            let text = match content {
                ContentBlock::Text(text) => text.text.trim().to_string(),
                ContentBlock::Thinking(thinking) => thinking.thinking.trim().to_string(),
                ContentBlock::ToolCall(_) => String::new(),
            };
            if self.last_block_texts.get(&index) != Some(&text) {
                markdown.borrow_mut().set_text(text.clone());
                self.last_block_texts.insert(index, text);
            }
        }
    }

    /// Port of `rebuild`.
    fn rebuild(&mut self, message: &AssistantMessage) {
        {
            let mut content = self.content_container.borrow_mut();
            content.clear();
        }
        self.block_markdowns.clear();
        self.last_block_texts.clear();

        let has_visible_content = message.content.iter().any(is_visible_content);

        if has_visible_content {
            self.content_container
                .borrow_mut()
                .add_child(Rc::new(RefCell::new(Spacer::new(1))) as Rc<RefCell<dyn Component>>);
        }

        // Render content in order
        for (index, content) in message.content.iter().enumerate() {
            match content {
                ContentBlock::Text(text) if !text.text.trim().is_empty() => {
                    // Assistant text messages with no background - trim the text.
                    // Set paddingY=0 to avoid extra spacing before tool executions.
                    let trimmed = text.text.trim().to_string();
                    let mermaid_transform = self.mermaid_transform.clone();
                    let is_streaming = self.is_streaming;
                    let mut options = MarkdownOptions {
                        base_url: self.base_url.clone(),
                        ..Default::default()
                    };
                    if let Some(mermaid_transform) = mermaid_transform {
                        options.transform =
                            Some(Rc::new(move |markdown: &str, available_width: usize| {
                                (*mermaid_transform)(
                                    markdown.to_string(),
                                    available_width as f64,
                                    is_streaming,
                                )
                            }));
                    }
                    let markdown = Rc::new(RefCell::new(Markdown::new(
                        trimmed.clone(),
                        1,
                        0,
                        clone_tui_markdown_theme(&self.markdown_theme),
                        Some(DefaultTextStyle {
                            color: Some(Rc::new(assistant_prose)),
                            ..Default::default()
                        }),
                        options,
                    )));
                    self.block_markdowns.insert(index, Rc::clone(&markdown));
                    self.last_block_texts.insert(index, trimmed);
                    self.content_container
                        .borrow_mut()
                        .add_child(markdown as Rc<RefCell<dyn Component>>);
                }
                ContentBlock::Thinking(thinking) if !thinking.thinking.trim().is_empty() => {
                    // Add spacing only when another visible assistant content block
                    // follows. This avoids a superfluous blank line before separately
                    // rendered tool execution blocks.
                    let has_visible_content_after =
                        message.content[index + 1..].iter().any(is_visible_content);

                    let thinking_label = theme()
                        .bold(&theme().fg("thinkingText", &self.hidden_thinking_label.clone()));
                    if self.hide_thinking_block {
                        // Collapsed row: bold label, a one-line recap of the trace, and
                        // the hint. The row truncates the recap to the render width so it
                        // never wraps onto a second line on narrow terminals.
                        let recap =
                            thinking_recap(&thinking.thinking, &self.hidden_thinking_label, None);
                        self.content_container
                            .borrow_mut()
                            .add_child(Rc::new(RefCell::new(CollapsedThinkingRow {
                                label: thinking_label,
                                recap,
                                hint: expand_collapse_hint("app.thinking.toggle", false),
                            }))
                                as Rc<RefCell<dyn Component>>);
                        if has_visible_content_after {
                            self.content_container
                                .borrow_mut()
                                .add_child(Rc::new(RefCell::new(Spacer::new(1)))
                                    as Rc<RefCell<dyn Component>>);
                        }
                    } else {
                        // Expanded: the same label line with the collapse hint, then the
                        // trace. Thinking traces keep Markdown structure but stay quiet.
                        self.content_container
                            .borrow_mut()
                            .add_child(Rc::new(RefCell::new(Text::new(
                                format!(
                                    "{thinking_label} {}",
                                    expand_collapse_hint("app.thinking.toggle", true)
                                ),
                                1,
                                0,
                                None,
                            )))
                                as Rc<RefCell<dyn Component>>);
                        let trimmed = thinking.thinking.trim().to_string();
                        let default_style = DefaultTextStyle {
                            color: Some(Rc::new(|text: &str| theme().fg("thinkingText", text))),
                            bg_color: None,
                            bold: false,
                            italic: false,
                            strikethrough: false,
                            underline: false,
                        };
                        let markdown = Rc::new(RefCell::new(Markdown::new(
                            trimmed.clone(),
                            1,
                            0,
                            get_thinking_markdown_theme(&self.markdown_theme),
                            Some(default_style),
                            MarkdownOptions {
                                base_url: self.base_url.clone(),
                                ..Default::default()
                            },
                        )));
                        self.block_markdowns.insert(index, Rc::clone(&markdown));
                        self.last_block_texts.insert(index, trimmed);
                        self.content_container
                            .borrow_mut()
                            .add_child(markdown as Rc<RefCell<dyn Component>>);
                        if has_visible_content_after {
                            self.content_container
                                .borrow_mut()
                                .add_child(Rc::new(RefCell::new(Spacer::new(1)))
                                    as Rc<RefCell<dyn Component>>);
                        }
                    }
                }
                _ => {}
            }
        }

        let has_tool_calls = message
            .content
            .iter()
            .any(|content| matches!(content, ContentBlock::ToolCall(_)));
        self.has_tool_calls = has_tool_calls;
        if message.stop_reason == pi_ai::types::STOP_REASON_ABORTED {
            let abort_message = match &message.error_message {
                Some(error_message) if error_message != "Request was aborted" => {
                    error_message.clone()
                }
                _ => "Operation aborted".to_string(),
            };
            self.content_container
                .borrow_mut()
                .add_child(Rc::new(RefCell::new(Spacer::new(1))) as Rc<RefCell<dyn Component>>);
            let component = self.create_error_component(&abort_message, None);
            self.content_container.borrow_mut().add_child(component);
        } else if !has_tool_calls && message.stop_reason == pi_ai::types::STOP_REASON_ERROR {
            let error_msg = match &message.error_message {
                Some(error_message) if !error_message.is_empty() => error_message.clone(),
                _ => "Unknown error".to_string(),
            };
            self.content_container
                .borrow_mut()
                .add_child(Rc::new(RefCell::new(Spacer::new(1))) as Rc<RefCell<dyn Component>>);
            let component = self.create_error_component(&error_msg, Some("Error"));
            self.content_container.borrow_mut().add_child(component);
        }

        if has_tool_calls
            && (has_visible_content
                || message.stop_reason == pi_ai::types::STOP_REASON_ABORTED
                || !self.preceded_by_tool_activity)
        {
            self.content_container
                .borrow_mut()
                .add_child(Rc::new(RefCell::new(Spacer::new(1))) as Rc<RefCell<dyn Component>>);
        }
    }

    /// Port of `createErrorComponent`.
    fn create_error_component(
        &self,
        message: &str,
        prefix: Option<&str>,
    ) -> Rc<RefCell<dyn Component>> {
        let inline_login_recovery = format_inline_login_recovery_message(message);
        if let Some(inline_login_recovery) = inline_login_recovery {
            let text = match prefix {
                Some(prefix) => format!("{prefix}: {inline_login_recovery}"),
                None => inline_login_recovery,
            };
            return Rc::new(RefCell::new(Text::new(
                theme().fg("error", &text),
                1,
                0,
                None,
            )));
        }

        if !should_collapse_error_details(message) {
            let text = match prefix {
                Some(prefix) => format!("{prefix}: {message}"),
                None => message.to_string(),
            };
            return Rc::new(RefCell::new(Text::new(
                theme().fg("error", &text),
                1,
                0,
                None,
            )));
        }

        let text = match prefix {
            Some(prefix) => format!("{prefix}: {message}"),
            None => message.to_string(),
        };
        let summary = match prefix {
            Some(prefix) => format!("{prefix}: {}", summarize_error_details(message)),
            None => summarize_error_details(message),
        };
        Rc::new(RefCell::new(CollapsibleErrorComponent::new(
            CollapsibleErrorOptions {
                text,
                summary: Some(summary),
                expanded: Some(self.expanded),
                force_collapse: None,
                padding_x: None,
            },
        )))
    }
}

/// `isVisibleContent` predicate from `rebuild`.
fn is_visible_content(content: &ContentBlock) -> bool {
    match content {
        ContentBlock::Text(text) => !text.text.trim().is_empty(),
        ContentBlock::Thinking(thinking) => !thinking.thinking.trim().is_empty(),
        ContentBlock::ToolCall(_) => false,
    }
}

/// `pathToFileURL(`${resolve(cwd)}${sep}`).href`
fn path_to_file_url(path: &str) -> String {
    use std::path::{Component as PathComponent, Path, PathBuf};

    // Node resolve() is lexical: it collapses dots without resolving symlinks.
    let absolute = if Path::new(path).is_absolute() {
        PathBuf::from(path)
    } else {
        std::env::current_dir()
            .expect("resolve current working directory")
            .join(path)
    };
    let mut normalized = PathBuf::new();
    for component in absolute.components() {
        match component {
            PathComponent::CurDir => {}
            PathComponent::ParentDir => {
                normalized.pop();
            }
            component => normalized.push(component.as_os_str()),
        }
    }
    let is_directory =
        path.ends_with(std::path::MAIN_SEPARATOR) || (cfg!(windows) && path.ends_with('/'));
    let url = if is_directory {
        url::Url::from_directory_path(&normalized)
    } else {
        url::Url::from_file_path(&normalized)
    };
    url.expect("absolute file URL").to_string()
}

impl Component for AssistantMessageComponent {
    /// Port of the `render` override.
    fn render(&mut self, width: f64) -> Vec<String> {
        if self.dirty {
            if let Some(message) = self.last_message.clone() {
                self.reconcile(&message);
            }
            self.dirty = false;
        }
        let mut lines = self.container.render(width);
        if self.has_tool_calls || lines.is_empty() {
            return lines;
        }

        lines[0] = format!("{OSC133_ZONE_START}{}", lines[0]);
        let last = lines.len() - 1;
        lines[last] = format!("{OSC133_ZONE_END}{OSC133_ZONE_FINAL}{}", lines[last]);
        lines
    }

    fn get_selection_regions(&self) -> Vec<pi_tui::selection_metadata::TableCellSelectionRegion> {
        self.container.get_selection_regions()
    }

    fn invalidate(&mut self) {
        self.container.invalidate();
        // Force a full rebuild so theme-dependent children are recreated.
        self.last_signature = None;
        self.dirty = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modes::interactive::theme::theme::{get_markdown_theme, init_theme};
    use pi_ai::types::{
        Api, ContentBlock, Provider, TextContent, ThinkingContent, ToolCall, Usage,
    };
    use serde_json::Map;

    fn init() {
        init_theme(Some("prime"), false);
    }

    fn assistant(content: Vec<ContentBlock>, stop_reason: &str) -> AssistantMessage {
        AssistantMessage {
            role: "assistant".to_string(),
            content,
            api: Api::from("faux".to_string()),
            provider: Provider::from("faux".to_string()),
            model: "faux".to_string(),
            response_model: None,
            response_id: None,
            diagnostics: None,
            usage: Usage::zero(),
            stop_reason: stop_reason.to_string(),
            stop_reason_raw: None,
            error_message: None,
            timestamp: 0,
        }
    }

    fn text(value: &str) -> ContentBlock {
        ContentBlock::Text(TextContent::new(value))
    }

    fn thinking(value: &str) -> ContentBlock {
        ContentBlock::Thinking(ThinkingContent::new(value))
    }

    fn component() -> AssistantMessageComponent {
        AssistantMessageComponent::new(
            None,
            false,
            get_markdown_theme(),
            "Thinking...",
            Default::default(),
        )
    }

    fn plain(lines: &[String]) -> String {
        lines
            .iter()
            .map(|line| strip_ansi(line))
            .collect::<Vec<String>>()
            .join("\n")
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
                } else if chars.peek() == Some(&']') {
                    chars.next();
                    while let Some(next) = chars.next() {
                        if next == '\u{7}' {
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

    #[test]
    fn thinking_recap_prefers_the_last_bold_header() {
        init();
        let recap = thinking_recap("First line\n\n**Plan**\nDetails here", "Fallback", None);
        assert_eq!(recap, "Plan");
    }

    #[test]
    fn thinking_recap_strips_markdown_emphasis() {
        init();
        assert_eq!(
            thinking_recap("**Bold** and `code`", "Fallback", None),
            "Bold and code"
        );
        assert_eq!(
            thinking_recap("## Heading text", "Fallback", None),
            "Heading text"
        );
        assert_eq!(thinking_recap("", "Fallback", None), "Fallback");
    }

    #[test]
    fn text_blocks_render_and_carry_osc133_zones() {
        init();
        let mut component = component();
        component.update_content(assistant(vec![text("Hello")], "stop"), false);
        let lines = component.render(60.0);
        assert!(lines[0].starts_with(OSC133_ZONE_START));
        assert!(
            lines[lines.len() - 1].starts_with(&format!("{OSC133_ZONE_END}{OSC133_ZONE_FINAL}"))
        );
        assert!(plain(&lines).contains("Hello"));
    }

    #[test]
    fn tool_calls_suppress_the_osc133_zones() {
        init();
        let mut component = component();
        let call = ContentBlock::ToolCall(ToolCall {
            type_: pi_ai::types::TOOL_CALL_TYPE.to_string(),
            id: "call-1".to_string(),
            name: "bash".to_string(),
            arguments: Map::new(),
            thought_signature: None,
        });
        component.update_content(assistant(vec![call], "toolUse"), false);
        let lines = component.render(60.0);
        assert!(!lines[0].starts_with(OSC133_ZONE_START));
    }

    #[test]
    fn hidden_thinking_block_renders_the_collapsed_row() {
        init();
        let mut component = AssistantMessageComponent::new(
            None,
            true,
            get_markdown_theme(),
            "Thinking...",
            Default::default(),
        );
        component.update_content(
            assistant(vec![thinking("**Plan**\nworking")], "stop"),
            false,
        );
        let text = plain(&component.render(80.0));
        assert!(text.contains("Thinking..."));
        assert!(text.contains("Plan"));
        assert!(text.contains("to expand"));
    }

    #[test]
    fn expanded_thinking_block_renders_the_label_and_trace() {
        init();
        let mut component = component();
        component.update_content(assistant(vec![thinking("step one")], "stop"), false);
        let text = plain(&component.render(80.0));
        assert!(text.contains("Thinking..."));
        assert!(text.contains("to collapse"));
        assert!(text.contains("step one"));
    }

    #[test]
    fn streaming_flag_is_part_of_the_signature() {
        init();
        let mut component = component();
        component.update_content(assistant(vec![text("partial")], "stop"), true);
        let signature_streaming =
            component.compute_signature(&assistant(vec![text("partial")], "stop"));
        component.update_content(assistant(vec![text("partial")], "stop"), false);
        let signature_final =
            component.compute_signature(&assistant(vec![text("partial")], "stop"));
        assert_ne!(signature_streaming, signature_final);
    }

    #[test]
    fn aborted_messages_render_the_abort_error() {
        init();
        let mut component = component();
        let mut message = assistant(vec![text("partial")], "aborted");
        message.error_message = Some("Request was aborted".to_string());
        component.update_content(message, false);
        let text = plain(&component.render(80.0));
        assert!(text.contains("Operation aborted"));
    }

    #[test]
    fn error_stop_reason_without_tool_calls_renders_the_error_text() {
        init();
        let mut component = component();
        let mut message = assistant(Vec::new(), "error");
        message.error_message = Some("boom".to_string());
        component.update_content(message, false);
        let text = plain(&component.render(80.0));
        assert!(text.contains("Error: boom"));
    }

    #[test]
    fn inline_login_recovery_message_merges_the_suffix() {
        init();
        let message = format!("Bad credentials{}", login_recovery_suffix());
        let inline = format_inline_login_recovery_message(&message).expect("inline recovery");
        assert!(inline.contains("Bad credentials"));
        assert!(inline.contains(LOGIN_RECOVERY_MESSAGE));
        assert!(inline.contains(" \u{b7} "));
        assert!(format_inline_login_recovery_message("Bad credentials").is_none());
    }

    #[test]
    fn file_urls_use_native_paths_and_escape_reserved_characters() {
        let directory = if cfg!(windows) {
            "C:\\work\\"
        } else {
            "/work/"
        };
        let expected = if cfg!(windows) {
            "file:///C:/work/"
        } else {
            "file:///work/"
        };
        assert_eq!(path_to_file_url(directory), expected);
        let root = std::env::current_dir().unwrap();
        let path = root
            .join("folder with space")
            .join("..")
            .join("file #100%?.md");
        let expected = url::Url::from_file_path(root.join("file #100%?.md")).unwrap();
        assert_eq!(path_to_file_url(path.to_str().unwrap()), expected.as_str());
        assert!(expected.as_str().ends_with("file%20%23100%25%3F.md"));
    }
}
