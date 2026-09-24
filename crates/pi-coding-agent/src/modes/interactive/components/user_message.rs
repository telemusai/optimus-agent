//! Port of packages/coding-agent/src/modes/interactive/components/user-message.ts

use std::rc::Rc;
use std::sync::Arc;

use pi_tui::components::markdown::{Markdown, MarkdownTheme as TuiMarkdownTheme};
use pi_tui::components::r#box::Box_;
use pi_tui::selection_metadata::TableCellSelectionRegion;
use pi_tui::tui::Component;

use crate::core::slash_commands::{builtin_slash_command_takes_argument, parse_slash_command};
use crate::modes::interactive::theme::theme::{get_markdown_theme, theme, MarkdownTheme};

use super::prompt_highlight::PromptTokenMask;

/// A submitted prompt surface, independent of the editor and global accents.
fn submitted_prompt_background(content: &str) -> String {
    let palette = theme();
    if palette.name.as_deref() == Some("light") {
        return palette.get_user_message_background_color()(content);
    }
    let ansi = match palette.color_mode() {
        pi_tui::terminal_colors::TerminalColorMode::Truecolor => "\x1b[48;2;28;28;28m",
        _ => "\x1b[48;5;234m",
    };
    format!("{ansi}{content}\x1b[49m")
}

/// `OSC133_ZONE_START`
pub const OSC133_ZONE_START: &str = "\u{1b}]133;A\u{7}";
/// `OSC133_ZONE_END`
pub const OSC133_ZONE_END: &str = "\u{1b}]133;B\u{7}";
/// `OSC133_ZONE_FINAL`
pub const OSC133_ZONE_FINAL: &str = "\u{1b}]133;C\u{7}";

/// `isRecognizedSlashCommand` callback.
pub type IsRecognizedSlashCommand<'a> = &'a dyn Fn(&str) -> bool;

/// `theme.ts`'s `MarkdownTheme` carries `Arc` closures with `Send + Sync`; the
/// `pi-tui` markdown component holds `Rc` closures without those bounds. Private
/// port of the conversion the other message cards apply.
fn to_tui_markdown_theme(source: MarkdownTheme) -> TuiMarkdownTheme {
    fn rc(value: Arc<dyn Fn(&str) -> String + Send + Sync>) -> Rc<dyn Fn(&str) -> String> {
        Rc::new(move |text: &str| value(text))
    }

    TuiMarkdownTheme {
        heading: rc(source.heading),
        link: rc(source.link),
        link_url: rc(source.link_url),
        code: rc(source.code),
        code_block: rc(source.code_block),
        code_block_border: rc(source.code_block_border),
        quote: rc(source.quote),
        quote_border: rc(source.quote_border),
        hr: rc(source.hr),
        list_bullet: rc(source.list_bullet),
        bold: rc(source.bold),
        italic: rc(source.italic),
        strikethrough: rc(source.strikethrough),
        underline: rc(source.underline),
        highlight_code: Some(Rc::new(move |code: &str, language: Option<&str>| {
            (source.highlight_code)(code, language)
        })),
        code_block_indent: source.code_block_indent,
        math: Some(rc(source.math)),
        math_block: Some(rc(source.math_block)),
    }
}

struct HighlightedMarkdown {
    markdown: Markdown,
    mask: PromptTokenMask,
}

impl HighlightedMarkdown {
    fn new(
        text: &str,
        markdown_theme: TuiMarkdownTheme,
        command_end: usize,
        include_bare_separator: bool,
    ) -> Self {
        let mask = PromptTokenMask::new(text, command_end, include_bare_separator);
        let markdown = Markdown::new(
            mask.text.clone(),
            0,
            0,
            markdown_theme,
            Some(pi_tui::components::markdown::DefaultTextStyle {
                color: Some(Rc::new(|content: &str| {
                    theme().fg("userMessageText", content)
                })),
                ..Default::default()
            }),
            Default::default(),
        );
        Self { markdown, mask }
    }
}

impl Component for HighlightedMarkdown {
    fn render(&mut self, width: f64) -> Vec<String> {
        self.markdown
            .render(width)
            .into_iter()
            .map(|line| self.mask.restore_line(&line))
            .collect()
    }

    fn get_selection_regions(&self) -> Vec<TableCellSelectionRegion> {
        self.markdown
            .get_selection_regions()
            .iter()
            .map(|region| TableCellSelectionRegion {
                content: self.mask.restore_text(&region.content),
                ..region.clone()
            })
            .collect()
    }

    fn invalidate(&mut self) {
        self.markdown.invalidate();
    }
}

pub struct UserMessageComponent {
    content_box: Box_,
}

impl UserMessageComponent {
    pub fn new(
        text: &str,
        markdown_theme: MarkdownTheme,
        is_recognized_slash_command: IsRecognizedSlashCommand<'_>,
    ) -> Self {
        let command = parse_slash_command(text);
        let command_end = match &command {
            Some(command) if is_recognized_slash_command(&command.name) => command.name.len() + 1,
            _ => 0,
        };
        let include_bare_separator = match &command {
            Some(command) => command_end > 0 && builtin_slash_command_takes_argument(&command.name),
            None => false,
        };

        let mut content_box = Box_::new(
            2,
            1,
            Some(Box::new(|content: &str| {
                submitted_prompt_background(content)
            })),
        );
        content_box.add_child(Box::new(HighlightedMarkdown::new(
            text,
            to_tui_markdown_theme(markdown_theme),
            command_end,
            include_bare_separator,
        )));

        Self { content_box }
    }
}

impl Default for UserMessageComponent {
    fn default() -> Self {
        Self::new("", get_markdown_theme(), &|_name| false)
    }
}

impl Component for UserMessageComponent {
    fn render(&mut self, width: f64) -> Vec<String> {
        let mut lines = self.content_box.render(width);
        if lines.is_empty() {
            return lines;
        }

        let last = lines.len() - 1;
        lines[0] = format!("{OSC133_ZONE_START}{}", lines[0]);
        lines[last] = format!("{OSC133_ZONE_END}{OSC133_ZONE_FINAL}{}", lines[last]);
        lines
    }

    fn get_selection_regions(&self) -> Vec<TableCellSelectionRegion> {
        self.content_box.get_selection_regions()
    }

    fn invalidate(&mut self) {
        self.content_box.invalidate();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modes::interactive::theme::theme::get_markdown_theme;
    use pi_tui::utils::strip_ansi;

    #[test]
    fn wraps_the_render_in_osc133_zones() {
        let mut component =
            UserMessageComponent::new("hello", get_markdown_theme(), &|_name| false);
        let lines = component.render(20.0);
        assert!(lines[0].starts_with(OSC133_ZONE_START));
        assert!(lines[lines.len() - 1].starts_with(OSC133_ZONE_END));
        assert!(lines[lines.len() - 1].contains(OSC133_ZONE_FINAL));
        assert!(
            strip_ansi(&lines[0]).trim().starts_with("hello")
                || strip_ansi(&lines[1]).contains("hello")
        );
    }

    #[test]
    fn recognized_slash_command_marks_its_name_length() {
        let mut component =
            UserMessageComponent::new("/help me", get_markdown_theme(), &|name| name == "help");
        let lines = component.render(30.0);
        assert!(!lines.is_empty());
    }
}
