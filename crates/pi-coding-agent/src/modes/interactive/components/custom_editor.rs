//! Port of packages/coding-agent/src/modes/interactive/components/custom-editor.ts

use std::cell::RefCell;
use std::rc::Rc;

use pi_tui::components::editor::{Editor, EditorOptions, EditorTheme};
use pi_tui::keybindings::{get_keybindings, KeybindingsManager};
use pi_tui::tui::{Component, Focusable, TUI};
use pi_tui::utils::{truncate_to_width, visible_width};
use regex::Regex;

use crate::modes::interactive::components::prompt_highlight::ArgTokenHighlighter;

const COMMAND_TOKEN_PATTERN: &str = r"^(\s*)\/(\S+)";

/// `CustomEditorOptions extends EditorOptions`
pub struct CustomEditorOptions {
    pub padding_x: Option<f64>,
    pub autocomplete_max_visible: Option<f64>,
    pub prompt_prefix: Option<String>,
    pub placeholder: Option<String>,
    pub placeholder_color: Option<Box<dyn Fn(&str) -> String>>,
    pub is_argument_command: Option<Box<dyn Fn(&str) -> bool>>,
}

impl Default for CustomEditorOptions {
    fn default() -> Self {
        Self {
            padding_x: None,
            autocomplete_max_visible: None,
            prompt_prefix: None,
            placeholder: None,
            placeholder_color: None,
            is_argument_command: None,
        }
    }
}

/// Port of `CustomEditor`.
///
/// The TypeScript overrides the protected hooks of `Editor`
/// (`getPromptPrefix`, `formatPromptPrefix`, `getHiddenTextPrefixLength`,
/// `styleDisplayText`). In the port those hooks are private to `pi_tui`, so this
/// wrapper re-implements the two observable behaviours that do not depend on the
/// private layout internals - the command-token styling of the first line and the
/// placeholder/header rows spliced into the rendered output - and delegates
/// everything else to [`Editor`].
pub struct CustomEditor {
    editor: Editor,
    keybindings: KeybindingsManager,
    placeholder: Option<String>,
    placeholder_color: Box<dyn Fn(&str) -> String>,
    is_argument_command: Box<dyn Fn(&str) -> bool>,
    command_color: Option<Rc<dyn Fn(&str) -> String>>,
    background_color: Option<Rc<dyn Fn(&str) -> String>>,
    arg_token_highlighter: ArgTokenHighlighter,
    /// `public onAction(action, handler)` handlers, keyed by action id.
    pub action_handlers: indexmap::IndexMap<String, Box<dyn FnMut()>>,
    pub on_escape: Option<Box<dyn FnMut()>>,
    pub on_ctrl_d: Option<Box<dyn FnMut()>>,
    pub on_paste_image: Option<Box<dyn FnMut()>>,
    pub on_move_below_prompt: Option<Box<dyn FnMut() -> bool>>,
    pub on_agents_back: Option<Box<dyn FnMut() -> bool>>,
    /// When set, the returned line is rendered inside the top of the editor box.
    pub get_header_line: Option<Box<dyn FnMut() -> Option<String>>>,
    /// Handler for extension-registered shortcuts. Returns true if handled.
    pub on_extension_shortcut: Option<Box<dyn FnMut(&str) -> bool>>,
    default_prompt_prefix: String,
    configured_padding_x: usize,
}

impl CustomEditor {
    /// Port of the `CustomEditor` constructor.
    pub fn new(tui: Rc<RefCell<TUI>>, theme: EditorTheme, options: CustomEditorOptions) -> Self {
        let prompt_prefix = options
            .prompt_prefix
            .clone()
            .unwrap_or_else(|| "> ".to_string());
        let configured_padding_x = options.padding_x.unwrap_or(0.0).max(0.0).floor() as usize;
        let command_color = theme.command_color.clone();
        let background_color = theme.background_color.clone();

        let editor = Editor::new(
            tui,
            theme,
            EditorOptions {
                padding_x: options.padding_x,
                autocomplete_max_visible: options.autocomplete_max_visible,
                prompt_prefix: Some(prompt_prefix.clone()),
            },
        );

        Self {
            editor,
            keybindings: get_keybindings(),
            placeholder: options.placeholder,
            placeholder_color: options
                .placeholder_color
                .unwrap_or_else(|| Box::new(|text: &str| text.to_string())),
            is_argument_command: options
                .is_argument_command
                .unwrap_or_else(|| Box::new(|_| false)),
            command_color,
            background_color,
            arg_token_highlighter: ArgTokenHighlighter::new(),
            action_handlers: indexmap::IndexMap::new(),
            on_escape: None,
            on_ctrl_d: None,
            on_paste_image: None,
            on_move_below_prompt: None,
            on_agents_back: None,
            get_header_line: None,
            on_extension_shortcut: None,
            default_prompt_prefix: prompt_prefix,
            configured_padding_x,
        }
    }

    /// Register a handler for an app action.
    pub fn on_action(&mut self, action: &str, handler: Box<dyn FnMut()>) {
        self.action_handlers.insert(action.to_string(), handler);
    }

    /// Port of `setPlaceholder`.
    pub fn set_placeholder(&mut self, placeholder: Option<String>) {
        self.placeholder = placeholder;
        self.editor.invalidate();
    }

    /// Access to the wrapped editor (the TypeScript class extends `Editor`).
    pub fn editor(&self) -> &Editor {
        &self.editor
    }

    pub fn editor_mut(&mut self) -> &mut Editor {
        &mut self.editor
    }

    /// Port of `styleCommandToken`.
    fn style_command_token(
        &self,
        display_text: &str,
        layout_line_index: usize,
        line_text: &str,
        cursor_col: Option<usize>,
    ) -> String {
        let Some(command_color) = &self.command_color else {
            return display_text.to_string();
        };
        if layout_line_index != 0 {
            return display_text.to_string();
        }

        let pattern = CommandTokenPattern::get();
        let Some(captures) = pattern.captures(line_text) else {
            return display_text.to_string();
        };
        let token = captures
            .get(0)
            .map(|m| m.as_str().to_string())
            .unwrap_or_default();
        let leading_whitespace = captures
            .get(1)
            .map(|m| m.as_str().to_string())
            .unwrap_or_default();
        let name = captures
            .get(2)
            .map(|m| m.as_str().to_string())
            .unwrap_or_default();
        if !(self.is_argument_command)(&name) {
            return display_text.to_string();
        }

        let token_start = leading_whitespace.chars().count();
        let token_end = token.chars().count();

        if let Some(cursor_col) = cursor_col {
            if cursor_col < token_end {
                return display_text.to_string();
            }
        }

        let characters: Vec<char> = display_text.chars().collect();
        let before: String = characters[..token_start.min(characters.len())]
            .iter()
            .collect();
        let token_text: String = characters
            [token_start.min(characters.len())..token_end.min(characters.len())]
            .iter()
            .collect();
        let after: String = characters[token_end.min(characters.len())..]
            .iter()
            .collect();
        format!("{before}{}{after}", command_color(&token_text))
    }

    /// Port of `getBashPromptInfo`.
    fn get_bash_prompt_info(line: &str) -> Option<BashPromptInfo> {
        let trimmed_line = line.trim_start();
        let leading_whitespace_length = line.chars().count() - trimmed_line.chars().count();
        if let Some(_rest) = trimmed_line.strip_prefix("!!") {
            return Some(BashPromptInfo {
                prompt_prefix: "!! ".to_string(),
                hidden_text_prefix_length: leading_whitespace_length
                    + if trimmed_line.starts_with("!! ") {
                        3
                    } else {
                        2
                    },
            });
        }
        if trimmed_line.starts_with('!') {
            return Some(BashPromptInfo {
                prompt_prefix: "! ".to_string(),
                hidden_text_prefix_length: leading_whitespace_length
                    + if trimmed_line.starts_with("! ") { 2 } else { 1 },
            });
        }
        None
    }

    /// Port of the `getPromptPrefix` override.
    pub fn get_prompt_prefix(&self) -> String {
        let lines = self.editor.get_lines();
        let first_line = lines.first().cloned().unwrap_or_default();
        match Self::get_bash_prompt_info(&first_line) {
            Some(info) => info.prompt_prefix,
            None => self.default_prompt_prefix.clone(),
        }
    }

    /// Port of the `formatPromptPrefix` override.
    pub fn format_prompt_prefix(&self, prefix: &str) -> String {
        if prefix.starts_with('!') {
            (self.editor.border_color)(prefix)
        } else {
            prefix.to_string()
        }
    }

    /// Port of the `getHiddenTextPrefixLength` override.
    pub fn get_hidden_text_prefix_length(&self, line_index: usize, line: &str) -> f64 {
        if line_index != 0 {
            return 0.0;
        }
        match Self::get_bash_prompt_info(line) {
            Some(info) => info.hidden_text_prefix_length as f64,
            None => 0.0,
        }
    }

    /// Port of `getEffectivePaddingX`.
    fn get_effective_padding_x(&self, width: f64) -> usize {
        let max_padding = std::cmp::max(0, ((width - 1.0) / 2.0).floor() as i64) as usize;
        let configured_padding_x = std::cmp::min(self.configured_padding_x, max_padding);
        if self.background_color.is_some() {
            std::cmp::min(std::cmp::max(configured_padding_x, 2), max_padding)
        } else {
            configured_padding_x
        }
    }

    /// Port of `renderPlaceholderLine`.
    fn render_placeholder_line(&self, width: f64) -> String {
        let padding_x = self.get_effective_padding_x(width);
        let content_width = std::cmp::max(1, width.max(0.0).floor() as usize - padding_x * 2);
        let prompt_prefix_text = self.get_prompt_prefix();
        let prompt_prefix_width = std::cmp::min(
            visible_width(&prompt_prefix_text),
            content_width.saturating_sub(1),
        );
        let input_width = std::cmp::max(1, content_width - prompt_prefix_width);
        let prompt_prefix = if prompt_prefix_width > 0 {
            self.format_prompt_prefix(&truncate_to_width(
                &prompt_prefix_text,
                prompt_prefix_width as f64,
                "",
                false,
            ))
        } else {
            String::new()
        };
        let prompt_prefix_inset = if prompt_prefix_width > 0 {
            std::cmp::min(1, padding_x)
        } else {
            0
        };
        let prompt_leading_padding = " ".repeat(prompt_prefix_inset);
        let prompt_trailing_padding = " ".repeat(padding_x.saturating_sub(prompt_prefix_inset));
        let right_padding = " ".repeat(padding_x);
        let placeholder_width = input_width.saturating_sub(1);
        let placeholder_text = truncate_to_width(
            self.placeholder.as_deref().unwrap_or(""),
            placeholder_width as f64,
            "",
            false,
        );
        let cursor_marker = if self.editor.focused() && !self.editor.is_showing_autocomplete() {
            pi_tui::tui::CURSOR_MARKER
        } else {
            ""
        };
        let cursor_reset = if self.background_color.is_some() {
            "\u{1b}[27m"
        } else {
            "\u{1b}[0m"
        };
        let display_text = format!(
            "{cursor_marker}\u{1b}[7m {cursor_reset}{}",
            (self.placeholder_color)(&placeholder_text)
        );
        let padding =
            " ".repeat(placeholder_width.saturating_sub(visible_width(&placeholder_text)));
        let line = format!(
            "{prompt_leading_padding}{prompt_prefix}{prompt_trailing_padding}{display_text}{padding}{right_padding}"
        );
        let padded = format!(
            "{line}{}",
            " ".repeat(width.max(0.0).floor() as usize - visible_width(&line))
        );
        let padded = match self.background_color.as_ref() {
            Some(background) => background(&padded),
            None => padded,
        };
        // The TypeScript prefixes `getAutocompleteAnchorMarker()`, whose value is
        // private state inside `pi-tui`'s Editor; the port cannot read it, so the
        // placeholder row omits that zero-width marker. See blocked_on.
        padded
    }

    /// Port of `renderHeaderContentLine`.
    fn render_header_content_line(&self, content: &str, width: f64) -> String {
        let padding_x = self.get_effective_padding_x(width);
        let content_width = std::cmp::max(1, width.max(0.0).floor() as usize - padding_x * 2);
        let line = format!(
            "{}{}",
            " ".repeat(padding_x),
            truncate_to_width(content, content_width as f64, "", false)
        );
        let padded = format!(
            "{line}{}",
            " ".repeat(width.max(0.0).floor() as usize - visible_width(&line))
        );
        let Some(background) = self.background_color.as_ref() else {
            return padded;
        };
        // Truncation may inject full ANSI resets; wrap each segment so the
        // background survives past them instead of falling back to the terminal's.
        padded
            .split("\u{1b}[0m")
            .map(|segment| background(segment))
            .collect::<Vec<String>>()
            .join("\u{1b}[0m")
    }

    /// Port of `isCursorAtEnd`.
    fn is_cursor_at_end(&self) -> bool {
        let lines = self.editor.get_lines();
        let (cursor_line, cursor_col) = self.editor.get_cursor();
        cursor_line == lines.len().saturating_sub(1)
            && cursor_col
                == lines
                    .get(cursor_line)
                    .map(|line| line.chars().count())
                    .unwrap_or(0)
    }

    /// Port of `splitRepeatedKeybinding`.
    fn split_repeated_keybinding(&self, data: &str, keybinding: &str) -> Option<Vec<String>> {
        let characters: Vec<char> = data.chars().collect();
        let mut inputs: Vec<String> = Vec::new();
        let mut offset = 0usize;

        while offset < characters.len() {
            let mut matched: Option<String> = None;
            for end in (offset + 1)..=characters.len() {
                let candidate: String = characters[offset..end].iter().collect();
                if self.keybindings_match(&candidate, keybinding) {
                    matched = Some(candidate);
                    offset = end;
                    break;
                }
            }
            match matched {
                None => return None,
                Some(matched) => inputs.push(matched),
            }
        }

        if inputs.len() > 1 {
            Some(inputs)
        } else {
            None
        }
    }

    /// `this.keybindings.matches(data, keybinding)`.
    fn keybindings_match(&self, data: &str, keybinding: &str) -> bool {
        self.keybindings.matches(data, keybinding)
    }
}

/// `{ promptPrefix, hiddenTextPrefixLength }` from `getBashPromptInfo`.
struct BashPromptInfo {
    prompt_prefix: String,
    hidden_text_prefix_length: usize,
}

/// Compiled `COMMAND_TOKEN_PATTERN` for `^(\s*)\/(\S+)`.
struct CommandTokenPattern;

impl CommandTokenPattern {
    fn get() -> &'static Regex {
        use std::sync::OnceLock;
        static PATTERN: OnceLock<Regex> = OnceLock::new();
        PATTERN
            .get_or_init(|| Regex::new(COMMAND_TOKEN_PATTERN).expect("valid command token pattern"))
    }
}

impl Component for CustomEditor {
    /// Port of the `render` override.
    fn render(&mut self, width: f64) -> Vec<String> {
        let lines_source = self.editor.get_lines();
        let first_line = lines_source.first().cloned().unwrap_or_default();
        let is_argument_command_line = match CommandTokenPattern::get().captures(&first_line) {
            Some(captures) => captures
                .get(2)
                .map(|name| (self.is_argument_command)(name.as_str()))
                .unwrap_or(false),
            None => false,
        };
        self.arg_token_highlighter
            .reset(&lines_source, is_argument_command_line);
        let mut lines = Component::render(&mut self.editor, width);
        if let Some(placeholder) = self.placeholder.clone() {
            if self.editor.get_text().is_empty() && lines.len() >= 2 {
                let placeholder_line = self.render_placeholder_line(width);
                let mut next = vec![lines[0].clone(), placeholder_line];
                next.extend(lines[2..].iter().cloned());
                lines = next;
            }
            let _ = placeholder;
        }

        // `getHeaderLine?.()` - the returned line is rendered inside the top of
        // the editor box.
        if lines.len() >= 2 {
            if let Some(get_header_line) = self.get_header_line.as_mut() {
                if let Some(header_line) = get_header_line() {
                    let header_content = self.render_header_content_line(&header_line, width);
                    let header_blank = self.render_header_content_line("", width);
                    let mut next = vec![lines[0].clone(), header_content, header_blank];
                    next.extend(lines[1..].iter().cloned());
                    lines = next;
                }
            }
        }
        lines
    }

    /// Port of the `handleInput` override.
    fn handle_input(&mut self, data: &str) {
        // Check extension-registered shortcuts first
        if let Some(on_extension_shortcut) = self.on_extension_shortcut.as_mut() {
            if on_extension_shortcut(data) {
                return;
            }
        }

        let repeated_clear_input = self.split_repeated_keybinding(data, "app.input.clear");
        if let Some(repeated_clear_input) = repeated_clear_input {
            for input in repeated_clear_input {
                self.handle_input(&input);
            }
            return;
        }

        // Check for paste image keybinding
        if self.keybindings_match(data, "app.clipboard.pasteImage") {
            if let Some(on_paste_image) = self.on_paste_image.as_mut() {
                on_paste_image();
            }
            return;
        }

        // Check app keybindings first
        if self.editor.get_text().is_empty() && self.keybindings_match(data, "app.sidebar.focus") {
            if let Some(on_agents_back) = self.on_agents_back.as_mut() {
                if on_agents_back() {
                    return;
                }
            }
        }

        // Clear input
        if self.keybindings_match(data, "app.input.clear") {
            let had_autocomplete = self.editor.is_showing_autocomplete();
            if had_autocomplete {
                self.editor.cancel_autocomplete();
            }
            if let Some(on_escape) = self.on_escape.as_mut() {
                on_escape();
                return;
            }
            if let Some(handler) = self.action_handlers.get_mut("app.input.clear") {
                handler();
                return;
            }
            if had_autocomplete {
                return;
            }
            Component::handle_input(&mut self.editor, data);
            return;
        }

        // Exit (Ctrl+D) - only when editor is empty
        if self.keybindings_match(data, "app.exit") {
            if self.editor.get_text().is_empty() {
                if let Some(on_ctrl_d) = self.on_ctrl_d.as_mut() {
                    on_ctrl_d();
                    return;
                }
                if let Some(handler) = self.action_handlers.get_mut("app.exit") {
                    handler();
                    return;
                }
            }
            // Fall through to editor handling for delete-char-forward when not empty
        }

        // Check all other app actions. A raw "\n" is Shift+Enter's newline in some
        // terminals, so it goes to the editor even though it decodes as ctrl+j.
        // `keybindingsMatch` reads only the keybindings, so resolve the matches
        // before taking the mutable borrow of the handler map.
        let text_is_empty = self.editor.get_text().is_empty();
        let matching_action = self
            .action_handlers
            .keys()
            .find(|action| {
                data != "\n"
                    && *action != "app.input.clear"
                    && *action != "app.exit"
                    && (*action != "app.shortcuts" || text_is_empty)
                    && self.keybindings_match(data, action)
            })
            .cloned();
        let mut handled = false;
        for (action, handler) in self.action_handlers.iter_mut() {
            if Some(action.as_str()) == matching_action.as_deref() {
                if (action == "app.clear" || action == "app.interrupt")
                    && self.editor.is_showing_autocomplete()
                {
                    self.editor.cancel_autocomplete();
                }
                handler();
                handled = true;
                break;
            }
        }
        if handled {
            return;
        }

        if self.keybindings_match(data, "tui.editor.cursorDown")
            && !self.editor.is_showing_autocomplete()
            && !self.editor.is_history_navigation_active()
            && self.is_cursor_at_end()
        {
            if let Some(on_move_below_prompt) = self.on_move_below_prompt.as_mut() {
                if on_move_below_prompt() {
                    return;
                }
            }
        }

        // Pass to parent for editor handling
        Component::handle_input(&mut self.editor, data);
    }

    fn invalidate(&mut self) {
        Component::invalidate(&mut self.editor);
    }

    fn as_focusable(&mut self) -> Option<&mut dyn Focusable> {
        Some(&mut self.editor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modes::interactive::theme::theme::{get_editor_theme, init_theme};
    use pi_tui::terminal::ProcessTerminal;

    /// `theme.ts`'s `EditorTheme` carries `Box` closures with `Send + Sync`; the
    /// `pi-tui` editor takes `Rc` closures without those bounds. Private port of the
    /// conversion the other editor host applies (`extension-editor.ts`). The
    /// constructor itself already takes the `pi-tui` type.
    fn to_tui_editor_theme(
        source: crate::modes::interactive::theme::theme::EditorTheme,
    ) -> EditorTheme {
        EditorTheme {
            border_color: Rc::new(move |text: &str| (source.border_color)(text)),
            background_color: source.background_color.map(|color| {
                let color: Rc<dyn Fn(&str) -> String> = Rc::new(move |text: &str| (color)(text));
                color
            }),
            autocomplete_background_color: Some({
                let color: Rc<dyn Fn(&str) -> String> =
                    Rc::new(move |text: &str| (source.autocomplete_background_color)(text));
                color
            }),
            select_list: pi_tui::components::select_list::SelectListTheme {
                selected_prefix: source.select_list.selected_prefix,
                selected_text: source.select_list.selected_text,
                description: source.select_list.description,
                argument_hint: Some(source.select_list.argument_hint),
                source_tag: Some(source.select_list.source_tag),
                scroll_info: source.select_list.scroll_info,
                no_match: source.select_list.no_match,
            },
            command_color: Some({
                let color: Rc<dyn Fn(&str) -> String> =
                    Rc::new(move |text: &str| (source.command_color)(text));
                color
            }),
        }
    }

    fn init() {
        init_theme(Some("prime"), false);
        crate::core::keybindings::KeybindingsManager::new(Default::default(), None).install();
    }

    fn editor(options: CustomEditorOptions) -> CustomEditor {
        let tui = Rc::new(RefCell::new(TUI::new(
            Box::new(ProcessTerminal::new()),
            None,
        )));
        CustomEditor::new(tui, to_tui_editor_theme(get_editor_theme()), options)
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

    #[test]
    fn bash_prompt_info_detects_both_prefixes() {
        let info = CustomEditor::get_bash_prompt_info("!! ls").expect("info");
        assert_eq!(info.prompt_prefix, "!! ");
        assert_eq!(info.hidden_text_prefix_length, 3);

        let info = CustomEditor::get_bash_prompt_info("!ls").expect("info");
        assert_eq!(info.prompt_prefix, "! ");
        assert_eq!(info.hidden_text_prefix_length, 1);

        let info = CustomEditor::get_bash_prompt_info("  ! ls").expect("info");
        assert_eq!(info.prompt_prefix, "! ");
        assert_eq!(info.hidden_text_prefix_length, 4);

        assert!(CustomEditor::get_bash_prompt_info("hello").is_none());
    }

    #[test]
    fn prompt_prefix_switches_to_bash_mode_for_a_bang_line() {
        init();
        let mut editor = editor(CustomEditorOptions::default());
        assert_eq!(editor.get_prompt_prefix(), "> ");
        editor.editor_mut().set_text("! echo hi");
        assert_eq!(editor.get_prompt_prefix(), "! ");
        editor.editor_mut().set_text("!! echo hi");
        assert_eq!(editor.get_prompt_prefix(), "!! ");
    }

    #[test]
    fn format_prompt_prefix_colors_bang_prefixes_only() {
        init();
        let editor = editor(CustomEditorOptions::default());
        assert_eq!(editor.format_prompt_prefix("> "), "> ");
        assert_ne!(editor.format_prompt_prefix("! "), "! ");
    }

    #[test]
    fn hidden_text_prefix_length_only_applies_to_the_first_line() {
        init();
        let editor = editor(CustomEditorOptions::default());
        assert_eq!(editor.get_hidden_text_prefix_length(0, "!ls"), 1.0);
        assert_eq!(editor.get_hidden_text_prefix_length(1, "!ls"), 0.0);
    }

    #[test]
    fn placeholder_replaces_the_empty_second_line() {
        init();
        let mut editor = editor(CustomEditorOptions {
            placeholder: Some("Type a message".to_string()),
            ..Default::default()
        });
        let lines = editor.render(40.0);
        assert!(lines.len() >= 2);
        assert!(strip_ansi(&lines[1]).contains("Type a message"));
    }

    #[test]
    fn header_line_is_inserted_above_the_editor_content() {
        init();
        let mut editor = editor(CustomEditorOptions::default());
        editor.get_header_line = Some(Box::new(|| Some("Session recap".to_string())));
        let lines = editor.render(40.0);
        assert!(strip_ansi(&lines[1]).contains("Session recap"));
        assert_eq!(strip_ansi(&lines[2]).trim(), "");
    }

    #[test]
    fn app_action_handlers_fire_before_the_editor() {
        init();
        let mut editor = editor(CustomEditorOptions::default());
        let fired = Rc::new(std::cell::Cell::new(false));
        let fired_for_handler = Rc::clone(&fired);
        editor.on_action(
            "app.tools.expand",
            Box::new(move || fired_for_handler.set(true)),
        );
        let key = pi_tui::keybindings::get_keybindings().get_keys("app.tools.expand");
        let key = key.first().cloned().expect("ctrl+o binding");
        assert_eq!(key, "ctrl+o");
        editor.handle_input("\x0f");
        assert!(fired.get());
    }

    #[test]
    fn ctrl_d_on_an_empty_editor_requests_exit() {
        init();
        let mut editor = editor(CustomEditorOptions::default());
        let exited = Rc::new(std::cell::Cell::new(false));
        let exited_for_handler = Rc::clone(&exited);
        editor.on_ctrl_d = Some(Box::new(move || exited_for_handler.set(true)));
        let keys = pi_tui::keybindings::get_keybindings().get_keys("app.exit");
        assert_eq!(keys[0], "ctrl+d");
        editor.handle_input("\x04");
        assert!(exited.get());
    }

    #[test]
    fn left_edits_nonempty_prompt_instead_of_leaving_the_chat() {
        init();
        let mut editor = editor(CustomEditorOptions::default());
        let back = Rc::new(std::cell::Cell::new(false));
        let called = back.clone();
        editor.on_agents_back = Some(Box::new(move || { called.set(true); true }));
        editor.editor_mut().set_text("/hep");
        editor.handle_input("\x1b[1;1D");
        editor.handle_input("l");
        editor.handle_input("\x1b[1;1C");
        assert_eq!(editor.editor().get_text(), "/help");
        assert!(!back.get());
        editor.editor_mut().set_text("");
        editor.handle_input("\x1b[1;1D");
        assert!(back.get(), "empty-prompt navigation remains available");
    }

    #[test]
    fn repeated_clear_keybinding_splits_into_individual_inputs() {
        init();
        let editor = editor(CustomEditorOptions::default());
        let keys = pi_tui::keybindings::get_keybindings().get_keys("app.input.clear");
        assert_eq!(keys.first().map(String::as_str), Some("escape"));
        let single = "\x1b".to_string();
        assert!(editor
            .split_repeated_keybinding(&single, "app.input.clear")
            .is_none());
        assert_eq!(
            editor.split_repeated_keybinding(&format!("{single}{single}"), "app.input.clear"),
            Some(vec![single.clone(), single])
        );
    }
}
