//! Port of packages/coding-agent/src/modes/interactive/components/extension-editor.ts
//!
//! `spawnSync(editor, [...args, tmpFile], { stdio: "inherit", shell })` runs the
//! external editor; the port uses `std::process::Command` with inherited
//! stdio, which is the same synchronous contract.

use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;

use pi_tui::components::editor::{Editor, EditorOptions, EditorTheme as TuiEditorTheme};
use pi_tui::components::select_list::SelectListTheme as TuiSelectListTheme;
use pi_tui::components::spacer::Spacer;
use pi_tui::components::text::Text;
use pi_tui::keybindings::get_keybindings;
use pi_tui::tui::Focusable as _;
use pi_tui::tui::{Component, Container, Focusable, TuiStopOptions, TUI};

use super::super::theme::theme::{get_editor_theme, theme};
use super::dynamic_border::{ColorFn, DynamicBorder};
use super::keybinding_hints::{key_hint, KeyTextOptions};

/// `theme.ts`'s `EditorTheme` carries `Box` closures with `Send + Sync`; the
/// `pi-tui` editor takes `Rc` closures without those bounds. Private port of the
/// conversion the other selectors apply to `SelectListTheme`.
fn to_tui_editor_theme(source: super::super::theme::theme::EditorTheme) -> TuiEditorTheme {
    TuiEditorTheme {
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
        select_list: TuiSelectListTheme {
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

/// Port of `KeybindingsManager` from `core/keybindings.ts` (another slice).
/// Only `matches` is observable here.
pub trait KeybindingsManager {
    fn matches(&self, data: &str, keybinding: &str) -> bool;
}

/// Port of the app keybinding manager wrapper used by interactive mode.
#[derive(Clone, Default)]
pub struct AppKeybindingsManager;

impl KeybindingsManager for AppKeybindingsManager {
    fn matches(&self, data: &str, keybinding: &str) -> bool {
        get_keybindings().matches(data, keybinding)
    }
}

/// Port of `ExtensionEditorComponent extends Container implements Focusable`.
pub struct ExtensionEditorComponent {
    container: Container,
    editor: Rc<RefCell<Editor>>,
    on_submit_callback: Box<dyn FnMut(String)>,
    on_cancel_callback: Box<dyn FnMut()>,
    tui: Rc<RefCell<TUI>>,
    keybindings: Arc<dyn KeybindingsManager + Send + Sync>,
    external_editor_error: Rc<RefCell<Text>>,
    focused: bool,
    /// `this.editor.onSubmit = (text) => this.onSubmitCallback(text)`.
    /// The pi-tui editor owns the callback, so the submitted text is routed
    /// through this slot and delivered from `handle_input`.
    submitted: Rc<RefCell<Option<String>>>,
}

impl ExtensionEditorComponent {
    pub fn new(
        tui: Rc<RefCell<TUI>>,
        keybindings: Arc<dyn KeybindingsManager + Send + Sync>,
        title: &str,
        prefill: Option<&str>,
        on_submit: Box<dyn FnMut(String)>,
        on_cancel: Box<dyn FnMut()>,
        options: EditorOptions,
    ) -> Self {
        let mut container = Container::new();

        container.add_child(Rc::new(RefCell::new(DynamicBorder::new(border_color()))));
        container.add_child(Rc::new(RefCell::new(Spacer::new(1))));

        container.add_child(Rc::new(RefCell::new(Text::new(
            theme().fg("accent", title),
            1,
            0,
            None,
        ))));
        container.add_child(Rc::new(RefCell::new(Spacer::new(1))));

        let mut editor = Editor::new(
            Rc::clone(&tui),
            to_tui_editor_theme(get_editor_theme()),
            options,
        );
        if let Some(prefill) = prefill {
            editor.set_text(prefill);
        }
        let submitted: Rc<RefCell<Option<String>>> = Rc::new(RefCell::new(None));
        {
            let submitted = Rc::clone(&submitted);
            editor.on_submit = Some(Box::new(move |text: &str| {
                *submitted.borrow_mut() = Some(text.to_string());
            }));
        }
        let editor = Rc::new(RefCell::new(editor));
        container.add_child(Rc::clone(&editor) as Rc<RefCell<dyn Component>>);

        container.add_child(Rc::new(RefCell::new(Spacer::new(1))));

        let has_external_editor = process_env_visual_editor().is_some();
        let hint = format!(
            "{}{}{}{}",
            key_hint("tui.select.confirm", "submit", &KeyTextOptions::default()),
            format!(
                "  {}",
                key_hint("tui.input.newLine", "newline", &KeyTextOptions::default())
            ),
            format!(
                "  {}",
                key_hint("tui.select.cancel", "cancel", &KeyTextOptions::default())
            ),
            if has_external_editor {
                format!(
                    "  {}",
                    key_hint(
                        "app.editor.external",
                        "external editor",
                        &KeyTextOptions::default()
                    )
                )
            } else {
                String::new()
            }
        );
        container.add_child(Rc::new(RefCell::new(Text::new(hint, 1, 0, None))));

        let external_editor_error = Rc::new(RefCell::new(Text::new(String::new(), 1, 0, None)));
        container.add_child(Rc::clone(&external_editor_error) as Rc<RefCell<dyn Component>>);

        container.add_child(Rc::new(RefCell::new(Spacer::new(1))));

        container.add_child(Rc::new(RefCell::new(DynamicBorder::new(border_color()))));

        Self {
            container,
            editor,
            on_submit_callback: on_submit,
            on_cancel_callback: on_cancel,
            tui,
            keybindings,
            external_editor_error,
            focused: false,
            submitted,
        }
    }

    /// Port of `handleInput(keyData)`.
    pub fn handle_input(&mut self, key_data: &str) {
        let kb = get_keybindings();
        if kb.matches(key_data, "tui.select.cancel") {
            (self.on_cancel_callback)();
            return;
        }

        if self.keybindings.matches(key_data, "app.editor.external") {
            self.open_external_editor();
            return;
        }

        Component::handle_input(&mut *self.editor.borrow_mut(), key_data);
        let submitted = self.submitted.borrow_mut().take();
        if let Some(text) = submitted {
            (self.on_submit_callback)(text);
        }
    }

    /// Port of `openExternalEditor()`.
    pub fn open_external_editor(&mut self) {
        let editor_cmd = match process_env_visual_editor() {
            Some(command) => command,
            None => return,
        };
        self.set_external_editor_error(String::new());

        let current_text = self.editor.borrow().get_text();
        let tmp_file =
            std::env::temp_dir().join(format!("pi-extension-editor-{}.md", now_millis()));
        let mut command = match external_editor_command(&editor_cmd, &tmp_file) {
            Ok(command) => command,
            Err(error) => {
                self.set_external_editor_error(error);
                return;
            }
        };

        if let Err(error) = std::fs::write(&tmp_file, &current_text) {
            self.set_external_editor_error(format!("Could not write the editor draft: {error}"));
            return;
        }
        self.tui.borrow_mut().stop(TuiStopOptions::default());

        let mut error_message = None;
        match command.status() {
            Ok(status) if status.success() => match std::fs::read_to_string(&tmp_file) {
                Ok(content) => {
                    let new_content = strip_trailing_newline(&content);
                    self.editor.borrow_mut().set_text(&new_content);
                }
                Err(error) => {
                    error_message = Some(format!("Could not read the editor draft: {error}"));
                }
            },
            Ok(status) => {
                error_message = Some(format!("External editor exited with {status}"));
            }
            Err(error) => {
                error_message = Some(format!("Could not start the external editor: {error}"));
            }
        }

        // `finally`: always clean up, restart the TUI and force a full re-render.
        let _ = std::fs::remove_file(&tmp_file);
        self.tui.borrow_mut().start();
        self.tui.borrow_mut().request_render_forced();
        if let Some(error) = error_message {
            self.set_external_editor_error(error);
        }
    }

    fn set_external_editor_error(&mut self, message: String) {
        let text = if message.is_empty() {
            message
        } else {
            theme().fg("error", &message)
        };
        self.external_editor_error.borrow_mut().set_text(text);
    }
}

pub(in crate::modes::interactive) fn external_editor_command(
    editor_command: &str,
    draft_path: &Path,
) -> Result<std::process::Command, String> {
    let parts = parse_editor_command_for(editor_command, cfg!(windows))?;
    let program = parts
        .first()
        .filter(|program| !program.is_empty())
        .ok_or_else(|| "No editor executable configured.".to_string())?;
    let mut command = std::process::Command::new(program);
    command.args(&parts[1..]);
    command.arg(draft_path);
    Ok(command)
}

fn parse_editor_command_for(input: &str, windows: bool) -> Result<Vec<String>, String> {
    let characters: Vec<char> = input.chars().collect();
    let mut parts = Vec::new();
    let mut index = 0;
    while index < characters.len() {
        while index < characters.len() && characters[index].is_whitespace() {
            index += 1;
        }
        if index == characters.len() {
            break;
        }

        let mut part = String::new();
        if windows {
            let mut quoted = false;
            while index < characters.len() {
                let character = characters[index];
                if character.is_whitespace() && !quoted {
                    break;
                }
                if character == '\\' {
                    let start = index;
                    while index < characters.len() && characters[index] == '\\' {
                        index += 1;
                    }
                    let backslashes = index - start;
                    if characters.get(index) == Some(&'"') {
                        part.extend(std::iter::repeat_n('\\', backslashes / 2));
                        if backslashes % 2 == 1 {
                            part.push('"');
                        } else {
                            quoted = !quoted;
                        }
                        index += 1;
                    } else {
                        part.extend(std::iter::repeat_n('\\', backslashes));
                    }
                    continue;
                }
                if character == '"' {
                    quoted = !quoted;
                    index += 1;
                    continue;
                }
                part.push(character);
                index += 1;
            }
            if quoted {
                return Err("Editor command has an unterminated quoted string.".to_string());
            }
        } else {
            let mut quote = None;
            while index < characters.len() {
                let character = characters[index];
                if quote.is_none() && character.is_whitespace() {
                    break;
                }
                if character == '\'' && quote != Some('"') {
                    quote = if quote == Some('\'') {
                        None
                    } else {
                        Some('\'')
                    };
                    index += 1;
                    continue;
                }
                if character == '"' && quote != Some('\'') {
                    quote = if quote == Some('"') { None } else { Some('"') };
                    index += 1;
                    continue;
                }
                if character == '\\' && quote != Some('\'') {
                    index += 1;
                    let Some(escaped) = characters.get(index).copied() else {
                        return Err("Editor command has a trailing escape.".to_string());
                    };
                    part.push(escaped);
                    index += 1;
                    continue;
                }
                part.push(character);
                index += 1;
            }
            if quote.is_some() {
                return Err("Editor command has an unterminated quoted string.".to_string());
            }
        }
        parts.push(part);
    }
    Ok(parts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_editor_command_preserves_quoted_nonascii_executable_and_arguments() {
        let parts = parse_editor_command_for(
            r#""C:\Program Files\Éditeur\editor.exe" --wait "Série A""#,
            true,
        )
        .unwrap();
        assert_eq!(
            parts,
            [
                "C:\\Program Files\\Éditeur\\editor.exe",
                "--wait",
                "Série A"
            ]
        );

        let command = external_editor_command(
            r#""C:\Program Files\Éditeur\editor.exe" --wait"#,
            Path::new("draft.md"),
        );
        if cfg!(windows) {
            let command = command.unwrap();
            assert_eq!(
                command.get_program(),
                "C:\\Program Files\\Éditeur\\editor.exe"
            );
            assert_eq!(
                command.get_args().collect::<Vec<_>>(),
                ["--wait", "draft.md"]
            );
        }
    }

    #[test]
    fn nonwindows_editor_command_keeps_plain_configuration_and_supports_quotes() {
        assert_eq!(
            parse_editor_command_for("vim --clean", false).unwrap(),
            ["vim", "--clean"]
        );
        assert_eq!(
            parse_editor_command_for("'/tmp/Éditeur spécial' --wait", false).unwrap(),
            ["/tmp/Éditeur spécial", "--wait"]
        );
    }

    #[test]
    fn malformed_editor_commands_are_errors() {
        assert!(parse_editor_command_for("\"unterminated", true).is_err());
        assert!(parse_editor_command_for("\"unterminated", false).is_err());
        assert!(external_editor_command("   ", Path::new("draft.md")).is_err());
    }
}

/// `new DynamicBorder()`'s default color: `theme.fg("border", str)`.
fn border_color() -> ColorFn {
    Box::new(|text: &str| theme().fg("border", text))
}

pub(in crate::modes::interactive) fn strip_trailing_newline(content: &str) -> String {
    content
        .strip_suffix('\n')
        .map(|text| text.to_string())
        .unwrap_or_else(|| content.to_string())
}

/// `process.env.VISUAL || process.env.EDITOR`.
pub(in crate::modes::interactive) fn process_env_visual_editor() -> Option<String> {
    match std::env::var("VISUAL") {
        Ok(value) if !value.is_empty() => Some(value),
        _ => match std::env::var("EDITOR") {
            Ok(value) if !value.is_empty() => Some(value),
            _ => None,
        },
    }
}

/// `Date.now()`.
fn now_millis() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or(0)
}

impl Focusable for ExtensionEditorComponent {
    fn focused(&self) -> bool {
        self.focused
    }

    fn set_focused(&mut self, focused: bool) {
        self.focused = focused;
        self.editor.borrow_mut().set_focused(focused);
    }
}

impl Component for ExtensionEditorComponent {
    fn render(&mut self, width: f64) -> Vec<String> {
        Component::render(&mut self.container, width)
    }

    fn handle_input(&mut self, data: &str) {
        ExtensionEditorComponent::handle_input(self, data);
    }

    fn invalidate(&mut self) {
        Component::invalidate(&mut self.container);
    }

    fn as_focusable(&mut self) -> Option<&mut dyn Focusable> {
        Some(self)
    }
}
