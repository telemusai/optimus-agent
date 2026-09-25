//! Port of packages/coding-agent/src/modes/interactive/components/login-dialog.ts

use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use pi_ai::utils::oauth::get_oauth_providers;
use pi_tui::components::input::Input;
use pi_tui::components::spacer::Spacer;
use pi_tui::components::text::Text;
use pi_tui::keybindings::get_keybindings;
use pi_tui::terminal_image::{get_capabilities, hyperlink};
use pi_tui::tui::{Component, Focusable, TUI};
use pi_tui::utils::truncate_to_width;

use crate::modes::interactive::theme::theme::theme;
use crate::themes::prime_logo::PRIME_BUTTERFLY_LOGO;
use crate::utils::child_process::{exec_file_hidden, SpawnOptions};
use crate::utils::clipboard::copy_to_clipboard;

use super::keybinding_hints::{format_key_text, key_hint, KeyTextOptions};

/// `PRIME_INFERENCE_PROVIDER_ID`
pub const PRIME_INFERENCE_PROVIDER_ID: &str = "prime-inference";

/// `MenuPanel` (components/menu-panel.ts) belongs to another slice, so this
/// module keeps the parts of the panel the login dialog uses; see
/// evidence/status/ca-interactive-components-3.json -> blocked_on.
const PANEL_PADDING_X: usize = 2;
const PANEL_PADDING_Y: usize = 1;
const FIELD_PADDING_X: usize = 2;
const ANSI_RESET: &str = "\x1b[0m";

/// `getMenuPanelInnerWidth`.
pub fn get_menu_panel_inner_width(width: f64) -> usize {
    let safe_width = (width.max(0.0).floor() as usize).max(PANEL_PADDING_X * 2 + 1);
    (safe_width - PANEL_PADDING_X * 2).max(1)
}

fn apply_background(text: &str, background: &dyn Fn(&str) -> String) -> String {
    text.split(ANSI_RESET)
        .map(|segment| background(segment))
        .collect::<Vec<String>>()
        .join(ANSI_RESET)
}

fn padded_background_line(
    text: &str,
    width: usize,
    padding_x: usize,
    background: Option<&(dyn Fn(&str) -> String + Send + Sync)>,
) -> String {
    let inner_width = (width.saturating_sub(padding_x * 2)).max(1);
    let content = truncate_to_width(text, inner_width as f64, "", false);
    let right_padding =
        " ".repeat(inner_width.saturating_sub(pi_tui::utils::visible_width(&content)));
    let content_span = format!("{}{content}", " ".repeat(padding_x));
    let trailing_span = format!("{right_padding}{}", " ".repeat(padding_x));
    match background {
        Some(background) => {
            apply_background(&content_span, background) + &background(&trailing_span)
        }
        None => content_span + &trailing_span,
    }
}

fn surface_line(text: &str, width: usize) -> String {
    let background = theme().get_editor_background_color();
    padded_background_line(text, width, PANEL_PADDING_X, background.as_deref())
}

fn surface_wrapped_lines(text: &str, width: usize) -> Vec<String> {
    let inner_width = width.saturating_sub(PANEL_PADDING_X * 2).max(1);
    pi_tui::utils::wrap_text_with_ansi(text, inner_width)
        .into_iter()
        .map(|content| surface_line(&content, width))
        .collect()
}

/// Private `MenuPanel` stand-in covering `title` and `subtitle`.
struct MenuPanel {
    title: String,
    subtitle: Option<String>,
    children: Vec<Box<dyn Component>>,
}

impl MenuPanel {
    fn new(title: &str, subtitle: Option<&str>) -> Self {
        Self {
            title: title.to_string(),
            subtitle: subtitle.map(|value| value.to_string()),
            children: Vec::new(),
        }
    }

    fn add_child(&mut self, component: Box<dyn Component>) {
        self.children.push(component);
    }
}

impl Component for MenuPanel {
    fn render(&mut self, width: f64) -> Vec<String> {
        let width = width.max(0.0).floor() as usize;
        let safe_width = width.max(PANEL_PADDING_X * 2 + 1);
        let inner_width = get_menu_panel_inner_width(width as f64);
        let mut lines: Vec<String> = Vec::new();

        for _ in 0..PANEL_PADDING_Y {
            lines.push(surface_line("", safe_width));
        }
        let has_title = !self.title.trim().is_empty();
        let subtitle = self.subtitle.as_ref().map(|value| value.trim().to_string());
        let has_subtitle = subtitle
            .as_deref()
            .map(|value| !value.is_empty())
            .unwrap_or(false);
        let has_header = has_title || has_subtitle;
        if has_title {
            lines.push(surface_line(
                &theme().bold(&theme().fg("text", &self.title)),
                safe_width,
            ));
        }
        if let Some(subtitle) = subtitle.as_ref().filter(|_| has_subtitle) {
            lines.extend(surface_wrapped_lines(
                &theme().fg("muted", subtitle),
                safe_width,
            ));
        }
        if has_header {
            lines.push(surface_line("", safe_width));
        }

        for child in self.children.iter_mut() {
            let child_lines = child.render(inner_width as f64);
            for line in child_lines {
                lines.push(surface_line(&line, safe_width));
            }
        }

        for _ in 0..PANEL_PADDING_Y {
            lines.push(surface_line("", safe_width));
        }
        lines
    }

    fn invalidate(&mut self) {
        for child in self.children.iter_mut() {
            child.invalidate();
        }
    }
}

/// Private `MenuSearchInput` stand-in (`fillsMenuPanel` is true).
struct MenuSearchInput {
    input: Input,
    placeholder: String,
}

impl MenuSearchInput {
    fn new(placeholder: &str) -> Self {
        Self {
            input: Input::new(),
            placeholder: placeholder.to_string(),
        }
    }

    fn get_value(&self) -> String {
        self.input.get_value().to_string()
    }

    fn get_cursor(&self) -> usize {
        self.input.get_cursor()
    }

    fn set_value(&mut self, value: String) {
        self.input.set_value(value);
    }

    fn strip_input_prompt(line: &str) -> String {
        match line.strip_prefix("> ") {
            Some(rest) => rest.to_string(),
            None => line.to_string(),
        }
    }
}

impl Focusable for MenuSearchInput {
    fn focused(&self) -> bool {
        self.input.focused()
    }

    fn set_focused(&mut self, focused: bool) {
        self.input.set_focused(focused);
    }
}

impl Component for MenuSearchInput {
    fn render(&mut self, width: f64) -> Vec<String> {
        let safe_width = (width.max(0.0).floor() as usize).max(FIELD_PADDING_X * 2 + 1);
        let inner_width = (safe_width.saturating_sub(FIELD_PADDING_X * 2)).max(1);
        let focused = Focusable::focused(&self.input);
        let content = if self.get_value().is_empty() && !focused {
            theme().fg("dim", &self.placeholder)
        } else {
            let rendered = self.input.render(inner_width as f64 + 2.0);
            let first = rendered.first().cloned().unwrap_or_default();
            MenuSearchInput::strip_input_prompt(&first)
        };
        let background = theme().get_editor_background_color();
        vec![padded_background_line(
            &content,
            safe_width,
            FIELD_PADDING_X,
            background.as_deref(),
        )]
    }

    fn handle_input(&mut self, data: &str) {
        self.input.handle_input(data);
    }

    fn invalidate(&mut self) {
        self.input.invalidate();
    }

    fn as_focusable(&mut self) -> Option<&mut dyn pi_tui::tui::Focusable> {
        Some(&mut self.input)
    }
}

struct SharedLoginInput(Rc<RefCell<MenuSearchInput>>);

impl Component for SharedLoginInput {
    fn render(&mut self, width: f64) -> Vec<String> {
        self.0.borrow_mut().render(width)
    }

    fn handle_input(&mut self, data: &str) {
        self.0.borrow_mut().handle_input(data);
    }

    fn invalidate(&mut self) {
        self.0.borrow_mut().invalidate();
    }
}

/// `PRIME_LOGO_LINES`.
fn prime_logo_lines() -> Vec<String> {
    PRIME_BUTTERFLY_LOGO
        .split('\n')
        .map(|line| line.to_string())
        .collect()
}

/// `PRIME_LOGO_WIDTH`.
fn prime_logo_width() -> usize {
    prime_logo_lines()
        .iter()
        .map(|line| pi_tui::utils::visible_width(line))
        .max()
        .unwrap_or(0)
}

/// Port of `centeredLine`.
pub fn centered_line(text: &str, width: usize) -> String {
    let safe_width = width.max(1);
    let content = truncate_to_width(text, safe_width as f64, "", false);
    let padding = safe_width.saturating_sub(pi_tui::utils::visible_width(&content));
    let left = padding / 2;
    format!(
        "{}{content}{}",
        " ".repeat(left),
        " ".repeat(padding - left)
    )
}

/// Port of `isTextEntryKeybinding`.
pub fn is_text_entry_keybinding(key: &str) -> bool {
    let parts: Vec<String> = key
        .to_lowercase()
        .split('+')
        .map(|part| part.to_string())
        .collect();
    let key_part = parts.last().cloned();
    !parts.contains(&"ctrl".to_string())
        && !parts.contains(&"alt".to_string())
        && (key_part.as_deref() == Some("space")
            || key_part
                .map(|part| part.chars().count() == 1)
                .unwrap_or(false))
}

/// Port of `isPrintableInput`.
pub fn is_printable_input(data: &str) -> bool {
    data.chars().count() == 1 && data >= " " && data != "\u{7f}"
}

/// Port of `PrimeLoginHeader`.
pub struct PrimeLoginHeader;

impl Component for PrimeLoginHeader {
    fn render(&mut self, width: f64) -> Vec<String> {
        let safe_width = (width.max(0.0).floor() as usize).max(1);
        let logo_lines = prime_logo_lines();
        let logo_width = prime_logo_width();
        let effective_logo_width = logo_width.min(safe_width);
        let mut lines: Vec<String> = logo_lines
            .iter()
            .map(|line| {
                let padded_logo_line = format!(
                    "{line}{}",
                    " ".repeat(logo_width.saturating_sub(pi_tui::utils::visible_width(line)))
                );
                centered_line(
                    &theme().fg(
                        "text",
                        &truncate_to_width(
                            &padded_logo_line,
                            effective_logo_width as f64,
                            "",
                            false,
                        ),
                    ),
                    safe_width,
                )
            })
            .collect();
        lines.push(centered_line("", safe_width));
        lines.push(centered_line(
            &theme().bold(&theme().fg("text", "Login to Prime Inference")),
            safe_width,
        ));
        lines.push(centered_line(
            &theme().fg(
                "muted",
                "Connect your Prime Intellect account to enable Prime Inference models.",
            ),
            safe_width,
        ));
        lines
    }

    fn invalidate(&mut self) {
        // Header render is derived from the current theme.
    }
}

/// Port of `LoginDialogComponent`.
pub struct LoginDialogComponent {
    tui: Rc<RefCell<TUI>>,
    content_children: Vec<Box<dyn Component>>,
    input: Rc<RefCell<MenuSearchInput>>,
    on_complete: Box<dyn FnMut(bool, Option<String>)>,
    is_prime_inference: bool,
    provider_id: String,
    provider_name: String,
    title: String,
    /// `new AbortController()`; `AbortSignal` is the abort flag.
    abort_controller: Arc<AtomicBool>,
    input_resolver: Option<tokio::sync::oneshot::Sender<String>>,
    input_rejecter: Option<tokio::sync::oneshot::Sender<String>>,
    /// True only while the editable paste field is actually shown in the panel.
    /// Tracks visibility directly rather than inferring it from `inputResolver`,
    /// which can outlive the field when a new screen clears the content.
    input_visible: bool,
    continue_resolver: Option<tokio::sync::oneshot::Sender<()>>,
    auth_url: Option<String>,
    auth_actions_text: Option<String>,
    focused: bool,
}

impl LoginDialogComponent {
    pub fn new(
        tui: Rc<RefCell<TUI>>,
        provider_id: &str,
        on_complete: Box<dyn FnMut(bool, Option<String>)>,
        provider_name_override: Option<&str>,
        title_override: Option<&str>,
    ) -> Self {
        let provider_info = get_oauth_providers()
            .into_iter()
            .find(|provider| provider.id == provider_id);
        let provider_name = provider_name_override
            .map(|value| value.to_string())
            .or_else(|| provider_info.as_ref().map(|provider| provider.name.clone()))
            .unwrap_or_else(|| provider_id.to_string());
        let is_prime_inference = provider_id == PRIME_INFERENCE_PROVIDER_ID;
        let title = title_override
            .map(|value| value.to_string())
            .unwrap_or_else(|| format!("Login to {provider_name}"));

        let dialog = Self {
            tui,
            content_children: Vec::new(),
            input: Rc::new(RefCell::new(MenuSearchInput::new("Paste value"))),
            on_complete,
            is_prime_inference,
            provider_id: provider_id.to_string(),
            provider_name,
            title,
            abort_controller: Arc::new(AtomicBool::new(false)),
            input_resolver: None,
            input_rejecter: None,
            input_visible: false,
            continue_resolver: None,
            auth_url: None,
            auth_actions_text: None,
            focused: false,
        };

        dialog
    }

    pub fn provider_id(&self) -> &str {
        &self.provider_id
    }

    pub fn provider_name(&self) -> &str {
        &self.provider_name
    }

    pub fn title(&self) -> &str {
        &self.title
    }

    /// Port of the `signal` getter.
    pub fn signal(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.abort_controller)
    }

    pub fn signal_aborted(&self) -> bool {
        self.abort_controller.load(Ordering::SeqCst)
    }

    /// Port of `cancel`.
    pub fn cancel(&mut self) {
        self.abort_controller.store(true, Ordering::SeqCst);
        // Dropping the sender rejects the pending Rust input future.
        self.input_rejecter = None;
        self.input_resolver = None;
        if self.continue_resolver.is_some() {
            self.continue_resolver = None;
        }
        (self.on_complete)(false, Some("Login cancelled".to_string()));
    }

    /// Called by `onAuth` - show URL and optional instructions.
    pub fn show_auth(&mut self, url: &str, instructions: Option<&str>) {
        self.start_content();
        self.auth_url = Some(url.to_string());
        self.add_section_title("Browser sign-in");
        self.add_muted_text(
            "The sign-in page should already be opening. If it did not open, use the link below.",
        );
        self.content_children.push(Box::new(Spacer::new(1)));
        self.add_label("Sign-in link");
        let linked_url = if get_capabilities().hyperlinks {
            hyperlink(url, url)
        } else {
            url.to_string()
        };
        self.content_children.push(Box::new(Text::new(
            theme().fg("text", &linked_url),
            0,
            0,
            None,
        )));
        self.auth_actions_text = Some(self.get_auth_actions_text(None));
        self.content_children.push(Box::new(Text::new(
            self.auth_actions_text.clone().unwrap_or_default(),
            0,
            0,
            None,
        )));

        if let Some(instructions) = instructions {
            self.content_children.push(Box::new(Spacer::new(1)));
            self.add_instructions(instructions);
        }

        // Try to open browser
        let platform = crate::core::tools::path_utils::process_platform();
        let (command, args) = if platform == "darwin" {
            ("open".to_string(), vec![url.to_string()])
        } else if platform == "win32" {
            let system_root =
                std::env::var("SystemRoot").unwrap_or_else(|_| "C:\\Windows".to_string());
            (
                Path::new(&system_root)
                    .join("System32")
                    .join("rundll32.exe")
                    .to_string_lossy()
                    .into_owned(),
                vec!["url.dll,FileProtocolHandler".to_string(), url.to_string()],
            )
        } else {
            ("xdg-open".to_string(), vec![url.to_string()])
        };
        let _ = exec_file_hidden(&command, &args, SpawnOptions::default());

        self.tui.borrow_mut().request_render();
    }

    /// Show input for manual code/URL entry (for callback server providers).
    pub fn show_manual_input(&mut self, prompt: &str) -> tokio::sync::oneshot::Receiver<String> {
        self.add_section_spacer();
        self.add_section_title("Manual fallback");
        self.add_muted_text(prompt);
        self.content_children
            .push(Box::new(SharedLoginInput(Rc::clone(&self.input))));
        self.input_visible = true;
        self.auth_actions_text = Some(self.get_auth_actions_text(None));
        self.content_children.push(Box::new(Text::new(
            theme().fg(
                "muted",
                &key_hint("tui.select.cancel", "cancel", &KeyTextOptions::default()),
            ),
            0,
            0,
            None,
        )));
        self.tui.borrow_mut().request_render();

        self.wait_for_input()
    }

    /// Wait for the next submission of the already-visible input.
    pub fn wait_for_input(&mut self) -> tokio::sync::oneshot::Receiver<String> {
        let (sender, receiver) = tokio::sync::oneshot::channel::<String>();
        self.input_resolver = Some(sender);
        receiver
    }

    /// Called by `onPrompt` - show prompt and wait for input.
    /// Note: does NOT clear content, appends to existing (preserves URL from
    /// `showAuth`).
    pub fn show_prompt(
        &mut self,
        message: &str,
        placeholder: Option<&str>,
    ) -> tokio::sync::oneshot::Receiver<String> {
        if self.provider_id == "github-copilot" && self.auth_url.is_none() {
            // This prompt precedes device authorization; it is not a network wait.
            self.start_content();
            self.add_muted_text(&key_hint(
                "tui.select.confirm",
                "use github.com, or enter your GitHub Enterprise domain below",
                &KeyTextOptions::default(),
            ));
        }
        self.add_section_spacer();
        self.add_section_title(message);
        if let Some(placeholder) = placeholder {
            self.content_children.push(Box::new(Text::new(
                theme().fg("muted", &format!("e.g., {placeholder}")),
                0,
                0,
                None,
            )));
        }
        self.content_children
            .push(Box::new(SharedLoginInput(Rc::clone(&self.input))));
        self.input_visible = true;
        self.auth_actions_text = Some(self.get_auth_actions_text(None));
        self.content_children.push(Box::new(Text::new(
            theme().fg(
                "muted",
                &format!(
                    "{}  {}",
                    key_hint("tui.select.confirm", "submit", &KeyTextOptions::default()),
                    key_hint("tui.select.cancel", "cancel", &KeyTextOptions::default())
                ),
            ),
            0,
            0,
            None,
        )));

        self.input.borrow_mut().set_value(String::new());
        self.tui.borrow_mut().request_render();

        self.wait_for_input()
    }

    /// Show informational text without prompting for input.
    pub fn show_info(&mut self, lines: Vec<String>) {
        self.start_content();
        for line in lines {
            self.content_children
                .push(Box::new(Text::new(line, 0, 0, None)));
        }
        self.content_children.push(Box::new(Spacer::new(1)));
        self.content_children.push(Box::new(Text::new(
            theme().fg(
                "muted",
                &key_hint("tui.select.cancel", "close", &KeyTextOptions::default()),
            ),
            0,
            0,
            None,
        )));
        self.tui.borrow_mut().request_render();
    }

    pub fn show_continue_info(&mut self, lines: Vec<String>) -> tokio::sync::oneshot::Receiver<()> {
        self.start_content();
        for line in lines {
            self.content_children
                .push(Box::new(Text::new(line, 0, 0, None)));
        }
        self.content_children.push(Box::new(Spacer::new(1)));
        self.content_children.push(Box::new(Text::new(
            theme().fg(
                "muted",
                &format!(
                    "{}  {}",
                    key_hint("tui.select.confirm", "continue", &KeyTextOptions::default()),
                    key_hint("tui.select.cancel", "cancel", &KeyTextOptions::default())
                ),
            ),
            0,
            0,
            None,
        )));
        self.tui.borrow_mut().request_render();

        let (sender, receiver) = tokio::sync::oneshot::channel::<()>();
        self.continue_resolver = Some(sender);
        receiver
    }

    /// Show waiting message (for polling flows like GitHub Copilot).
    pub fn show_waiting(&mut self, message: &str) {
        self.add_section_spacer();
        self.content_children.push(Box::new(Text::new(
            theme().fg("accent", message),
            0,
            0,
            None,
        )));
        self.content_children.push(Box::new(Text::new(
            theme().fg(
                "muted",
                &key_hint("tui.select.cancel", "cancel", &KeyTextOptions::default()),
            ),
            0,
            0,
            None,
        )));
        self.tui.borrow_mut().request_render();
    }

    /// Called by `onProgress`.
    pub fn show_progress(&mut self, message: &str) {
        if self.content_children.is_empty() {
            self.start_content();
            self.add_section_title("Preparing authentication");
        }
        self.content_children.push(Box::new(Text::new(
            theme().fg("muted", message),
            0,
            0,
            None,
        )));
        self.tui.borrow_mut().request_render();
    }

    /// Port of `startContent`.
    fn start_content(&mut self) {
        self.content_children.clear();
        self.auth_url = None;
        self.auth_actions_text = None;
        // The cleared panel no longer shows the paste field.
        self.input_visible = false;
        if self.is_prime_inference {
            self.content_children.push(Box::new(PrimeLoginHeader));
            self.content_children.push(Box::new(Spacer::new(1)));
            return;
        }
        self.content_children.push(Box::new(Spacer::new(1)));
    }

    /// Port of `addSectionSpacer`.
    fn add_section_spacer(&mut self) {
        if self.content_children.is_empty() {
            self.start_content();
            return;
        }
        self.content_children.push(Box::new(Spacer::new(1)));
    }

    /// Port of `addInstructions`.
    fn add_instructions(&mut self, instructions: &str) {
        let code = parse_verification_code(instructions);
        if let Some(code) = code {
            self.add_label("Verification code");
            self.content_children.push(Box::new(Text::new(
                theme().bold(&theme().fg("text", &code)),
                0,
                0,
                None,
            )));
            return;
        }
        self.add_label("Next step");
        self.content_children.push(Box::new(Text::new(
            theme().fg("text", instructions),
            0,
            0,
            None,
        )));
    }

    /// Port of `addSectionTitle`.
    fn add_section_title(&mut self, text: &str) {
        self.content_children.push(Box::new(Text::new(
            theme().bold(&theme().fg("text", text)),
            0,
            0,
            None,
        )));
    }

    /// Port of `addLabel`.
    fn add_label(&mut self, text: &str) {
        self.content_children
            .push(Box::new(Text::new(theme().fg("muted", text), 0, 0, None)));
    }

    /// Port of `addMutedText`.
    fn add_muted_text(&mut self, text: &str) {
        self.content_children
            .push(Box::new(Text::new(theme().fg("muted", text), 0, 0, None)));
    }

    /// Port of `getAuthActionsText`.
    fn get_auth_actions_text(&self, status: Option<&str>) -> String {
        let configured_copy_keys = get_keybindings().get_keys("app.clipboard.copyLoginUrl");
        let copy_keys: Vec<String> = if self.input_visible {
            configured_copy_keys
                .into_iter()
                .filter(|key| !is_text_entry_keybinding(key))
                .collect()
        } else {
            configured_copy_keys.into_iter().take(1).collect()
        };
        let copy_hint = if !copy_keys.is_empty() {
            Some(
                theme().fg("dim", &format_key_text(&copy_keys.join("/"), None))
                    + &theme().fg(
                        "muted",
                        &format!(
                            " {}",
                            if status == Some("failed") {
                                "retry"
                            } else {
                                "copy"
                            }
                        ),
                    ),
            )
        } else {
            None
        };
        let status_text = match status {
            Some("copied") => Some(theme().fg("success", "Copied sign-in link")),
            Some("forwarded") => Some(theme().fg("muted", "Sign-in link sent to terminal (clipboard unconfirmed)")),
            Some("failed") => Some(theme().fg("error", "Failed to copy sign-in link")),
            _ => None,
        };
        [
            status_text,
            copy_hint,
            Some(key_hint(
                "tui.select.cancel",
                "cancel",
                &KeyTextOptions::default(),
            )),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<String>>()
        .join("  ")
    }

    /// Port of `copyAuthUrl`.
    pub async fn copy_auth_url(&mut self) {
        let url = match &self.auth_url {
            Some(url) => url.clone(),
            None => return,
        };
        if self.auth_actions_text.is_none() {
            return;
        }

        let status = match copy_to_clipboard(&url).await {
            Ok(crate::utils::clipboard::ClipboardOutcome::LocalBackendAccepted) => Some("copied"),
            Ok(crate::utils::clipboard::ClipboardOutcome::TerminalForwarded) => Some("forwarded"),
            Err(_) => Some("failed"),
        };
        if self.auth_url.as_deref() == Some(url.as_str()) {
            self.auth_actions_text = Some(self.get_auth_actions_text(status));
            self.tui.borrow_mut().request_render();
        }
    }

    /// Port of `handleInput`.
    pub fn handle_input(&mut self, data: &str) {
        let kb = get_keybindings();

        if self.auth_url.is_some()
            && kb.matches(data, "app.clipboard.copyLoginUrl")
            && (!self.input_visible || !is_printable_input(data))
        {
            // `void this.copyAuthUrl()` - the caller drives the async copy.
            return;
        }

        // Left arrow acts as "back" like Esc. While the editable field is actually
        // shown, only treat it as back at the start of the text so left still moves
        // the cursor mid-edit; on info/continue screens there is no field to guard.
        let back_guard_cursor = if self.input_visible {
            Some(self.input.borrow().get_cursor())
        } else {
            None
        };
        if kb.matches(data, "tui.select.cancel") || should_treat_as_back(data, back_guard_cursor) {
            self.cancel();
            return;
        }

        if self.continue_resolver.is_some() && kb.matches(data, "tui.select.confirm") {
            let resolve = self.continue_resolver.take();
            if let Some(resolve) = resolve {
                let _ = resolve.send(());
            }
            return;
        }

        if kb.matches(data, "tui.input.submit") || data == "\n" {
            if let Some(resolve) = self.input_resolver.take() {
                let _ = resolve.send(self.input.borrow().get_value());
                self.input_rejecter = None;
                if self.provider_id == "github-copilot" && self.auth_url.is_none() {
                    self.start_content();
                    self.show_waiting("Requesting GitHub sign-in link...");
                }
                return;
            }
        }

        // Pass to input
        self.input.borrow_mut().handle_input(data);
    }
}

/// `/^(?:Code|Enter code):\s*(.+)$/i` - the first capture group, if any.
fn parse_verification_code(instructions: &str) -> Option<String> {
    let trimmed = instructions.trim();
    let lower = trimmed.to_lowercase();
    for prefix in ["code:", "enter code:"] {
        if lower.starts_with(prefix) {
            let rest = &trimmed[prefix.len()..];
            let value = rest.trim_start();
            if value.is_empty() {
                return None;
            }
            return Some(value.to_string());
        }
    }
    None
}

/// Port of `shouldTreatAsBack` (components/modal-back.ts, another slice: this
/// module keeps a private copy; see the slice status file).
pub fn should_treat_as_back(data: &str, cursor: Option<usize>) -> bool {
    if !get_keybindings().matches(data, "app.modal.back") {
        return false;
    }
    match cursor {
        Some(cursor) => cursor == 0,
        None => true,
    }
}

impl Component for LoginDialogComponent {
    fn render(&mut self, width: f64) -> Vec<String> {
        // `this.addChild(panel)`: the content container is the panel's child.
        let content_children = std::mem::take(&mut self.content_children);
        let mut panel = MenuPanel::new(
            if self.is_prime_inference {
                ""
            } else {
                self.title.as_str()
            },
            if self.is_prime_inference {
                None
            } else {
                Some("Complete this step to continue setup.")
            },
        );
        for child in content_children {
            panel.add_child(child);
        }
        let mut lines = panel.render(width);
        self.content_children = panel.children;
        lines.shrink_to_fit();
        lines
    }

    fn handle_input(&mut self, data: &str) {
        LoginDialogComponent::handle_input(self, data);
    }

    fn invalidate(&mut self) {
        for child in self.content_children.iter_mut() {
            child.invalidate();
        }
    }

    fn as_focusable(&mut self) -> Option<&mut dyn pi_tui::tui::Focusable> {
        Some(self)
    }
}

impl pi_tui::tui::Focusable for LoginDialogComponent {
    fn focused(&self) -> bool {
        self.focused
    }

    fn set_focused(&mut self, focused: bool) {
        self.focused = focused;
        self.input.borrow_mut().set_focused(focused);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pi_tui::terminal::ProcessTerminal;

    fn tui() -> Rc<RefCell<TUI>> {
        crate::modes::interactive::theme::theme::init_theme(Some("prime"), false);
        crate::core::keybindings::KeybindingsManager::new(Default::default(), None).install();
        Rc::new(RefCell::new(TUI::new(
            Box::new(ProcessTerminal::new()),
            Some(false),
        )))
    }

    fn dialog(provider_id: &str) -> LoginDialogComponent {
        LoginDialogComponent::new(tui(), provider_id, Box::new(|_, _| {}), None, None)
    }

    #[test]
    fn centered_line_pads_to_the_width() {
        assert_eq!(centered_line("ab", 6), "  ab  ");
        assert_eq!(centered_line("abcdef", 6), "abcdef");
        assert_eq!(centered_line("abcdefg", 6).chars().count(), 6);
        assert_eq!(centered_line("", 3), "   ");
        assert_eq!(centered_line("x", 0), "x");
    }

    #[test]
    fn text_entry_keybindings_are_single_unmodified_keys() {
        assert!(is_text_entry_keybinding("space"));
        assert!(is_text_entry_keybinding("a"));
        assert!(is_text_entry_keybinding("shift+a"));
        assert!(!is_text_entry_keybinding("ctrl+a"));
        assert!(!is_text_entry_keybinding("alt+a"));
        assert!(!is_text_entry_keybinding("escape"));
    }

    #[test]
    fn printable_input_is_a_single_visible_character() {
        assert!(is_printable_input("a"));
        assert!(is_printable_input(" "));
        assert!(!is_printable_input(""));
        assert!(!is_printable_input("ab"));
        assert!(!is_printable_input("\u{7f}"));
        assert!(!is_printable_input("\u{1b}"));
    }

    #[test]
    fn the_prime_inference_header_hides_the_title_and_subtitle() {
        let mut component = dialog(PRIME_INFERENCE_PROVIDER_ID);
        assert!(component.is_prime_inference);
        let lines = component.render(60.0);
        assert!(lines
            .iter()
            .all(|line| !line.contains("Complete this step")));

        let mut other = dialog("anthropic");
        assert!(!other.is_prime_inference);
        assert_eq!(other.title(), "Login to Anthropic (Claude Pro/Max)");
        assert!(other
            .render(60.0)
            .iter()
            .any(|line| line.contains("Complete this step to continue setup.")));
    }

    #[test]
    fn verification_codes_are_extracted_from_instructions() {
        assert_eq!(
            parse_verification_code("Code: 1234"),
            Some("1234".to_string())
        );
        assert_eq!(
            parse_verification_code("enter code: xy-z"),
            Some("xy-z".to_string())
        );
        assert_eq!(
            parse_verification_code("  CODE:  42  "),
            Some("42".to_string())
        );
        assert_eq!(parse_verification_code("code:"), None);
        assert_eq!(parse_verification_code("paste the url"), None);
    }

    #[test]
    fn auth_actions_filter_text_entry_keys_while_the_input_is_visible() {
        let mut component = dialog("anthropic");
        component.start_content();
        assert!(!component.input_visible);
        let hidden = component.get_auth_actions_text(None);
        component.input_visible = true;
        let visible = component.get_auth_actions_text(None);
        assert!(hidden.contains("cancel"));
        assert!(visible.contains("cancel"));
        assert_eq!(
            component
                .get_auth_actions_text(Some("copied"))
                .contains("Copied sign-in link"),
            true
        );
        assert!(component
            .get_auth_actions_text(Some("failed"))
            .contains("Failed to copy sign-in link"));
        let forwarded = component.get_auth_actions_text(Some("forwarded"));
        assert!(forwarded.contains("clipboard unconfirmed"));
        assert!(!forwarded.contains("Copied sign-in link"));
    }

    #[test]
    fn show_auth_renders_the_link_and_instructions() {
        let mut component = dialog("anthropic");
        component.show_auth("https://example.test/auth", Some("Code: 9999"));
        let rendered = component.render(70.0);
        assert!(rendered.iter().any(|line| line.contains("Browser sign-in")));
        assert!(rendered.iter().any(|line| line.contains("9999")));
        assert!(rendered
            .iter()
            .any(|line| line.contains("Verification code")));
    }

    #[test]
    fn show_info_clears_previous_content_and_marks_the_input_hidden() {
        let mut component = dialog("anthropic");
        component.show_manual_input("paste");
        assert!(component.input_visible);
        component.show_info(vec!["line one".to_string()]);
        assert!(!component.input_visible);
        let rendered = component.render(60.0);
        assert!(rendered.iter().any(|line| line.contains("line one")));
        assert!(rendered.iter().any(|line| line.contains("close")));
    }

    #[test]
    fn show_prompt_resets_the_input_value() {
        let mut component = dialog("anthropic");
        component
            .input
            .borrow_mut()
            .set_value("leftover".to_string());
        let mut receiver = component.show_prompt("Enter the code", Some("abc"));
        assert_eq!(component.input.borrow().get_value(), "");
        let rendered = component.render(60.0);
        assert!(rendered.iter().any(|line| line.contains("Enter the code")));
        assert!(rendered.iter().any(|line| line.contains("e.g., abc")));
        assert!(rendered.iter().any(|line| line.contains("submit")));
        component.handle_input("test-code");
        assert!(component
            .render(60.0)
            .iter()
            .any(|line| line.contains("test-code")));
        component.handle_input("\r");
        assert_eq!(receiver.try_recv().unwrap(), "test-code");
    }

    #[test]
    fn mounted_copilot_domain_prompt_accepts_blank_and_enterprise_input() {
        for value in ["", "company.ghe.com"] {
            let ui = tui();
            let component = Rc::new(RefCell::new(LoginDialogComponent::new(
                ui.clone(), "github-copilot", Box::new(|_, _| {}), None, None,
            )));
            component.borrow_mut().show_progress("Starting sign-in...");
            let handle = ui.borrow_mut().show_overlay(component.clone(), Default::default());
            let mut receiver = component.borrow_mut().show_prompt(
                "GitHub Enterprise URL/domain (blank for github.com)", Some("company.ghe.com"),
            );
            assert!(handle.is_focused());
            assert!(component.borrow().input.borrow().focused());
            let rendered = component.borrow_mut().render(90.0).join("\n");
            assert!(rendered.contains("use github.com"), "{rendered}");
            assert!(!rendered.contains("Starting sign-in"), "{rendered}");
            assert!(!rendered.contains("Preparing authentication"), "{rendered}");
            let focused = ui.borrow().focused_component().unwrap();
            focused.borrow_mut().handle_input(&format!("\x1b[200~{value}\x1b[201~"));
            focused.borrow_mut().handle_input("\r");
            assert_eq!(receiver.try_recv().unwrap(), value);
            let waiting = component.borrow_mut().render(90.0).join("\n");
            assert!(waiting.contains("Requesting GitHub sign-in link"), "{waiting}");
            assert!(!waiting.contains("GitHub Enterprise URL/domain"), "{waiting}");
            handle.hide();
            ui.borrow_mut().sync_overlays();
            assert!(!ui.borrow().has_overlay());
        }
    }

    #[test]
    fn mounted_copilot_prompt_cancellation_does_not_wait_for_authentication() {
        for key in ["\x1b", "\x03"] {
            let ui = tui();
            let cancelled = Rc::new(std::cell::Cell::new(false));
            let report = cancelled.clone();
            let component = Rc::new(RefCell::new(LoginDialogComponent::new(
                ui.clone(), "github-copilot",
                Box::new(move |success, error| {
                    assert!(!success);
                    assert_eq!(error.as_deref(), Some("Login cancelled"));
                    report.set(true);
                }), None, None,
            )));
            let handle = ui.borrow_mut().show_overlay(component.clone(), Default::default());
            let mut receiver = component.borrow_mut().show_prompt(
                "GitHub Enterprise URL/domain (blank for github.com)", None,
            );
            let focused = ui.borrow().focused_component().unwrap();
            focused.borrow_mut().handle_input(key);
            assert!(cancelled.get(), "cancellation must settle synchronously");
            assert!(component.borrow().signal_aborted());
            assert_eq!(receiver.try_recv(), Err(tokio::sync::oneshot::error::TryRecvError::Closed));
            handle.hide();
            ui.borrow_mut().sync_overlays();
            assert!(!ui.borrow().has_overlay());
        }
    }

    #[test]
    fn show_waiting_and_show_progress_append_to_the_panel() {
        let mut component = dialog("anthropic");
        component.show_waiting("waiting for browser");
        let rendered = component.render(60.0);
        assert!(rendered
            .iter()
            .any(|line| line.contains("waiting for browser")));
        component.show_progress("polling");
        let rendered = component.render(60.0);
        assert!(rendered.iter().any(|line| line.contains("polling")));
        assert!(rendered
            .iter()
            .any(|line| line.contains("waiting for browser")));
        assert!(!rendered
            .iter()
            .any(|line| line.contains("Preparing authentication")));
        let mut fresh = dialog("anthropic");
        fresh.show_progress("polling");
        assert!(fresh
            .render(60.0)
            .iter()
            .any(|line| line.contains("Preparing authentication")));
    }

    #[test]
    fn escape_and_left_arrow_cancel_and_report_it() {
        let reported: Rc<RefCell<Vec<(bool, Option<String>)>>> = Rc::new(RefCell::new(Vec::new()));
        let sink = Rc::clone(&reported);
        let mut component = LoginDialogComponent::new(
            tui(),
            "anthropic",
            Box::new(move |success, message| sink.borrow_mut().push((success, message))),
            None,
            None,
        );
        component.handle_input("\u{1b}");
        assert!(component.signal_aborted());
        assert_eq!(
            reported.borrow()[0],
            (false, Some("Login cancelled".to_string()))
        );
    }

    #[test]
    fn a_guarded_left_arrow_only_cancels_at_column_zero() {
        let mut component = dialog("anthropic");
        component.input_visible = true;
        component.input.borrow_mut().handle_input("typed");
        assert_eq!(component.input.borrow().get_cursor(), 5);
        // Each left arrow moves the cursor until it reaches the start.
        for cursor in (0..5).rev() {
            component.handle_input("\u{1b}[D");
            assert!(!component.signal_aborted());
            assert_eq!(component.input.borrow().get_cursor(), cursor);
        }
        // Now the cursor is at column 0.
        component.handle_input("\u{1b}[D");
        assert!(component.signal_aborted());
    }

    #[test]
    fn should_treat_as_back_uses_the_modal_binding_and_the_cursor() {
        crate::core::keybindings::KeybindingsManager::new(Default::default(), None).install();
        assert!(get_keybindings().matches("\u{1b}[D", "app.modal.back"));
        assert!(should_treat_as_back("\u{1b}[D", None));
        assert!(should_treat_as_back("\u{1b}[D", Some(0)));
        assert!(!should_treat_as_back("\u{1b}[D", Some(3)));
        assert!(!should_treat_as_back("z", None));
    }

    #[test]
    fn cancel_closes_a_pending_input_request() {
        let mut component = dialog("anthropic");
        let mut receiver = component.show_manual_input("paste the code");
        component.handle_input("\x1b");
        assert!(component.signal_aborted());
        assert_eq!(
            receiver.try_recv(),
            Err(tokio::sync::oneshot::error::TryRecvError::Closed)
        );
    }

    #[test]
    fn continue_info_resolves_on_confirm() {
        let mut component = dialog("anthropic");
        let mut receiver = component.show_continue_info(vec!["ready".to_string()]);
        let rendered = component.render(60.0);
        assert!(rendered.iter().any(|line| line.contains("ready")));
        assert!(rendered.iter().any(|line| line.contains("continue")));
        component.handle_input("\r");
        assert!(receiver.try_recv().is_ok());
    }
}
