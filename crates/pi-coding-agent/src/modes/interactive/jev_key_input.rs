//! Masked API-key entry component for `/jev key`.
//!
//! The logic lives in `jev_menu.rs` ([`JevKeyInputState`]) so it is unit-testable
//! and include-able by `tests/jev_ui_tests.rs`. This file is the thin TUI
//! wrapper: it renders the masked buffer and routes input, nothing else.
//!
//! Security properties (JEV_BRIEF "Mask API input", DESIGN.md section 6):
//!
//! * The rendered text is a fixed-length mask. The value is never formatted, so
//!   it cannot reach the transcript, the render cache, history or a debug dump.
//! * `JevKeyInputState` has a manual redacting `Debug`.
//! * Paste is supported; newlines/tabs are stripped so a paste cannot break the
//!   single-line contract.
//! * Validation is asynchronous and never blocks the UI: the component only
//!   exposes the state, and the owner answers later with `apply_validation`.
//! * Escape and Ctrl+C cancel immediately, through the configurable
//!   `tui.select.cancel` / `app.interrupt` ids (no hardcoded keys).

use pi_tui::tui::Component;

use super::jev_menu::{binding_hint, JevKeyInputState, KeyInputState, KEY_INPUT_PROMPT};
use crate::modes::interactive::theme::theme::theme;

/// The masked key-entry dialog.
///
/// The component owns only [`JevKeyInputState`]; the owner (the command task in
/// `native_host_commands::run`) decides when to validate, cancel or store.
#[derive(Debug, Default)]
pub struct JevKeyInputComponent {
    input: JevKeyInputState,
    focused: bool,
    /// Set by the owner once the dialog must close.
    closed: bool,
}

impl JevKeyInputComponent {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn input_mut(&mut self) -> &mut JevKeyInputState {
        &mut self.input
    }

    pub fn state(&self) -> KeyInputState {
        self.input.state()
    }

    pub fn closed(&self) -> bool {
        self.closed
    }

    /// The cancel key as displayed, resolved from the live binding.
    pub fn cancel_hint(&self) -> String {
        binding_hint("tui.select.cancel", "cancel")
    }

    /// Handle one input chunk and report the resulting state.
    pub fn handle_key(&mut self, data: &str) -> KeyInputState {
        let state = self.input.handle_key(data);
        if matches!(state, KeyInputState::Cancelled) {
            // Cancel closes the dialog immediately; the owner hides the overlay
            // on its next frame instead of waiting for a reply.
            self.closed = true;
        }
        state
    }

    /// The lines the render pass produces, before the width clamp.
    pub fn lines(&self) -> Vec<String> {
        let mut lines = vec![theme().fg("accent", KEY_INPUT_PROMPT)];
        lines.push(theme().fg("muted", &self.input.masked_line()));
        lines.push(theme().fg("dim", &self.input.status_line()));
        if matches!(self.state(), KeyInputState::Editing | KeyInputState::Validating) {
            lines.push(theme().fg("dim", &self.cancel_hint()));
        }
        lines
    }
}

impl Component for JevKeyInputComponent {
    fn render(&mut self, width: f64) -> Vec<String> {
        self.lines()
            .into_iter()
            .map(|line| pi_tui::utils::truncate_to_width(&line, width.max(1.0), "\u{2026}", false))
            .collect()
    }

    fn handle_input(&mut self, data: &str) {
        self.handle_key(data);
    }

    fn invalidate(&mut self) {}

    fn as_focusable(&mut self) -> Option<&mut dyn pi_tui::tui::Focusable> {
        Some(self)
    }
}

impl pi_tui::tui::Focusable for JevKeyInputComponent {
    fn focused(&self) -> bool {
        self.focused
    }

    fn set_focused(&mut self, focused: bool) {
        self.focused = focused;
    }
}
