//! Jev menu component: the interactive `/jev` overlay component.
//!
//! Split from `jev_menu.rs` so the PURE logic there stays self-contained and an
//! integration test can include it with `#[path]`. This file holds only rendering
//! and input routing; every decision is delegated to `JevMenuState` /
//! `JevMenuAction` in the pure module.

use pi_jev::types::JevMode;
use pi_tui::tui::Component;

use super::jev_menu::{JevMenuAction, JevMenuState, JEV_BOUNDARY_NOTICE, JEV_DISCLOSURE_NOTICE};
use crate::modes::interactive::theme::theme::theme;

/// The `/jev` menu overlay.
///
/// `Debug` is manual because the action callback is not `Debug`; it prints the
/// state instead, so a debug dump stays useful and never carries user input.
pub struct JevMenuComponent {
    state: JevMenuState,
    on_action: Box<dyn FnMut(JevMenuAction)>,
}

impl std::fmt::Debug for JevMenuComponent {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("JevMenuComponent")
            .field("state", &self.state)
            .finish()
    }
}

impl JevMenuComponent {
    pub fn new(active_mode: JevMode, on_action: Box<dyn FnMut(JevMenuAction)>) -> Self {
        Self {
            state: JevMenuState::new(active_mode),
            on_action,
        }
    }

    pub fn state(&self) -> &JevMenuState {
        &self.state
    }

    /// Route one key through the state machine and report the resulting action.
    /// Exposed so the owner can drive the dialog without the TUI trait.
    pub fn dispatch(&mut self, data: &str) -> JevMenuAction {
        let action = self.state.handle_key(data);
        (self.on_action)(action.clone());
        action
    }
}

impl Component for JevMenuComponent {
    fn render(&mut self, width: f64) -> Vec<String> {
        let mut lines = vec![theme().bold(&theme().fg("text", "Jev"))];
        lines.push(theme().fg("dim", JEV_BOUNDARY_NOTICE));
        lines.extend(pi_tui::utils::wrap_text_with_ansi(JEV_DISCLOSURE_NOTICE, width.max(1.0) as usize));
        lines.extend(self.state.body_lines());
        lines
            .into_iter()
            .map(|line| pi_tui::utils::truncate_to_width(&line, width.max(1.0), "\u{2026}", false))
            .collect()
    }

    fn handle_input(&mut self, data: &str) {
        self.dispatch(data);
    }

    fn invalidate(&mut self) {}

    fn as_focusable(&mut self) -> Option<&mut dyn pi_tui::tui::Focusable> {
        Some(self)
    }
}

impl pi_tui::tui::Focusable for JevMenuComponent {
    /// The overlay is focused while it is open: the owner loop routes keys to the
    /// `command_dialog` component whenever no cancellation token is active.
    fn focused(&self) -> bool {
        true
    }

    fn set_focused(&mut self, _focused: bool) {}
}
