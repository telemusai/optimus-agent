//! Jev footer segment: the status indicator next to the model/effort tray.
//!
//! Rendering path (chosen to avoid touching `native_host.rs`, a REPAIR-OVERLAP
//! file): the footer text is published through the EXISTING extension status
//! surface
//!
//! ```text
//! AgentConnectionEvent::ExtensionUiRequest {
//!     request: { method: "setStatus", payload: { statusKey: "jev", statusText } }
//! }
//! ```
//!
//! `native_host.rs:2327-2331` already routes `setStatus` into
//! `extension_surfaces.set_status(key, text)` and
//! `native_host_extensions::Statuses` renders the collected statuses directly
//! under the model/effort tray, so the footer segment needs no new host surface
//! and no edit to the shared host.
//!
//! Truth-table rules (JEV_BRIEF "Footer next to model/effort"):
//!
//! | state | text | colour |
//! |---|---|---|
//! | `Off` | `* Jev Off` | red (`error`) |
//! | `Compare` | `* Jev Compare` | cyan (`accent`) |
//! | checking | `* Jev checking` | amber (`warning`) |
//! | unavailable | `* Jev unavailable` | amber (`warning`) |
//! | fallback | `* Jev fallback` | amber (`warning`) |
//! | green `* Jev On` | RESERVED for a future genuinely Active healthy mode | never produced |
//!
//! Green is not merely unused: [`JevFooterState::is_green`] is a constant
//! `false`, [`JevFooterState::color_key`] has no `success` arm, and
//! [`JEV_GREEN_RESERVED_NOTICE`] documents the reservation, so a future edit
//! that wants a green footer must deliberately change the pure module.
//!
//! The segment renders from a snapshot the caller owns. Nothing here reads the
//! mode store, the credential store, or the network, so a render pass can never
//! trigger I/O or an RPC (JEV_BRIEF test item "no per-render RPC/network work").

use pi_jev::types::JevMode;

use crate::modes::agent_connection::types as wire;
use super::jev_menu::{
    footer_clear_payload, footer_state, footer_text, CredentialStatus, JevFooterState,
    JevPipelineStatus,
};

/// A cached footer snapshot: rebuilt when something actually changed (mode set,
/// credential change, pipeline event), then rendered for free.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JevFooterSnapshot {
    pub mode: JevMode,
    pub credential: CredentialStatus,
    pub pipeline: JevPipelineStatus,
}

impl Default for JevFooterSnapshot {
    fn default() -> Self {
        Self {
            mode: JevMode::Off,
            credential: CredentialStatus::resolve(false, false, false),
            pipeline: JevPipelineStatus::default(),
        }
    }
}

impl JevFooterSnapshot {
    /// The state the footer shows. Derived on every call, never stored, so a
    /// snapshot cannot carry a stale state.
    pub fn state(&self) -> JevFooterState {
        footer_state(self.mode, &self.credential, &self.pipeline)
    }

    /// Plain text (no ANSI) so width measurement is exact.
    pub fn text(&self) -> String {
        footer_text(self.state())
    }

    /// The themed segment the payload carries.
    ///
    /// The text goes through the same `theme().fg` call the rest of the interactive
    /// host uses for status surfaces, and the colour comes from
    /// [`footer_color_key`](super::jev_menu::footer_color_key), so the reserved
    /// green can only be reached by changing the pure module.
    fn themed_text(&self) -> String {
        let state = self.state();
        let text = self.text();
        crate::modes::interactive::theme::theme::theme()
            .fg(super::jev_menu::footer_color_key(state), &text)
    }

    /// The host event that publishes this snapshot. The dispatch task holds no UI
    /// handle, so it cannot measure the terminal: the host's status row truncates
    /// to the live width when it renders and the labelled form is therefore safe.
    /// The measured (dot-only) form lives in `jev_menu::footer_segment` and
    /// `jev_menu::footer_status_payload`, which `tests/jev_ui_tests.rs` pins.
    pub fn published_event(&self) -> wire::AgentConnectionEvent {
        wire::AgentConnectionEvent::ExtensionUiRequest {
            request: wire::AgentConnectionExtensionUiRequest {
                id: JEV_FOOTER_REQUEST_ID.to_string(),
                method: "setStatus".to_string(),
                payload: serde_json::json!({
                    "statusKey": super::jev_menu::JEV_STATUS_KEY,
                    "statusText": self.themed_text(),
                }),
            },
        }
    }
}

/// Stable request id for the footer publish (visible in daemon logs).
pub const JEV_FOOTER_REQUEST_ID: &str = "jev-footer";
