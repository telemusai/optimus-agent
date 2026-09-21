//! Jev footer segments: the status indicators on the model/effort tray row.
//!
//! Rendering path (chosen to avoid touching `native_host.rs`, a REPAIR-OVERLAP
//! file): the footer text is published through the EXISTING extension status
//! surface
//!
//! ```text
//! AgentConnectionEvent::ExtensionUiRequest {
//!     request: { method: "setStatus", payload: { statusKey, statusText, statusCompactText } }
//! }
//! ```
//!
//! `native_host.rs` routes `setStatus` into `extension_surfaces.set_status`, and
//! `native_host_extensions::Statuses` renders the `jev` and `jev-compact`
//! segments ON the tray row, immediately after the location label (the model and
//! effort tray), while every other extension status keeps its own line below it.
//!
//! Two independent segments are published under two status keys:
//!
//! * `jev` — the decision mode. Truth table (JEV_BRIEF "Footer next to
//!   model/effort", green/red revision):
//!
//!   | state | text | colour |
//!   |---|---|---|
//!   | `Off` | `● Jev Off` | red (`error`) |
//!   | `Compare` | `● Jev On (Compare)` | green (`success`) |
//!   | `Active` | `● Jev On (Active)` | green (`success`) |
//!   | `CompareAndActive` | `● Jev On (Compare + Active)` | green (`success`) |
//!   | checking | `● Jev checking` | amber (`warning`) |
//!   | unavailable | `● Jev unavailable` | amber (`warning`) |
//!   | fallback | `● Jev fallback` | amber (`warning`) |
//!
//!   The label names the truthful effective mode, so a green segment can only
//!   mean "on and healthy in the mode it names".
//! * `jev-compact` — the independent compaction state: `● Jev compact on`
//!   (green), `● Jev compact off` (red), `● Jev compact unknown` (amber).
//!   Turning the decision mode off never removes or flips this dot.
//!
//! Every segment additionally carries `statusCompactText`, the SHORT LABELLED
//! narrow form the tray row uses when the full row does not fit: `● Jev C On` /
//! `● Jev A On` / `● Jev C+A On` for the decision segment and `● Jev Cmp on` /
//! `● Jev Cmp off` for compaction. Never a bare dot: two bare dots cannot be
//! told apart, and the state must stay readable in text. Receivers that do not
//! know the optional field ignore it; senders that omit it degrade to
//! left-truncation instead of segment compaction.
//!
//! The segments render from a snapshot the caller owns. Nothing here reads the
//! mode store, the credential store, or the network, so a render pass can never
//! trigger I/O or an RPC (JEV_BRIEF test item "no per-render RPC/network work").

use pi_jev::types::JevMode;

use super::jev_menu::{
    footer_color_key, footer_compact_text, footer_compaction_compact_text, footer_compaction_text,
    footer_state, footer_text, CredentialStatus, JevCompactionState, JevFooterState,
    JevPipelineStatus, JEV_COMPACT_STATUS_KEY, JEV_STATUS_KEY,
};
use crate::modes::agent_connection::types as wire;

/// A cached footer snapshot: rebuilt when something actually changed (mode set,
/// credential change, compaction change, pipeline event), then rendered for free.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JevFooterSnapshot {
    pub mode: JevMode,
    pub credential: CredentialStatus,
    pub pipeline: JevPipelineStatus,
    /// The independent compaction state for the session. `Unknown` only when no
    /// settings source could resolve it; never inferred from the decision mode.
    pub compaction: JevCompactionState,
}

impl Default for JevFooterSnapshot {
    fn default() -> Self {
        Self {
            mode: JevMode::Off,
            credential: CredentialStatus::resolve(false, false, false),
            pipeline: JevPipelineStatus::default(),
            compaction: JevCompactionState::Unknown,
        }
    }
}

impl JevFooterSnapshot {
    /// The state the decision segment shows. Derived on every call, never
    /// stored, so a snapshot cannot carry a stale state.
    pub fn state(&self) -> JevFooterState {
        footer_state(self.mode, &self.credential, &self.pipeline)
    }

    /// The state the compaction segment shows.
    pub fn compaction_state(&self) -> JevCompactionState {
        self.compaction
    }

    /// Plain decision text (no ANSI) so width measurement is exact.
    pub fn text(&self) -> String {
        footer_text(self.state())
    }

    /// Plain compaction text (no ANSI).
    pub fn compaction_text(&self) -> String {
        footer_compaction_text(self.compaction)
    }

    /// The themed decision segment the payload carries.
    ///
    /// The text goes through the same `theme().fg` call the rest of the
    /// interactive host uses for status surfaces, and the colour comes from
    /// [`footer_color_key`](super::jev_menu::footer_color_key), so a green
    /// segment can only be reached by changing the pure module.
    fn themed_text(&self) -> String {
        let state = self.state();
        crate::modes::interactive::theme::theme::theme().fg(footer_color_key(state), &self.text())
    }

    /// The themed SHORT LABELLED narrow form of the decision segment (the
    /// `statusCompactText` the tray row uses when the row must narrow): the dot
    /// plus a label that still names the state, e.g. `● Jev C On`. Never a bare
    /// dot: two bare dots cannot be told apart.
    fn themed_compact(&self) -> String {
        let state = self.state();
        crate::modes::interactive::theme::theme::theme()
            .fg(footer_color_key(state), &footer_compact_text(state))
    }

    /// The themed compaction segment.
    fn themed_compaction_text(&self) -> String {
        let state = self.compaction;
        crate::modes::interactive::theme::theme::theme()
            .fg(state.color_key(), &self.compaction_text())
    }

    /// The themed SHORT LABELLED narrow form of the compaction segment, e.g.
    /// `● Jev Cmp on`.
    fn themed_compaction_compact(&self) -> String {
        let state = self.compaction;
        crate::modes::interactive::theme::theme::theme()
            .fg(state.color_key(), &footer_compaction_compact_text(state))
    }

    /// The host events that publish this snapshot: the decision segment first,
    /// then the independent compaction dot. The dispatch task holds no UI
    /// handle, so it cannot measure the terminal: the tray row truncates to the
    /// live width when it renders and uses `statusCompactText` when the full
    /// row does not fit.
    pub fn published_events(&self) -> Vec<wire::AgentConnectionEvent> {
        vec![
            wire::AgentConnectionEvent::ExtensionUiRequest {
                request: wire::AgentConnectionExtensionUiRequest {
                    id: JEV_FOOTER_REQUEST_ID.to_string(),
                    method: "setStatus".to_string(),
                    payload: serde_json::json!({
                        "statusKey": JEV_STATUS_KEY,
                        "statusText": self.themed_text(),
                        "statusCompactText": self.themed_compact(),
                    }),
                },
            },
            wire::AgentConnectionEvent::ExtensionUiRequest {
                request: wire::AgentConnectionExtensionUiRequest {
                    id: format!("{JEV_FOOTER_REQUEST_ID}-compact"),
                    method: "setStatus".to_string(),
                    payload: serde_json::json!({
                        "statusKey": JEV_COMPACT_STATUS_KEY,
                        "statusText": self.themed_compaction_text(),
                        "statusCompactText": self.themed_compaction_compact(),
                    }),
                },
            },
        ]
    }
}

/// Stable request id for the footer publish (visible in daemon logs).
pub const JEV_FOOTER_REQUEST_ID: &str = "jev-footer";

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot(mode: JevMode, compaction: JevCompactionState) -> JevFooterSnapshot {
        JevFooterSnapshot {
            mode,
            credential: CredentialStatus::resolve(true, false, false),
            pipeline: JevPipelineStatus::default(),
            compaction,
        }
    }

    #[test]
    fn both_segments_are_published_independently() {
        crate::modes::interactive::theme::theme::init_theme(Some("dark"), false);
        let snapshot = snapshot(JevMode::Off, JevCompactionState::On);
        let events = snapshot.published_events();
        assert_eq!(events.len(), 2);
        let decision = &events[0];
        let compaction = &events[1];
        let wire::AgentConnectionEvent::ExtensionUiRequest {
            request: decision_request,
        } = decision
        else {
            panic!("decision event shape");
        };
        let wire::AgentConnectionEvent::ExtensionUiRequest {
            request: compaction_request,
        } = compaction
        else {
            panic!("compaction event shape");
        };
        assert_eq!(
            decision_request.payload["statusKey"],
            serde_json::json!(JEV_STATUS_KEY)
        );
        assert!(decision_request.payload["statusText"]
            .as_str()
            .unwrap()
            .contains("Jev Off"));
        assert_eq!(
            compaction_request.payload["statusKey"],
            serde_json::json!(JEV_COMPACT_STATUS_KEY)
        );
        assert!(compaction_request.payload["statusText"]
            .as_str()
            .unwrap()
            .contains("Jev compact on"));
        // Decision off does NOT imply compaction off: the compaction dot is
        // green while the decision dot is red. The narrow forms are SHORT
        // LABELLED, never bare dots.
        assert!(compaction_request.payload["statusCompactText"]
            .as_str()
            .unwrap()
            .contains("Jev Cmp on"));
        assert!(decision_request.payload["statusCompactText"]
            .as_str()
            .unwrap()
            .contains("Jev Off"));
    }

    #[test]
    fn compaction_unknown_is_its_own_label() {
        let snapshot = snapshot(JevMode::Compare, JevCompactionState::Unknown);
        assert!(snapshot.compaction_text().contains("Jev compact unknown"));
    }
}
