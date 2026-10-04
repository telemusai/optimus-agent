//! Short-lived UI errors. Compaction recovery and connection state own their
//! separate lifecycles; these notices never modify the stored conversation.
use super::*;
use std::time::{Duration, Instant};

pub(super) const ERROR_DURATION: Duration = Duration::from_secs(5);

struct ErrorNotice {
    text: Text,
    expires_at: Instant,
}

impl super::super::interactive_mode_services::Component for ErrorNotice {
    fn render(&self, width: usize) -> Vec<String> {
        let mut lines = vec![String::new()];
        lines.extend(super::super::interactive_mode_services::Component::render(&self.text, width));
        lines
    }

    fn as_any(&self) -> &dyn std::any::Any { self }
}

impl InteractiveMode {
    pub(super) fn show_error_at(&mut self, message: &str, now: Instant) {
        let expires_at = now + ERROR_DURATION;
        self.chat_container.add_child(Box::new(ErrorNotice {
            text: Text::new(theme().fg("error", &format!("Error: {message}")), 1, 0),
            expires_at,
        }));
        self.next_error_expiry = Some(self.next_error_expiry.map_or(expires_at, |next| next.min(expires_at)));
        self.last_status_spacer_index = None;
        self.last_status_text_index = None;
        self.ui.request_render();
    }

    /// Called by the owner loop even when idle. Do not scan chat rows on each
    /// tick: only visit notices once the next expiry is due.
    pub(super) fn expire_error_notices(&mut self, now: Instant) -> bool {
        if self.next_error_expiry.is_none_or(|next| now < next) { return false; }
        self.next_error_expiry = None;
        let before = self.chat_container.len();
        self.chat_container.children.retain(|child| {
            let Some(notice) = child.as_any().downcast_ref::<ErrorNotice>() else { return true; };
            if now >= notice.expires_at { return false; }
            self.next_error_expiry = Some(self.next_error_expiry.map_or(notice.expires_at, |next| next.min(notice.expires_at)));
            true
        });
        if before == self.chat_container.len() { return false; }
        // Removing a notice shifts indices; no later status may overwrite an
        // unrelated row using an index cached before removal.
        self.last_status_spacer_index = None;
        self.last_status_text_index = None;
        self.ui.request_render();
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::tests::{chat_lines, test_mode};

    #[test]
    fn transient_errors_expire_independently_and_remove_their_spacing() {
        let now = Instant::now();
        let mut mode = test_mode();
        mode.show_error_at("Kiro network request failed", now);
        assert!(chat_lines(&mode).join("\n").contains("Kiro network request failed"));
        assert!(!mode.expire_error_notices(now + ERROR_DURATION - Duration::from_millis(1)));
        mode.show_error_at("Newer error", now + Duration::from_secs(2));
        assert!(mode.expire_error_notices(now + ERROR_DURATION));
        let lines = chat_lines(&mode);
        assert_eq!(lines.len(), 2, "old notice leaves no blank spacer behind");
        assert!(lines.join("\n").contains("Newer error"));
        assert!(!lines.join("\n").contains("Kiro network request failed"));
        assert!(mode.expire_error_notices(now + Duration::from_secs(7)));
        assert!(chat_lines(&mode).is_empty());
        assert!(!mode.expire_error_notices(now + Duration::from_secs(8)));
    }

    #[test]
    fn transient_error_expiry_preserves_status_warning_and_compaction_notice() {
        let now = Instant::now();
        let mut mode = test_mode();
        mode.show_error_at("temporary", now);
        mode.show_status("new status", "dim");
        mode.show_warning("warning");
        mode.show_compaction_error("compaction failed");
        assert!(mode.expire_error_notices(now + ERROR_DURATION));
        mode.show_status("latest status", "dim");
        let visible = chat_lines(&mode).join("\n");
        for text in ["new status", "warning", "compaction failed", "latest status"] {
            assert!(visible.contains(text), "missing {text}: {visible}");
        }
        assert!(!visible.contains("temporary"));
    }
}
