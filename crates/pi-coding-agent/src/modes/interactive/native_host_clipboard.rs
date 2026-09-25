//! Clipboard feedback owns a separate UI slot, never a durable status row.
use std::time::{Duration, Instant};

const NOTICE_DURATION: Duration = Duration::from_secs(3);

#[derive(Default)]
pub(super) struct Notice {
    active: Option<(String, Instant)>,
}

impl Notice {
    pub(super) fn show(&mut self, message: String, now: Instant) {
        self.active = Some((message, now + NOTICE_DURATION));
    }

    pub(super) fn text(&self) -> Option<&str> {
        self.active.as_ref().map(|(message, _)| message.as_str())
    }

    // Called by the owner loop even while idle; true requests a repaint.
    pub(super) fn expire(&mut self, now: Instant) -> bool {
        if self.active.as_ref().is_some_and(|(_, deadline)| now >= *deadline) {
            self.active = None;
            return true;
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::*;
    use crate::utils::clipboard::ClipboardOutcome;

    #[test]
    fn clipboard_notice_expires_at_deadline_and_requests_one_repaint() {
        let now = Instant::now();
        for outcome in [ClipboardOutcome::LocalBackendAccepted, ClipboardOutcome::TerminalForwarded] {
            let mut notice = Notice::default();
            notice.show(outcome.status().into(), now);
            assert!(!notice.expire(now + NOTICE_DURATION - Duration::from_millis(1)));
            assert_eq!(notice.text(), Some(outcome.status()));
            assert!(notice.expire(now + NOTICE_DURATION));
            assert_eq!(notice.text(), None);
            assert!(!notice.expire(now + NOTICE_DURATION));
        }
    }

    #[test]
    fn clipboard_notice_repeated_copy_restarts_deadline() {
        let now = Instant::now();
        let mut notice = Notice::default();
        notice.show("same message".into(), now);
        notice.show("same message".into(), now + Duration::from_secs(2));
        assert!(!notice.expire(now + NOTICE_DURATION));
        assert_eq!(notice.text(), Some("same message"));
        assert!(notice.expire(now + Duration::from_secs(5)));
    }

    #[test]
    fn clipboard_notice_command_and_mouse_feedback_preserve_newer_status() {
        let now = Instant::now();
        let mode = Rc::new(RefCell::new(super::super::tests::stash_mode("clipboard-test")));
        let mut transcript = Transcript::new(mode.clone());
        for outcome in [ClipboardOutcome::LocalBackendAccepted, ClipboardOutcome::TerminalForwarded] {
            let events = CommandOutput::ClipboardNotice(outcome.status().into()).into_events();
            let [HostEvent::ClipboardNotice(message)] = events.as_slice() else {
                panic!("/copy must use the same transient event as mouse selection");
            };
            transcript.clipboard_notice.show(message.clone(), now);
            let visible = transcript.render(120.0).join("\n");
            assert!(visible.contains(outcome.status()));
            mode.borrow_mut().show_status("Newer unrelated status", "dim");
            assert!(transcript.clipboard_notice.expire(now + NOTICE_DURATION));
            let visible = transcript.render(120.0).join("\n");
            assert!(!visible.contains(outcome.status()));
            assert!(visible.contains("Newer unrelated status"));
        }
    }

    #[test]
    fn clipboard_notice_direct_report_and_session_reset() {
        let mode = Rc::new(RefCell::new(super::super::tests::stash_mode("clipboard-reset")));
        let (send, receive) = mpsc::channel();
        CommandOutput::ClipboardNotice("copy feedback".into()).report(&mode, &send);
        let HostEvent::ClipboardNotice(message) = receive.try_recv().unwrap() else {
            panic!("direct reporting must also remain transient");
        };
        let mut transcript = Transcript::new(mode);
        transcript.clipboard_notice.show(message, Instant::now());
        transcript.replace(Vec::new());
        assert_eq!(transcript.clipboard_notice.text(), None);
    }
}
