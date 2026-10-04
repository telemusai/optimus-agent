//! Only pending error decorations participate in the timer, so a long chat
//! does not require walking every assistant message on every idle frame.
use super::*;
use std::collections::VecDeque;
use std::rc::Weak;
use super::super::transient_errors::ERROR_DURATION;

#[derive(Default)]
pub(super) struct PendingErrors {
    pending: VecDeque<(Instant, Weak<RefCell<AssistantMessageComponent>>)>,
}

impl PendingErrors {
    pub(super) fn push(&mut self, component: &Rc<RefCell<AssistantMessageComponent>>, now: Instant) {
        self.pending.push_back((now + ERROR_DURATION, Rc::downgrade(component)));
    }

    pub(super) fn expire(&mut self, now: Instant) -> bool {
        let mut changed = false;
        while self.pending.front().is_some_and(|(deadline, _)| now >= *deadline) {
            if let Some(component) = self.pending.pop_front().and_then(|(_, component)| component.upgrade()) {
                changed |= component.borrow_mut().dismiss_request_error();
            }
        }
        changed
    }

    /// Loading saved history must not bring previously dismissed errors back.
    pub(super) fn dismiss_all(&mut self) {
        for (_, component) in self.pending.drain(..) {
            if let Some(component) = component.upgrade() {
                component.borrow_mut().dismiss_request_error();
            }
        }
    }
}

pub(super) fn expire_notices(
    mode: &Rc<RefCell<InteractiveMode>>,
    transcript: &Rc<RefCell<Transcript>>,
    ui: &Rc<RefCell<TUI>>,
    now: Instant,
) {
    let notice_expired = mode.borrow_mut().expire_error_notices(now);
    let request_expired = transcript.borrow_mut().request_errors.expire(now);
    if notice_expired || request_expired {
        ui.borrow_mut().request_render();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pi_ai::types::{AssistantMessage, ContentBlock, TextContent};

    fn failure(text: &str) -> AgentMessage {
        AssistantMessage {
            stop_reason: pi_ai::types::STOP_REASON_ERROR.into(),
            error_message: Some(text.into()),
            ..Default::default()
        }.into()
    }

    fn transcript() -> Transcript {
        Transcript::new(Rc::new(RefCell::new(super::super::tests::stash_mode("transient-error-test"))))
    }

    #[test]
    fn request_error_disappears_while_idle_and_newer_error_gets_its_own_deadline() {
        let mut transcript = transcript();
        transcript.message(failure("Kiro network request failed"), false);
        let first_deadline = transcript.request_errors.pending[0].0;
        assert!(transcript.render(100.0).join("\n").contains("Kiro network request failed"));
        assert!(!transcript.request_errors.expire(first_deadline - Duration::from_millis(1)));
        transcript.message(failure("Newer error"), false);
        // Advance the second deadline explicitly: no sleeps or wall-clock race.
        transcript.request_errors.pending[1].0 = first_deadline + Duration::from_secs(2);
        assert!(transcript.request_errors.expire(first_deadline));
        let visible = transcript.render(100.0).join("\n");
        assert!(!visible.contains("Kiro network request failed"));
        assert!(visible.contains("Newer error"));
        assert!(transcript.request_errors.expire(first_deadline + Duration::from_secs(2)));
        assert!(!transcript.render(100.0).join("\n").contains("Newer error"));
        assert!(!transcript.request_errors.expire(first_deadline + Duration::from_secs(3)));
    }

    #[test]
    fn request_error_expiry_preserves_partial_response_and_later_success() {
        let mut transcript = transcript();
        let message = AssistantMessage {
            content: vec![ContentBlock::Text(TextContent::new("Partial response"))],
            stop_reason: pi_ai::types::STOP_REASON_ERROR.into(),
            error_message: Some("Kiro network request failed".into()),
            ..Default::default()
        };
        transcript.message(message.into(), false);
        transcript.message(AssistantMessage {
            content: vec![ContentBlock::Text(TextContent::new("Successful next reply"))],
            ..Default::default()
        }.into(), false);
        assert!(transcript.request_errors.expire(Instant::now() + ERROR_DURATION));
        transcript.apply_chat_detail();
        transcript.invalidate();
        let visible = transcript.render(100.0).join("\n");
        assert!(visible.contains("Partial response"));
        assert!(visible.contains("Successful next reply"));
        assert!(!visible.contains("Kiro network request failed"));
    }

    #[test]
    fn restoring_or_replacing_chat_does_not_resurrect_request_errors() {
        let mut transcript = transcript();
        transcript.replace(vec![failure("Old live error")]);
        transcript.replace_history(vec![failure("Old history error")], 1.0);
        let visible = transcript.render(100.0).join("\n");
        assert!(!visible.contains("Old live error"));
        assert!(!visible.contains("Old history error"));
        assert!(transcript.request_errors.pending.is_empty());
        transcript.message(failure("New session error"), false);
        transcript.replace(Vec::new());
        assert!(!transcript.request_errors.expire(Instant::now() + ERROR_DURATION));
    }

    #[test]
    fn transient_errors_clear_from_the_painted_frame_without_typing() {
        let h = super::super::ui_tests::FrameHarness::new("idle-error-expiry");
        h.editor.borrow_mut().editor_mut().set_text("my next prompt");
        apply_event(&h.mode, &h.transcript, wire::AgentConnectionSessionEvent::MessageEnd {
            message: failure("Kiro network request failed"),
        });
        apply_event(&h.mode, &h.transcript, wire::AgentConnectionSessionEvent::AutoRetryStart {
            attempt: 1.0, max_attempts: 3.0, delay_ms: 1000.0,
            error_message: "Temporary retry error".into(),
        });
        h.mode.borrow_mut().patch_connection_state(|state| state.is_streaming = false);
        let before = h.paint().join("\n");
        assert!(before.contains("Kiro network request failed"));
        assert!(before.contains("Temporary retry error"));
        expire_notices(&h.mode, &h.transcript, &h.ui, Instant::now() + ERROR_DURATION);
        let after = h.paint().join("\n");
        assert!(!after.contains("Kiro network request failed"), "{after}");
        assert!(!after.contains("Temporary retry error"), "{after}");
        assert!(after.contains("my next prompt"));
        assert_eq!(h.mode.borrow().get_retry_attempt(), 1.0, "expiry does not end an active retry");
    }
}
