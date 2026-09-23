//! Presentation only: the stored completion and its agent wakeup stay unchanged.
use pi_agent_core::types::CustomAgentMessage;
use pi_tui::tui::Component;
use pi_tui::utils::truncate_to_width;

use crate::core::messages::{AsyncBashCompletionDetails, ASYNC_BASH_COMPLETION_CUSTOM_TYPE};
use crate::modes::interactive::theme::theme::theme;

pub struct ShellCompletion {
    label: String,
    failed: bool,
}

impl ShellCompletion {
    pub fn from_message(message: &CustomAgentMessage) -> Option<Self> {
        let CustomAgentMessage::Custom {
            custom_type,
            details,
            ..
        } = message
        else {
            return None;
        };
        if custom_type != ASYNC_BASH_COMPLETION_CUSTOM_TYPE {
            return None;
        }
        Some(Self::from_details(details.as_ref()))
    }

    pub(super) fn from_details(details: Option<&serde_json::Value>) -> Self {
        let details = details.and_then(|value| {
            serde_json::from_value::<AsyncBashCompletionDetails>(value.clone()).ok()
        });
        match details {
            Some(details) => Self {
                label: format!(
                    "Shell finished — exit {} (PID {}).",
                    details.exit_code, details.pid
                ),
                failed: details.exit_code != 0,
            },
            None => Self {
                label: "Shell finished — details unavailable.".into(),
                failed: false,
            },
        }
    }
}

impl Component for ShellCompletion {
    fn render(&mut self, width: f64) -> Vec<String> {
        let color = if self.failed { "warning" } else { "dim" };
        vec![truncate_to_width(
            &theme().fg(color, &self.label),
            width.max(0.0),
            "…",
            false,
        )]
    }
    fn invalidate(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::messages::{convert_to_llm, create_async_bash_completion_message};
    use pi_agent_core::types::{AgentMessage, CustomMessageContent};
    use pi_tui::utils::{strip_ansi, visible_width};

    #[test]
    fn shell_notice_is_one_line_and_keeps_the_full_agent_message() {
        crate::modes::interactive::theme::theme::init_theme(Some("prime"), false);
        let message = create_async_bash_completion_message(
            AsyncBashCompletionDetails {
                pid: 33504,
                command: "echo private-path\nsecond command".into(),
                exit_code: 0,
            },
            1,
        );
        let before = serde_json::to_value(&message).unwrap();
        let mut component = ShellCompletion::from_message(&message).unwrap();
        assert_eq!(
            strip_ansi(&component.render(80.0)[0]),
            "Shell finished — exit 0 (PID 33504)."
        );
        for width in [0, 1, 5, 20, 80, 120] {
            let lines = component.render(width as f64);
            assert_eq!(lines.len(), 1);
            assert!(visible_width(&lines[0]) <= width);
            assert!(!lines[0].contains('\n'));
        }
        assert_eq!(serde_json::to_value(&message).unwrap(), before);
        let agent = convert_to_llm(&[AgentMessage::Custom(message)], &Default::default());
        let encoded = serde_json::to_string(&agent).unwrap();
        assert!(encoded.contains("private-path"));
        assert!(encoded.contains("BashHandle"));
    }

    #[test]
    fn shell_notice_failure_and_missing_metadata_never_dump_the_command() {
        crate::modes::interactive::theme::theme::init_theme(Some("prime"), false);
        let mut message = create_async_bash_completion_message(
            AsyncBashCompletionDetails {
                pid: 42,
                command: "secret fixture command".into(),
                exit_code: -1,
            },
            1,
        );
        assert_eq!(
            strip_ansi(
                &ShellCompletion::from_message(&message)
                    .unwrap()
                    .render(80.0)[0]
            ),
            "Shell finished — exit -1 (PID 42)."
        );
        if let CustomAgentMessage::Custom {
            details, content, ..
        } = &mut message
        {
            *details = Some(serde_json::json!({"pid": "42\nINJECTED", "exitCode": 0}));
            *content = CustomMessageContent::Text("untrusted\nmultiline command".into());
        }
        assert_eq!(
            strip_ansi(
                &ShellCompletion::from_message(&message)
                    .unwrap()
                    .render(80.0)[0]
            ),
            "Shell finished — details unavailable."
        );
    }
}
