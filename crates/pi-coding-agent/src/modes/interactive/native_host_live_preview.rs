//! A bounded, presentation-only tail of the current model response.
use super::*;
use pi_ai::types::{ContentBlock, Message};
use pi_tui::utils::{strip_ansi, wrap_text_with_ansi};

const MAX_TAIL_CHARS: usize = 2048;

#[derive(Default)]
pub(super) struct LivePreview {
    visible: bool,
    streaming: bool,
    label: String,
    tail: String,
    output_tokens: Option<u64>,
}

fn clean_tail(text: &str, limit: usize) -> String {
    let start = text
        .char_indices()
        .rev()
        .nth(limit.saturating_sub(1))
        .map(|(offset, _)| offset)
        .unwrap_or(0);
    strip_ansi(&text[start..])
        .chars()
        .map(|c| match c {
            '\n' => '\n',
            '\t' => ' ',
            c if c.is_control() => ' ',
            c => c,
        })
        .collect()
}

impl LivePreview {
    pub fn observe(&mut self, event: &wire::AgentConnectionSessionEvent) {
        use pi_agent_core::types::AgentEvent;
        use wire::AgentConnectionSessionEvent as Event;
        match event.type_name() {
            "agent_start" => {
                *self = Self {
                    visible: true,
                    streaming: true,
                    ..Self::default()
                };
            }
            "agent_end" => self.streaming = false,
            _ => {}
        }
        let message = match event {
            Event::MessageStart { message }
            | Event::MessageUpdate { message, .. }
            | Event::MessageEnd { message }
            | Event::Agent(
                AgentEvent::MessageStart { message }
                | AgentEvent::MessageUpdate { message, .. }
                | AgentEvent::MessageEnd { message },
            ) => message,
            _ => return,
        };
        if let AgentMessage::Message(Message::Assistant(message)) = message {
            self.update(message, event.type_name() != "message_end");
        }
    }

    pub fn update(&mut self, message: &pi_ai::types::AssistantMessage, streaming: bool) {
        self.visible = true;
        self.streaming = streaming;
        self.tail.clear();
        self.label.clear();
        self.output_tokens = (message.usage.output.is_finite() && message.usage.output > 0.0)
            .then_some(message.usage.output as u64);
        // Providers already decode partial tool arguments for the normal transcript.
        // Reuse those strings rather than exposing JSON or counting stream chunks as tokens.
        for block in message.content.iter().rev() {
            let (label, text) = match block {
                ContentBlock::Text(text) => ("text", text.text.as_str()),
                ContentBlock::ToolCall(tool) => {
                    let text = [
                        "code",
                        "command",
                        "content",
                        "patch",
                        "newText",
                        "new_string",
                        "new_text",
                    ]
                    .iter()
                    .find_map(|key| tool.arguments.get(*key).and_then(serde_json::Value::as_str))
                    .unwrap_or("Preparing tool call…");
                    (tool.name.as_str(), text)
                }
                _ => continue,
            };
            if !text.is_empty() {
                self.label = clean_tail(label, 32).replace('\n', " ");
                self.tail = clean_tail(text, MAX_TAIL_CHARS);
                break;
            }
        }
    }

    pub fn height(&self, width: usize, budget: usize) -> usize {
        if !self.visible || width < 60 || budget < 6 {
            0
        } else {
            5.min(budget - 3)
        }
    }

    /// Rows occupy the right side of the session frame; the input keeps its own budget.
    pub fn render(&self, width: usize, height: usize) -> Vec<String> {
        if height < 3 || width < 60 {
            return Vec::new();
        }
        let palette = theme();
        let panel_width = 56.min(width - 6);
        let inner = panel_width - 4;
        let tokens = self
            .output_tokens
            .map(|n| format!("{n} tokens"))
            .unwrap_or_else(|| {
                if self.streaming {
                    "tokens pending"
                } else {
                    "tokens unavailable"
                }
                .into()
            });
        let title = format!(
            " {} · {} ",
            if self.streaming {
                "LIVE"
            } else {
                "LAST OUTPUT"
            },
            tokens
        );
        let title = truncate_to_width(&title, (panel_width - 2) as f64, "…", false);
        let mut rows = vec![palette.fg(
            "border",
            &format!(
                "┌{title}{}┐",
                "─".repeat((panel_width - 2).saturating_sub(pi_tui::utils::visible_width(&title)))
            ),
        )];
        let content = if self.tail.is_empty() {
            vec![if self.streaming {
                "Waiting for text or code…".into()
            } else {
                "No text output".into()
            }]
        } else {
            wrap_text_with_ansi(&self.tail, inner)
        };
        let count = height - 2;
        let start = content.len().saturating_sub(count);
        for index in 0..count {
            let text = content.get(start + index).map(String::as_str).unwrap_or("");
            rows.push(format!(
                "{} {} {}",
                palette.fg("border", "│"),
                palette.fg(
                    if self.label == "text" {
                        "text"
                    } else {
                        "thinkingText"
                    },
                    &truncate_to_width(text, inner as f64, "…", true)
                ),
                palette.fg("border", "│")
            ));
        }
        let label = truncate_to_width(
            &format!(" {} ", self.label),
            (panel_width - 2) as f64,
            "…",
            false,
        );
        rows.push(palette.fg(
            "border",
            &format!(
                "└{label}{}┘",
                "─".repeat((panel_width - 2).saturating_sub(pi_tui::utils::visible_width(&label)))
            ),
        ));
        rows.into_iter()
            .map(|row| {
                native_neon::surface(
                    &format!(
                        "{}{}{} {}",
                        palette.fg("border", "│"),
                        " ".repeat(width - panel_width - 3),
                        row,
                        palette.fg("border", "│")
                    ),
                    width,
                )
            })
            .collect()
    }
}

#[cfg(test)]
#[path = "native_host_live_preview_tests.rs"]
mod tests;
