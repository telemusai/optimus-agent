//! Port of
//! packages/coding-agent/src/modes/interactive/components/compaction-outcome-message.ts

use pi_tui::components::r#box::Box_;
use pi_tui::components::text::Text;
use pi_tui::tui::{Component, Container};
use std::cell::RefCell;
use std::rc::Rc;

use crate::core::messages::{CompactionOutcomeDetails, COMPACTION_OUTCOME_SKIPPED};
use crate::modes::interactive::theme::theme::theme;

/// Renders a durable unsuccessful automatic-compaction outcome.
pub struct CompactionOutcomeMessageComponent {
    container: Container,
}

impl CompactionOutcomeMessageComponent {
    /// Port of the `CompactionOutcomeMessageComponent` constructor.
    ///
    /// The TypeScript takes the `CompactionOutcomeMessage` custom message; the
    /// port takes its two fields, because `CustomMessage<CompactionOutcomeDetails>`
    /// carries `details` as `serde_json::Value` in core/messages.rs.
    pub fn new(content: &str, details: &CompactionOutcomeDetails) -> Self {
        let mut container = Container::new();
        let color = if details.outcome == COMPACTION_OUTCOME_SKIPPED {
            "warning"
        } else {
            "error"
        };
        let background = theme().get_user_message_background_color();
        let mut content_box = Box_::new(2, 1, Some(Box::new(move |text: &str| background(text))));
        let text = theme().fg(color, content);
        content_box.add_child(Box::new(Text::new(text, 0, 0, None)));
        container.add_child(Rc::new(RefCell::new(content_box)) as Rc<RefCell<dyn Component>>);
        Self { container }
    }

    /// Port of `setExpanded` - the outcome component ignores expansion.
    pub fn set_expanded(&mut self, _expanded: bool) {}

    /// Convenience accessor for the wrapped children (the TypeScript class
    /// extends `Container`, so the port exposes the same child list).
    pub fn children(&self) -> &[Rc<RefCell<dyn Component>>] {
        &self.container.children
    }
}

impl Component for CompactionOutcomeMessageComponent {
    fn render(&mut self, width: f64) -> Vec<String> {
        self.container.render(width)
    }

    fn get_selection_regions(&self) -> Vec<pi_tui::selection_metadata::TableCellSelectionRegion> {
        self.container.get_selection_regions()
    }

    fn invalidate(&mut self) {
        self.container.invalidate();
    }
}

/// Port of `MalformedCompactionOutcomeMessageComponent`.
pub struct MalformedCompactionOutcomeMessageComponent {
    container: Container,
}

impl MalformedCompactionOutcomeMessageComponent {
    /// Port of the `MalformedCompactionOutcomeMessageComponent` constructor.
    pub fn new() -> Self {
        let mut container = Container::new();
        let background = theme().get_user_message_background_color();
        let mut content_box = Box_::new(2, 1, Some(Box::new(move |text: &str| background(text))));
        let text = theme().fg("error", "[Malformed compaction outcome message]");
        content_box.add_child(Box::new(Text::new(text, 0, 0, None)));
        container.add_child(Rc::new(RefCell::new(content_box)) as Rc<RefCell<dyn Component>>);
        Self { container }
    }

    /// Port of `setExpanded` - ignored.
    pub fn set_expanded(&mut self, _expanded: bool) {}

    pub fn children(&self) -> &[Rc<RefCell<dyn Component>>] {
        &self.container.children
    }
}

impl Default for MalformedCompactionOutcomeMessageComponent {
    fn default() -> Self {
        Self::new()
    }
}

impl Component for MalformedCompactionOutcomeMessageComponent {
    fn render(&mut self, width: f64) -> Vec<String> {
        self.container.render(width)
    }

    fn get_selection_regions(&self) -> Vec<pi_tui::selection_metadata::TableCellSelectionRegion> {
        self.container.get_selection_regions()
    }

    fn invalidate(&mut self) {
        self.container.invalidate();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn init() {
        crate::modes::interactive::theme::theme::init_theme(Some("prime"), false);
    }

    #[test]
    fn outcome_message_renders_one_line() {
        init();
        let details = CompactionOutcomeDetails {
            reason: crate::core::messages::COMPACTION_OUTCOME_REASON_THRESHOLD.to_string(),
            outcome: COMPACTION_OUTCOME_SKIPPED.to_string(),
            ..Default::default()
        };
        let mut component = CompactionOutcomeMessageComponent::new("Compaction skipped", &details);
        let lines = component.render(40.0);
        assert_eq!(lines.len(), 3);
        assert!(lines[1].contains("Compaction skipped"));
    }

    #[test]
    fn malformed_message_renders_the_error_label() {
        init();
        let mut component = MalformedCompactionOutcomeMessageComponent::new();
        let lines = component.render(60.0);
        assert_eq!(lines.len(), 3);
        assert!(lines[1].contains("[Malformed compaction outcome message]"));
        let wrapped = component.render(40.0);
        assert_eq!(wrapped.len(), 4);
        assert!(wrapped[1].contains("[Malformed compaction outcome"));
        assert!(wrapped[2].contains("message]"));
    }
}
