//! Port of packages/coding-agent/src/modes/interactive/components/user-message-selector.ts

use pi_tui::components::spacer::Spacer;
use pi_tui::components::text::Text;
use pi_tui::keybindings::get_keybindings;
use pi_tui::tui::Component;
use pi_tui::utils::truncate_to_width;

use crate::modes::interactive::theme::theme::theme;

use super::show_images_selector::DynamicBorder;

/// Port of `UserMessageItem`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct UserMessageItem {
    pub id: String,
    pub text: String,
    pub timestamp: Option<String>,
}

/// Port of `UserMessageList`.
pub struct UserMessageList {
    messages: Vec<UserMessageItem>,
    selected_index: usize,
    max_visible: usize,
    pub on_select: Option<Box<dyn FnMut(&str)>>,
    pub on_cancel: Option<Box<dyn FnMut()>>,
}

impl UserMessageList {
    pub fn new(messages: Vec<UserMessageItem>, initial_selected_id: Option<&str>) -> Self {
        // Session history is chronological; default to the latest fork point.
        let initial_index = initial_selected_id
            .and_then(|id| messages.iter().position(|message| message.id == id))
            .map(|index| index as i64)
            .unwrap_or(-1);
        let selected_index = if initial_index >= 0 {
            initial_index as usize
        } else {
            messages.len().saturating_sub(1).max(0)
        };
        Self {
            messages,
            selected_index,
            max_visible: 10,
            on_select: None,
            on_cancel: None,
        }
    }

    pub fn selected_index(&self) -> usize {
        self.selected_index
    }
}

impl Component for UserMessageList {
    fn render(&mut self, width: f64) -> Vec<String> {
        let width = width.max(0.0).floor() as usize;
        let mut lines: Vec<String> = Vec::new();

        if self.messages.is_empty() {
            lines.push(theme().fg("muted", "  No user messages found"));
            return lines;
        }

        let start_index = (self.selected_index as i64 - (self.max_visible / 2) as i64).clamp(
            0,
            self.messages.len().saturating_sub(self.max_visible) as i64,
        ) as usize;
        let end_index = (start_index + self.max_visible).min(self.messages.len());

        for i in start_index..end_index {
            let message = &self.messages[i];
            let is_selected = i == self.selected_index;

            let normalized_message = normalize_newlines(&message.text).trim().to_string();

            let cursor = if is_selected {
                theme().fg("accent", "\u{203a} ")
            } else {
                "  ".to_string()
            };
            let max_msg_width = width.saturating_sub(2) as f64;
            let truncated_msg = truncate_to_width(&normalized_message, max_msg_width, "", false);
            let message_text = if is_selected {
                theme().bold(&truncated_msg)
            } else {
                truncated_msg
            };
            let message_line = format!("{cursor}{message_text}");

            lines.push(message_line);

            let position = i + 1;
            let metadata = format!("  Message {position} of {}", self.messages.len());
            let metadata_line = theme().fg("muted", &metadata);
            lines.push(metadata_line);
            lines.push(String::new());
        }

        if start_index > 0 || end_index < self.messages.len() {
            let scroll_info = theme().fg(
                "muted",
                &format!("  ({}/{})", self.selected_index + 1, self.messages.len()),
            );
            lines.push(scroll_info);
        }

        lines
    }

    fn handle_input(&mut self, key_data: &str) {
        let kb = get_keybindings();
        if kb.matches(key_data, "tui.select.up") {
            self.selected_index = if self.selected_index == 0 {
                self.messages.len().saturating_sub(1)
            } else {
                self.selected_index - 1
            };
        } else if kb.matches(key_data, "tui.select.down") {
            self.selected_index = if self.selected_index == self.messages.len().saturating_sub(1) {
                0
            } else {
                self.selected_index + 1
            };
        } else if kb.matches(key_data, "tui.select.confirm") {
            let selected = self.messages.get(self.selected_index).map(|m| m.id.clone());
            if let (Some(id), Some(callback)) = (selected, self.on_select.as_mut()) {
                callback(&id);
            }
        } else if kb.matches(key_data, "tui.select.cancel") {
            if let Some(callback) = self.on_cancel.as_mut() {
                callback();
            }
        }
    }

    fn invalidate(&mut self) {}
}

/// `message.text.replace(/\n/g, " ")`.
fn normalize_newlines(text: &str) -> String {
    text.replace('\n', " ")
}

/// Port of `UserMessageSelectorComponent`.
pub struct UserMessageSelectorComponent {
    /// The leading children pushed before `messageList`.
    leading: Vec<Box<dyn Component>>,
    message_list: UserMessageList,
    /// `new Spacer(1)` pushed after `messageList`.
    trailing_spacer: Spacer,
    /// The second `DynamicBorder` pushed after `messageList`.
    trailing_border: DynamicBorder,
}

impl UserMessageSelectorComponent {
    pub fn new(
        messages: Vec<UserMessageItem>,
        on_select: Box<dyn FnMut(&str)>,
        on_cancel: Box<dyn FnMut()>,
        initial_selected_id: Option<&str>,
    ) -> Self {
        let leading: Vec<Box<dyn Component>> = vec![
            Box::new(Spacer::new(1)),
            Box::new(Text::new(theme().bold("Fork from Message"), 1, 0, None)),
            Box::new(Text::new(
                theme().fg(
                    "muted",
                    "Select a user message to copy the active path up to that point into a new session",
                ),
                1,
                0,
                None,
            )),
            Box::new(Spacer::new(1)),
            Box::new(DynamicBorder::new(None)),
            Box::new(Spacer::new(1)),
        ];

        let mut message_list = UserMessageList::new(messages, initial_selected_id);
        message_list.on_select = Some(on_select);
        message_list.on_cancel = Some(on_cancel);

        Self {
            leading,
            message_list,
            trailing_spacer: Spacer::new(1),
            trailing_border: DynamicBorder::new(None),
        }
    }

    /// Port of `getMessageList`.
    pub fn get_message_list(&mut self) -> &mut UserMessageList {
        &mut self.message_list
    }
}

impl Component for UserMessageSelectorComponent {
    fn render(&mut self, width: f64) -> Vec<String> {
        let mut lines: Vec<String> = Vec::new();
        for child in self.leading.iter_mut() {
            lines.extend(child.render(width));
        }
        lines.extend(self.message_list.render(width));
        lines.extend(self.trailing_spacer.render(width));
        lines.extend(self.trailing_border.render(width));
        lines
    }

    fn handle_input(&mut self, data: &str) {
        self.message_list.handle_input(data);
    }

    fn invalidate(&mut self) {
        for child in self.leading.iter_mut() {
            child.invalidate();
        }
        self.message_list.invalidate();
        self.trailing_spacer.invalidate();
        self.trailing_border.invalidate();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    fn messages() -> Vec<UserMessageItem> {
        (0..25)
            .map(|index| UserMessageItem {
                id: format!("m{index}"),
                text: format!("message {index}"),
                timestamp: None,
            })
            .collect()
    }

    #[test]
    fn early_and_late_selections_stay_inside_the_fork_dialog_viewport() {
        crate::modes::interactive::theme::theme::init_theme(Some("prime"), false);
        for selected in [0usize, 4, 20] {
            let id = format!("m{selected}");
            let mut list = UserMessageList::new(messages(), Some(&id));
            let rendered = list.render(100.0).join("\n");
            assert!(
                rendered.contains(&format!("Message {} of 25", selected + 1)),
                "selection {selected} is missing from viewport: {rendered}"
            );
            assert!(rendered.contains(&format!("message {selected}")));
        }
    }

    #[test]
    fn fork_dialog_up_down_enter_selects_the_visible_message() {
        let selected = Rc::new(RefCell::new(Vec::<String>::new()));
        let sink = Rc::clone(&selected);
        let mut list = UserMessageList::new(messages(), Some("m4"));
        list.on_select = Some(Box::new(move |id| sink.borrow_mut().push(id.to_string())));

        list.handle_input("\u{1b}[A");
        assert_eq!(list.selected_index(), 3);
        list.handle_input("\u{1b}[B");
        assert_eq!(list.selected_index(), 4);
        list.handle_input("\r");

        assert_eq!(selected.borrow().as_slice(), ["m4"]);
    }
}
