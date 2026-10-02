//! Local prompt previews. These are never model messages or durable history.
use super::*;
use std::collections::VecDeque;

#[derive(Default)]
pub(super) struct Receipts(VecDeque<Receipt>);

struct Receipt {
    id: String,
    text: String,
    component: UserMessageComponent,
    accepted: bool,
    queued: bool,
}

impl Receipts {
    pub fn begin(&mut self, id: &str, text: &str, mode: &InteractiveMode) {
        // Match the acknowledgement tracker's bound, even if a host never replies.
        if self.0.len() == 32 { self.0.pop_front(); }
        self.0.push_back(Receipt {
            id: id.into(), text: text.into(), accepted: false, queued: false,
            component: UserMessageComponent::new(text, mode.get_markdown_theme_with_settings(), &|_| false),
        });
    }

    pub fn reply(&mut self, id: &str, success: bool) {
        if let Some(index) = self.0.iter().position(|receipt| receipt.id == id) {
            if success { self.0[index].accepted = true; }
            else { self.0.remove(index); }
        }
    }

    pub fn message(&mut self, text: &str) {
        // Repeated identical prompts are separate submissions; consume only one.
        if let Some(index) = self.0.iter().position(|receipt| receipt.text == text) {
            self.0.remove(index);
        }
    }

    pub fn queue(&mut self, texts: impl Iterator<Item = String>) -> bool {
        if self.0.is_empty() { return false; }
        let mut counts = HashMap::<String, usize>::new();
        for text in texts { *counts.entry(text).or_default() += 1; }
        let mut changed = false;
        self.0.retain_mut(|receipt| {
            let count = counts.entry(receipt.text.clone()).or_default();
            if *count > 0 {
                *count -= 1;
                changed |= !receipt.queued;
                receipt.queued = true;
                true
            } else if receipt.queued {
                // The authoritative queue has consumed or cancelled this prompt.
                changed = true;
                false
            } else { true }
        });
        changed
    }

    pub fn clear(&mut self) { self.0.clear(); }

    pub fn invalidate(&mut self) {
        for receipt in &mut self.0 { receipt.component.invalidate(); }
    }

    pub fn render(&mut self, width: f64) -> Vec<String> {
        let mut lines = Vec::new();
        for receipt in self.0.iter_mut().filter(|receipt| !receipt.queued) {
            lines.extend(receipt.component.render(width));
            let label = if receipt.accepted { "Accepted — waiting for chat update" }
                else { "Sending — not yet confirmed by host" };
            lines.push(theme().fg("dim", label));
        }
        lines
    }
}
