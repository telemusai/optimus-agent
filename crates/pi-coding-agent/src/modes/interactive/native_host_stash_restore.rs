//! Restore only the stash captured by the latest accepted editor submission.
use super::*;
use crate::modes::interactive::prompt_stash_state::PromptStash;

pub(super) struct PendingRestore {
    ticket_id: String,
    session_id: String,
    stash: PromptStash,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepted_submit_restores_images_and_paste_but_never_overwrites_new_typing() {
        let mut controller = super::super::tests::stash_mode("restore-after-submit");
        controller.options.prompt_stash_store = Some(Arc::new(std::sync::Mutex::new(
            crate::modes::interactive::prompt_stash_state::ClientPromptStashStore::new(),
        )));
        controller.apply_connection_state_snapshot(local::AgentConnectionState {
            session_id: "restore-after-submit".into(),
            ..Default::default()
        });
        let mode = Rc::new(RefCell::new(controller));
        let ui = Rc::new(RefCell::new(TUI::new(
            Box::new(pi_tui::terminal::ProcessTerminal::new()),
            Some(false),
        )));
        let editor = Rc::new(RefCell::new(CustomEditor::new(
            ui,
            editor_theme(),
            CustomEditorOptions::default(),
        )));
        let session = stash_session(&mode.borrow(), "restore-after-submit");
        let draft = PromptStashCapture {
            text: "draft [image #7] [paste #1 +2 lines]".into(),
            expanded_text: "draft [image #7] one\ntwo".into(),
            images: vec![(7, ImageContent::new("fixture", "image/png"))],
            paste_snapshot: Some(pi_tui::editor_component::EditorPasteSnapshot {
                pastes: vec![(1, "one\ntwo".into())],
                paste_counter: 1,
            }),
        };
        session.handle_prompt_stash(&draft, true);
        let mut pending = capture(&mode.borrow(), "new");
        finish(&mut pending, "old", true, &mode, &editor);
        assert!(
            pending.is_some(),
            "out-of-order acknowledgement must not consume latest capture"
        );
        editor.borrow_mut().editor_mut().set_text("new typing");
        finish(&mut pending, "new", true, &mode, &editor);
        assert_eq!(editor.borrow().editor().get_text(), "new typing");
        assert!(session.state().stash.is_some());
        editor.borrow_mut().editor_mut().set_text("");
        let mut pending = capture(&mode.borrow(), "rejected");
        finish(&mut pending, "rejected", false, &mode, &editor);
        assert!(editor.borrow().editor().get_text().is_empty());
        assert!(session.state().stash.is_some());
        let mut pending = capture(&mode.borrow(), "accepted");
        finish(&mut pending, "accepted", true, &mode, &editor);
        assert_eq!(editor.borrow().editor().get_text(), draft.text);
        assert_eq!(
            editor.borrow().editor().get_expanded_text(),
            draft.expanded_text
        );
        assert_eq!(mode.borrow().pasted_images.get(&7).unwrap().data, "fixture");
        assert!(session.state().stash.is_none());

        session.handle_prompt_stash(&draft, true);
        let mut pending = capture(&mode.borrow(), "previous-session");
        editor.borrow_mut().editor_mut().set_text("");
        mode.borrow_mut()
            .patch_connection_state(|state| state.session_id = "other-session".into());
        finish(&mut pending, "previous-session", true, &mode, &editor);
        assert!(editor.borrow().editor().get_text().is_empty());
        assert!(session.state().stash.is_some());
    }
}

pub(super) fn capture(mode: &InteractiveMode, ticket_id: &str) -> Option<PendingRestore> {
    let session_id = mode.connection_state.as_ref()?.session_id.clone();
    let stash = stash_session(mode, &session_id).state().stash?;
    Some(PendingRestore {
        ticket_id: ticket_id.into(),
        session_id,
        stash,
    })
}

pub(super) fn finish(
    pending: &mut Option<PendingRestore>,
    ticket_id: &str,
    accepted: bool,
    mode: &Rc<RefCell<InteractiveMode>>,
    editor: &Rc<RefCell<CustomEditor>>,
) {
    if !pending
        .as_ref()
        .is_some_and(|pending| pending.ticket_id == ticket_id)
    {
        return;
    }
    let pending = pending.take().unwrap();
    if !accepted
        || mode
            .borrow()
            .connection_state
            .as_ref()
            .is_none_or(|state| state.session_id != pending.session_id)
    {
        return;
    }
    let session = stash_session(&mode.borrow(), &pending.session_id);
    let outcome = session.restore_prompt_stash_if_editor_empty(
        Some(&pending.stash),
        &editor.borrow().editor().get_text(),
        true,
    );
    if let Some(outcome) = outcome {
        if let Some(images) = pending.stash.images {
            let mut mode = mode.borrow_mut();
            for (id, image) in images {
                mode.next_image_marker_id = mode.next_image_marker_id.max(id.saturating_add(1));
                mode.pasted_images.insert(id, image);
            }
        }
        apply_prompt_stash_outcome(mode, editor, &outcome);
    }
}
