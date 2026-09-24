use super::*;

fn session(id: &str, cwd: &str, modified: &str) -> SessionSummary {
    SessionSummary { id: id.into(), session_id: id.into(), cwd: cwd.into(),
        session_name: Some(format!("Chat {id}")), modified: Some(modified.into()), ..Default::default() }
}

#[test]
fn ui_repair_sidebar_groups_saved_cwds_and_keeps_identity_when_activity_reorders() {
    let temp = tempfile::tempdir().unwrap();
    let mut state = State::new("/a".into(), temp.path().join("folders.json"));
    let a = session("a", "/a", "2026-09-24T00:00:00Z");
    let b = session("b", "/b", "2026-09-24T01:00:00Z");
    state.update(vec![a.clone(), b.clone()], true);
    assert_eq!(state.groups[0].cwd, "/b");
    state.selected = Some(Item::Session("a".into()));
    state.active = Some("b".into());
    let mut latest = a; latest.last_activity_at = Some("2026-09-24T02:00:00Z".into());
    state.update(vec![latest, b], true);
    assert_eq!(state.groups[0].cwd, "/a");
    assert_eq!(state.selected, Some(Item::Session("a".into())));
    assert_eq!(state.active.as_deref(), Some("b"));
    assert_eq!(state.selected_cwd().as_deref(), Some("/a"));
    state.selected = Some(Item::Folder(path_key("/a")));
    assert!(state.confirm().is_none());
    assert!(!state.items().contains(&Item::Session("a".into())));
    assert!(state.confirm().is_none());
    assert!(state.items().contains(&Item::Session("a".into())));
    state.move_by(1);
    assert_eq!(state.confirm().unwrap().session_id, "a");
    assert_eq!(state.active.as_deref(), Some("b"), "Enter returns an explicit request; navigation cannot change active chat");
}

#[test]
fn ui_repair_folder_validation_persistence_and_no_creation() {
    let temp = tempfile::tempdir().unwrap();
    let folder = temp.path().join("ordinary folder");
    std::fs::create_dir(&folder).unwrap();
    let store = temp.path().join("settings/folders.json");
    let missing = temp.path().join("missing");
    assert!(remember_folder(&store, &missing.to_string_lossy()).is_err());
    assert!(!missing.exists()); assert!(!store.exists());
    let file = temp.path().join("not-a-directory"); std::fs::write(&file, "fixture").unwrap();
    assert!(remember_folder(&store, &file.to_string_lossy()).unwrap_err().contains("file"));
    assert!(validate_folder("relative/path").is_err());
    let path = folder.to_string_lossy();
    let added = remember_folder(&store, &format!("\"{path}\"")).unwrap();
    assert_eq!(added.len(), 1);
    assert_eq!(remember_folder(&store, &path).unwrap().len(), 1);
    let reopened = State::new(temp.path().to_string_lossy().into_owned(), store.clone());
    assert!(reopened.groups.iter().any(|g| path_key(&g.cwd) == path_key(&path)));
    assert!(reopened.groups.iter().all(|g| g.sessions.is_empty()));
    assert_eq!(std::fs::read_dir(&folder).unwrap().count(), 0, "Add must not create a chat, repo or other files");
    std::fs::write(&store, "broken settings").unwrap();
    assert!(remember_folder(&store, &path).is_err());
    assert_eq!(std::fs::read_to_string(&store).unwrap(), "broken settings");
}

#[test]
fn ui_repair_sidebar_legend_uses_configured_bindings() {
    use crate::core::keybindings::{KeybindingsConfig, KeybindingSetting, KeybindingsManager};
    let mut bindings = KeybindingsConfig::new();
    bindings.insert("app.sidebar.addFolder".into(), KeybindingSetting::Single("alt+f".into()));
    bindings.insert("app.agents.delete".into(), KeybindingSetting::Single("ctrl+k".into()));
    KeybindingsManager::new(bindings, None).install();
    assert_eq!(key_label("app.sidebar.addFolder"), "Alt+F");
    assert_eq!(key_label("app.agents.delete"), "Ctrl+K");
    let temp = tempfile::tempdir().unwrap();
    let state = Rc::new(RefCell::new(State::new("C:/work".into(), temp.path().join("folders.json"))));
    let painted = Pane(state).render_with_height(34.0, 24).join("\n");
    assert!(strip_ansi(&painted).contains("Alt+F Add folder"));
    assert!(strip_ansi(&painted).contains("Ctrl+K Delete"));
    KeybindingsManager::new(Default::default(), None).install();
}
