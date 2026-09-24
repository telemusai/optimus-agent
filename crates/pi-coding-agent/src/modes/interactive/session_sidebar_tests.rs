use super::*;
use crate::modes::interactive::theme::theme::init_theme;

fn session(id: &str, cwd: &str, modified: &str) -> SessionSummary {
    SessionSummary { id: id.into(), session_id: id.into(), cwd: cwd.into(),
        session_name: Some(format!("Chat {id}")), modified: Some(modified.into()), ..Default::default() }
}

fn group_label(state: &State, cwd: &str) -> String {
    state.groups.iter().find(|g| path_key(&g.cwd) == path_key(cwd))
        .unwrap_or_else(|| panic!("missing group for {cwd}")).label.clone()
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
    bindings.insert("app.sidebar.location".into(), KeybindingSetting::Single("alt+l".into()));
    KeybindingsManager::new(bindings, None).install();
    assert_eq!(key_label("app.sidebar.addFolder"), "Alt+F");
    assert_eq!(key_label("app.agents.delete"), "Ctrl+K");
    init_theme(Some("neon"), false);
    let temp = tempfile::tempdir().unwrap();
    let state = Rc::new(RefCell::new(State::new("C:/work".into(), temp.path().join("folders.json"))));
    let rows = Pane(state.clone()).render_with_height(34.0, 24);
    assert_eq!(strip_ansi(&rows[state.borrow().location_footer_row()]).trim_end_matches([' ', '│']), "C:/work");
    assert!(strip_ansi(rows.last().unwrap()).contains("Alt+L Location / copy"));
    let painted = rows.join("\n");
    assert!(strip_ansi(&painted).contains("Alt+F Add folder"));
    assert!(strip_ansi(&painted).contains("Ctrl+K Delete"));
    assert!(strip_ansi(&painted).contains("Alt+L Location / copy"));
    KeybindingsManager::new(Default::default(), None).install();
}

#[test]
fn ui008_group_labels_name_projects_and_disambiguate_identical_names() {
    let temp = tempfile::tempdir().unwrap();
    let mut state = State::new("/nowhere".into(), temp.path().join("folders.json"));
    state.update(vec![
        session("a", "C:/batches/optimus-sidebar-fixes-074529/source", "2026-09-24T01:00:00Z"),
        session("b", "C:/other/optimus-main-update-234513/source", "2026-09-24T02:00:00Z"),
        session("c", "C:/work/WSP Financial Tracker", "2026-09-24T03:00:00Z"),
    ], true);
    assert_eq!(group_label(&state, "C:/batches/optimus-sidebar-fixes-074529/source"), "optimus-sidebar-fixes-074529",
        "a generic 'source' folder shows the meaningful project name");
    assert_eq!(group_label(&state, "C:/other/optimus-main-update-234513/source"), "optimus-main-update-234513");
    assert_eq!(group_label(&state, "C:/work/WSP Financial Tracker"), "WSP Financial Tracker",
        "a meaningful project folder name is kept");
    state.selected = Some(Item::Folder(path_key("C:/batches/optimus-sidebar-fixes-074529/source")));
    assert_eq!(state.selected_cwd().as_deref(), Some("C:/batches/optimus-sidebar-fixes-074529/source"),
        "the full path stays available for confirmation");
    state.update(vec![
        session("d", "C:/clients/wsp/source", "2026-09-24T04:00:00Z"),
        session("e", "C:/internal/wsp/source", "2026-09-24T05:00:00Z"),
    ], true);
    assert_eq!(group_label(&state, "C:/clients/wsp/source"), "clients/wsp",
        "same-named projects are distinguished by their location");
    assert_eq!(group_label(&state, "C:/internal/wsp/source"), "internal/wsp");
    assert!(state.groups.iter().any(|g| path_key(&g.cwd) == path_key("C:/clients/wsp/source")),
        "grouping identity stays the repository cwd");
}

#[test]
fn ui011_saved_and_live_rows_of_one_chat_merge_into_its_repository_once() {
    let temp = tempfile::tempdir().unwrap();
    let mut state = State::new("/nowhere".into(), temp.path().join("folders.json"));
    let mut saved = session("a", "C:/repo-alpha", "2026-09-24T00:00:00Z");
    saved.session_id = String::new(); // old transcript without header id: identity falls back to the file
    saved.session_file = Some("C:/sessions/alpha.jsonl".into());
    let mut live = session("live-sid", "", "2026-09-24T03:00:00Z");
    live.id = "daemon-1".into();
    live.active_session_id = Some("daemon-1".into());
    live.is_session_active = true;
    live.session_file = Some("C:/sessions/alpha.jsonl".into());
    state.update(vec![saved, live], true);
    let repo = state.groups.iter().find(|g| path_key(&g.cwd) == path_key("C:/repo-alpha")).unwrap();
    assert_eq!(repo.sessions.len(), 1, "the same chat appears exactly once");
    let merged = &repo.sessions[0];
    assert_eq!(merged.cwd, "C:/repo-alpha", "the live row inherits its saved chat's real repository");
    assert_eq!(merged.session_id, "live-sid", "the live daemon identity is canonical");
    assert_eq!(merged.active_session_id.as_deref(), Some("daemon-1"));
    assert_eq!(merged.session_name.as_deref(), Some("Chat live-sid"));
    assert!(is_active_chat(merged));
    assert!(state.groups.iter().all(|g| path_key(&g.cwd) != path_key("")),
        "no separate recent/active group exists");
    assert_eq!(state.items().iter().filter(|i| matches!(i, Item::Session(_))).count(), 1);
    let mut partial = session("live-sid", "", "2026-09-24T04:00:00Z");
    partial.active_session_id = Some("daemon-1".into());
    partial.is_streaming = true;
    partial.session_file = Some("C:/sessions/alpha.jsonl".into());
    state.update(vec![partial], false);
    let repo = state.groups.iter().find(|g| path_key(&g.cwd) == path_key("C:/repo-alpha")).unwrap();
    assert_eq!(repo.sessions.len(), 1);
    assert_eq!(repo.sessions[0].cwd, "C:/repo-alpha", "partial refreshes keep the repository group");
    let mut current = session("b", "C:/repo-beta", "2026-09-24T05:00:00Z");
    current.active_session_id = Some("daemon-9".into());
    current.is_session_active = true;
    state.set_current(current.clone());
    let mut daemon_row = current.clone();
    daemon_row.cwd = String::new(); // daemon active rows arrive without cwd
    state.update(vec![daemon_row], true);
    let beta = state.groups.iter().find(|g| path_key(&g.cwd) == path_key("C:/repo-beta")).unwrap();
    assert_eq!(beta.sessions.len(), 1);
    assert_eq!(beta.sessions[0].cwd, "C:/repo-beta",
        "opening or switching chats never moves a chat out of its repository group");
    // an open chat whose live identity differs from its saved row still shows once
    let mut saved_old = session("", "C:/repo-gamma", "2026-09-24T06:00:00Z");
    saved_old.session_id = String::new();
    saved_old.session_file = Some("C:/sessions/gamma.jsonl".into());
    saved_old.session_name = Some("Old gamma".into());
    state.update(vec![saved_old], true);
    state.selected = Some(Item::Session(path_key("C:/sessions/gamma.jsonl")));
    let mut opened = session("gamma-live", "C:/repo-gamma", "2026-09-24T07:00:00Z");
    opened.active_session_id = Some("daemon-g".into());
    opened.is_session_active = true;
    opened.session_file = Some("C:/sessions/gamma.jsonl".into());
    opened.session_name = None;
    state.set_current(opened);
    assert_eq!(state.active.as_deref(), Some("gamma-live"));
    assert_eq!(state.selected, Some(Item::Session("gamma-live".into())));
    let gamma = state.groups.iter().find(|g| path_key(&g.cwd) == path_key("C:/repo-gamma")).unwrap();
    assert_eq!(gamma.sessions.len(), 1);
    assert_eq!(gamma.sessions[0].session_id, "gamma-live");
    assert_eq!(gamma.sessions[0].session_name.as_deref(), Some("Old gamma"));
    assert_eq!(state.items().iter().filter(|i| matches!(i, Item::Session(_))).count(), 1);
    #[cfg(windows)]
    {
        let mut alias = session("case-alias", "", "2026-09-24T08:00:00Z");
        alias.active_session_id = Some("daemon-c".into());
        alias.session_file = Some("c:/SESSIONS/GAMMA.JSONL".into());
        state.update(vec![alias], false);
        assert_eq!(state.active.as_deref(), Some("case-alias"));
        assert_eq!(state.selected, Some(Item::Session("case-alias".into())));
        assert_eq!(state.items().iter().filter(|i| matches!(i, Item::Session(_))).count(), 1,
            "case-insensitive session_file aliases are the same chat on Windows");
    }
}

#[test]
fn ui011_unknown_location_chats_are_never_fabricated_or_merged() {
    let temp = tempfile::tempdir().unwrap();
    let mut state = State::new("C:/initial-repo".into(), temp.path().join("folders.json"));
    let mut x = session("x", "", "2026-09-24T01:00:00Z");
    x.active_session_id = Some("dx".into());
    x.is_streaming = true;
    let mut y = session("y", "", "2026-09-24T02:00:00Z");
    y.active_session_id = Some("dy".into());
    state.update(vec![x, y], true);
    assert_eq!(state.items().iter().filter(|i| matches!(i, Item::Session(_))).count(), 2,
        "both chats stay visible exactly once");
    assert!(state.groups.iter().filter(|g| !g.sessions.is_empty())
        .all(|g| path_key(&g.cwd) == path_key("")),
        "chats without a known repository share one honest group");
    let unplaced = state.groups.iter().find(|g| !g.sessions.is_empty()).unwrap();
    assert_eq!(unplaced.label, "Unknown location");
    assert_eq!(unplaced.sessions.len(), 2);
    assert!(unplaced.sessions.iter().all(|s| s.cwd.is_empty()));
    let identities: HashSet<String> = unplaced.sessions.iter().map(identity).collect();
    assert_eq!(identities.len(), 2, "unrelated chats are never merged");
    assert!(state.groups.iter().any(|g| path_key(&g.cwd) == path_key("C:/initial-repo") && g.sessions.is_empty()),
        "chats are never fabricated into the initial folder");
}

#[test]
fn ui012_activity_predicate_reads_daemon_facts_and_ignores_saved_rows() {
    let base = session("s", "C:/repo", "2026-09-24T00:00:00Z");
    assert!(!is_active_chat(&base), "a saved idle chat is not active");
    let mut variant = base.clone(); variant.active_session_id = Some("d".into());
    assert!(is_active_chat(&variant), "daemon residency is activity");
    let mut variant = base.clone(); variant.is_session_active = true;
    assert!(is_active_chat(&variant), "the session's own live flag is activity");
    let mut variant = base.clone(); variant.is_streaming = true;
    assert!(is_active_chat(&variant), "streaming is activity");
    let mut variant = base.clone(); variant.is_compacting = true;
    assert!(is_active_chat(&variant), "compaction is activity");
    let mut variant = base.clone(); variant.activity = "working".into();
    assert!(is_active_chat(&variant), "foreground work is activity");
    let mut variant = base.clone(); variant.has_running_rlm_children = Some(true);
    assert!(is_active_chat(&variant), "delegated child work is activity");
    let mut variant = base.clone(); variant.has_running_rlm_children = Some(false);
    assert!(!is_active_chat(&variant));
    let mut variant = base; variant.activity = "idle".into();
    assert!(!is_active_chat(&variant), "an idle row is not marked active");
}

#[test]
fn ui012_every_actually_active_chat_keeps_its_asterisk_including_noncurrent() {
    init_theme(Some("neon"), false);
    let temp = tempfile::tempdir().unwrap();
    let state = Rc::new(RefCell::new(State::new("/nowhere".into(), temp.path().join("folders.json"))));
    let mut working = session("w", "C:/repo", "2026-09-24T03:00:00Z");
    working.active_session_id = Some("dw".into());
    working.activity = "working".into();
    let mut streaming = session("s", "C:/repo", "2026-09-24T02:00:00Z");
    streaming.is_streaming = true;
    let saved = session("idle", "C:/repo", "2026-09-24T01:00:00Z");
    state.borrow_mut().update(vec![working, streaming, saved], true);
    state.borrow_mut().active = Some("idle".into()); // the open chat is the saved, inactive one
    state.borrow_mut().selected = Some(Item::Session("idle".into()));
    let painted = strip_ansi(&Pane(state.clone()).render_with_height(40.0, 24).join("\n"));
    assert!(painted.contains("* Chat w"), "a resident working chat keeps its asterisk");
    assert!(painted.contains("* Chat s"), "a streaming chat keeps its asterisk");
    assert!(!painted.contains("* Chat idle"), "a saved inactive chat is not marked active");
    state.borrow_mut().active = Some("w".into());
    let painted = strip_ansi(&Pane(state.clone()).render_with_height(40.0, 24).join("\n"));
    assert!(painted.contains("* Chat s"), "switching the viewed chat keeps other asterisks");
    assert!(painted.contains("* Chat w"));
}

#[test]
fn ui013_current_chat_name_is_red_and_independent_of_keyboard_highlight() {
    init_theme(Some("neon"), false);
    let temp = tempfile::tempdir().unwrap();
    let state = Rc::new(RefCell::new(State::new("/nowhere".into(), temp.path().join("folders.json"))));
    state.borrow_mut().update(vec![
        session("one", "C:/repo", "2026-09-24T01:00:00Z"),
        session("two", "C:/repo", "2026-09-24T02:00:00Z"),
    ], true);
    state.borrow_mut().active = Some("one".into());
    state.borrow_mut().selected = Some(Item::Session("two".into()));
    let selection = theme().get_selection_background_color();
    let one = format!("{:<38}", "    Chat one");
    let two = format!("{:<38}", "    Chat two");
    let folder = format!("{:<38}", "v repo");
    let red_one = format!("{}{}", theme().fg("text", "    "), theme().fg("error", &one[4..]));
    let selected_red_one = format!("{}{}", theme().fg("accent", "    "), theme().fg("error", &one[4..]));
    let selected_red_two = format!("{}{}", theme().fg("accent", "    "), theme().fg("error", &two[4..]));
    let painted = Pane(state.clone()).render_with_height(40.0, 24).join("\n");
    assert!(painted.contains(&red_one), "the open chat's name is red while another row is highlighted");
    assert!(painted.contains(&selection(&theme().fg("accent", &two))),
        "the highlighted row keeps its unchanged selection colours");
    state.borrow_mut().selected = Some(Item::Session("one".into()));
    let painted = Pane(state.clone()).render_with_height(40.0, 24).join("\n");
    assert!(painted.contains(&selection(&selected_red_one)),
        "the open chat keeps its red name and selection background when highlighted");
    state.borrow_mut().selected = Some(Item::Session("two".into()));
    let painted = Pane(state.clone()).render_with_height(40.0, 24).join("\n");
    assert!(painted.contains(&red_one), "keyboard navigation alone never turns a name red");
    state.borrow_mut().active = Some("two".into());
    let painted = Pane(state.clone()).render_with_height(40.0, 24).join("\n");
    assert!(painted.contains(&selection(&selected_red_two)), "opening another chat moves the red name to it");
    assert!(!painted.contains(&theme().fg("error", &one[4..])));
    assert!(painted.contains(&theme().fg("text", &one)), "other session rows keep their colour");
    assert!(painted.contains(&theme().fg("muted", &folder)), "unselected folders keep their colour");
    state.borrow_mut().selected = Some(Item::Folder(path_key("C:/repo")));
    let painted = Pane(state.clone()).render_with_height(40.0, 24).join("\n");
    assert!(painted.contains(&selection(&theme().fg("accent", &folder))),
        "highlighted folders keep their original accent and selection background");
    state.borrow_mut().focused = false;
    let painted = Pane(state.clone()).render_with_height(40.0, 24).join("\n");
    assert!(painted.contains(&selection(&theme().fg("text", &folder))),
        "unfocused folder selection keeps its original text colour");
    assert!(painted.contains(&theme().fg("error", &two[4..])), "the current name stays red without sidebar focus");
}

#[test]
fn ui009_selected_full_paths_stay_available_beyond_pane_width() {
    let temp = tempfile::tempdir().unwrap();
    let mut state = State::new("/nowhere".into(), temp.path().join("folders.json"));
    let long_repo = "C:/very/deep/directory-tree/with-a-long-repository-name";
    let mut chat = session("deep", long_repo, "2026-09-24T01:00:00Z");
    chat.session_file = Some("C:/sessions/deep-chat-file.jsonl".into());
    state.update(vec![chat], true);
    state.selected = Some(Item::Session("deep".into()));
    assert_eq!(state.selected_cwd().as_deref(), Some(long_repo));
    assert_eq!(state.selected_session_file().as_deref(), Some("C:/sessions/deep-chat-file.jsonl"));
    state.selected = Some(Item::Folder(path_key(long_repo)));
    assert_eq!(state.selected_cwd().as_deref(), Some(long_repo));
    assert_eq!(state.selected_session_file(), None, "folders have no chat file");
}

#[test]
fn ui006_focus_selects_a_running_child_of_the_current_chat_only() {
    let temp = tempfile::tempdir().unwrap();
    let mut state = State::new("/nowhere".into(), temp.path().join("folders.json"));
    let mut parent = session("p", "C:/repo", "2026-09-24T05:00:00Z");
    parent.active_session_id = Some("dp".into());
    parent.session_file = Some("C:/sessions/p.jsonl".into());
    let mut running = session("c1", "C:/repo", "2026-09-24T04:00:00Z");
    running.runtime_kind = Some("subagent".into());
    running.lifecycle = "live".into();
    running.parent_active_session_id = Some("dp".into());
    running.active_session_id = Some("dc1".into());
    running.is_streaming = true;
    let mut resident = session("c2", "C:/repo", "2026-09-24T06:00:00Z");
    resident.runtime_kind = Some("subagent".into());
    resident.lifecycle = "live".into();
    resident.parent_session_id = Some("p".into());
    resident.active_session_id = Some("dc2".into());
    let mut stranger_active = session("r1", "C:/repo", "2026-09-24T03:00:00Z");
    stranger_active.active_session_id = Some("dr".into());
    let mut other_child = session("c3", "C:/repo", "2026-09-24T07:00:00Z");
    other_child.runtime_kind = Some("subagent".into());
    other_child.lifecycle = "live".into();
    other_child.parent_active_session_id = Some("dz".into());
    other_child.active_session_id = Some("dc3".into());
    let mut saved_child = session("c4", "C:/repo", "2026-09-24T08:00:00Z");
    saved_child.runtime_kind = Some("subagent".into());
    saved_child.lifecycle = "live".into();
    saved_child.parent_session_path = Some("C:/sessions/p.jsonl".into());
    state.update(vec![parent, running, resident, stranger_active, other_child, saved_child], true);
    state.active = Some("p".into());
    let children = state.current_running_children();
    let ids: Vec<&str> = children.iter().map(|c| c.session_id.as_str()).collect();
    assert_eq!(ids, ["c2", "c1"],
        "only the current chat's actually-active direct children, newest first");
    state.selected = Some(Item::Session("r1".into()));
    assert!(state.focus_current_running_child());
    let aimed = state.selected_session().unwrap();
    assert_eq!(aimed.session_id, "c2", "focus moves off an unrelated row onto a running child");
    assert_eq!(state.selected, Some(Item::Session("c2".into())));
    state.selected = Some(Item::Session("c1".into()));
    assert!(state.focus_current_running_child());
    let aimed = state.selected_session().unwrap();
    assert_eq!(aimed.session_id, "c1", "a selection already on a running child is kept");
    assert_eq!(state.selected, Some(Item::Session("c1".into())));
    state.active = None;
    assert!(!state.focus_current_running_child());
    assert_eq!(state.selected, Some(Item::Session("c1".into())), "without an open chat nothing is re-aimed");
    state.active = Some("p".into());
    state.collapsed.insert(path_key("C:/repo"));
    state.selected = Some(Item::Session("r1".into()));
    assert!(state.focus_current_running_child());
    assert!(state.items().contains(&Item::Session("c2".into())),
        "a collapsed child group is expanded so the aimed row is visible");
    state.collapsed.insert(path_key("C:/repo"));
    assert!(state.focus_current_running_child());
    assert!(state.items().contains(&Item::Session("c2".into())),
        "an already-selected child is also revealed when its group is collapsed");
}

#[test]
fn ui008_single_letter_projects_and_generic_checkout_ancestors_keep_their_names() {
    let temp = tempfile::tempdir().unwrap();
    let mut state = State::new("/nowhere".into(), temp.path().join("folders.json"));
    state.update(vec![
        session("x", "C:/work/x/source", "2026-09-24T01:00:00Z"),
        session("y", "C:/work/y/src", "2026-09-24T02:00:00Z"),
        session("main", "C:/work/main/repo/code/workspace", "2026-09-24T03:00:00Z"),
        session("z", "C:/work/z", "2026-09-24T04:00:00Z"),
    ], true);
    assert_eq!(group_label(&state, "C:/work/x/source"), "x");
    assert_eq!(group_label(&state, "C:/work/y/src"), "y");
    assert_eq!(group_label(&state, "C:/work/main/repo/code/workspace"), "main");
    assert_eq!(group_label(&state, "C:/work/z"), "z");
    assert!(is_generic_segment("C:"));
    assert!(!is_generic_segment("C"), "a letter without a colon is a real project name");
}

#[test]
fn ui011_same_chat_rekeys_current_and_selection_across_full_and_partial_refreshes() {
    init_theme(Some("neon"), false);
    let temp = tempfile::tempdir().unwrap();
    let mut state = State::new("/nowhere".into(), temp.path().join("folders.json"));
    let mut saved = session("legacy", "C:/project/source", "2026-09-24T01:00:00Z");
    saved.session_id.clear();
    saved.session_file = Some("C:/sessions/parent.jsonl".into());
    saved.session_name = Some("Parent".into());
    let mut child = session("child", "C:/project/source", "2026-09-24T02:00:00Z");
    child.runtime_kind = Some("subagent".into());
    child.lifecycle = "live".into();
    child.active_session_id = Some("dc".into());
    child.parent_session_path = saved.session_file.clone();
    let mut stranger = child.clone();
    stranger.id = "stranger".into();
    stranger.session_id = "stranger".into();
    stranger.active_session_id = Some("ds".into());
    stranger.parent_session_path = Some("C:/sessions/unrelated.jsonl".into());
    stranger.parent_session_id = Some(String::new());
    state.update(vec![saved.clone(), child.clone(), stranger.clone()], true);
    state.set_current(saved.clone());
    assert_eq!(state.current_running_children().iter().map(identity).collect::<Vec<_>>(), ["child"],
        "an empty legacy session id cannot link an unrelated child");
    state.selected = Some(Item::Session(identity(&saved)));
    let mut live = session("live", "", "2026-09-24T03:00:00Z");
    live.active_session_id = Some("dp".into());
    live.session_file = saved.session_file.clone();
    live.session_name = None;
    state.update(vec![saved, live.clone(), child.clone(), stranger.clone()], true);
    assert_eq!(state.active.as_deref(), Some("live"));
    assert_eq!(state.selected, Some(Item::Session("live".into())));
    assert_eq!(state.selected_cwd().as_deref(), Some("C:/project/source"));
    assert_eq!(state.sessions.len(), 3, "saved and live twins occupy one row");
    assert_eq!(state.current_running_children().iter().map(identity).collect::<Vec<_>>(), ["child"]);
    let state = Rc::new(RefCell::new(state));
    let painted = Pane(state.clone()).render_with_height(40.0, 24).join("\n");
    let red_name = theme().fg("error", &format!("{:<34}", "Parent"));
    assert!(painted.contains(&red_name), "the current name stays red after canonicalization");
    {
        let mut state = state.borrow_mut();
        live.session_id = "reattached".into();
        state.update(vec![live.clone(), child.clone(), stranger.clone()], false);
        assert_eq!(state.active.as_deref(), Some("reattached"));
        assert_eq!(state.selected, Some(Item::Session("reattached".into())));
        assert_eq!(state.selected_session_file().as_deref(), Some("C:/sessions/parent.jsonl"));
        assert_eq!(state.current_running_children().iter().map(identity).collect::<Vec<_>>(), ["child"]);
        live.cwd = "C:/project/source".into();
        live.session_file = None;
        state.update(vec![live.clone(), child.clone(), stranger.clone()], true);
        assert_eq!(state.selected_session_file().as_deref(), Some("C:/sessions/parent.jsonl"),
            "a same-id full refresh retains the known file even when cwd is present");
        assert_eq!(state.current_running_children().iter().map(identity).collect::<Vec<_>>(), ["child"],
            "file-linked direct children remain reachable after refresh");
        state.selected = Some(Item::Session("stranger".into()));
        live.session_id = "latest".into();
        live.session_file = Some("C:/sessions/parent.jsonl".into());
        state.update(vec![live, child, stranger], false);
        assert_eq!(state.active.as_deref(), Some("latest"));
        assert_eq!(state.selected, Some(Item::Session("stranger".into())),
            "rekeying the current chat does not move independent keyboard selection");
        assert!(state.focus_current_running_child());
        assert_eq!(state.selected_session().unwrap().session_id, "child");
    }
    let painted = Pane(state.clone()).render_with_height(40.0, 24).join("\n");
    assert!(painted.contains(&red_name), "focusing a child does not move the red current-name indicator");
}

#[test]
fn ui011_identity_rekey_never_infers_a_chat_from_its_name_or_repository() {
    init_theme(Some("neon"), false);
    let temp = tempfile::tempdir().unwrap();
    let state = Rc::new(RefCell::new(State::new("C:/repo".into(), temp.path().join("folders.json"))));
    let mut old = session("old", "C:/repo", "2026-09-24T01:00:00Z");
    old.session_name = Some("Shared name".into());
    old.session_file = Some("C:/sessions/old.jsonl".into());
    state.borrow_mut().set_current(old.clone());
    state.borrow_mut().selected = Some(Item::Session("old".into()));
    let mut unrelated = old;
    unrelated.id = "unrelated".into();
    unrelated.session_id = "unrelated".into();
    unrelated.session_file = Some("C:/sessions/unrelated.jsonl".into());
    unrelated.active_session_id = Some("du".into());
    state.borrow_mut().update(vec![unrelated], true);
    assert_eq!(state.borrow().active.as_deref(), Some("old"));
    assert_ne!(state.borrow().selected, Some(Item::Session("unrelated".into())));
    assert!(state.borrow().current_running_children().is_empty());
    assert!(!state.borrow_mut().focus_current_running_child());
    let painted = Pane(state).render_with_height(40.0, 24).join("\n");
    assert!(!painted.contains(&theme().fg("error", &format!("{:<34}", "Shared name"))),
        "an unrelated replacement is never painted as the open chat");
}

#[test]
fn ui012_partial_active_refresh_clears_stale_delegated_activity() {
    init_theme(Some("neon"), false);
    let temp = tempfile::tempdir().unwrap();
    let state = Rc::new(RefCell::new(State::new("C:/repo".into(), temp.path().join("folders.json"))));
    let mut old = session("old", "C:/repo", "2026-09-24T01:00:00Z");
    old.active_session_id = Some("do".into());
    old.is_session_active = true;
    old.is_streaming = true;
    old.is_compacting = true;
    old.activity = "working".into();
    old.has_running_rlm_children = Some(true);
    state.borrow_mut().set_current(old);
    let mut live = session("live", "C:/repo", "2026-09-24T02:00:00Z");
    live.has_running_rlm_children = Some(true);
    state.borrow_mut().update(vec![live], false);
    {
        let state = state.borrow();
        let old = state.sessions.iter().find(|row| row.session_id == "old").unwrap();
        assert!(!is_active_chat(old));
        assert_eq!(old.has_running_rlm_children, None);
        let live = state.sessions.iter().find(|row| row.session_id == "live").unwrap();
        assert!(is_active_chat(live), "fresh delegated activity is still authoritative");
        assert_eq!(state.active.as_deref(), Some("old"), "open-chat identity is not activity");
    }
    let painted = strip_ansi(&Pane(state.clone()).render_with_height(40.0, 24).join("\n"));
    assert!(!painted.contains("* Chat old"), "the omitted inactive row loses its stale asterisk");
    assert!(painted.contains("* Chat live"));
    state.borrow_mut().update(Vec::new(), false);
    assert!(state.borrow().sessions.iter().all(|row| !is_active_chat(row)));
}

#[test]
fn ui006_opening_a_child_preserves_its_known_direct_parent_linkage() {
    let temp = tempfile::tempdir().unwrap();
    let mut state = State::new("C:/repo".into(), temp.path().join("folders.json"));
    let mut parent = session("parent", "C:/repo", "2026-09-24T01:00:00Z");
    parent.active_session_id = Some("dp".into());
    let mut child = session("child", "C:/repo", "2026-09-24T02:00:00Z");
    child.active_session_id = Some("dc".into());
    child.session_file = Some("C:/sessions/child.jsonl".into());
    child.runtime_kind = Some("subagent".into());
    child.lifecycle = "live".into();
    child.parent_active_session_id = Some("dp".into());
    state.update(vec![parent.clone(), child.clone()], true);
    state.set_current(parent.clone());
    assert!(state.focus_current_running_child());
    let mut opened = session("child-open", "C:/repo", "2026-09-24T03:00:00Z");
    opened.active_session_id = Some("dc".into());
    opened.session_file = child.session_file;
    opened.is_session_active = true;
    state.set_current(opened.clone());
    assert_eq!(state.selected, Some(Item::Session("child-open".into())));
    state.set_current(parent.clone());
    assert_eq!(state.current_running_children().iter().map(identity).collect::<Vec<_>>(), ["child-open"],
        "the host's minimal connection snapshot does not erase proven child linkage");
    state.update(vec![parent.clone(), opened.clone()], false);
    assert_eq!(state.current_running_children().len(), 1);
    state.update(vec![parent, opened], true);
    assert!(state.focus_current_running_child(), "full refreshes also retain omitted linkage for the same child");
    assert_eq!(state.selected_session().unwrap().session_id, "child-open");
}
