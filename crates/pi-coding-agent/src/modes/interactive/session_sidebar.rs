//! Session navigation is a local projection. Browsing never attaches a worker.
use std::cell::RefCell;
use std::collections::{BTreeMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use pi_tui::components::input::Input;
use pi_tui::keybindings::get_keybindings;
use pi_tui::tui::{Component, Focusable};
use pi_tui::utils::{strip_ansi, truncate_to_width, wrap_text_with_ansi};
use serde::{Deserialize, Serialize};

use crate::modes::daemon::daemon_session_list::SessionSummary;
use super::theme::theme::theme;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Item {
    Folder(String),
    Session(String),
}

#[derive(Clone, Debug)]
pub(crate) struct Group {
    pub cwd: String,
    pub sessions: Vec<SessionSummary>,
    activity: i64,
}

pub(crate) fn path_key(path: &str) -> String {
    let path = path.strip_prefix(r"\\?\").unwrap_or(path).replace('\\', "/");
    let path = path.trim_end_matches('/');
    if cfg!(windows) { path.to_lowercase() } else { path.to_string() }
}

pub(crate) fn identity(session: &SessionSummary) -> String {
    if !session.session_id.is_empty() { session.session_id.clone() }
    else { session.session_file.as_deref().map(path_key).unwrap_or_else(|| session.id.clone()) }
}

fn activity(session: &SessionSummary) -> i64 {
    [&session.last_activity_at, &session.modified, &session.created].into_iter()
        .filter_map(|value| value.as_deref())
        .filter_map(|value| chrono::DateTime::parse_from_rfc3339(value).ok())
        .map(|date| date.timestamp_millis()).max().unwrap_or(0)
}

pub(crate) fn label(session: &SessionSummary) -> String {
    session.session_name.as_deref().filter(|s| !s.trim().is_empty())
        .or(session.first_message.as_deref().filter(|s| !s.trim().is_empty()))
        .unwrap_or("Untitled chat").to_string()
}

fn clean(value: &str) -> String {
    strip_ansi(value).chars().map(|c| if c.is_control() { ' ' } else { c }).collect()
}

#[derive(Default, Serialize, Deserialize)]
struct RememberedFolders {
    #[serde(default)]
    folders: Vec<String>,
}

fn load_folders(path: &Path) -> Result<Vec<String>, String> {
    match std::fs::read_to_string(path) {
        Ok(text) => serde_json::from_str::<RememberedFolders>(&text).map(|s| s.folders)
            .map_err(|error| format!("Cannot read remembered folders: {error}")),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(error) => Err(format!("Cannot read remembered folders: {error}")),
    }
}

pub(crate) fn validate_folder(value: &str) -> Result<String, String> {
    let value = value.trim().trim_matches('"');
    if value.is_empty() { return Err("Enter an existing folder/repo path.".into()); }
    let path = Path::new(value);
    if !path.is_absolute() { return Err("Enter a full, absolute folder/repo path.".into()); }
    let metadata = std::fs::metadata(path).map_err(|_| "Folder is missing or inaccessible. Check the path and permissions.".to_string())?;
    if !metadata.is_dir() { return Err("This path is a file. Enter an existing directory.".into()); }
    std::fs::read_dir(path).map_err(|_| "Cannot access this folder. Check its permissions.".to_string())?;
    let canonical = std::fs::canonicalize(path).map_err(|_| "Cannot resolve this folder path.".to_string())?;
    let path = canonical.to_string_lossy();
    // Preserve UNC paths while removing the Windows extended-length display prefix.
    Ok(if let Some(unc) = path.strip_prefix(r"\\?\UNC\") { format!(r"\\{unc}") }
        else { path.strip_prefix(r"\\?\").unwrap_or(&path).to_string() })
}

#[cfg(test)]
pub(crate) fn remember_folder(store: &Path, value: &str) -> Result<Vec<String>, String> {
    let folder = validate_folder(value)?;
    persist_folder(store, folder)
}

pub(crate) fn persist_folder(store: &Path, folder: String) -> Result<Vec<String>, String> {
    let mut folders = load_folders(store)?;
    if !folders.iter().any(|s| path_key(s) == path_key(&folder)) { folders.push(folder); }
    let parent = store.parent().ok_or("Folder settings path has no parent")?;
    std::fs::create_dir_all(parent).map_err(|e| format!("Cannot save folders: {e}"))?;
    let mut file = tempfile::NamedTempFile::new_in(parent).map_err(|e| format!("Cannot save folders: {e}"))?;
    file.write_all(serde_json::to_string_pretty(&RememberedFolders { folders: folders.clone() })
        .map_err(|e| e.to_string())?.as_bytes()).map_err(|e| e.to_string())?;
    file.as_file().sync_all().map_err(|e| e.to_string())?;
    file.persist(store).map_err(|e| format!("Cannot save folders: {e}"))?;
    Ok(folders)
}

pub(crate) struct State {
    pub groups: Vec<Group>,
    pub selected: Option<Item>,
    pub active: Option<String>,
    pub focused: bool,
    pub status: String,
    pub busy: bool,
    pub store: PathBuf,
    remembered: Vec<String>,
    initial_cwd: String,
    collapsed: HashSet<String>,
    sessions: Vec<SessionSummary>,
    top: usize,
    visible: usize,
}

impl State {
    pub fn new(cwd: String, store: PathBuf) -> Self {
        let (remembered, status) = match load_folders(&store) {
            Ok(folders) => (folders, String::new()),
            Err(error) => (Vec::new(), error),
        };
        let mut state = Self { groups: Vec::new(), selected: Some(Item::Folder(path_key(&cwd))),
            active: None, focused: true, status, busy: false, store, remembered,
            initial_cwd: cwd, collapsed: HashSet::new(), sessions: Vec::new(), top: 0, visible: 1 };
        state.rebuild();
        state
    }

    pub fn update(&mut self, sessions: Vec<SessionSummary>, full: bool) {
        if full { self.sessions = sessions; }
        else {
            for saved in &mut self.sessions {
                saved.active_session_id = None;
                saved.is_session_active = false;
                saved.is_streaming = false;
                saved.is_compacting = false;
                saved.activity = "idle".into();
            }
            for session in sessions { self.upsert(session); }
        }
        self.rebuild();
    }

    fn upsert(&mut self, session: SessionSummary) {
        if let Some(old) = self.sessions.iter_mut().find(|s| identity(s) == identity(&session)) { *old = session; }
        else { self.sessions.push(session); }
    }

    pub fn set_current(&mut self, session: SessionSummary) {
        let changed = self.active.as_ref() != Some(&identity(&session));
        self.active = Some(identity(&session));
        self.upsert(session);
        self.rebuild();
        if changed { self.focused = false; }
    }

    pub fn folders_added(&mut self, folders: Vec<String>, value: &str) {
        self.remembered = folders;
        self.rebuild();
        self.selected = Some(Item::Folder(path_key(value)));
        self.status = "Folder added. New creates a chat here.".into();
    }

    fn rebuild(&mut self) {
        let mut groups = BTreeMap::<String, Group>::new();
        for cwd in self.remembered.iter().chain(std::iter::once(&self.initial_cwd)) {
            groups.entry(path_key(cwd)).or_insert_with(|| Group { cwd: cwd.clone(), sessions: Vec::new(), activity: 0 });
        }
        for session in &self.sessions {
            let group = groups.entry(path_key(&session.cwd)).or_insert_with(|| Group { cwd: session.cwd.clone(), sessions: Vec::new(), activity: 0 });
            group.activity = group.activity.max(activity(session));
            if !group.sessions.iter().any(|s| identity(s) == identity(session)) { group.sessions.push(session.clone()); }
        }
        self.groups = groups.into_values().collect();
        for group in &mut self.groups {
            group.sessions.sort_by(|a, b| activity(b).cmp(&activity(a)).then_with(|| identity(a).cmp(&identity(b))));
        }
        self.groups.sort_by(|a, b| b.activity.cmp(&a.activity).then_with(|| path_key(&a.cwd).cmp(&path_key(&b.cwd))));
        let items = self.items();
        if self.selected.as_ref().is_none_or(|s| !items.contains(s)) { self.selected = items.first().cloned(); }
    }

    pub fn items(&self) -> Vec<Item> {
        let mut items = Vec::new();
        for group in &self.groups {
            let key = path_key(&group.cwd);
            items.push(Item::Folder(key.clone()));
            if !self.collapsed.contains(&key) { items.extend(group.sessions.iter().map(|s| Item::Session(identity(s)))); }
        }
        items
    }

    pub fn selected_session(&self) -> Option<SessionSummary> {
        let Item::Session(id) = self.selected.as_ref()? else { return None; };
        self.sessions.iter().find(|s| identity(s) == *id).cloned()
    }

    pub fn selected_cwd(&self) -> Option<String> {
        match self.selected.as_ref()? {
            Item::Folder(key) => self.groups.iter().find(|g| path_key(&g.cwd) == *key).map(|g| g.cwd.clone()),
            Item::Session(_) => self.selected_session().map(|s| s.cwd),
        }
    }

    pub fn move_by(&mut self, delta: isize) {
        let items = self.items();
        if items.is_empty() { return; }
        let current = self.selected.as_ref().and_then(|s| items.iter().position(|i| i == s)).unwrap_or(0);
        let next = (current as isize + delta).clamp(0, items.len() as isize - 1) as usize;
        self.selected = Some(items[next].clone());
    }

    pub fn confirm(&mut self) -> Option<SessionSummary> {
        match self.selected.clone()? {
            Item::Folder(key) => {
                if !self.collapsed.remove(&key) { self.collapsed.insert(key); }
                None
            }
            Item::Session(_) => self.selected_session(),
        }
    }

    /// Mouse hover/press selects only. It never opens a conversation.
    pub fn pointer(&mut self, row: usize) {
        if row > 0 && row <= self.visible {
            if let Some(item) = self.items().get(self.top + row - 1) { self.selected = Some(item.clone()); }
        }
    }
}

pub(crate) fn key_label(action: &str) -> String {
    let keys = get_keybindings().get_keys(action);
    if keys.is_empty() { return "Unbound".into(); }
    keys.iter().map(|key| key.split('+').map(|part| match part {
        "ctrl" => "Ctrl".into(), "alt" => "Alt".into(), "shift" => "Shift".into(),
        "left" => "Left".into(), "right" => "Right".into(), "enter" => "Enter".into(),
        "escape" => "Esc".into(), other => other.to_uppercase(),
    }).collect::<Vec<String>>().join("+")).collect::<Vec<_>>().join("/")
}

pub(crate) struct Pane(pub Rc<RefCell<State>>);
impl Component for Pane {
    fn render(&mut self, width: f64) -> Vec<String> { self.render_with_height(width, 24) }
    fn render_with_height(&mut self, width: f64, height: usize) -> Vec<String> {
        let width = width.max(1.0) as usize;
        let inner = width.saturating_sub(2);
        let mut state = self.0.borrow_mut();
        let mut lines = vec![theme().fg(if state.focused { "accent" } else { "muted" },
            if state.focused { "> Sessions" } else { "  Sessions" })];
        let legend = [ ("app.agents.new", "New"), ("app.sidebar.addFolder", "Add folder"),
            ("app.agents.rename", "Rename"), ("app.agents.delete", "Delete / stop"),
            ("tui.select.confirm", "Open / toggle"), ("app.sidebar.chat", "Chat") ];
        let list_height = height.saturating_sub(legend.len() + 4).max(1);
        state.visible = list_height;
        let items = state.items();
        let selected = state.selected.as_ref().and_then(|s| items.iter().position(|i| i == s)).unwrap_or(0);
        if selected < state.top { state.top = selected; }
        if selected >= state.top + list_height { state.top = selected + 1 - list_height; }
        for item in items.iter().skip(state.top).take(list_height) {
            let text = match item {
                Item::Folder(key) => {
                    let group = state.groups.iter().find(|g| path_key(&g.cwd) == *key).unwrap();
                    let basename = group.cwd.trim_end_matches(['/', '\\']).rsplit(['/', '\\']).next().unwrap_or(&group.cwd);
                    format!("{} {}", if state.collapsed.contains(key) { ">" } else { "v" }, clean(basename))
                }
                Item::Session(id) => {
                    let session = state.sessions.iter().find(|s| identity(s) == *id).unwrap();
                    format!("  {} {}", if state.active.as_ref() == Some(id) { "*" }
                        else if session.activity == "working" { ">" } else { " " }, clean(&label(session)))
                }
            };
            let row = truncate_to_width(&text, inner as f64, "…", true);
            lines.push(if state.selected.as_ref() == Some(item) {
                (theme().get_selection_background_color())(&theme().fg(if state.focused { "accent" } else { "text" }, &row))
            } else { theme().fg(if matches!(item, Item::Folder(_)) { "muted" } else { "text" }, &row) });
        }
        while lines.len() < height.saturating_sub(legend.len() + 3) { lines.push(String::new()); }
        lines.push(theme().fg("dim", &state.selected_cwd().map(|s| clean(&s)).unwrap_or_default()));
        lines.push(theme().fg(if state.busy { "muted" } else { "dim" }, &clean(&state.status)));
        lines.push(theme().fg("borderMuted", &"─".repeat(inner)));
        lines.extend(legend.iter().map(|(key, text)| theme().fg("dim", &format!("{} {text}", key_label(key)))));
        lines.truncate(height);
        while lines.len() < height { lines.push(String::new()); }
        lines.into_iter().map(|line| format!("{} {}", truncate_to_width(&line, inner as f64, "…", true), theme().fg("borderMuted", "│"))).collect()
    }
    fn invalidate(&mut self) {}
}

#[derive(Clone)]
pub(crate) enum DialogKind { AddFolder, Rename(SessionSummary), Delete(SessionSummary) }

pub(crate) struct Dialog {
    pub kind: DialogKind,
    pub input: Input,
    pub error: String,
    pub busy: bool,
}
impl Dialog {
    pub fn new(kind: DialogKind) -> Self {
        let mut input = Input::new();
        input.set_focused(true);
        if let DialogKind::Rename(session) = &kind { input.handle_input(&label(session)); }
        Self { kind, input, error: String::new(), busy: false }
    }
}
impl Component for Dialog {
    fn render(&mut self, width: f64) -> Vec<String> {
        let inner = (width as usize).saturating_sub(4).max(1);
        let (title, field, help) = match &self.kind {
            DialogKind::AddFolder => ("Add folder", "Folder/repo path", "Existing directory only. This does not create a chat or directory.".to_string()),
            DialogKind::Rename(_) => ("Rename chat", "Chat name", "Only the saved chat name changes.".to_string()),
            DialogKind::Delete(session) => if session.active_session_id.is_some() {
                ("Stop session?", "", format!("Stop {} and keep its history. Repository files are not changed.", clean(&label(session))))
            } else { ("Delete saved chat?", "", format!("Delete {} and its chat history. Repository files are not changed.", clean(&label(session)))) },
        };
        let mut lines = vec![theme().fg("accent", &format!(" {title}")), String::new()];
        lines.extend(wrap_text_with_ansi(&help, inner));
        if !field.is_empty() {
            lines.push(String::new()); lines.push(theme().fg("muted", field));
            lines.extend(self.input.render(inner as f64));
        }
        if !self.error.is_empty() { lines.extend(wrap_text_with_ansi(&theme().fg("error", &clean(&self.error)), inner)); }
        lines.push(String::new());
        lines.push(if self.busy { "Working…".into() } else { format!("{} Confirm  {} Cancel", key_label("tui.select.confirm"), key_label("tui.select.cancel")) });
        let border = theme().fg("border", &"─".repeat(inner));
        let mut output = vec![format!("┌{border}┐")];
        output.extend(lines.into_iter().map(|line| format!("│{}│", truncate_to_width(&line, inner as f64, "…", true))));
        output.push(format!("└{border}┘"));
        output
    }
    fn invalidate(&mut self) {}
}

#[cfg(test)]
#[path = "session_sidebar_tests.rs"]
mod tests;
