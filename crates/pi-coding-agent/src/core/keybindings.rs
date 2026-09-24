//! Port of packages/coding-agent/src/core/keybindings.ts

use indexmap::IndexMap;
use pi_tui::keybindings::{
    KeybindingDefinition, KeybindingsConfig as TuiKeybindingsConfig,
    KeybindingsManager as TuiKeybindingsManager,
};
use serde_json::{Map, Value};

/// Port of `interface AppKeybindings` (the `declare module` merge into the TUI
/// `Keybindings` interface). TypeScript keys that interface by action id, so the
/// port keeps the ids as data and `AppKeybinding` as the id type.
pub type AppKeybinding = &'static str;

/// `keyof AppKeybindings` in declaration order.
pub const APP_KEYBINDINGS: [AppKeybinding; 59] = [
    "app.interrupt",
    "app.clear",
    "app.input.clear",
    "app.shortcuts",
    "app.exit",
    "app.suspend",
    "app.model.select",
    "app.model.toggleScope",
    "app.configuration.previousTab",
    "app.tools.expand",
    "app.messages.expand",
    "app.edits.expand",
    "app.thinking.toggle",
    "app.subagents.focus",
    "app.heartbeats.open",
    "app.heartbeats.openSelected",
    "app.editor.external",
    "app.prompt.stash",
    "app.message.followUp",
    "app.message.navigateOlder",
    "app.message.navigateNewer",
    "app.message.moveEarlier",
    "app.message.moveLater",
    "app.clipboard.pasteImage",
    "app.clipboard.copyLoginUrl",
    "app.session.new",
    "app.session.tree",
    "app.session.fork",
    "app.session.resume",
    "app.agents.back",
    "app.agents.open",
    "app.modal.back",
    "app.agents.reply",
    "app.agents.new",
    "app.agents.delete",
    "app.agents.program",
    "app.agents.rename",
    "app.agents.inactiveCollapse",
    "app.agents.expand",
    "app.tree.foldOrUp",
    "app.tree.unfoldOrDown",
    "app.tree.editLabel",
    "app.tree.toggleLabelTimestamp",
    "app.models.save",
    "app.models.enableAll",
    "app.models.clearAll",
    "app.models.toggleProvider",
    "app.models.reorderUp",
    "app.models.reorderDown",
    "app.tree.filter.default",
    "app.tree.filter.noTools",
    "app.tree.filter.userOnly",
    "app.tree.filter.labeledOnly",
    "app.tree.filter.all",
    "app.tree.filter.cycleForward",
    "app.tree.filter.cycleBackward",
    // SHARED FILE EDIT (core/keybindings.rs, DEFAULT-only addition by jev-ui lane):
    // the `/jev` menu's explicit cancel action. It has NO default key of its own
    // (`default_keys: vec![]`), so Escape/Ctrl+C keep resolving through the TUI
    // `tui.select.cancel` / `app.interrupt` bindings and a user override of those
    // still cancels the dialog. The id exists so the intent is configurable and
    // named instead of hardcoded inside the component.
    "app.jev.cancel",
    "app.stats.toggleSubagents",
    "app.stats.nextSection",
];

/// `Keybinding` (the merged id type) and `KeyId`; the TUI crate models both as
/// strings already.
pub type Keybinding = String;
pub type KeyId = String;

/// Port of `KeybindingsConfig`.
///
/// TypeScript allows a single key or a list per action and keeps that shape
/// observable (`getResolvedBindings`, `getUserBindings`), so the port stores the
/// two shapes in an enum instead of normalizing to a list.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub enum KeybindingSetting {
    Single(String),
    List(Vec<String>),
}

impl KeybindingSetting {
    /// The declared key list for this setting (`Array.isArray` branch).
    pub fn keys(&self) -> Vec<String> {
        match self {
            KeybindingSetting::Single(key) => vec![key.clone()],
            KeybindingSetting::List(keys) => keys.clone(),
        }
    }

    /// JSON shape the TypeScript keeps: a single key stays a string.
    pub fn to_json(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }
}

/// `Record<string, KeyId | KeyId[] | undefined>`; an absent action is `None`.
pub type KeybindingsConfig = IndexMap<String, KeybindingSetting>;

/// `KEYBINDINGS` - `{ ...TUI_KEYBINDINGS, ...app entries }` as one definitions map.
///
/// Key order is observable (`orderKeybindingsConfig` walks `Object.keys`), so the
/// TUI entries are inserted first and the app entries follow in declaration order.
pub fn keybindings() -> IndexMap<String, KeybindingDefinition> {
    let mut map: IndexMap<String, KeybindingDefinition> = pi_tui::keybindings::tui_keybindings();
    let is_win32 = crate::utils::pi_user_agent::process_platform() == "win32";
    map.insert(
        "app.interrupt".to_string(),
        KeybindingDefinition {
            default_keys: vec![],
            default_keys_is_single: false,
            description: Some("Interrupt current operation".to_string()),
            default_key_scope: None,
        },
    );
    map.insert(
        "app.clear".to_string(),
        KeybindingDefinition {
            default_keys: vec![String::from("ctrl+c")],
            default_keys_is_single: true,
            description: Some("Interrupt current operation, then exit".to_string()),
            default_key_scope: None,
        },
    );
    map.insert(
        "app.input.clear".to_string(),
        KeybindingDefinition {
            default_keys: vec![String::from("escape")],
            default_keys_is_single: true,
            description: Some("Interrupt response or clear prompt".to_string()),
            default_key_scope: None,
        },
    );
    map.insert(
        "app.shortcuts".to_string(),
        KeybindingDefinition {
            default_keys: vec![String::from("?")],
            default_keys_is_single: true,
            description: Some("Show keyboard shortcuts".to_string()),
            default_key_scope: None,
        },
    );
    map.insert(
        "app.exit".to_string(),
        KeybindingDefinition {
            default_keys: vec![String::from("ctrl+d")],
            default_keys_is_single: true,
            description: Some("Exit when editor is empty".to_string()),
            default_key_scope: None,
        },
    );
    let (default_keys, default_keys_is_single) = if is_win32 { (vec![], false) } else { (vec![String::from("ctrl+z")], true) };
    map.insert(
        "app.suspend".to_string(),
        KeybindingDefinition {
            default_keys,
            default_keys_is_single,
            description: Some("Suspend to background".to_string()),
            default_key_scope: None,
        },
    );
    map.insert(
        "app.model.select".to_string(),
        KeybindingDefinition {
            default_keys: vec![String::from("ctrl+l")],
            default_keys_is_single: true,
            description: Some("Open model selector".to_string()),
            default_key_scope: None,
        },
    );
    map.insert(
        "app.model.toggleScope".to_string(),
        KeybindingDefinition {
            default_keys: vec![String::from("alt+s")],
            default_keys_is_single: true,
            description: Some("Toggle model selector scope".to_string()),
            default_key_scope: None,
        },
    );
    map.insert(
        "app.configuration.previousTab".to_string(),
        KeybindingDefinition {
            default_keys: vec![String::from("shift+tab")],
            default_keys_is_single: true,
            description: Some("Select previous configuration tab".to_string()),
            default_key_scope: None,
        },
    );
    map.insert(
        "app.tools.expand".to_string(),
        KeybindingDefinition {
            default_keys: vec![String::from("ctrl+o")],
            default_keys_is_single: true,
            description: Some("Toggle tool output".to_string()),
            default_key_scope: Some("editor".to_string()),
        },
    );
    map.insert(
        "app.messages.expand".to_string(),
        KeybindingDefinition {
            default_keys: vec![String::from("ctrl+p")],
            default_keys_is_single: true,
            description: Some("Toggle agent message expansion".to_string()),
            default_key_scope: Some("editor".to_string()),
        },
    );
    map.insert(
        "app.edits.expand".to_string(),
        KeybindingDefinition {
            default_keys: vec![String::from("ctrl+j")],
            default_keys_is_single: true,
            description: Some("Toggle edit diffs".to_string()),
            default_key_scope: Some("editor".to_string()),
        },
    );
    map.insert(
        "app.thinking.toggle".to_string(),
        KeybindingDefinition {
            default_keys: vec![String::from("ctrl+t")],
            default_keys_is_single: true,
            description: Some("Toggle thinking blocks".to_string()),
            default_key_scope: Some("editor".to_string()),
        },
    );
    map.insert(
        "app.subagents.focus".to_string(),
        KeybindingDefinition {
            default_keys: vec![String::from("alt+a")],
            default_keys_is_single: true,
            description: Some("Open child agents".to_string()),
            default_key_scope: None,
        },
    );
    map.insert(
        "app.heartbeats.open".to_string(),
        KeybindingDefinition {
            default_keys: vec![String::from("ctrl+r")],
            default_keys_is_single: true,
            description: Some("Manage heartbeats".to_string()),
            default_key_scope: None,
        },
    );
    map.insert(
        "app.heartbeats.openSelected".to_string(),
        KeybindingDefinition {
            default_keys: vec![String::from("right")],
            default_keys_is_single: true,
            description: Some("Open selected heartbeat".to_string()),
            default_key_scope: None,
        },
    );
    map.insert(
        "app.editor.external".to_string(),
        KeybindingDefinition {
            default_keys: vec![String::from("ctrl+g")],
            default_keys_is_single: true,
            description: Some("Open external editor".to_string()),
            default_key_scope: None,
        },
    );
    map.insert(
        "app.prompt.stash".to_string(),
        KeybindingDefinition {
            default_keys: vec![String::from("ctrl+s")],
            default_keys_is_single: true,
            description: Some("Stash or restore draft prompt".to_string()),
            default_key_scope: None,
        },
    );
    map.insert(
        "app.message.followUp".to_string(),
        KeybindingDefinition {
            default_keys: vec![String::from("alt+enter")],
            default_keys_is_single: true,
            description: Some("Queue follow-up message".to_string()),
            default_key_scope: None,
        },
    );
    map.insert(
        "app.message.navigateOlder".to_string(),
        KeybindingDefinition {
            default_keys: vec![String::from("alt+up")],
            default_keys_is_single: true,
            description: Some("Select older pending message".to_string()),
            default_key_scope: None,
        },
    );
    map.insert(
        "app.message.navigateNewer".to_string(),
        KeybindingDefinition {
            default_keys: vec![String::from("alt+down")],
            default_keys_is_single: true,
            description: Some("Select newer pending message or draft".to_string()),
            default_key_scope: None,
        },
    );
    map.insert(
        "app.message.moveEarlier".to_string(),
        KeybindingDefinition {
            default_keys: vec![String::from("ctrl+alt+up")],
            default_keys_is_single: true,
            description: Some("Move selected pending message earlier".to_string()),
            default_key_scope: None,
        },
    );
    map.insert(
        "app.message.moveLater".to_string(),
        KeybindingDefinition {
            default_keys: vec![String::from("ctrl+alt+down")],
            default_keys_is_single: true,
            description: Some("Move selected pending message later".to_string()),
            default_key_scope: None,
        },
    );
    let (default_keys, default_keys_is_single) = if is_win32 { (vec![String::from("alt+v")], true) } else { (vec![String::from("ctrl+v")], true) };
    map.insert(
        "app.clipboard.pasteImage".to_string(),
        KeybindingDefinition {
            default_keys,
            default_keys_is_single,
            description: Some("Paste image from clipboard".to_string()),
            default_key_scope: None,
        },
    );
    map.insert(
        "app.clipboard.copyLoginUrl".to_string(),
        KeybindingDefinition {
            default_keys: vec![String::from("c"), String::from("alt+c")],
            default_keys_is_single: false,
            description: Some("Copy login URL".to_string()),
            default_key_scope: None,
        },
    );
    map.insert(
        "app.session.new".to_string(),
        KeybindingDefinition {
            default_keys: vec![],
            default_keys_is_single: false,
            description: Some("Start a new session".to_string()),
            default_key_scope: None,
        },
    );
    map.insert(
        "app.session.tree".to_string(),
        KeybindingDefinition {
            default_keys: vec![],
            default_keys_is_single: false,
            description: Some("Open session tree".to_string()),
            default_key_scope: None,
        },
    );
    map.insert(
        "app.session.fork".to_string(),
        KeybindingDefinition {
            default_keys: vec![],
            default_keys_is_single: false,
            description: Some("Fork current session".to_string()),
            default_key_scope: None,
        },
    );
    map.insert(
        "app.session.resume".to_string(),
        KeybindingDefinition {
            default_keys: vec![],
            default_keys_is_single: false,
            description: Some("Resume a session".to_string()),
            default_key_scope: None,
        },
    );
    map.insert(
        "app.agents.back".to_string(),
        KeybindingDefinition {
            default_keys: vec![String::from("left")],
            default_keys_is_single: true,
            description: Some("Return to parent agent scope".to_string()),
            default_key_scope: None,
        },
    );
    map.insert(
        "app.agents.open".to_string(),
        KeybindingDefinition {
            default_keys: vec![String::from("right")],
            default_keys_is_single: true,
            description: Some("Drill into selected agent".to_string()),
            default_key_scope: None,
        },
    );
    map.insert(
        "app.modal.back".to_string(),
        KeybindingDefinition {
            default_keys: vec![String::from("left")],
            default_keys_is_single: true,
            description: Some("Go back / close the current dialog".to_string()),
            default_key_scope: None,
        },
    );
    map.insert(
        "app.agents.reply".to_string(),
        KeybindingDefinition {
            default_keys: vec![String::from("space")],
            default_keys_is_single: true,
            description: Some("Reply to selected agent".to_string()),
            default_key_scope: None,
        },
    );
    map.insert(
        "app.agents.new".to_string(),
        KeybindingDefinition {
            default_keys: vec![String::from("ctrl+n")],
            default_keys_is_single: true,
            description: Some("Start a new session from the agents view".to_string()),
            default_key_scope: None,
        },
    );
    map.insert(
        "app.agents.delete".to_string(),
        KeybindingDefinition {
            default_keys: vec![String::from("ctrl+x")],
            default_keys_is_single: true,
            description: Some("Stop or delete selected agent".to_string()),
            default_key_scope: None,
        },
    );
    map.insert(
        "app.agents.program".to_string(),
        KeybindingDefinition {
            default_keys: vec![String::from("ctrl+o")],
            default_keys_is_single: true,
            description: Some("Show the program that spawned subagents".to_string()),
            default_key_scope: None,
        },
    );
    map.insert(
        "app.agents.rename".to_string(),
        KeybindingDefinition {
            default_keys: vec![String::from("ctrl+r")],
            default_keys_is_single: true,
            description: Some("Rename selected agent session".to_string()),
            default_key_scope: None,
        },
    );
    map.insert(
        "app.agents.inactiveCollapse".to_string(),
        KeybindingDefinition {
            default_keys: vec![String::from("alt+i")],
            default_keys_is_single: true,
            description: Some("Show or hide inactive sessions".to_string()),
            default_key_scope: None,
        },
    );
    map.insert(
        "app.agents.expand".to_string(),
        KeybindingDefinition {
            default_keys: vec![String::from("alt+right")],
            default_keys_is_single: true,
            description: Some("Expand or collapse selected agent subagents".to_string()),
            default_key_scope: None,
        },
    );
    map.insert(
        "app.tree.foldOrUp".to_string(),
        KeybindingDefinition {
            default_keys: vec![String::from("ctrl+left"), String::from("alt+left")],
            default_keys_is_single: false,
            description: Some("Fold tree branch or move up".to_string()),
            default_key_scope: None,
        },
    );
    map.insert(
        "app.tree.unfoldOrDown".to_string(),
        KeybindingDefinition {
            default_keys: vec![String::from("ctrl+right"), String::from("alt+right")],
            default_keys_is_single: false,
            description: Some("Unfold tree branch or move down".to_string()),
            default_key_scope: None,
        },
    );
    map.insert(
        "app.tree.editLabel".to_string(),
        KeybindingDefinition {
            default_keys: vec![String::from("shift+l")],
            default_keys_is_single: true,
            description: Some("Edit tree label".to_string()),
            default_key_scope: None,
        },
    );
    map.insert(
        "app.tree.toggleLabelTimestamp".to_string(),
        KeybindingDefinition {
            default_keys: vec![String::from("shift+t")],
            default_keys_is_single: true,
            description: Some("Toggle tree label timestamps".to_string()),
            default_key_scope: None,
        },
    );
    map.insert(
        "app.models.save".to_string(),
        KeybindingDefinition {
            default_keys: vec![String::from("ctrl+s")],
            default_keys_is_single: true,
            description: Some("Save model selection".to_string()),
            default_key_scope: None,
        },
    );
    map.insert(
        "app.models.enableAll".to_string(),
        KeybindingDefinition {
            default_keys: vec![String::from("ctrl+a")],
            default_keys_is_single: true,
            description: Some("Enable all models".to_string()),
            default_key_scope: None,
        },
    );
    map.insert(
        "app.models.clearAll".to_string(),
        KeybindingDefinition {
            default_keys: vec![String::from("ctrl+x")],
            default_keys_is_single: true,
            description: Some("Clear all models".to_string()),
            default_key_scope: None,
        },
    );
    map.insert(
        "app.models.toggleProvider".to_string(),
        KeybindingDefinition {
            default_keys: vec![String::from("ctrl+p")],
            default_keys_is_single: true,
            description: Some("Toggle all models for provider".to_string()),
            default_key_scope: None,
        },
    );
    map.insert(
        "app.models.reorderUp".to_string(),
        KeybindingDefinition {
            default_keys: vec![String::from("alt+up")],
            default_keys_is_single: true,
            description: Some("Move model up in order".to_string()),
            default_key_scope: None,
        },
    );
    map.insert(
        "app.models.reorderDown".to_string(),
        KeybindingDefinition {
            default_keys: vec![String::from("alt+down")],
            default_keys_is_single: true,
            description: Some("Move model down in order".to_string()),
            default_key_scope: None,
        },
    );
    map.insert(
        "app.tree.filter.default".to_string(),
        KeybindingDefinition {
            default_keys: vec![String::from("ctrl+d")],
            default_keys_is_single: true,
            description: Some("Tree filter: default view".to_string()),
            default_key_scope: None,
        },
    );
    map.insert(
        "app.tree.filter.noTools".to_string(),
        KeybindingDefinition {
            default_keys: vec![String::from("ctrl+t")],
            default_keys_is_single: true,
            description: Some("Tree filter: hide tool results".to_string()),
            default_key_scope: None,
        },
    );
    map.insert(
        "app.tree.filter.userOnly".to_string(),
        KeybindingDefinition {
            default_keys: vec![String::from("ctrl+u")],
            default_keys_is_single: true,
            description: Some("Tree filter: user messages only".to_string()),
            default_key_scope: None,
        },
    );
    map.insert(
        "app.tree.filter.labeledOnly".to_string(),
        KeybindingDefinition {
            default_keys: vec![String::from("ctrl+l")],
            default_keys_is_single: true,
            description: Some("Tree filter: labeled entries only".to_string()),
            default_key_scope: None,
        },
    );
    map.insert(
        "app.tree.filter.all".to_string(),
        KeybindingDefinition {
            default_keys: vec![String::from("ctrl+a")],
            default_keys_is_single: true,
            description: Some("Tree filter: show all entries".to_string()),
            default_key_scope: None,
        },
    );
    map.insert(
        "app.tree.filter.cycleForward".to_string(),
        KeybindingDefinition {
            default_keys: vec![String::from("ctrl+o")],
            default_keys_is_single: true,
            description: Some("Tree filter: cycle forward".to_string()),
            default_key_scope: None,
        },
    );
    map.insert(
        "app.tree.filter.cycleBackward".to_string(),
        KeybindingDefinition {
            default_keys: vec![String::from("shift+ctrl+o")],
            default_keys_is_single: true,
            description: Some("Tree filter: cycle backward".to_string()),
            default_key_scope: None,
        },
    );
    // SHARED FILE EDIT (core/keybindings.rs, DEFAULT-only addition by jev-ui lane):
    // see `APP_KEYBINDINGS`. No default key on purpose: the dialog cancels through
    // the TUI `tui.select.cancel` / `app.interrupt` ids, and this id documents the
    // action so a user can bind it without touching component code.
    map.insert(
        "app.jev.cancel".to_string(),
        KeybindingDefinition {
            default_keys: vec![],
            default_keys_is_single: false,
            description: Some("Cancel the /jev dialog".to_string()),
            default_key_scope: None,
        },
    );
    for (action, key, description) in [
        ("app.stats.toggleSubagents", "s", "Include or exclude subagents in statistics"),
        ("app.stats.nextSection", "tab", "Show the next statistics section"),
    ] {
        map.insert(action.into(), KeybindingDefinition {
            default_keys: vec![key.into()],
            default_keys_is_single: true,
            description: Some(description.into()),
            default_key_scope: None,
        });
    }
    map
}

/// `KeybindingDefinitions`.
pub type KeybindingDefinitions = IndexMap<String, KeybindingDefinition>;

/// `KEYBINDING_NAME_MIGRATIONS`.
pub const KEYBINDING_NAME_MIGRATIONS: [(&str, &str); 59] = [
    ("app.message.dequeue", "app.message.navigateOlder"),
    ("cursorUp", "tui.editor.cursorUp"),
    ("cursorDown", "tui.editor.cursorDown"),
    ("cursorLeft", "tui.editor.cursorLeft"),
    ("cursorRight", "tui.editor.cursorRight"),
    ("cursorWordLeft", "tui.editor.cursorWordLeft"),
    ("cursorWordRight", "tui.editor.cursorWordRight"),
    ("cursorLineStart", "tui.editor.cursorLineStart"),
    ("cursorLineEnd", "tui.editor.cursorLineEnd"),
    ("jumpForward", "tui.editor.jumpForward"),
    ("jumpBackward", "tui.editor.jumpBackward"),
    ("pageUp", "tui.editor.pageUp"),
    ("pageDown", "tui.editor.pageDown"),
    ("deleteCharBackward", "tui.editor.deleteCharBackward"),
    ("deleteCharForward", "tui.editor.deleteCharForward"),
    ("deleteWordBackward", "tui.editor.deleteWordBackward"),
    ("deleteWordForward", "tui.editor.deleteWordForward"),
    ("deleteToLineStart", "tui.editor.deleteToLineStart"),
    ("deleteToLineEnd", "tui.editor.deleteToLineEnd"),
    ("yank", "tui.editor.yank"),
    ("yankPop", "tui.editor.yankPop"),
    ("undo", "tui.editor.undo"),
    ("newLine", "tui.input.newLine"),
    ("submit", "tui.input.submit"),
    ("tab", "tui.input.tab"),
    ("copy", "tui.input.copy"),
    ("selectUp", "tui.select.up"),
    ("selectDown", "tui.select.down"),
    ("selectPageUp", "tui.select.pageUp"),
    ("selectPageDown", "tui.select.pageDown"),
    ("selectConfirm", "tui.select.confirm"),
    ("selectCancel", "tui.select.cancel"),
    ("interrupt", "app.interrupt"),
    ("clear", "app.clear"),
    ("clearInput", "app.input.clear"),
    ("exit", "app.exit"),
    ("suspend", "app.suspend"),
    ("selectModel", "app.model.select"),
    ("expandTools", "app.tools.expand"),
    ("toggleThinking", "app.thinking.toggle"),
    ("focusSubagents", "app.subagents.focus"),
    ("externalEditor", "app.editor.external"),
    ("followUp", "app.message.followUp"),
    ("dequeue", "app.message.navigateOlder"),
    ("pasteImage", "app.clipboard.pasteImage"),
    ("newSession", "app.session.new"),
    ("tree", "app.session.tree"),
    ("fork", "app.session.fork"),
    ("resume", "app.session.resume"),
    ("agentsBack", "app.agents.back"),
    ("agentsReply", "app.agents.reply"),
    ("agentsNew", "app.agents.new"),
    ("agentsDelete", "app.agents.delete"),
    ("agentsProgram", "app.agents.program"),
    ("agentsRename", "app.agents.rename"),
    ("treeFoldOrUp", "app.tree.foldOrUp"),
    ("treeUnfoldOrDown", "app.tree.unfoldOrDown"),
    ("treeEditLabel", "app.tree.editLabel"),
    ("treeToggleLabelTimestamp", "app.tree.toggleLabelTimestamp"),
];

/// `isRecord`.
fn is_record(value: &Value) -> bool {
    matches!(value, Value::Object(_))
}

/// `isLegacyKeybindingName`.
fn is_legacy_keybinding_name(key: &str) -> bool {
    KEYBINDING_NAME_MIGRATIONS.iter().any(|(legacy, _)| *legacy == key)
}

/// `toKeybindingsConfig`.
///
/// Only string and string-array entries survive; numbers, `null` and mixed
/// arrays are dropped exactly like the TypeScript `continue` branches.
fn to_keybindings_config(value: &Value) -> IndexMap<String, KeybindingSetting> {
    let mut config: IndexMap<String, KeybindingSetting> = IndexMap::new();
    if !is_record(value) {
        return config;
    }
    let Value::Object(map) = value else {
        return config;
    };
    for (key, binding) in map {
        if let Value::String(text) = binding {
            config.insert(key.clone(), KeybindingSetting::Single(text.clone()));
            continue;
        }
        if let Value::Array(entries) = binding {
            if entries.iter().all(|entry| entry.is_string()) {
                config.insert(
                    key.clone(),
                    KeybindingSetting::List(
                        entries
                            .iter()
                            .map(|entry| entry.as_str().unwrap_or_default().to_string())
                            .collect(),
                    ),
                );
            }
        }
    }
    config
}

/// `migrateKeybindingsConfig`.
pub fn migrate_keybindings_config(raw_config: &Map<String, Value>) -> (Map<String, Value>, bool) {
    let mut config: Map<String, Value> = Map::new();
    let mut migrated = false;

    for (key, value) in raw_config {
        let next_key = if is_legacy_keybinding_name(key) {
            KEYBINDING_NAME_MIGRATIONS
                .iter()
                .find(|(legacy, _)| *legacy == key.as_str())
                .map(|(_, next)| *next)
                .unwrap_or(key.as_str())
        } else {
            key.as_str()
        };
        if next_key != key {
            migrated = true;
        }
        if key != next_key && raw_config.contains_key(next_key) {
            migrated = true;
            continue;
        }
        config.insert(next_key.to_string(), value.clone());
    }

    (order_keybindings_config(&config), migrated)
}

/// `orderKeybindingsConfig`.
fn order_keybindings_config(config: &Map<String, Value>) -> Map<String, Value> {
    let mut ordered: Map<String, Value> = Map::new();
    for keybinding in keybindings().keys() {
        if let Some(value) = config.get(keybinding) {
            ordered.insert(keybinding.clone(), value.clone());
        }
    }

    let mut extras: Vec<String> = config
        .keys()
        .filter(|key| !ordered.contains_key(*key))
        .cloned()
        .collect();
    extras.sort();
    for key in extras {
        if let Some(value) = config.get(&key) {
            ordered.insert(key, value.clone());
        }
    }

    ordered
}

/// `loadRawConfig`.
fn load_raw_config(path: &str) -> Option<Map<String, Value>> {
    if !std::path::Path::new(path).exists() {
        return None;
    }
    let content = std::fs::read_to_string(path).ok()?;
    match serde_json::from_str::<Value>(&content) {
        Ok(Value::Object(map)) => Some(map),
        _ => None,
    }
}
/// `KeybindingsManager extends TuiKeybindingsManager`.
///
/// The application class adds the config file path, `create`, `reload` and
/// `getEffectiveConfig`; every other member is delegated to the TUI manager so
/// resolution, matching and conflict reporting stay in one place.
pub struct KeybindingsManager {
    inner: TuiKeybindingsManager,
    user_bindings: KeybindingsConfig,
    config_path: Option<String>,
}

/// `KeybindingsConfig` -> the TUI manager's normalized key lists.
fn to_tui_config(user_bindings: &KeybindingsConfig) -> TuiKeybindingsConfig {
    let mut config: TuiKeybindingsConfig = IndexMap::new();
    for (keybinding, setting) in user_bindings {
        config.insert(keybinding.clone(), setting.keys());
    }
    config
}

impl KeybindingsManager {
    /// Install application and editor bindings for components on this UI thread.
    /// Mirrors `setKeybindings(this.keybindings)` during interactive startup.
    pub fn install(&self) {
        pi_tui::keybindings::set_keybindings(TuiKeybindingsManager::new(
            keybindings(),
            to_tui_config(&self.user_bindings),
        ));
    }

    /// `constructor(userBindings: KeybindingsConfig = {}, configPath?: string)`.
    pub fn new(user_bindings: KeybindingsConfig, config_path: Option<String>) -> Self {
        Self {
            inner: TuiKeybindingsManager::new(keybindings(), to_tui_config(&user_bindings)),
            user_bindings,
            config_path,
        }
    }

    /// `KeybindingsManager.create(agentDir = getAgentDir())`.
    pub fn create(agent_dir: Option<&str>) -> Self {
        let agent_dir = match agent_dir {
            Some(dir) => dir.to_string(),
            None => crate::config::get_agent_dir(),
        };
        let config_path = join_path(&agent_dir, "keybindings.json");
        let user_bindings = Self::load_from_file(&config_path);
        Self::new(user_bindings, Some(config_path))
    }

    /// `reload()`.
    pub fn reload(&mut self) {
        let Some(config_path) = self.config_path.clone() else {
            return;
        };
        let user_bindings = Self::load_from_file(&config_path);
        self.set_user_bindings(user_bindings);
    }

    /// `getEffectiveConfig()`.
    pub fn get_effective_config(&self) -> KeybindingsConfig {
        self.get_resolved_bindings()
    }

    /// `loadFromFile(path)`.
    fn load_from_file(path: &str) -> KeybindingsConfig {
        let Some(raw_config) = load_raw_config(path) else {
            return KeybindingsConfig::new();
        };
        let (migrated, _migrated_flag) = migrate_keybindings_config(&raw_config);
        to_keybindings_config(&Value::Object(migrated))
    }

    /// `matches(data, keybinding)`.
    pub fn matches(&self, data: &str, keybinding: &str) -> bool {
        self.inner.matches(data, keybinding)
    }

    /// `getKeys(keybinding)`.
    pub fn get_keys(&self, keybinding: &str) -> Vec<String> {
        self.inner.get_keys(keybinding)
    }

    /// `getDefinition(keybinding)`.
    pub fn get_definition(&self, keybinding: &str) -> Option<KeybindingDefinition> {
        self.inner.get_definition(keybinding)
    }

    /// `getConflicts()`.
    pub fn get_conflicts(&self) -> Vec<pi_tui::keybindings::KeybindingConflict> {
        self.inner.get_conflicts()
    }

    /// `setUserBindings(userBindings)`.
    pub fn set_user_bindings(&mut self, user_bindings: KeybindingsConfig) {
        self.inner.set_user_bindings(to_tui_config(&user_bindings));
        self.user_bindings = user_bindings;
    }

    /// `getUserBindings()` - the configured shape, unchanged.
    pub fn get_user_bindings(&self) -> KeybindingsConfig {
        self.user_bindings.clone()
    }

    /// `getResolvedBindings()` - single keys stay single, multi-key entries stay arrays.
    pub fn get_resolved_bindings(&self) -> KeybindingsConfig {
        let mut resolved: KeybindingsConfig = IndexMap::new();
        for id in keybindings().keys() {
            let keys = self.inner.get_keys(id);
            let setting = if keys.len() == 1 {
                KeybindingSetting::Single(keys[0].clone())
            } else {
                KeybindingSetting::List(keys)
            };
            resolved.insert(id.clone(), setting);
        }
        resolved
    }
}

/// `path.join` for the one join this module performs.
fn join_path(base: &str, leaf: &str) -> String {
    std::path::Path::new(base).join(leaf).to_string_lossy().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setting(value: &str) -> KeybindingSetting {
        KeybindingSetting::Single(value.to_string())
    }

    fn list(values: &[&str]) -> KeybindingSetting {
        KeybindingSetting::List(values.iter().map(|value| value.to_string()).collect())
    }

    #[test]
    fn ui014_session_list_keys_remain_without_sidebar_actions() {
        let definitions = keybindings();
        assert!(definitions.keys().all(|key| !key.starts_with("app.sidebar.")));
        assert_eq!(definitions["app.agents.back"].default_keys, vec!["left"]);
        assert_eq!(definitions["app.agents.open"].default_keys, vec!["right"]);
        assert!(definitions.contains_key("app.session.resume"));
    }

    #[test]
    fn keybindings_spread_tui_entries_first_then_app_entries() {
        let definitions = keybindings();
        let tui = pi_tui::keybindings::tui_keybindings();
        assert_eq!(definitions.len(), tui.len() + APP_KEYBINDINGS.len());

        let keys: Vec<String> = definitions.keys().cloned().collect();
        let first_app = keys.iter().position(|key| key == "app.interrupt").unwrap();
        // Every TUI entry precedes the app block.
        for key in tui.keys() {
            let index = keys.iter().position(|candidate| candidate == key).unwrap();
            assert!(index < first_app);
        }
        // App entries keep declaration order after the TUI block.
        let app_order: Vec<String> = keys[first_app..].to_vec();
        let expected: Vec<String> = APP_KEYBINDINGS.iter().map(|key| key.to_string()).collect();
        assert_eq!(app_order, expected);
    }

    #[test]
    fn app_defaults_match_the_typescript_literals() {
        let definitions = keybindings();
        assert!(definitions["app.interrupt"].default_keys.is_empty());
        assert!(!definitions["app.interrupt"].default_keys_is_single);
        assert_eq!(definitions["app.clear"].default_keys, vec!["ctrl+c".to_string()]);
        assert!(definitions["app.clear"].default_keys_is_single);
        assert_eq!(
            definitions["app.tools.expand"].default_key_scope.as_deref(),
            Some("editor")
        );
        assert_eq!(
            definitions["app.clipboard.copyLoginUrl"].default_keys,
            vec!["c".to_string(), "alt+c".to_string()]
        );
        // `process.platform === "win32" ? [] : "ctrl+z"`.
        let suspend = &definitions["app.suspend"];
        if crate::utils::pi_user_agent::process_platform() == "win32" {
            assert!(suspend.default_keys.is_empty());
            assert!(!suspend.default_keys_is_single);
        } else {
            assert_eq!(suspend.default_keys, vec!["ctrl+z".to_string()]);
            assert!(suspend.default_keys_is_single);
        }
    }

    #[test]
    fn migration_rewrites_old_names_and_keeps_the_namespaced_value() {
        let mut raw: Map<String, Value> = Map::new();
        raw.insert("cursorUp".to_string(), serde_json::json!(["up", "ctrl+p"]));
        raw.insert("expandTools".to_string(), serde_json::json!("ctrl+x"));
        raw.insert("app.message.dequeue".to_string(), serde_json::json!("alt+u"));
        let (config, migrated) = migrate_keybindings_config(&raw);
        assert!(migrated);
        assert_eq!(config["tui.editor.cursorUp"], serde_json::json!(["up", "ctrl+p"]));
        assert_eq!(config["app.tools.expand"], serde_json::json!("ctrl+x"));
        assert_eq!(config["app.message.navigateOlder"], serde_json::json!("alt+u"));
        assert!(!config.contains_key("cursorUp"));

        let mut both: Map<String, Value> = Map::new();
        both.insert("expandTools".to_string(), serde_json::json!("ctrl+x"));
        both.insert("app.tools.expand".to_string(), serde_json::json!("ctrl+y"));
        let (config, migrated) = migrate_keybindings_config(&both);
        assert!(migrated);
        assert_eq!(config["app.tools.expand"], serde_json::json!("ctrl+y"));
        assert!(!config.contains_key("expandTools"));
    }

    #[test]
    fn orders_known_keybindings_first_then_sorted_extras() {
        let mut raw: Map<String, Value> = Map::new();
        raw.insert("zzzCustom".to_string(), serde_json::json!("ctrl+z"));
        raw.insert("aaaCustom".to_string(), serde_json::json!("ctrl+a"));
        raw.insert("tui.input.submit".to_string(), serde_json::json!("enter"));
        let (config, migrated) = migrate_keybindings_config(&raw);
        assert!(!migrated);
        let keys: Vec<String> = config.keys().cloned().collect();
        assert_eq!(
            keys,
            vec![
                "tui.input.submit".to_string(),
                "aaaCustom".to_string(),
                "zzzCustom".to_string()
            ]
        );
    }

    #[test]
    fn loads_old_names_before_the_file_is_rewritten() {
        let dir = std::env::temp_dir().join(format!(
            "pi-keybindings-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("keybindings.json");
        std::fs::write(
            &path,
            serde_json::json!({ "selectConfirm": "enter", "interrupt": "ctrl+x" }).to_string(),
        )
        .unwrap();

        let manager = KeybindingsManager::create(Some(&dir.to_string_lossy()));
        let user = manager.get_user_bindings();
        assert_eq!(user["tui.select.confirm"], setting("enter"));
        assert_eq!(user["app.interrupt"], setting("ctrl+x"));
        let effective = manager.get_effective_config();
        assert_eq!(effective["tui.select.confirm"], setting("enter"));
        assert_eq!(effective["app.interrupt"], setting("ctrl+x"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn explicit_editor_bindings_take_precedence_over_application_defaults() {
        let mut config: KeybindingsConfig = IndexMap::new();
        config.insert("tui.editor.cursorUp".to_string(), list(&["up", "ctrl+p"]));
        config.insert("tui.editor.cursorDown".to_string(), list(&["down", "ctrl+n"]));
        let manager = KeybindingsManager::new(config, None);

        assert_eq!(
            manager.get_keys("tui.editor.cursorUp"),
            vec!["up".to_string(), "ctrl+p".to_string()]
        );
        assert!(manager.get_keys("app.messages.expand").is_empty());
        assert_eq!(manager.get_keys("app.models.toggleProvider"), vec!["ctrl+p".to_string()]);
        assert_eq!(manager.get_keys("app.agents.new"), vec!["ctrl+n".to_string()]);
    }

    #[test]
    fn reports_an_application_default_retained_against_an_editor_binding() {
        let mut config: KeybindingsConfig = IndexMap::new();
        config.insert("tui.editor.cursorUp".to_string(), list(&["up", "ctrl+p"]));
        config.insert("app.messages.expand".to_string(), setting("ctrl+p"));
        let manager = KeybindingsManager::new(config, None);

        let conflicts = manager.get_conflicts();
        let expected = vec!["tui.editor.cursorUp".to_string(), "app.messages.expand".to_string()];
        assert!(conflicts
            .iter()
            .any(|conflict| conflict.key == "ctrl+p" && conflict.keybindings == expected));
        assert_eq!(manager.get_keys("app.messages.expand"), vec!["ctrl+p".to_string()]);
    }

    #[test]
    fn resolved_bindings_keep_the_single_key_shape() {
        let manager = KeybindingsManager::new(KeybindingsConfig::new(), None);
        let resolved = manager.get_resolved_bindings();
        assert!(matches!(resolved["tui.input.submit"], KeybindingSetting::Single(_)));
        assert!(matches!(resolved["tui.editor.cursorLeft"], KeybindingSetting::List(_)));
        assert_eq!(resolved["app.interrupt"], list(&[]));
    }

    #[test]
    fn to_keybindings_config_drops_non_string_entries() {
        let parsed = serde_json::json!({
            "app.clear": "ctrl+c",
            "tui.select.cancel": ["escape", "ctrl+c"],
            "badNumber": 3,
            "badMixed": ["escape", 4],
        });
        let config = to_keybindings_config(&parsed);
        assert_eq!(config.len(), 2);
        assert_eq!(config["app.clear"], setting("ctrl+c"));
        assert_eq!(config["tui.select.cancel"], list(&["escape", "ctrl+c"]));
    }
}
