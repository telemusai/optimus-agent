//! Port of packages/tui/src/keybindings.ts.

use crate::keys::{matches_key, matches_option_composed_key};
use indexmap::IndexMap;
use std::cell::RefCell;

/// Port of the `Keybindings` interface. TypeScript keys that interface by action
/// id, so an action id is the unit the interface enumerates.
pub type Keybindings = String;

/// Port of `Keybinding = keyof Keybindings`.
pub type Keybinding = String;

/// Port of the `KeybindingsConfig` record: user overrides by action id.
pub type KeybindingsConfig = IndexMap<String, Vec<String>>;

#[derive(Clone, Copy, Debug, Default)]
pub struct KeyMatchOptions {
    pub option_composed: bool,
}

/// Port of the `KeybindingDefinitions` record.
pub type KeybindingDefinitions = IndexMap<String, KeybindingDefinition>;

/// Port of the `TUI_KEYBINDINGS` constant.
pub static TUI_KEYBINDINGS: once_cell::sync::Lazy<KeybindingDefinitions> =
    once_cell::sync::Lazy::new(tui_keybindings);

/// Definition of one keybinding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeybindingDefinition {
    /// `defaultKeys` in the TypeScript definition (one key or a list).
    pub default_keys: Vec<String>,
    /// True when `defaultKeys` was a single string, false when it was an array.
    pub default_keys_is_single: bool,
    pub description: Option<String>,
    pub default_key_scope: Option<String>,
}

impl KeybindingDefinition {
    fn single(key: &str, description: &str) -> Self {
        Self {
            default_keys: vec![key.to_string()],
            default_keys_is_single: true,
            description: Some(description.to_string()),
            default_key_scope: None,
        }
    }

    fn single_scoped(key: &str, description: &str, scope: &str) -> Self {
        Self {
            default_keys: vec![key.to_string()],
            default_keys_is_single: true,
            description: Some(description.to_string()),
            default_key_scope: Some(scope.to_string()),
            }
    }

    fn list(keys: &[&str], description: &str) -> Self {
        Self {
            default_keys: keys.iter().map(|k| k.to_string()).collect(),
            default_keys_is_single: false,
            description: Some(description.to_string()),
            default_key_scope: None,
        }
    }

    fn list_scoped(keys: &[&str], description: &str, scope: &str) -> Self {
        Self {
            default_keys: keys.iter().map(|k| k.to_string()).collect(),
            default_keys_is_single: false,
            description: Some(description.to_string()),
            default_key_scope: Some(scope.to_string()),
        }
    }
}

/// The built-in TUI keybinding definitions (`TUI_KEYBINDINGS`).
pub fn tui_keybindings() -> IndexMap<String, KeybindingDefinition> {
    let mut map: IndexMap<String, KeybindingDefinition> = IndexMap::new();
    map.insert(
        "tui.editor.cursorUp".to_string(),
        KeybindingDefinition::single_scoped("up", "Move cursor up", "editor"),
    );
    map.insert(
        "tui.editor.cursorDown".to_string(),
        KeybindingDefinition::single_scoped("down", "Move cursor down", "editor"),
    );
    map.insert(
        "tui.editor.cursorLeft".to_string(),
        KeybindingDefinition::list_scoped(&["left", "ctrl+b"], "Move cursor left", "editor"),
    );
    map.insert(
        "tui.editor.cursorRight".to_string(),
        KeybindingDefinition::list_scoped(&["right", "ctrl+f"], "Move cursor right", "editor"),
    );
    map.insert(
        "tui.editor.cursorWordLeft".to_string(),
        KeybindingDefinition::list_scoped(&["alt+left", "ctrl+left", "alt+b"], "Move cursor word left", "editor"),
    );
    map.insert(
        "tui.editor.cursorWordRight".to_string(),
        KeybindingDefinition::list_scoped(&["alt+right", "ctrl+right", "alt+f"], "Move cursor word right", "editor"),
    );
    map.insert(
        "tui.editor.cursorLineStart".to_string(),
        KeybindingDefinition::list_scoped(&["home", "ctrl+a"], "Move to line start", "editor"),
    );
    map.insert(
        "tui.editor.cursorLineEnd".to_string(),
        KeybindingDefinition::list_scoped(&["end", "ctrl+e"], "Move to line end", "editor"),
    );
    map.insert(
        "tui.editor.jumpForward".to_string(),
        KeybindingDefinition::single_scoped("ctrl+]", "Jump forward to character", "editor"),
    );
    map.insert(
        "tui.editor.jumpBackward".to_string(),
        KeybindingDefinition::single_scoped("ctrl+alt+]", "Jump backward to character", "editor"),
    );
    map.insert(
        "tui.editor.pageUp".to_string(),
        KeybindingDefinition::single_scoped("pageUp", "Page up", "editor"),
    );
    map.insert(
        "tui.editor.pageDown".to_string(),
        KeybindingDefinition::single_scoped("pageDown", "Page down", "editor"),
    );
    map.insert(
        "tui.editor.deleteCharBackward".to_string(),
        KeybindingDefinition::single_scoped("backspace", "Delete character backward", "editor"),
    );
    map.insert(
        "tui.editor.deleteCharForward".to_string(),
        KeybindingDefinition::list_scoped(&["delete", "ctrl+d"], "Delete character forward", "editor"),
    );
    map.insert(
        "tui.editor.deleteWordBackward".to_string(),
        KeybindingDefinition::list_scoped(&["ctrl+w", "alt+backspace"], "Delete word backward", "editor"),
    );
    map.insert(
        "tui.editor.deleteWordForward".to_string(),
        KeybindingDefinition::list_scoped(&["alt+d", "alt+delete"], "Delete word forward", "editor"),
    );
    map.insert(
        "tui.editor.deleteToLineStart".to_string(),
        KeybindingDefinition::single_scoped("ctrl+u", "Delete to line start", "editor"),
    );
    map.insert(
        "tui.editor.deleteToLineEnd".to_string(),
        KeybindingDefinition::single_scoped("ctrl+k", "Delete to line end", "editor"),
    );
    map.insert(
        "tui.editor.yank".to_string(),
        KeybindingDefinition::single_scoped("ctrl+y", "Yank", "editor"),
    );
    map.insert(
        "tui.editor.yankPop".to_string(),
        KeybindingDefinition::single_scoped("alt+y", "Yank pop", "editor"),
    );
    map.insert(
        "tui.editor.undo".to_string(),
        KeybindingDefinition::single_scoped("ctrl+-", "Undo", "editor"),
    );
    map.insert(
        "tui.input.newLine".to_string(),
        KeybindingDefinition::single_scoped("shift+enter", "Insert newline", "editor"),
    );
    map.insert(
        "tui.input.submit".to_string(),
        KeybindingDefinition::single_scoped("enter", "Submit input", "editor"),
    );
    map.insert(
        "tui.input.tab".to_string(),
        KeybindingDefinition::single_scoped("tab", "Tab / autocomplete", "editor"),
    );
    map.insert(
        "tui.input.copy".to_string(),
        KeybindingDefinition::single_scoped("ctrl+c", "Copy selection", "editor"),
    );
    map.insert(
        "tui.viewport.pageUp".to_string(),
        KeybindingDefinition::single("pageUp", "Scroll transcript up a page (fullscreen)"),
    );
    map.insert(
        "tui.viewport.pageDown".to_string(),
        KeybindingDefinition::single("pageDown", "Scroll transcript down a page (fullscreen)"),
    );
    map.insert(
        "tui.viewport.top".to_string(),
        KeybindingDefinition::single("shift+alt+up", "Scroll transcript to top (fullscreen)"),
    );
    map.insert(
        "tui.viewport.follow".to_string(),
        KeybindingDefinition::single("ctrl+shift+down", "Scroll to bottom and follow output (fullscreen)"),
    );
    map.insert(
        "tui.select.up".to_string(),
        KeybindingDefinition::single("up", "Move selection up"),
    );
    map.insert(
        "tui.select.down".to_string(),
        KeybindingDefinition::single("down", "Move selection down"),
    );
    map.insert(
        "tui.select.pageUp".to_string(),
        KeybindingDefinition::single("pageUp", "Selection page up"),
    );
    map.insert(
        "tui.select.pageDown".to_string(),
        KeybindingDefinition::single("pageDown", "Selection page down"),
    );
    map.insert(
        "tui.select.confirm".to_string(),
        KeybindingDefinition::single("enter", "Confirm selection"),
    );
    map.insert(
        "tui.select.cancel".to_string(),
        KeybindingDefinition::list(&["escape", "ctrl+c"], "Cancel selection"),
    );
    map
}

/// A key that is claimed by more than one keybinding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeybindingConflict {
    pub key: String,
    pub keybindings: Vec<String>,
}

fn normalize_keys(keys: Option<&Vec<String>>) -> Vec<String> {
    match keys {
        None => Vec::new(),
        Some(list) => {
            let mut seen: Vec<String> = Vec::new();
            for key in list {
                if !seen.contains(key) {
                    seen.push(key.clone());
                }
            }
            seen
        }
    }
}

/// Resolved keybindings with user overrides applied.
#[derive(Debug, Clone)]
pub struct KeybindingsManager {
    definitions: IndexMap<String, KeybindingDefinition>,
    user_bindings: IndexMap<String, Vec<String>>,
    keys_by_id: IndexMap<String, Vec<String>>,
    conflicts: Vec<KeybindingConflict>,
}

impl KeybindingsManager {
    pub fn new(
        definitions: IndexMap<String, KeybindingDefinition>,
        user_bindings: IndexMap<String, Vec<String>>,
    ) -> Self {
        let mut manager = Self {
            definitions,
            user_bindings,
            keys_by_id: IndexMap::new(),
            conflicts: Vec::new(),
        };
        manager.rebuild();
        manager
    }

    fn rebuild(&mut self) {
        self.keys_by_id.clear();
        self.conflicts.clear();

        let mut explicit_claims: IndexMap<String, Vec<String>> = IndexMap::new();
        let mut added_claims: IndexMap<String, Vec<String>> = IndexMap::new();
        for (keybinding, keys) in self.user_bindings.clone() {
            let definition = match self.definitions.get(&keybinding) {
                Some(d) => d.clone(),
                None => continue,
            };
            let defaults = normalize_keys(Some(&definition.default_keys));
            for key in normalize_keys(Some(&keys)) {
                let claimants = explicit_claims.entry(key.clone()).or_default();
                if !claimants.contains(&keybinding) {
                    claimants.push(keybinding.clone());
                }
                if !defaults.contains(&key) {
                    let added = added_claims.entry(key).or_default();
                    if !added.contains(&keybinding) {
                        added.push(keybinding.clone());
                    }
                }
            }
        }

        for (key, keybindings) in &explicit_claims {
            if keybindings.len() > 1 {
                self.conflicts.push(KeybindingConflict {
                    key: key.clone(),
                    keybindings: keybindings.clone(),
                });
            }
        }

        for (id, definition) in self.definitions.clone() {
            let user_keys = self.user_bindings.get(&id).cloned();
            let keys = match user_keys {
                None => normalize_keys(Some(&definition.default_keys))
                    .into_iter()
                    .filter(|key| {
                        let scope = match &definition.default_key_scope {
                            Some(s) => s,
                            None => return true,
                        };
                        let claimants = added_claims.get(key).cloned().unwrap_or_default();
                        !claimants.iter().any(|claimant| {
                            self.definitions
                                .get(claimant)
                                .and_then(|d| d.default_key_scope.as_deref())
                                == Some(scope.as_str())
                        })
                    })
                    .collect(),
                Some(keys) => normalize_keys(Some(&keys)),
            };
            self.keys_by_id.insert(id, keys);
        }
    }

    pub fn matches(&self, data: &str, keybinding: &str) -> bool {
        self.matches_with_options(data, keybinding, KeyMatchOptions::default())
    }

    pub fn matches_with_options(&self, data: &str, keybinding: &str, options: KeyMatchOptions) -> bool {
        match self.keys_by_id.get(keybinding) {
            Some(keys) => keys.iter().any(|key| {
                matches_key(data, key)
                    || (options.option_composed && matches_option_composed_key(data, key))
            }),
            None => false,
        }
    }

    pub fn get_keys(&self, keybinding: &str) -> Vec<String> {
        self.keys_by_id.get(keybinding).cloned().unwrap_or_default()
    }

    pub fn get_definition(&self, keybinding: &str) -> Option<KeybindingDefinition> {
        self.definitions.get(keybinding).cloned()
    }

    pub fn get_conflicts(&self) -> Vec<KeybindingConflict> {
        self.conflicts.clone()
    }

    pub fn set_user_bindings(&mut self, user_bindings: IndexMap<String, Vec<String>>) {
        self.user_bindings = user_bindings;
        self.rebuild();
    }

    pub fn get_user_bindings(&self) -> IndexMap<String, Vec<String>> {
        self.user_bindings.clone()
    }

    /// `getResolvedBindings` - single keys stay single, multi-key entries stay arrays.
    pub fn get_resolved_bindings(&self) -> IndexMap<String, serde_json::Value> {
        let mut resolved: IndexMap<String, serde_json::Value> = IndexMap::new();
        for id in self.definitions.keys() {
            let keys = self.keys_by_id.get(id).cloned().unwrap_or_default();
            let value = if keys.len() == 1 {
                serde_json::Value::String(keys[0].clone())
            } else {
                serde_json::Value::Array(
                    keys.iter()
                        .map(|k| serde_json::Value::String(k.clone()))
                        .collect(),
                )
            };
            resolved.insert(id.clone(), value);
        }
        resolved
    }
}

thread_local! {
    static GLOBAL_KEYBINDINGS: RefCell<Option<KeybindingsManager>> = const { RefCell::new(None) };
}

pub fn set_keybindings(keybindings: KeybindingsManager) {
    GLOBAL_KEYBINDINGS.with(|k| *k.borrow_mut() = Some(keybindings));
}

pub fn get_keybindings() -> KeybindingsManager {
    GLOBAL_KEYBINDINGS.with(|k| {
        let mut slot = k.borrow_mut();
        if slot.is_none() {
            *slot = Some(KeybindingsManager::new(tui_keybindings(), IndexMap::new()));
        }
        slot.as_ref().unwrap().clone()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_resolve_and_match() {
        let manager = KeybindingsManager::new(tui_keybindings(), IndexMap::new());
        assert_eq!(manager.get_keys("tui.input.submit"), vec!["enter".to_string()]);
        assert!(manager.matches("\r", "tui.input.submit"));
        assert!(manager.matches("\x1b", "tui.select.cancel"));
        assert!(!manager.matches("\x1b", "tui.input.submit"));
    }

    #[test]
    fn user_bindings_override_defaults() {
        let mut user: IndexMap<String, Vec<String>> = IndexMap::new();
        user.insert("tui.input.submit".to_string(), vec!["ctrl+s".to_string()]);
        let manager = KeybindingsManager::new(tui_keybindings(), user);
        assert_eq!(manager.get_keys("tui.input.submit"), vec!["ctrl+s".to_string()]);
        assert!(manager.matches("\x13", "tui.input.submit"));
        assert!(!manager.matches("\r", "tui.input.submit"));
    }

    #[test]
    fn option_composed_matching_respects_remapped_and_disabled_bindings() {
        let options = KeyMatchOptions { option_composed: true };
        for keys in [vec!["alt+s".into()], vec!["alt+a".into()], vec![]] {
            let accepts_option_s = keys == ["alt+s"];
            let manager = KeybindingsManager::new(tui_keybindings(), IndexMap::from([
                ("tui.input.submit".into(), keys),
            ]));
            assert!(!manager.matches("ß", "tui.input.submit"));
            assert_eq!(manager.matches_with_options("ß", "tui.input.submit", options), accepts_option_s);
        }
    }

    #[test]
    fn conflicting_explicit_claims_are_reported() {
        let mut user: IndexMap<String, Vec<String>> = IndexMap::new();
        user.insert("tui.input.submit".to_string(), vec!["ctrl+s".to_string()]);
        user.insert("tui.input.tab".to_string(), vec!["ctrl+s".to_string()]);
        let manager = KeybindingsManager::new(tui_keybindings(), user);
        let conflicts = manager.get_conflicts();
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].key, "ctrl+s");
        assert_eq!(conflicts[0].keybindings.len(), 2);
    }

    #[test]
    fn resolved_bindings_keep_single_vs_list_shape() {
        let manager = KeybindingsManager::new(tui_keybindings(), IndexMap::new());
        let resolved = manager.get_resolved_bindings();
        assert!(resolved.get("tui.input.submit").unwrap().is_string());
        assert!(resolved.get("tui.editor.cursorLeft").unwrap().is_array());
    }
}
