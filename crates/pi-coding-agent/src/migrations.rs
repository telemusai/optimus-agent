//! Port of packages/coding-agent/src/migrations.ts
//!
//! One-time migrations that run on startup.

use std::path::Path;

use crate::config::{CONFIG_DIR_NAME, get_agent_dir, get_bin_dir, get_sessions_dir};
use crate::utils::atomic_file::{realpath_if_present_sync, write_file_atomic_sync, WriteFileAtomicOptions};
use crate::utils::file_lines::read_first_line_sync;

const MIGRATION_GUIDE_URL: &str =
    "https://github.com/earendil-works/pi-mono/blob/main/packages/coding-agent/CHANGELOG.md#extensions-migration";
const EXTENSIONS_DOC_URL: &str =
    "https://github.com/earendil-works/pi-mono/blob/main/packages/coding-agent/docs/extensions.md";

fn join_path(base: &str, name: &str) -> String {
    Path::new(base).join(name).to_string_lossy().to_string()
}

fn parent_dir(path: &str) -> String {
    Path::new(path)
        .parent()
        .map(|parent| parent.to_string_lossy().to_string())
        .unwrap_or_default()
}

fn base_name(path: &str) -> String {
    Path::new(path)
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_default()
}

fn green(message: &str) -> String {
    format!("\u{1b}[32m{message}\u{1b}[39m")
}

fn yellow(message: &str) -> String {
    format!("\u{1b}[33m{message}\u{1b}[39m")
}

fn dim(message: &str) -> String {
    format!("\u{1b}[2m{message}\u{1b}[22m")
}

/// `JSON.parse(readFileSync(path, "utf-8"))`.
fn read_json(path: &str) -> Option<serde_json::Value> {
    let content = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&content).ok()
}

/// `JSON.stringify(value, null, 2)`.
fn stringify_pretty(value: &serde_json::Value) -> String {
    serde_json::to_string_pretty(value).unwrap_or_default()
}

/// `migrateAuthToAuthJson`.
///
/// @returns Provider names that were migrated.
pub fn migrate_auth_to_auth_json() -> Vec<String> {
    migrate_auth_in_dir(&get_agent_dir(), write_file_atomic_sync)
}

fn migrate_auth_in_dir(
    agent_dir: &str,
    write_auth: impl FnOnce(&str, &str, WriteFileAtomicOptions) -> std::io::Result<()>,
) -> Vec<String> {
    let auth_path = join_path(agent_dir, "auth.json");
    let oauth_path = join_path(&agent_dir, "oauth.json");
    let settings_path = join_path(&agent_dir, "settings.json");

    // Skip if auth.json already exists
    if Path::new(&auth_path).exists() {
        return Vec::new();
    }

    let mut migrated: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();
    let mut providers: Vec<String> = Vec::new();

    let mut oauth_readable = false;
    if Path::new(&oauth_path).exists() {
        if let Some(serde_json::Value::Object(oauth)) = read_json(&oauth_path) {
            for (provider, credential) in oauth {
                let mut entry: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();
                entry.insert("type".to_string(), serde_json::Value::String("oauth".to_string()));
                if let serde_json::Value::Object(fields) = credential {
                    for (key, value) in fields {
                        entry.insert(key, value);
                    }
                }
                migrated.insert(provider.clone(), serde_json::Value::Object(entry));
                providers.push(provider);
            }
            oauth_readable = true;
        }
    }

    let mut settings_without_api_keys: Option<String> = None;
    let mut settings_mode: Option<u32> = None;
    if Path::new(&settings_path).exists() {
        if let Some(mut settings) = read_json(&settings_path) {
            settings_mode = std::fs::metadata(&settings_path)
                .ok()
                .map(|metadata| mode_of(&metadata) & 0o777);
            if let Some(serde_json::Value::Object(api_keys)) = settings.get("apiKeys").cloned() {
                for (provider, key) in api_keys {
                    let already_migrated = migrated.contains_key(&provider);
                    if !already_migrated {
                        if let serde_json::Value::String(key) = key {
                            let mut entry: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();
                            entry.insert("type".to_string(), serde_json::Value::String("api_key".to_string()));
                            entry.insert("key".to_string(), serde_json::Value::String(key));
                            migrated.insert(provider.clone(), serde_json::Value::Object(entry));
                            providers.push(provider);
                        }
                    }
                }
                if let Some(object) = settings.as_object_mut() {
                    object.remove("apiKeys");
                }
                settings_without_api_keys = Some(stringify_pretty(&settings));
            }
        }
    }

    // The destination must be durable before any source is destroyed.
    if migrated.is_empty() {
        return Vec::new();
    }
    {
        if std::fs::create_dir_all(parent_dir(&auth_path)).is_err() {
            return Vec::new();
        }
        let real_path = realpath_if_present_sync(&auth_path).unwrap_or_else(|_| auth_path.clone());
        if write_auth(
            &real_path,
            &stringify_pretty(&serde_json::Value::Object(migrated)),
            WriteFileAtomicOptions {
                mode: Some(0o600),
                fsync: true,
                fsync_dir: true,
                before_rename: None,
            },
        ).is_err() {
            // Includes directory-sync failure after rename: keep both sources
            // and do not report success when durability is uncertain.
            return Vec::new();
        }
    }
    // Source cleanup is best-effort: with auth.json durable, leftovers are inert.
    if oauth_readable {
        let _ = std::fs::rename(&oauth_path, format!("{oauth_path}.migrated"));
    }
    if let Some(settings_without_api_keys) = settings_without_api_keys {
        let real_path = realpath_if_present_sync(&settings_path).unwrap_or_else(|_| settings_path.clone());
        let _ = write_file_atomic_sync(
            &real_path,
            &settings_without_api_keys,
            WriteFileAtomicOptions {
                mode: settings_mode,
                fsync: false,
                fsync_dir: false,
                before_rename: None,
            },
        );
    }

    providers
}

#[cfg(unix)]
fn mode_of(metadata: &std::fs::Metadata) -> u32 {
    use std::os::unix::fs::MetadataExt;
    metadata.mode()
}

#[cfg(not(unix))]
fn mode_of(_metadata: &std::fs::Metadata) -> u32 {
    0
}

/// `migrateSessionsFromAgentRoot`.
///
/// Bug in v0.30.0: Sessions were saved to ~/.pi/agent/ instead of
/// ~/.pi/agent/sessions/. This migration moves them to the configured session root.
pub fn migrate_sessions_from_agent_root() {
    let agent_dir = get_agent_dir();

    // Find all .jsonl files directly in agentDir (not in subdirectories)
    let Ok(entries) = std::fs::read_dir(&agent_dir) else {
        return;
    };
    let files: Vec<String> = entries
        .flatten()
        .map(|entry| entry.file_name().to_string_lossy().to_string())
        .filter(|name| name.ends_with(".jsonl"))
        .map(|name| join_path(&agent_dir, &name))
        .collect();

    if files.is_empty() {
        return;
    }

    for file in files {
        let Some(first_line) = read_first_line_sync(&file, 0) else {
            continue;
        };
        if first_line.trim().is_empty() {
            continue;
        }
        let Ok(header) = serde_json::from_str::<serde_json::Value>(&first_line) else {
            continue;
        };
        if header.get("type").and_then(|value| value.as_str()) != Some("session") {
            continue;
        }

        let correct_dir = get_sessions_dir(Some(&agent_dir));

        // Create directory if needed
        if !Path::new(&correct_dir).exists() {
            let _ = std::fs::create_dir_all(&correct_dir);
        }

        // Move the file
        let new_path = join_path(&correct_dir, &base_name(&file));

        if Path::new(&new_path).exists() {
            continue; // Skip if target exists
        }

        let _ = std::fs::rename(&file, &new_path);
    }
}

fn is_session_jsonl_file(file_path: &str) -> bool {
    let Some(first_line) = read_first_line_sync(file_path, 0) else {
        return false;
    };
    if first_line.trim().is_empty() {
        return false;
    }
    let Ok(header) = serde_json::from_str::<serde_json::Value>(&first_line) else {
        return false;
    };
    header.get("type").and_then(|value| value.as_str()) == Some("session")
        && header.get("id").map(|value| value.is_string()).unwrap_or(false)
}

/// `/^--.+--$/`.
fn is_legacy_session_dir_name(name: &str) -> bool {
    if !name.starts_with("--") || !name.ends_with("--") {
        return false;
    }
    name.len() > 4
}

/// `migrateLegacySessionDirsToSessionRoot`.
///
/// Older versions stored sessions under ~/.prime/agent/sessions/--cwd--/*.jsonl.
/// The daemon list/continue paths now scan the flat session root, so move any
/// existing nested JSONL session files up one level.
pub fn migrate_legacy_session_dirs_to_session_root() {
    let agent_dir = get_agent_dir();
    let sessions_dir = get_sessions_dir(Some(&agent_dir));

    let Ok(entries) = std::fs::read_dir(&sessions_dir) else {
        return;
    };
    let names: Vec<String> = entries
        .flatten()
        .filter(|entry| entry.file_type().map(|kind| kind.is_dir()).unwrap_or(false))
        .map(|entry| entry.file_name().to_string_lossy().to_string())
        .collect();

    for name in names {
        if !is_legacy_session_dir_name(&name) {
            continue;
        }

        let legacy_dir = join_path(&sessions_dir, &name);
        let Ok(legacy_entries) = std::fs::read_dir(&legacy_dir) else {
            continue;
        };
        let files: Vec<String> = legacy_entries
            .flatten()
            .map(|entry| entry.file_name().to_string_lossy().to_string())
            .filter(|file| file.ends_with(".jsonl"))
            .collect();

        for file in files {
            let old_path = join_path(&legacy_dir, &file);
            let mut new_path = join_path(&sessions_dir, &file);
            if !is_session_jsonl_file(&old_path) {
                continue;
            }
            if Path::new(&new_path).exists() {
                if files_have_same_content(&old_path, &new_path) {
                    // Already migrated; leave the legacy copy alone.
                    continue;
                }
                // A different session shares the basename; move it under a unique name
                // so it stays discoverable by the flat-root list and continue paths.
                new_path = unique_session_root_path(&sessions_dir, &file);
            }
            // Leave the legacy file in place if it cannot be moved.
            let _ = std::fs::rename(&old_path, &new_path);
        }

        // Ignore cleanup errors; migrated files are already in the flat root.
        if std::fs::read_dir(&legacy_dir)
            .map(|entries| entries.flatten().next().is_none())
            .unwrap_or(false)
        {
            let _ = std::fs::remove_dir(&legacy_dir);
        }
    }
}

fn files_have_same_content(a: &str, b: &str) -> bool {
    let (Ok(a_meta), Ok(b_meta)) = (std::fs::metadata(a), std::fs::metadata(b)) else {
        return false;
    };
    if a_meta.len() != b_meta.len() {
        return false;
    }
    match (std::fs::read_to_string(a), std::fs::read_to_string(b)) {
        (Ok(a_content), Ok(b_content)) => a_content == b_content,
        _ => false,
    }
}

fn unique_session_root_path(sessions_dir: &str, file: &str) -> String {
    let base = file.strip_suffix(".jsonl").unwrap_or(file);
    let mut n = 1usize;
    loop {
        let candidate = join_path(sessions_dir, &format!("{base}-{n}.jsonl"));
        if !Path::new(&candidate).exists() {
            return candidate;
        }
        n += 1;
    }
}

/// `migrateCommandsToPrompts`.
///
/// Works for both regular directories and symlinks.
fn migrate_commands_to_prompts(base_dir: &str, label: &str) -> bool {
    let commands_dir = join_path(base_dir, "commands");
    let prompts_dir = join_path(base_dir, "prompts");

    if Path::new(&commands_dir).exists() && !Path::new(&prompts_dir).exists() {
        match std::fs::rename(&commands_dir, &prompts_dir) {
            Ok(()) => {
                println!("{}", green(&format!("Migrated {label} commands/ \u{2192} prompts/")));
                return true;
            }
            Err(error) => {
                println!(
                    "{}",
                    yellow(&format!("Warning: Could not migrate {label} commands/ to prompts/: {error}"))
                );
            }
        }
    }
    false
}

fn migrate_keybindings_config_file() {
    let config_path = join_path(&get_agent_dir(), "keybindings.json");
    if !Path::new(&config_path).exists() {
        return;
    }

    let Ok(content) = std::fs::read_to_string(&config_path) else {
        return;
    };
    let parsed: serde_json::Value = match serde_json::from_str(&content) {
        Ok(parsed) => parsed,
        // Ignore malformed files during migration.
        Err(_) => return,
    };
    let serde_json::Value::Object(raw_config) = parsed else {
        return;
    };

    let (config, migrated) = migrate_keybindings_config(&raw_config);
    if !migrated {
        return;
    }
    let _ = std::fs::write(
        &config_path,
        format!("{}\n", stringify_pretty(&serde_json::Value::Object(config))),
    );
}

/// `migrateKeybindingsConfig` from `core/keybindings.ts`.
///
/// `core/keybindings.ts` belongs to another slice; this is the exact rename
/// table and ordering rule the migration needs, kept private to this module.
fn migrate_keybindings_config(
    raw_config: &serde_json::Map<String, serde_json::Value>,
) -> (serde_json::Map<String, serde_json::Value>, bool) {
    let mut config: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();
    let mut migrated = false;

    for (key, value) in raw_config {
        let next_key = KEYBINDING_NAME_MIGRATIONS
            .iter()
            .find(|(legacy, _)| legacy == key)
            .map(|(_legacy, next)| *next)
            .unwrap_or(key.as_str());
        if next_key != key {
            migrated = true;
        }
        if key != next_key && raw_config.contains_key(next_key) {
            migrated = true;
            continue;
        }
        config.insert(next_key.to_string(), value.clone());
    }

    (order_keybindings_config(config), migrated)
}

/// `Object.keys(KEYBINDINGS)` from `core/keybindings.ts`, which spreads
/// `TUI_KEYBINDINGS` first and then adds the app entries in this order.
/// `core/keybindings.ts` belongs to another slice; the order itself is what the
/// migration needs, so it is kept private here.
const APP_KEYBINDING_ORDER: [&str; 57] = [
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
    "app.jev.cancel",
];

fn order_keybindings_config(
    config: serde_json::Map<String, serde_json::Value>,
) -> serde_json::Map<String, serde_json::Value> {
    let mut ordered: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();
    for (keybinding, _definition) in pi_tui::keybindings::tui_keybindings() {
        if let Some(value) = config.get(&keybinding) {
            ordered.insert(keybinding.clone(), value.clone());
        }
    }
    for keybinding in APP_KEYBINDING_ORDER {
        if let Some(value) = config.get(keybinding) {
            ordered.insert(keybinding.to_string(), value.clone());
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

/// `KEYBINDING_NAME_MIGRATIONS` from `core/keybindings.ts`.
const KEYBINDING_NAME_MIGRATIONS: [(&str, &str); 59] = [
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

/// `migrateToolsToBin`.
fn migrate_tools_to_bin() {
    let agent_dir = get_agent_dir();
    let tools_dir = join_path(&agent_dir, "tools");
    let bin_dir = get_bin_dir();

    if !Path::new(&tools_dir).exists() {
        return;
    }

    let binaries = ["fd", "rg", "fd.exe", "rg.exe"];
    let mut moved_any = false;

    for bin in binaries {
        let old_path = join_path(&tools_dir, bin);
        let new_path = join_path(&bin_dir, bin);

        if !Path::new(&old_path).exists() {
            continue;
        }
        if !Path::new(&bin_dir).exists() {
            let _ = std::fs::create_dir_all(&bin_dir);
        }
        if !Path::new(&new_path).exists() {
            if std::fs::rename(&old_path, &new_path).is_ok() {
                moved_any = true;
            }
        } else {
            // Target exists, just delete the old one
            let _ = std::fs::remove_file(&old_path);
        }
    }

    if moved_any {
        println!("{}", green("Migrated managed binaries tools/ \u{2192} bin/"));
    }
}

/// `checkDeprecatedExtensionDirs`.
///
/// `tools/` may contain fd/rg binaries extracted by pi, so only warn if it has
/// other files.
fn check_deprecated_extension_dirs(base_dir: &str, label: &str) -> Vec<String> {
    let hooks_dir = join_path(base_dir, "hooks");
    let tools_dir = join_path(base_dir, "tools");
    let mut warnings: Vec<String> = Vec::new();

    if Path::new(&hooks_dir).exists() {
        warnings.push(format!("{label} hooks/ directory found. Hooks have been renamed to extensions."));
    }

    if Path::new(&tools_dir).exists() {
        // Check if tools/ contains anything other than fd/rg (auto-extracted binaries).
        if let Ok(entries) = std::fs::read_dir(&tools_dir) {
            let custom_tools: Vec<String> = entries
                .flatten()
                .map(|entry| entry.file_name().to_string_lossy().to_string())
                // Ignore .DS_Store and other hidden files
                .filter(|name| {
                    let lower = name.to_lowercase();
                    lower != "fd" && lower != "rg" && lower != "fd.exe" && lower != "rg.exe" && !name.starts_with('.')
                })
                .collect();
            if !custom_tools.is_empty() {
                warnings.push(format!(
                    "{label} tools/ directory contains custom tools. Custom tools have been merged into extensions."
                ));
            }
        }
    }

    warnings
}

/// `migrateExtensionSystem`: migrate commands/ to prompts/ and collect warnings
/// about deprecated directories.
fn migrate_extension_system(cwd: &str) -> Vec<String> {
    let agent_dir = get_agent_dir();
    let project_dir = join_path(cwd, CONFIG_DIR_NAME);

    // Migrate commands/ to prompts/
    migrate_commands_to_prompts(&agent_dir, "Global");
    migrate_commands_to_prompts(&project_dir, "Project");

    // Check for deprecated directories
    let mut warnings = check_deprecated_extension_dirs(&agent_dir, "Global");
    warnings.extend(check_deprecated_extension_dirs(&project_dir, "Project"));

    warnings
}

/// `showDeprecationWarnings`: print the warnings and wait for one keypress.
pub async fn show_deprecation_warnings(warnings: &[String]) {
    if warnings.is_empty() {
        return;
    }

    for warning in warnings {
        println!("{}", yellow(&format!("Warning: {warning}")));
    }
    println!("{}", yellow("\nMove your extensions to the extensions/ directory."));
    println!("{}", yellow(&format!("Migration guide: {MIGRATION_GUIDE_URL}")));
    println!("{}", yellow(&format!("Documentation: {EXTENSIONS_DOC_URL}")));
    println!("{}", dim("\nPress any key to continue..."));

    // `process.stdin.setRawMode(true)` then wait for one chunk of input.
    read_raw_keypress().await;
    println!();
}

async fn read_raw_keypress() {
    use std::io::Read;
    let stdin = std::io::stdin();
    let read = tokio::task::spawn_blocking(move || {
        let mut buffer = [0u8; 1];
        let mut handle = stdin.lock();
        let _ = handle.read(&mut buffer);
    });
    let _ = read.await;
}

/// `runMigrations`: called once on startup.
pub struct MigrationsResult {
    pub migrated_auth_providers: Vec<String>,
    pub deprecation_warnings: Vec<String>,
}

/// `runMigrations(cwd)`.
pub fn run_migrations(cwd: &str) -> MigrationsResult {
    let migrated_auth_providers = migrate_auth_to_auth_json();
    migrate_sessions_from_agent_root();
    migrate_legacy_session_dirs_to_session_root();
    migrate_tools_to_bin();
    migrate_keybindings_config_file();
    let deprecation_warnings = migrate_extension_system(cwd);
    MigrationsResult {
        migrated_auth_providers,
        deprecation_warnings,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Environment-variable tests share the process environment, so they take turns.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn temp_agent_dir(label: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "pi-migrations-{label}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn legacy_session_dir_names_match_the_typescript_regex() {
        assert!(is_legacy_session_dir_name("--cwd--"));
        assert!(is_legacy_session_dir_name("--a b c--"));
        assert!(!is_legacy_session_dir_name("--"));
        assert!(!is_legacy_session_dir_name("-cwd-"));
        assert!(!is_legacy_session_dir_name("cwd"));
        assert!(!is_legacy_session_dir_name("--cwd"));
    }

    #[test]
    fn unique_session_root_path_skips_existing_candidates() {
        let dir = temp_agent_dir("unique");
        let dir_string = dir.to_string_lossy().to_string();
        let existing = join_path(&dir_string, "s-1.jsonl");
        std::fs::write(&existing, "").unwrap();
        let next = unique_session_root_path(&dir_string, "s.jsonl");
        assert_eq!(next, join_path(&dir_string, "s-2.jsonl"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn keybinding_migration_renames_and_orders() {
        let mut raw: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();
        raw.insert("cursorUp".to_string(), serde_json::json!("ctrl+p"));
        raw.insert("zzzCustom".to_string(), serde_json::json!("ctrl+z"));
        let (config, migrated) = migrate_keybindings_config(&raw);
        assert!(migrated);
        let keys: Vec<&String> = config.keys().collect();
        assert!(keys.contains(&&"tui.editor.cursorUp".to_string()));
        assert!(!keys.contains(&&"cursorUp".to_string()));
        // Ordering: known keybindings first, then unknown extras sorted.
        let cursor_index = keys.iter().position(|key| *key == "tui.editor.cursorUp").unwrap();
        let extra_index = keys.iter().position(|key| *key == "zzzCustom").unwrap();
        assert!(cursor_index < extra_index);
    }

    #[test]
    fn keybinding_migration_keeps_an_existing_target_and_drops_the_legacy_key() {
        let mut raw: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();
        raw.insert("cursorUp".to_string(), serde_json::json!("ctrl+p"));
        raw.insert("tui.editor.cursorUp".to_string(), serde_json::json!("up"));
        let (config, migrated) = migrate_keybindings_config(&raw);
        assert!(migrated);
        assert_eq!(config.get("tui.editor.cursorUp"), Some(&serde_json::json!("up")));
        assert!(!config.contains_key("cursorUp"));
    }

    #[test]
    fn keybinding_migration_reports_no_change_for_clean_configs() {
        let mut raw: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();
        raw.insert("tui.editor.cursorUp".to_string(), serde_json::json!("up"));
        let (_config, migrated) = migrate_keybindings_config(&raw);
        assert!(!migrated);
    }

    #[test]
    fn auth_migration_preserves_sources_on_destination_failure() {
        for after_commit in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let oauth = r#"{"synthetic":{"access":"test-only"}}"#;
            let settings = r#"{"apiKeys":{"other":"test-only"},"theme":"dark"}"#;
            std::fs::write(dir.path().join("oauth.json"), oauth).unwrap();
            std::fs::write(dir.path().join("settings.json"), settings).unwrap();
            let result = migrate_auth_in_dir(dir.path().to_str().unwrap(), |path, data, options| {
                assert!(options.fsync && options.fsync_dir);
                assert_eq!(options.mode, Some(0o600));
                if after_commit { std::fs::write(path, data)?; }
                Err(std::io::Error::other("injected destination failure"))
            });
            assert!(result.is_empty());
            assert_eq!(std::fs::read_to_string(dir.path().join("oauth.json")).unwrap(), oauth);
            assert_eq!(std::fs::read_to_string(dir.path().join("settings.json")).unwrap(), settings);
            assert!(!dir.path().join("oauth.json.migrated").exists());
        }
    }

    #[test]
    fn auth_migration_writes_auth_json_and_removes_the_api_keys_block() {
        let _guard = ENV_LOCK.lock().unwrap();
        let dir = temp_agent_dir("auth");
        let dir_string = dir.to_string_lossy().to_string();
        std::env::set_var("PRIME_AGENT_CODING_AGENT_DIR", &dir_string);

        std::fs::write(
            join_path(&dir_string, "oauth.json"),
            "{\"anthropic\":{\"access\":\"a\"}}",
        )
        .unwrap();
        std::fs::write(
            join_path(&dir_string, "settings.json"),
            "{\"apiKeys\":{\"openai\":\"sk-1\"},\"theme\":\"dark\"}",
        )
        .unwrap();

        let providers = migrate_auth_to_auth_json();
        assert_eq!(providers, vec!["anthropic".to_string(), "openai".to_string()]);

        let auth: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(join_path(&dir_string, "auth.json")).unwrap()).unwrap();
        assert_eq!(auth["anthropic"]["type"], serde_json::json!("oauth"));
        assert_eq!(auth["anthropic"]["access"], serde_json::json!("a"));
        assert_eq!(auth["openai"]["type"], serde_json::json!("api_key"));
        assert_eq!(auth["openai"]["key"], serde_json::json!("sk-1"));

        assert!(std::path::Path::new(&format!("{}/oauth.json.migrated", dir_string)).exists());
        let settings: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(join_path(&dir_string, "settings.json")).unwrap()).unwrap();
        assert!(settings.get("apiKeys").is_none());
        assert_eq!(settings["theme"], serde_json::json!("dark"));

        // A second run is a no-op because auth.json now exists.
        assert!(migrate_auth_to_auth_json().is_empty());

        std::env::remove_var("PRIME_AGENT_CODING_AGENT_DIR");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn session_root_migration_moves_legacy_dirs_into_the_flat_root() {
        let _guard = ENV_LOCK.lock().unwrap();
        let dir = temp_agent_dir("sessions");
        let dir_string = dir.to_string_lossy().to_string();
        let previous_agent_dir = std::env::var_os("PRIME_AGENT_CODING_AGENT_DIR");
        let previous_session_dir = std::env::var_os("PRIME_AGENT_SESSION_DIR");
        std::env::set_var("PRIME_AGENT_CODING_AGENT_DIR", &dir_string);

        let sessions = join_path(&dir_string, "sessions");
        // The configured session root takes precedence over the agent directory.
        // Keep this fixture isolated when the test runner supplies that override.
        std::env::set_var("PRIME_AGENT_SESSION_DIR", &sessions);
        let legacy = join_path(&sessions, "--tmp-cwd--");
        std::fs::create_dir_all(&legacy).unwrap();
        std::fs::write(
            join_path(&legacy, "s.jsonl"),
            "{\"type\":\"session\",\"id\":\"abc\"}\n{\"type\":\"message\"}\n",
        )
        .unwrap();
        // A file that is not a session header stays behind.
        std::fs::write(join_path(&legacy, "other.jsonl"), "{\"type\":\"message\"}\n").unwrap();

        migrate_legacy_session_dirs_to_session_root();

        assert!(std::path::Path::new(&join_path(&sessions, "s.jsonl")).exists());
        assert!(!std::path::Path::new(&join_path(&legacy, "s.jsonl")).exists());
        assert!(std::path::Path::new(&join_path(&legacy, "other.jsonl")).exists());

        match previous_agent_dir {
            Some(value) => std::env::set_var("PRIME_AGENT_CODING_AGENT_DIR", value),
            None => std::env::remove_var("PRIME_AGENT_CODING_AGENT_DIR"),
        }
        match previous_session_dir {
            Some(value) => std::env::set_var("PRIME_AGENT_SESSION_DIR", value),
            None => std::env::remove_var("PRIME_AGENT_SESSION_DIR"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn deprecated_extension_dirs_warn_only_for_custom_tools() {
        let dir = temp_agent_dir("deprecated");
        let dir_string = dir.to_string_lossy().to_string();
        assert!(check_deprecated_extension_dirs(&dir_string, "Global").is_empty());

        std::fs::create_dir_all(join_path(&dir_string, "hooks")).unwrap();
        let warnings = check_deprecated_extension_dirs(&dir_string, "Global");
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("hooks/ directory found"));

        std::fs::create_dir_all(join_path(&dir_string, "tools")).unwrap();
        std::fs::write(join_path(&dir_string, "tools/rg.exe"), "").unwrap();
        assert_eq!(check_deprecated_extension_dirs(&dir_string, "Global").len(), 1);

        std::fs::write(join_path(&dir_string, "tools/custom.js"), "").unwrap();
        let warnings = check_deprecated_extension_dirs(&dir_string, "Global");
        assert_eq!(warnings.len(), 2);
        assert!(warnings[1].contains("contains custom tools"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn migration_result_reports_both_fields() {
        let result = MigrationsResult {
            migrated_auth_providers: Vec::new(),
            deprecation_warnings: Vec::new(),
        };
        assert!(result.migrated_auth_providers.is_empty());
        assert!(result.deprecation_warnings.is_empty());
    }
}
