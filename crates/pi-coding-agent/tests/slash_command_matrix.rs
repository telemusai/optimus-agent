//! Validate the native command registry and dispatch owners without the removed
//! TypeScript reference tree. Handler presence is not a runtime behavior test;
//! individual commands retain their focused native tests.
use pi_coding_agent::core::slash_commands::{
    builtin_slash_commands, is_builtin_slash_command_name, parse_session_slash_command,
    parse_slash_command, resolve_builtin_slash_command_name, resolve_leading_builtin_slash_command,
};
use std::collections::BTreeSet;

const HOST: &str = include_str!("../src/modes/interactive/native_host.rs");
const COMMANDS: &[&str] = &[
    "autonomous",
    "btw",
    "changelog",
    "clone",
    "compact",
    "context",
    "copy",
    "effort",
    "export",
    "fast",
    "fork",
    "fullscreen",
    "goal",
    "heartbeat",
    "heartbeats",
    "hotkeys",
    "import",
    "jev",
    "login",
    "logout",
    "logs",
    "mcp",
    "model",
    "monitor",
    "name",
    "new",
    "quit",
    "refine",
    "reload",
    "resume",
    "rlm-max-depth",
    "scoped-models",
    "session",
    "settings",
    "share",
    "system-prompt",
    "traces",
    "tree",
    "update",
];

fn handler_arms(source: &str) -> BTreeSet<String> {
    let start = source
        .find("async fn run_builtin_command(")
        .expect("native command dispatcher");
    let end = source[start..]
        .find("\n}")
        .expect("dispatcher closing brace")
        + start;
    source[start..end]
        .lines()
        .filter(|line| line.starts_with("        \""))
        .filter_map(|line| line.split_once("=>"))
        .flat_map(|(pattern, _)| pattern.split('|'))
        .filter_map(|pattern| {
            pattern
                .trim()
                .strip_prefix('"')
                .and_then(|p| p.split_once('"'))
                .map(|(name, _)| name.to_owned())
        })
        .collect()
}

fn has_owner(name: &str, source: &str) -> bool {
    match name {
        "quit" => source.contains("matches!(text.trim(), \"/quit\" | \"/exit\")"),
        "hotkeys" => source.contains("if text.trim() == \"/hotkeys\""),
        "login" => source.contains("if name == \"login\""),
        _ => handler_arms(source).contains(name),
    }
}

#[test]
fn every_native_command_and_alias_has_a_dispatch_owner() {
    let registry = builtin_slash_commands();
    let names: BTreeSet<_> = registry
        .iter()
        .map(|command| command.name.as_str())
        .collect();
    assert_eq!(names, COMMANDS.iter().copied().collect());
    assert_eq!(names.len(), registry.len(), "duplicate canonical commands");
    let aliases: BTreeSet<_> = registry
        .iter()
        .flat_map(|command| {
            command
                .aliases
                .clone()
                .unwrap_or_default()
                .into_iter()
                .map(|alias| (alias, command.name.clone()))
        })
        .collect();
    assert_eq!(
        aliases,
        [
            ("clear", "new"),
            ("rename", "name"),
            ("side", "btw"),
            ("thinking", "effort"),
            ("usage", "context")
        ]
        .into_iter()
        .map(|(alias, name)| (alias.to_string(), name.to_string()))
        .collect()
    );
    for command in registry {
        let mut forms = vec![command.name.clone()];
        forms.extend(command.aliases.clone().unwrap_or_default());
        for name in forms {
            let text = format!("/{name} sample-argument");
            let parsed = parse_slash_command(&text).expect("command parses");
            assert_eq!(parsed.name, name);
            assert!(is_builtin_slash_command_name(&name));
            assert_eq!(resolve_builtin_slash_command_name(&name), command.name);
            let leading = resolve_leading_builtin_slash_command(&text).unwrap();
            assert_eq!(leading.name, command.name);
            assert_eq!(leading.is_alias, name != command.name);
            if command.execution.as_deref() == Some("session") {
                assert_eq!(
                    parse_session_slash_command(&text).unwrap().name,
                    command.name
                );
            } else {
                assert!(parse_session_slash_command(&text).is_none());
                assert!(
                    has_owner(&command.name, HOST),
                    "No native dispatch owner for /{name}"
                );
            }
        }
    }
}

#[test]
fn missing_handlers_are_detected_in_direct_grouped_and_input_dispatch() {
    assert!(has_owner("settings", HOST));
    let missing = HOST.replace(
        "        \"settings\" => {",
        "        \"missing-settings\" => {",
    );
    assert!(!has_owner("settings", &missing));
    assert!(has_owner("monitor", HOST));
    let missing = HOST.replace("| \"monitor\" |", "| \"missing-monitor\" |");
    assert!(!has_owner("monitor", &missing));
    assert!(has_owner("hotkeys", HOST));
    let missing = HOST.replace(
        "if text.trim() == \"/hotkeys\"",
        "if text.trim() == \"/missing-hotkeys\"",
    );
    assert!(!has_owner("hotkeys", &missing));
    assert!(has_owner("context", &missing));
}
