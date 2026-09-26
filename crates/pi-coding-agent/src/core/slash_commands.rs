//! Port of packages/coding-agent/src/core/slash-commands.ts

use std::collections::HashMap;
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use crate::core::source_info::SourceInfo;

/// `APP_NAME` from `config.ts` (package.json `piConfig.name` = "prime-agent").
pub const APP_NAME: &str = "prime-agent";

pub type SlashCommandSource = String;

pub const SLASH_COMMAND_SOURCE_EXTENSION: &str = "extension";
pub const SLASH_COMMAND_SOURCE_PROMPT: &str = "prompt";
pub const SLASH_COMMAND_SOURCE_SKILL: &str = "skill";

/// `SlashCommandInfo`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SlashCommandInfo {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub source: SlashCommandSource,
    #[serde(rename = "sourceInfo")]
    pub source_info: SourceInfo,
}

pub const SESSION_SLASH_COMMAND_NAMES: [&str; 5] = ["compact", "refine", "goal", "autonomous", "mode"];

pub type SessionSlashCommandName = String;

pub fn is_session_slash_command_name(value: &str) -> bool {
    SESSION_SLASH_COMMAND_NAMES.contains(&value)
}

/// `SessionSlashCommand`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionSlashCommand {
    pub name: SessionSlashCommandName,
    pub args: String,
    pub text: String,
}

/// `RefineCommandOptions`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RefineCommandOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rollback_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub global: Option<bool>,
}

/// Slash-command argument separator class: `[\t\p{Zs}]`.
fn is_argument_separator(ch: char) -> bool {
    ch == '\t' || (ch.is_whitespace() && ch != '\n' && ch != '\r' && ch != '\u{2028}' && ch != '\u{2029}')
}

/// `parseRefineCommandOptions`.
pub fn parse_refine_command_options(args: &str) -> Result<RefineCommandOptions, String> {
    let mut rest = args.trim().to_string();
    let mut global = false;
    if starts_with_global_flag(&rest) {
        global = true;
        rest = strip_global_flag(&rest).trim().to_string();
    }
    if rest == "rollback" {
        return Err("Usage: /refine rollback <refinement-id>".to_string());
    }
    // Slash-command args keep their original separators (tabs, Unicode spaces);
    // match the subcommand with the same class parseSlashCommand splits on.
    if let Some(matched) = rollback_prefix(&rest) {
        let mut rollback_id = rest[matched.len()..].trim().to_string();
        if rollback_id == "--global" {
            return Err("Usage: /refine rollback <refinement-id>".to_string());
        }
        if has_trailing_global_flag(&rollback_id) {
            global = true;
            rollback_id = strip_trailing_global_flag(&rollback_id).trim().to_string();
        }
        if rollback_id.is_empty() {
            return Err("Usage: /refine rollback <refinement-id>".to_string());
        }
        return Ok(RefineCommandOptions {
            instructions: None,
            rollback_id: Some(rollback_id),
            global: Some(global),
        });
    }
    Ok(RefineCommandOptions {
        instructions: if rest.is_empty() { None } else { Some(rest) },
        rollback_id: None,
        global: Some(global),
    })
}

fn starts_with_global_flag(value: &str) -> bool {
    let Some(rest) = value.strip_prefix("--global") else {
        return false;
    };
    rest.is_empty() || rest.chars().next().map(is_whitespace_class).unwrap_or(false)
}

fn strip_global_flag(value: &str) -> String {
    value["--global".len()..].to_string()
}

fn is_whitespace_class(ch: char) -> bool {
    ch.is_whitespace()
}

fn rollback_prefix(value: &str) -> Option<String> {
    let rest = value.strip_prefix("rollback")?;
    let next = rest.chars().next()?;
    if is_argument_separator(next) {
        Some(format!("rollback{next}"))
    } else {
        None
    }
}

fn has_trailing_global_flag(value: &str) -> bool {
    let mut chars: Vec<char> = value.chars().collect();
    let flag: Vec<char> = "--global".chars().collect();
    if chars.len() < flag.len() + 1 {
        return false;
    }
    if chars[chars.len() - flag.len()..] != flag[..] {
        return false;
    }
    let before = chars[chars.len() - flag.len() - 1];
    chars.truncate(chars.len() - flag.len() - 1);
    is_whitespace_class(before)
}

fn strip_trailing_global_flag(value: &str) -> String {
    let mut chars: Vec<char> = value.chars().collect();
    let flag: Vec<char> = "--global".chars().collect();
    if chars.len() >= flag.len() + 1 && chars[chars.len() - flag.len()..] == flag[..] {
        chars.truncate(chars.len() - flag.len() - 1);
    }
    chars.into_iter().collect()
}

/// `BuiltinSlashCommand`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BuiltinSlashCommand {
    pub name: String,
    pub description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution: Option<String>,
    /// Shown in autocomplete before the description, e.g. "[instructions]".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub argument_hint: Option<String>,
    /// Hidden names that resolve to this command without being shown as commands.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aliases: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub takes_argument: Option<bool>,
}

/// `ParsedSlashCommand`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParsedSlashCommand {
    pub name: String,
    pub args: String,
}

/// `ResolvedSlashCommand`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResolvedSlashCommand {
    pub name: String,
    pub args: String,
    pub original_name: String,
    pub is_alias: bool,
}

struct BuiltinSlashCommandAlias {
    name: &'static str,
    alias_for: &'static str,
}

/// Canonical command list, in declaration order.
fn canonical_builtin_slash_commands() -> Vec<BuiltinSlashCommand> {
    let mut commands: Vec<BuiltinSlashCommand> = Vec::new();
    let mut push = |name: &str,
                    description: String,
                    argument_hint: Option<&str>,
                    takes_argument: Option<bool>| {
        commands.push(BuiltinSlashCommand {
            name: name.to_string(),
            description,
            execution: None,
            argument_hint: argument_hint.map(str::to_string),
            aliases: None,
            takes_argument,
        });
    };
    push("settings", "Open settings menu".to_string(), None, None);
    push(
        "model",
        "Select model (opens selector UI)".to_string(),
        Some("[search]"),
        Some(true),
    );
    push(
        "effort",
        "Select reasoning/thinking level (opens selector UI)".to_string(),
        Some("[level]"),
        None,
    );
    push("fast", "Toggle OpenAI Fast mode".to_string(), None, None);
    push("mode", "Switch between IPython and Direct tools for this chat".into(), Some("[ipython|direct|toggle]"), Some(true));
    push(
        "scoped-models",
        "Enable/disable models for Ctrl+P cycling".to_string(),
        None,
        None,
    );
    push(
        "export",
        "Export session (HTML default, or specify path: .html/.jsonl)".to_string(),
        Some("[path]"),
        Some(true),
    );
    push(
        "import",
        "Import and resume a session from a JSONL file".to_string(),
        Some("<path.jsonl>"),
        Some(true),
    );
    push(
        "share",
        "Share session as a secret GitHub gist".to_string(),
        None,
        None,
    );
    push(
        "copy",
        "Copy last agent message to clipboard".to_string(),
        None,
        None,
    );
    push(
        "btw",
        "Ask a side question without adding it to the session; replies follow up, esc returns".to_string(),
        Some("<question>"),
        Some(true),
    );
    push(
        "name",
        "Set or show the session display name".to_string(),
        Some("[name]"),
        Some(true),
    );
    push("session", "Show session info".to_string(), None, None);
    push("stats", "Open live model usage, context, and JEV statistics; Esc closes".to_string(), None, None);
    push(
        "system-prompt",
        "Show the exact system prompt sent to the model".to_string(),
        None,
        None,
    );
    push(
        "logs",
        "Show where daemon and client logs are saved".to_string(),
        None,
        None,
    );
    push(
        "traces",
        "Preview, upload, or configure Prime Agent traces".to_string(),
        Some("[status|on|off|preview|upload|upload-current|upload-all|login]"),
        None,
    );
    push(
        "monitor",
        "Turn local performance monitoring ON or OFF".to_string(),
        Some("[status|on|off]"),
        None,
    );
    push(
        "context",
        "Show token, cost, and context usage for agent and sub-agents".to_string(),
        None,
        None,
    );
    push("changelog", "Show changelog entries".to_string(), None, None);
    push(
        "update",
        format!("Update {APP_NAME} and installed packages"),
        Some("[source|--self|--extensions]"),
        Some(true),
    );
    push(
        "hotkeys",
        "Show all keyboard shortcuts".to_string(),
        None,
        None,
    );
    push(
        "fork",
        "Create a new fork from a previous user message".to_string(),
        None,
        None,
    );
    push(
        "clone",
        "Duplicate the current session at the current position".to_string(),
        None,
        None,
    );
    push(
        "tree",
        "Navigate session tree (switch branches)".to_string(),
        None,
        None,
    );
    push(
        "login",
        "Configure provider authentication".to_string(),
        None,
        None,
    );
    push(
        "logout",
        "Remove provider authentication".to_string(),
        None,
        None,
    );
    push(
        "mcp",
        "Open MCP Connections or manage MCP integrations".to_string(),
        Some("[add|list|get|remove|login|logout]"),
        Some(true),
    );
    push(
        "new",
        "Start a new session, optionally named and/or with an initial prompt".to_string(),
        Some("[--name \"session name\" --] [prompt]"),
        Some(true),
    );
    push(
        "compact",
        "Compact the session context; optional instructions focus the summary".to_string(),
        Some("[instructions]"),
        None,
    );
    push(
        "refine",
        "Refine continual harness prompt notes, skills, subagents, and memory".to_string(),
        None,
        None,
    );
    push(
        "goal",
        "Set or view a persistent goal; supports pause, resume, and clear".to_string(),
        Some("[objective]"),
        Some(true),
    );
    push(
        "autonomous",
        "Set or view autonomous mode".to_string(),
        Some("[status|on|off]"),
        Some(true),
    );
    push(
        "rlm-max-depth",
        "Set/view the per-chat persistent RLM max depth immediately; never interrupts or queues the running turn"
            .to_string(),
        Some("[<int> [--global]]"),
        Some(true),
    );
    push(
        "heartbeat",
        "Set or view a persistent heartbeat; delivery defaults to steer, use --follow-up to queue; supports pause, resume, stop, and clear".to_string(),
        Some("[status|pause|resume|stop|[every <duration>] [--steer|--follow-up] <instruction>]"),
        Some(true),
    );
    push(
        "heartbeats",
        "View and manage all user and agent heartbeats".to_string(),
        None,
        None,
    );
    push(
        "resume",
        "Open the agents view, or resume a session by id or path".to_string(),
        Some("[id|path]"),
        Some(true),
    );
    push(
        "reload",
        "Reload keybindings, extensions, skills, prompts, and themes".to_string(),
        None,
        None,
    );
    push(
        "fullscreen",
        "Toggle fullscreen (alternate screen) rendering with scrollable transcript".to_string(),
        Some("[on|off]"),
        Some(true),
    );
    // SHARED FILE EDIT (core/slash_commands.rs, jev-ui lane): the canonical `/jev`
    // entry. It is a plain local built-in (no `execution: "session"`, so it is
    // resolved locally and never forwarded to the model). `takes_argument` is true
    // because `/jev <mode>` is a real form, and the argument hint lists exactly
    // what the UI accepts.
    push(
        "jev",
        "Jev System One: Off, Compare, Active, Compare + Active, compaction, feature gates, the global full-jev overlay, API key, the requested Jev model (model status/set/reset) and the model catalog"
            .to_string(),
        Some("[off|compare|active|compare-active|on|compact|feature|default|full-jev|status|key|models|model]"),
        Some(true),
    );
    push("quit", format!("Quit {APP_NAME}"), None, None);
    commands
}

fn builtin_slash_command_aliases() -> Vec<BuiltinSlashCommandAlias> {
    vec![
        BuiltinSlashCommandAlias { name: "clear", alias_for: "new" },
        BuiltinSlashCommandAlias { name: "usage", alias_for: "context" },
        BuiltinSlashCommandAlias { name: "thinking", alias_for: "effort" },
        BuiltinSlashCommandAlias { name: "rename", alias_for: "name" },
        BuiltinSlashCommandAlias { name: "side", alias_for: "btw" },
    ]
}

fn build_builtin_slash_commands() -> Result<Vec<BuiltinSlashCommand>, String> {
    let canonical = canonical_builtin_slash_commands();
    let mut canonical_by_name: HashMap<&str, usize> = HashMap::new();
    for (index, command) in canonical.iter().enumerate() {
        canonical_by_name.insert(command.name.as_str(), index);
    }
    let mut aliases_by_target: HashMap<String, Vec<String>> = HashMap::new();
    for alias in builtin_slash_command_aliases() {
        let Some(_) = canonical_by_name.get(alias.alias_for) else {
            return Err(format!(
                "Slash command alias '/{}' targets unknown command '/{}'",
                alias.name, alias.alias_for
            ));
        };
        aliases_by_target
            .entry(alias.alias_for.to_string())
            .or_default()
            .push(alias.name.to_string());
    }
    Ok(canonical
        .into_iter()
        .map(|mut command| {
            if is_session_slash_command_name(&command.name) {
                command.execution = Some("session".to_string());
            }
            if let Some(aliases) = aliases_by_target.get(&command.name) {
                command.aliases = Some(aliases.clone());
            }
            command
        })
        .collect())
}

fn builtin_commands() -> &'static Vec<BuiltinSlashCommand> {
    static COMMANDS: OnceLock<Vec<BuiltinSlashCommand>> = OnceLock::new();
    COMMANDS.get_or_init(|| build_builtin_slash_commands().expect("builtin slash commands must build"))
}

/// `BUILTIN_SLASH_COMMANDS`.
pub fn builtin_slash_commands() -> &'static Vec<BuiltinSlashCommand> {
    builtin_commands()
}

fn builtin_by_name() -> &'static HashMap<String, &'static BuiltinSlashCommand> {
    static BY_NAME: OnceLock<HashMap<String, &'static BuiltinSlashCommand>> = OnceLock::new();
    BY_NAME.get_or_init(|| {
        builtin_commands()
            .iter()
            .map(|command| (command.name.clone(), command))
            .collect()
    })
}

fn alias_to_name() -> &'static HashMap<String, String> {
    static ALIASES: OnceLock<HashMap<String, String>> = OnceLock::new();
    ALIASES.get_or_init(|| {
        let mut map = HashMap::new();
        for command in builtin_commands() {
            if let Some(aliases) = &command.aliases {
                for alias in aliases {
                    map.insert(alias.clone(), command.name.clone());
                }
            }
        }
        map
    })
}

/// `parseSlashCommand`: `/name args` with `\S+` name and trimmed args.
pub fn parse_slash_command(text: &str) -> Option<ParsedSlashCommand> {
    if !text.starts_with('/') {
        return None;
    }
    let rest = &text[1..];
    // `^\/(\S+)(?:\s+([\s\S]*))?$`
    let name_end = rest
        .char_indices()
        .find(|(_, ch)| ch.is_whitespace())
        .map(|(index, _)| index)
        .unwrap_or(rest.len());
    let name = &rest[..name_end];
    if name.is_empty() {
        return None;
    }
    let tail = &rest[name_end..];
    let args = if tail.is_empty() {
        String::new()
    } else {
        // `\s+` then `[\s\S]*` to end of string.
        if !tail.chars().next().map(char::is_whitespace).unwrap_or(false) {
            return None;
        }
        tail.trim_start_matches(char::is_whitespace).to_string()
    };
    // Model IDs are exact identifiers. Preserve trailing bytes only for the
    // narrow `/jev model set <id>` arm so whitespace-bearing IDs reach its
    // validator and are refused instead of being silently normalized into a
    // different accepted ID. Every other slash command keeps the established
    // outer-lexer trimming rule.
    let preserve_model_id_trailing = name == "jev"
        && args
            .get(.."model set ".len())
            .map(|prefix| prefix.eq_ignore_ascii_case("model set "))
            .unwrap_or(false);
    Some(ParsedSlashCommand {
        name: name.to_string(),
        args: if preserve_model_id_trailing {
            args
        } else {
            args.trim().to_string()
        },
    })
}

/// `resolveBuiltinSlashCommandName`.
pub fn resolve_builtin_slash_command_name(name: &str) -> String {
    alias_to_name()
        .get(name)
        .cloned()
        .unwrap_or_else(|| name.to_string())
}

/// `isBuiltinSlashCommandName`.
pub fn is_builtin_slash_command_name(name: &str) -> bool {
    builtin_by_name().contains_key(name) || alias_to_name().contains_key(name)
}

/// `builtinSlashCommandTakesArgument`.
pub fn builtin_slash_command_takes_argument(name: &str) -> bool {
    // /clear remains the no-argument compatibility alias even though /new accepts arguments.
    if name == "clear" {
        return false;
    }
    builtin_by_name()
        .get(&resolve_builtin_slash_command_name(name))
        .map(|command| command.takes_argument == Some(true))
        .unwrap_or(false)
}

/// `parseSlashCommand` + `resolveBuiltinSlashCommandName` for a submitted line
/// (interactive-mode.ts:4784-4786): `/name args` resolves its alias and reports
/// the canonical name, the original name, and the trimmed arguments.
///
/// Returns `None` when the line is not a slash command at all, or when the name
/// is neither a built-in nor an alias, so free text and extension commands keep
/// reaching the model.
pub fn resolve_leading_builtin_slash_command(text: &str) -> Option<ResolvedSlashCommand> {
    let parsed = parse_slash_command(text)?;
    if !is_builtin_slash_command_name(&parsed.name) {
        return None;
    }
    Some(resolve_slash_command(&parsed))
}

/// `resolveSlashCommand`.
pub fn resolve_slash_command(command: &ParsedSlashCommand) -> ResolvedSlashCommand {
    let name = resolve_builtin_slash_command_name(&command.name);
    ResolvedSlashCommand {
        name: name.clone(),
        args: command.args.clone(),
        original_name: command.name.clone(),
        is_alias: name != command.name,
    }
}

/// `parseSessionSlashCommand`.
pub fn parse_session_slash_command(text: &str) -> Option<SessionSlashCommand> {
    if text
        .chars()
        .any(|ch| ch == '\r' || ch == '\n' || ch == '\u{2028}' || ch == '\u{2029}')
    {
        return None;
    }
    let parsed = parse_slash_command(text)?;
    let name = resolve_builtin_slash_command_name(&parsed.name);
    let command = builtin_by_name().get(&name)?;
    if command.execution.as_deref() != Some("session") || !is_session_slash_command_name(&name) {
        return None;
    }
    Some(SessionSlashCommand {
        name,
        args: parsed.args,
        text: text.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_commands_keep_declaration_order_and_aliases() {
        let names: Vec<&str> = builtin_slash_commands()
            .iter()
            .map(|command| command.name.as_str())
            .collect();
        assert_eq!(names[0], "settings");
        assert_eq!(names[1], "model");
        assert_eq!(names.last().copied(), Some("quit"));

        let new = builtin_slash_commands()
            .iter()
            .find(|command| command.name == "new")
            .unwrap();
        assert_eq!(new.aliases.as_deref(), Some(["clear".to_string()].as_slice()));

        let compact = builtin_slash_commands()
            .iter()
            .find(|command| command.name == "compact")
            .unwrap();
        assert_eq!(compact.execution.as_deref(), Some("session"));
        assert_eq!(compact.takes_argument, None);
    }

    #[test]
    fn app_name_is_interpolated_into_descriptions() {
        let update = builtin_slash_commands()
            .iter()
            .find(|command| command.name == "update")
            .unwrap();
        assert_eq!(update.description, "Update prime-agent and installed packages");
        let quit = builtin_slash_commands()
            .iter()
            .find(|command| command.name == "quit")
            .unwrap();
        assert_eq!(quit.description, "Quit prime-agent");
    }

    #[test]
    fn parse_slash_command_trims_arguments() {
        let parsed = parse_slash_command("/compact   keep the plan  ").unwrap();
        assert_eq!(parsed.name, "compact");
        assert_eq!(parsed.args, "keep the plan");
        let no_args = parse_slash_command("/status").unwrap();
        assert_eq!(no_args.args, "");
        assert!(parse_slash_command("not a command").is_none());
        assert!(parse_slash_command("/").is_none());
    }

    #[test]
    fn jev_model_set_preserves_trailing_id_whitespace_for_refusal() {
        let trailing_space = parse_slash_command("/jev model set jev-safe ").unwrap();
        assert_eq!(trailing_space.args, "model set jev-safe ");
        let trailing_tab = parse_slash_command("/jev model set jev-safe\t").unwrap();
        assert_eq!(trailing_tab.args, "model set jev-safe\t");
        let safe = parse_slash_command("/jev model set jev-safe").unwrap();
        assert_eq!(safe.args, "model set jev-safe");
        // The established lexer rule is unchanged outside this narrow arm.
        assert_eq!(
            parse_slash_command("/compact keep the plan  ").unwrap().args,
            "keep the plan"
        );
    }

    #[test]
    fn resolve_and_alias_helpers_match_the_typescript() {
        assert_eq!(resolve_builtin_slash_command_name("clear"), "new");
        assert_eq!(resolve_builtin_slash_command_name("usage"), "context");
        assert_eq!(resolve_builtin_slash_command_name("model"), "model");
        assert!(is_builtin_slash_command_name("clear"));
        assert!(is_builtin_slash_command_name("thinking"));
        assert!(!is_builtin_slash_command_name("nope"));
        assert!(!builtin_slash_command_takes_argument("clear"));
        assert!(builtin_slash_command_takes_argument("new"));
        assert!(!builtin_slash_command_takes_argument("settings"));

        let resolved = resolve_slash_command(&ParsedSlashCommand {
            name: "clear".to_string(),
            args: String::new(),
        });
        assert_eq!(resolved.name, "new");
        assert_eq!(resolved.original_name, "clear");
        assert!(resolved.is_alias);
    }

    #[test]
    fn parse_session_slash_command_filters_by_execution_mode() {
        let parsed = parse_session_slash_command("/compact now").unwrap();
        assert_eq!(parsed.name, "compact");
        assert_eq!(parsed.args, "now");
        assert_eq!(parsed.text, "/compact now");
        assert!(parse_session_slash_command("/model").is_none());
        assert!(parse_session_slash_command("/compact\nsecond line").is_none());
    }

    #[test]
    fn parse_refine_command_options_handles_global_and_rollback() {
        let plain = parse_refine_command_options(" do the thing ").unwrap();
        assert_eq!(plain.instructions.as_deref(), Some("do the thing"));
        assert_eq!(plain.global, Some(false));
        assert_eq!(plain.rollback_id, None);

        let global = parse_refine_command_options("--global tighten the notes").unwrap();
        assert_eq!(global.instructions.as_deref(), Some("tighten the notes"));
        assert_eq!(global.global, Some(true));

        let rollback = parse_refine_command_options("rollback abc123").unwrap();
        assert_eq!(rollback.rollback_id.as_deref(), Some("abc123"));
        assert_eq!(rollback.instructions, None);

        let rollback_global = parse_refine_command_options("rollback abc123 --global").unwrap();
        assert_eq!(rollback_global.rollback_id.as_deref(), Some("abc123"));
        assert_eq!(rollback_global.global, Some(true));

        assert!(parse_refine_command_options("rollback").is_err());
        assert!(parse_refine_command_options("rollback --global").is_err());
        assert!(parse_refine_command_options("rollback   ").is_err());
        assert_eq!(
            parse_refine_command_options("rollback").unwrap_err(),
            "Usage: /refine rollback <refinement-id>"
        );

        let empty = parse_refine_command_options("   ").unwrap();
        assert_eq!(empty.instructions, None);
    }

    #[test]
    fn refine_global_flag_is_only_recognised_at_the_start() {
        let options = parse_refine_command_options("keep --global here").unwrap();
        assert_eq!(options.global, Some(false));
        assert_eq!(options.instructions.as_deref(), Some("keep --global here"));
    }

    #[test]
    fn session_slash_command_names_are_fixed() {
        assert_eq!(SESSION_SLASH_COMMAND_NAMES, ["compact", "refine", "goal", "autonomous", "mode"]);
        assert!(is_session_slash_command_name("goal"));
        assert!(!is_session_slash_command_name("model"));
    }
}
