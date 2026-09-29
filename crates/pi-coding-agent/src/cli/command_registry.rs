//! Port of packages/coding-agent/src/cli/command-registry.ts

use std::collections::BTreeSet;

/// TODO(slice): APP_NAME lives in `crate::config` (ca-root slice, not landed yet).
const APP_NAME: &str = "prime-agent";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandSpec {
    pub path: &'static [&'static str],
    pub usage: &'static str,
    pub summary: &'static str,
    pub description: Option<&'static str>,
    pub options: Option<&'static [&'static str]>,
    pub examples: Option<&'static [&'static str]>,
}

const fn spec(
    path: &'static [&'static str],
    usage: &'static str,
    summary: &'static str,
) -> CommandSpec {
    CommandSpec { path, usage, summary, description: None, options: None, examples: None }
}

pub const COMMAND_SPECS: &[CommandSpec] = &[
    spec(&["help"], "help [command]", "Show command help"),
    spec(&["agents"], "agents", "Search and open sessions"),
    CommandSpec {
        path: &["list"],
        usage: "list [--all] [--json]",
        summary: "List agents",
        description: None,
        options: Some(&["-a, --all  Include saved agents", "--json      Print JSON"]),
        examples: None,
    },
    spec(&["attach"], "attach <agent>", "Attach the interactive UI to an agent"),
    spec(&["stop"], "stop <agent> [--json]", "Stop an agent"),
    spec(&["rename"], "rename <agent> <name> [--json]", "Rename an agent"),
    CommandSpec {
        path: &["send"],
        usage: "send [--from <agent>] <agent> <message>",
        summary: "Send a message to an agent",
        description: None,
        options: Some(&[
            "--from <agent>  Identify the sending agent",
            "--steer         Deliver as steering when the agent is busy",
            "--follow-up     Queue the message after the current turn",
            "--json          Print JSON",
        ]),
        examples: None,
    },
    spec(
        &["schedule"],
        "schedule <list|add|cancel>",
        "Manage prompts that run later or on a recurring schedule",
    ),
    spec(&["schedule", "list"], "schedule list [--all] [agent] [--json]", "List scheduled prompts"),
    CommandSpec {
        path: &["schedule", "add"],
        usage: "schedule add <agent> <schedule> -- <message>",
        summary: "Schedule a prompt",
        description: Some("The schedule may be a cron expression or a supported one-time schedule."),
        options: None,
        examples: Some(&["schedule add worker \"0 9 * * 1-5\" -- \"Check open work\""]),
    },
    spec(&["schedule", "cancel"], "schedule cancel <job-id>", "Cancel a scheduled prompt"),
    spec(&["status"], "status [--json]", "Show background service status"),
    CommandSpec {
        path: &["doctor"],
        usage: "doctor [--fix] [--json]",
        summary: "Inspect and safely clean up background services",
        description: None,
        options: Some(&[
            "--fix   Remove stale sockets and stop idle orphaned services",
            "--json  Print JSON",
        ]),
        examples: None,
    },
    CommandSpec {
        path: &["shutdown"],
        usage: "shutdown [--force] [--json]",
        summary: "Stop every agent and background service",
        description: Some(
            "Without --force, an interactive confirmation is required. --force also kills unresponsive workers.",
        ),
        options: Some(&[
            "--force  Skip confirmation and kill unresponsive processes",
            "--json   Print JSON",
        ]),
        examples: None,
    },
    spec(&["mcp"], "mcp <add|list|get|remove>", "Manage user MCP servers"),
    CommandSpec {
        path: &["mcp", "add"],
        usage: "mcp add <name> --url <url> [--bearer-token-env-var <env>|--oauth] [--force]",
        summary: "Add an HTTP or stdio MCP server",
        description: Some(
            "For stdio, use: mcp add <name> [--cwd <dir>] [--env CHILD=SOURCE] -- <command> [args...]",
        ),
        options: None,
        examples: None,
    },
    spec(&["mcp", "list"], "mcp list", "List user MCP servers"),
    spec(&["mcp", "get"], "mcp get <name>", "Show a user MCP server"),
    spec(&["mcp", "remove"], "mcp remove <name>", "Remove a user MCP server"),
    CommandSpec {
        path: &["package"],
        usage: "package <install|remove|list|update>",
        summary: "Manage capability packages",
        description: Some("Packages can provide extensions, skills, prompts, and themes."),
        options: None,
        examples: None,
    },
    CommandSpec {
        path: &["package", "install"],
        usage: "package install <source> [--local]",
        summary: "Install a capability package",
        description: None,
        options: Some(&["--local  Install into the current project instead of the user configuration"]),
        examples: None,
    },
    CommandSpec {
        path: &["package", "remove"],
        usage: "package remove <source> [--local]",
        summary: "Remove a capability package",
        description: None,
        options: Some(&["--local  Remove from the current project configuration"]),
        examples: None,
    },
    spec(&["package", "list"], "package list", "List installed capability packages"),
    spec(&["package", "update"], "package update [source]", "Update capability packages"),
    spec(&["update"], "update [--force]", "Update Prime Agent"),
    spec(&["model"], "model list [search]", "Inspect available models"),
    spec(&["model", "list"], "model list [search]", "List available models"),
    spec(&["session"], "session export <file> [output]", "Manage saved sessions"),
    spec(&["session", "export"], "session export <file> [output]", "Export a saved session to HTML"),
    spec(&["config"], "config", "Configure package resources"),
];

/// `PUBLIC_COMMAND_NAMES`: every single-segment command name.
pub fn public_command_names() -> BTreeSet<&'static str> {
    COMMAND_SPECS
        .iter()
        .filter(|spec| spec.path.len() == 1)
        .map(|spec| spec.path[0])
        .collect()
}

/// `REMOVED_COMMAND_NAMES`.
pub fn removed_command_names() -> BTreeSet<&'static str> {
    ["app", "daemon", "install", "manage", "remove", "uninstall"].into_iter().collect()
}

pub struct TopLevelOptionGroup {
    pub heading: &'static str,
    pub options: &'static [(&'static str, &'static str)],
}

pub const TOP_LEVEL_OPTION_GROUPS: &[TopLevelOptionGroup] = &[
    TopLevelOptionGroup {
        heading: "Run options",
        options: &[
            ("-p, --print", "Print a response and exit"),
            ("--mode <text|json|rpc|acp|daemon>", "Select the output mode (default: text)"),
            ("--cwd <dir>", "Use a specific working directory"),
            ("--offline", "Disable startup network operations"),
            ("--verbose", "Force verbose startup"),
            ("--daemon-socket <path>", "Use a specific daemon socket"),
        ],
    },
    TopLevelOptionGroup {
        heading: "Model options",
        options: &[
            ("--provider <name>", "Select a model provider (including kiro)"),
            ("--model <id>", "Select a model"),
            ("--api-key <key>", "Use an API key for this run"),
            ("--models <patterns>", "Set comma-separated models for cycling"),
            ("--thinking <level>", "Set reasoning: off, minimal, low, medium, high, xhigh, max"),
        ],
    },
    TopLevelOptionGroup {
        heading: "Kiro environment",
        options: &[
            ("KIRO_API_KEY", "Use an authorized Kiro API key"),
            ("KIRO_API_REGION", "API key region (default: us-east-1)"),
            ("OPTIMUS_KIRO_CLI_AUTH=0", "Disable automatic reuse of the Kiro CLI login"),
            ("KIRO_CLI_DB_FILE", "Select the existing Kiro CLI credential database"),
        ],
    },
    TopLevelOptionGroup {
        heading: "Session options",
        options: &[
            ("-c, --continue", "Continue the previous session"),
            ("-r, --resume [path|id]", "Open the agents view, or resume a saved session"),
            ("--fork <path|id>", "Fork a saved session into a new session"),
            ("--session-dir <dir>", "Use a custom session directory"),
            ("--no-session", "Do not save the session"),
            ("--goal <objective>", "Seed a persistent goal for a new root session"),
            ("--goal-token-budget <n>", "Set a positive token budget for --goal"),
        ],
    },
    TopLevelOptionGroup {
        heading: "Tool and resource options",
        options: &[
            ("-t, --tools <list>", "Allowlist comma-separated tool names"),
            ("-nt, --no-tools", "Disable all tools by default"),
            ("-nbt, --no-builtin-tools", "Disable built-in tools by default"),
            ("-e, --extension <source>", "Load an extension (repeatable)"),
            ("-ne, --no-extensions", "Disable extension discovery"),
            ("--skill <path>", "Load a skill (repeatable)"),
            ("-ns, --no-skills", "Disable skill discovery"),
            ("--prompt-template <path>", "Load a prompt template (repeatable)"),
            ("-np, --no-prompt-templates", "Disable prompt template discovery"),
            ("--theme <path>", "Load a theme (repeatable)"),
            ("--no-themes", "Disable theme discovery"),
            ("-nc, --no-context-files", "Disable AGENTS.md and CLAUDE.md discovery"),
        ],
    },
    TopLevelOptionGroup {
        heading: "Prompt options",
        options: &[
            ("--system-prompt <text>", "Replace the default system prompt"),
            ("--append-system-prompt <text>", "Append to the system prompt (repeatable)"),
            ("--", "Treat all following arguments as messages"),
        ],
    },
    TopLevelOptionGroup {
        heading: "Autonomous options",
        options: &[
            ("--autonomous", "Continue until gates pass or a limit is reached"),
            ("--autonomous-gate <command>", "Run a completion gate (repeatable)"),
            ("--autonomous-gate-retries <n>", "Set positive retries per failed gate (default: 3)"),
            ("--autonomous-gate-timeout-ms <n>", "Set positive per-gate timeout in ms (default: 300000)"),
            ("--autonomous-max-continuations <n>", "Set positive follow-up limit (default: 3)"),
            ("--autonomous-max-turns <n>", "Set positive assistant-turn limit (default: 12)"),
            ("--autonomous-max-tokens <n>", "Set positive token limit (default: 80000)"),
            ("--autonomous-timeout-ms <n>", "Set positive wall-clock limit in ms (default: 1800000)"),
        ],
    },
    TopLevelOptionGroup {
        heading: "Help",
        options: &[("-v, --version", "Show version and exit"), ("-h, --help", "Show this help")],
    },
];

pub fn get_command_spec(path: &[&str]) -> Option<&'static CommandSpec> {
    COMMAND_SPECS.iter().find(|spec| {
        spec.path.len() == path.len()
            && spec.path.iter().zip(path.iter()).all(|(segment, candidate)| segment == candidate)
    })
}

pub fn get_child_command_specs(path: &[&str]) -> Vec<&'static CommandSpec> {
    COMMAND_SPECS
        .iter()
        .filter(|spec| {
            spec.path.len() == path.len() + 1
                && path.iter().zip(spec.path.iter()).all(|(segment, candidate)| segment == candidate)
        })
        .collect()
}

pub fn is_help_command_request(path: &[&str]) -> bool {
    if path.is_empty() || get_command_spec(path).is_some() {
        return true;
    }
    if removed_command_names().contains(path[0]) {
        return true;
    }
    if get_command_spec(&path[..1]).is_some() {
        return true;
    }
    let parent = &path[..path.len() - 1];
    let candidates: Vec<&str> = get_child_command_specs(parent)
        .iter()
        .map(|spec| spec.path[spec.path.len() - 1])
        .collect();
    find_command_suggestion(path[path.len() - 1], &candidates).is_some()
}

pub fn find_command_suggestion(input: &str, candidates: &[&str]) -> Option<String> {
    let mut closest: Option<(&str, usize)> = None;
    for candidate in candidates {
        let distance = edit_distance(input, candidate);
        if closest.map(|(_, best)| distance < best).unwrap_or(true) {
            closest = Some((candidate, distance));
        }
    }
    match closest {
        Some((candidate, distance))
            if distance <= std::cmp::max(2, (input.chars().count() as f64 / 3.0).floor() as usize) =>
        {
            Some(candidate.to_string())
        }
        _ => None,
    }
}

pub fn format_top_level_help() -> String {
    let commands: Vec<&CommandSpec> = COMMAND_SPECS.iter().filter(|spec| spec.path.len() == 1).collect();
    let command_width = commands
        .iter()
        .map(|spec| spec.path[0].chars().count())
        .max()
        .unwrap_or(0);
    let options = TOP_LEVEL_OPTION_GROUPS
        .iter()
        .map(|group| format_option_group(group.heading, group.options))
        .collect::<Vec<_>>()
        .join("\n\n");
    let command_lines = commands
        .iter()
        .map(|spec| format!("  {:<width$}  {}", spec.path[0], spec.summary, width = command_width))
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "{APP_NAME} - AI coding assistant with a Python REPL tool\n\nUsage:\n  {APP_NAME} [options] [@files...] [message...]\n  {APP_NAME} <command> [args...]\n\nOptions:\n{options}\n\nCommands:\n{command_lines}\n\nRun \"{APP_NAME} help <command>\" for command details."
    )
}

fn format_option_group(heading: &str, options: &[(&str, &str)]) -> String {
    let width = options.iter().map(|(option, _)| option.chars().count()).max().unwrap_or(0);
    let lines = options
        .iter()
        .map(|(option, summary)| format!("  {:<width$}  {}", option, summary, width = width))
        .collect::<Vec<_>>()
        .join("\n");
    format!("{}:\n{}", heading, lines)
}

pub fn format_command_help(path: &[&str]) -> Option<String> {
    let spec = get_command_spec(path)?;
    let children = get_child_command_specs(path);
    let mut sections: Vec<String> = vec![
        format!("Usage:\n  {} {}", APP_NAME, spec.usage),
        String::new(),
        format!("{}.", spec.summary),
    ];
    if let Some(description) = spec.description {
        sections.push(String::new());
        sections.push(description.to_string());
    }
    if !children.is_empty() {
        let width = children
            .iter()
            .map(|child| child.path[child.path.len() - 1].chars().count())
            .max()
            .unwrap_or(0);
        sections.push(String::new());
        sections.push("Commands:".to_string());
        for child in &children {
            sections.push(format!(
                "  {:<width$}  {}",
                child.path[child.path.len() - 1],
                child.summary,
                width = width
            ));
        }
    }
    if let Some(options) = spec.options {
        if !options.is_empty() {
            sections.push(String::new());
            sections.push("Options:".to_string());
            for option in options {
                sections.push(format!("  {}", option));
            }
        }
    }
    if let Some(examples) = spec.examples {
        if !examples.is_empty() {
            sections.push(String::new());
            sections.push("Examples:".to_string());
            for example in examples {
                sections.push(format!("  {} {}", APP_NAME, example));
            }
        }
    }
    Some(sections.join("\n"))
}

/// Levenshtein distance, mirroring the TypeScript rolling-row implementation.
fn edit_distance(left: &str, right: &str) -> usize {
    let left: Vec<char> = left.chars().collect();
    let right: Vec<char> = right.chars().collect();
    let mut previous: Vec<usize> = (0..=right.len()).collect();
    for left_index in 1..=left.len() {
        let mut diagonal = previous[0];
        previous[0] = left_index;
        for right_index in 1..=right.len() {
            let above = previous[right_index];
            previous[right_index] = std::cmp::min(
                std::cmp::min(previous[right_index] + 1, previous[right_index - 1] + 1),
                diagonal + usize::from(left[left_index - 1] != right[right_index - 1]),
            );
            diagonal = above;
        }
    }
    previous[right.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_specs_by_path() {
        assert!(get_command_spec(&["help"]).is_some());
        assert!(get_command_spec(&["schedule", "add"]).is_some());
        assert!(get_command_spec(&["nope"]).is_none());
        assert!(get_command_spec(&["help", "extra"]).is_none());
    }

    #[test]
    fn lists_child_specs() {
        let children: Vec<&str> = get_child_command_specs(&["package"])
            .iter()
            .map(|spec| spec.path[1])
            .collect();
        assert_eq!(children, vec!["install", "remove", "list", "update"]);
        assert!(get_child_command_specs(&["mcp"]).len() == 4);
    }

    #[test]
    fn public_and_removed_command_names() {
        let public = public_command_names();
        assert!(public.contains("help"));
        assert!(public.contains("agents"));
        assert!(!public.contains("install"));
        let removed = removed_command_names();
        assert!(removed.contains("daemon"));
        assert!(removed.contains("app"));
    }

    #[test]
    fn help_requests_cover_known_and_removed_paths() {
        assert!(is_help_command_request(&[]));
        assert!(is_help_command_request(&["help"]));
        assert!(is_help_command_request(&["daemon"]));
        assert!(is_help_command_request(&["package", "install"]));
        assert!(is_help_command_request(&["package", "instal"]));
        assert!(!is_help_command_request(&["nope", "zzzzzz"]));
    }

    #[test]
    fn suggestion_threshold_matches_the_typescript() {
        assert_eq!(find_command_suggestion("instal", &["install", "remove"]), Some("install".to_string()));
        assert_eq!(find_command_suggestion("xy", &["install"]), None);
        assert_eq!(find_command_suggestion("", &[]), None);
    }

    #[test]
    fn edit_distance_is_symmetric_for_short_words() {
        assert_eq!(edit_distance("kitten", "sitting"), 3);
        assert_eq!(edit_distance("", "abc"), 3);
        assert_eq!(edit_distance("same", "same"), 0);
    }

    #[test]
    fn top_level_help_lists_commands_and_options() {
        let help = format_top_level_help();
        assert!(help.starts_with("prime-agent - AI coding assistant with a Python REPL tool\n"));
        assert!(help.contains("\nOptions:\nRun options:\n  -p, --print"));
        assert!(help.contains("\nCommands:\n"));
        assert!(help.contains("  help      Show command help"));
        assert!(help.ends_with("Run \"prime-agent help <command>\" for command details."));
    }

    #[test]
    fn command_help_includes_children_options_and_examples() {
        let help = format_command_help(&["schedule"]).unwrap();
        assert!(help.starts_with("Usage:\n  prime-agent schedule <list|add|cancel>\n\nManage prompts"));
        assert!(help.contains("\nCommands:\n  list"));
        let help = format_command_help(&["schedule", "add"]).unwrap();
        assert!(help.contains("\nExamples:\n  prime-agent schedule add worker \"0 9 * * 1-5\" -- \"Check open work\""));
        assert!(format_command_help(&["nope"]).is_none());
    }
}
