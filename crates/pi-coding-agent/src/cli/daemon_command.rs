//! Port of packages/coding-agent/src/cli/daemon-command.ts
//!
//! TODO(slice): `DaemonClient`/daemon protocol (ca-daemon-b), `AgentCronJob`
//! formatting (ca-session-core), session resolver (ca-session), `spawnHidden`
//! (ca-utils), `isLocalPath` (ca-utils), `expandTildePath` (ca-root) and the
//! agent-session event shapes (ca-session) are not landed. Private local
//! stand-ins live at the bottom of this module and are listed in the slice
//! status file.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use std::sync::Arc;

use crate::core::session_id::matches_session_id_suffix;
use crate::modes::daemon::daemon_client::{DaemonClient, DaemonClientError, DaemonClientRequestOptions};
use crate::modes::daemon::daemon_protocol::DaemonResponse;

use super::args::is_valid_thinking_level;
use super::daemon_list_format::format_session_list_table;
use super::daemon_ps::{run_ps, run_reap, DaemonPsIo};

pub const DAEMON_CLIENT_COMMANDS: [&str; 22] = [
    "start",
    "ps",
    "list",
    "create",
    "attach",
    "detach",
    "kill",
    "rename",
    "prompt",
    "send",
    "agent-messages",
    "steer",
    "follow-up",
    "state",
    "messages",
    "stats",
    "commands",
    "cron",
    "retry",
    "restart",
    "shutdown",
    "open",
];

#[derive(Debug)]
pub struct ParsedDaemonClientCommand {
    pub command: String,
    pub socket_path: String,
    pub json: bool,
    pub positionals: Vec<String>,
}

/// Callback surface for `console.log` / `console.error` / `process.exitCode`.
pub struct DaemonCommandIo<'a> {
    pub log: &'a dyn Fn(&str),
    pub error: &'a dyn Fn(&str),
    pub set_exit_code: &'a dyn Fn(i32),
    pub stdin_is_tty: Option<bool>,
    pub prompt_yes_no: &'a dyn Fn(&str) -> bool,
    pub cwd: String,
}

pub async fn handle_daemon_command(args: &[String], io: &DaemonCommandIo<'_>) -> bool {
    if args.first().map(String::as_str) != Some("daemon") {
        return false;
    }

    match parse_daemon_client_command(&args[1..]) {
        Ok(parsed) => {
            if let Err(error) = run_daemon_client_command(parsed, io).await {
                (io.error)(&format!("\u{1b}[31mError: {}\u{1b}[39m", error));
                (io.set_exit_code)(1);
            }
            true
        }
        Err(error) => {
            (io.error)(&format!("\u{1b}[31mError: {}\u{1b}[39m", error));
            (io.set_exit_code)(1);
            true
        }
    }
}

fn parse_daemon_client_command(args: &[String]) -> Result<ParsedDaemonClientCommand, String> {
    let mut socket_path = default_daemon_socket_path();
    let mut json = false;
    let mut positionals: Vec<String> = Vec::new();
    let mut passthrough = false;
    let mut command: Option<String> = None;

    let mut index = 0usize;
    while index < args.len() {
        let arg = args[index].clone();

        if passthrough {
            positionals.push(arg);
            index += 1;
            continue;
        }

        // send/cron parse "--" themselves as an end-of-flags separator
        if arg == "--"
            && (command.as_deref() == Some("cron") || command.as_deref() == Some("send"))
        {
            positionals.push(arg);
            passthrough = true;
            index += 1;
            continue;
        }

        if arg == "--" {
            passthrough = true;
            index += 1;
            continue;
        }

        if arg == "--help" || arg == "-h" {
            match &command {
                None => command = Some("help".to_string()),
                Some(_) => positionals.push("help".to_string()),
            }
            index += 1;
            continue;
        }

        if arg == "--socket" || arg == "--daemon-socket" {
            let value = args.get(index + 1).cloned();
            let value = match value {
                Some(value) if !value.is_empty() => value,
                _ => return Err(format!("{} requires a value", arg)),
            };
            socket_path = normalize_socket_path(&value, None);
            index += 2;
            continue;
        }

        if arg == "--json" {
            json = true;
            index += 1;
            continue;
        }

        if command.is_none() && DAEMON_CLIENT_COMMANDS.contains(&arg.as_str()) {
            command = Some(arg);
            index += 1;
            continue;
        }

        positionals.push(arg);
        index += 1;
    }

    let command = command.unwrap_or_else(|| "open".to_string());
    Ok(ParsedDaemonClientCommand { command, socket_path, json, positionals })
}

async fn run_daemon_client_command(
    parsed: ParsedDaemonClientCommand,
    io: &DaemonCommandIo<'_>,
) -> Result<(), String> {
    if parsed.command == "open" {
        return run_open(parsed, io).await;
    }

    if parsed.command == "start" {
        return run_start(parsed, io).await;
    }

    if parsed.command == "ps" {
        return run_ps_command(parsed, io).await;
    }

    if parsed.command == "help" {
        return run_help(io);
    }

    let client = Arc::new(DaemonClient::new(&parsed.socket_path));
    client.connect(3000).await.map_err(|error| error.message())?;

    let result = run_connected_daemon_command(&client, &parsed, io).await;
    client.close().await;
    result
}

async fn run_connected_daemon_command(
    client: &Arc<DaemonClient>,
    parsed: &ParsedDaemonClientCommand,
    io: &DaemonCommandIo<'_>,
) -> Result<(), String> {
    let positionals = parsed.positionals.clone();
    match parsed.command.as_str() {
        "list" => run_list(client, &positionals, parsed.json, io).await,
        "create" => run_create(client, &positionals, parsed.json, io).await,
        "attach" => {
            let active_session_id = require_active_session_id(&positionals)?;
            if parsed.json {
                run_json_attach(client, &active_session_id, io).await
            } else {
                run_attach(client, &active_session_id, io).await
            }
        }
        "detach" => {
            let command = match positionals.first() {
                Some(active_session_id) => {
                    serde_json::json!({ "type": "detach", "activeSessionId": active_session_id })
                }
                None => serde_json::json!({ "type": "detach" }),
            };
            print_response_data(client, command, parsed.json, io).await
        }
        "kill" => {
            let active_session_id = require_active_session_id(&positionals)?;
            print_response_data(
                client,
                serde_json::json!({ "type": "kill", "activeSessionId": active_session_id }),
                parsed.json,
                io,
            )
            .await
        }
        "rename" => run_rename(client, &positionals, parsed.json, io).await,
        "prompt" => run_prompt(client, &positionals, io).await,
        "send" => run_send(client, &positionals, parsed.json, io).await,
        "agent-messages" => run_agent_messages(client, &positionals, parsed.json, io).await,
        "steer" => run_message_command(client, "steer", &positionals, parsed.json, io).await,
        "follow-up" => run_message_command(client, "follow_up", &positionals, parsed.json, io).await,
        "state" => {
            let active_session_id = require_active_session_id(&positionals)?;
            print_response_data(
                client,
                serde_json::json!({ "type": "get_state", "activeSessionId": active_session_id }),
                true,
                io,
            )
            .await
        }
        "messages" => {
            let active_session_id = require_active_session_id(&positionals)?;
            print_response_data(
                client,
                serde_json::json!({ "type": "get_messages", "activeSessionId": active_session_id }),
                true,
                io,
            )
            .await
        }
        "stats" => {
            let active_session_id = require_active_session_id(&positionals)?;
            print_response_data(
                client,
                serde_json::json!({ "type": "get_session_stats", "activeSessionId": active_session_id }),
                true,
                io,
            )
            .await
        }
        "commands" => {
            let active_session_id = require_active_session_id(&positionals)?;
            print_response_data(
                client,
                serde_json::json!({ "type": "get_commands", "activeSessionId": active_session_id }),
                true,
                io,
            )
            .await
        }
        "cron" => run_cron(client, &positionals, parsed.json, io).await,
        "retry" => {
            if positionals.len() != 1 {
                return Err("Usage: daemon retry <session>".to_string());
            }
            print_response_data(
                client,
                serde_json::json!({ "type": "retry_worker", "activeSessionId": positionals[0] }),
                parsed.json,
                io,
            )
            .await
        }
        "restart" => {
            if !positionals.is_empty() {
                return Err("Usage: daemon restart".to_string());
            }
            print_response_data(client, serde_json::json!({ "type": "restart" }), parsed.json, io).await
        }
        "shutdown" => run_shutdown(client, &positionals, parsed.json, io).await,
        other => Err(format!("Unknown daemon command: {}", other)),
    }
}

/// The daemon `create` request. Keys whose TypeScript value is `undefined` are
/// omitted instead of sent as `null`, matching `{ ...spread }` object building.
pub fn create_session_request(
    name: &str,
    config: Option<&serde_json::Map<String, serde_json::Value>>,
    session_path: Option<&str>,
    continue_recent: Option<bool>,
) -> serde_json::Value {
    let mut object = serde_json::Map::new();
    object.insert("type".to_string(), serde_json::json!("create"));
    object.insert("name".to_string(), serde_json::json!(name));
    if let Some(config) = config {
        object.insert("config".to_string(), serde_json::Value::Object(config.clone()));
    }
    if let Some(session_path) = session_path {
        object.insert("sessionPath".to_string(), serde_json::json!(session_path));
    }
    if let Some(continue_recent) = continue_recent {
        object.insert("continueRecent".to_string(), serde_json::json!(continue_recent));
    }
    serde_json::Value::Object(object)
}

async fn run_open(parsed: ParsedDaemonClientCommand, io: &DaemonCommandIo<'_>) -> Result<(), String> {
    let session_args = parse_session_args(&parsed.positionals, &io.cwd)?;
    if !can_connect_to_daemon(&parsed.socket_path, 250).await {
        let start = ParsedDaemonClientCommand {
            command: "start".to_string(),
            socket_path: parsed.socket_path.clone(),
            json: parsed.json,
            positionals: session_args.daemon_args.clone(),
        };
        run_start(start, io).await?;
    }

    let client = Arc::new(DaemonClient::new(&parsed.socket_path));
    client.connect(3000).await.map_err(|error| error.message())?;
    let result = run_open_attached(&client, &parsed.socket_path, session_args, io).await;
    client.close().await;
    result
}

async fn run_open_attached(
    client: &Arc<DaemonClient>,
    socket_path: &str,
    session_args: ParsedSessionArgs,
    io: &DaemonCommandIo<'_>,
) -> Result<(), String> {
    let _ = socket_path;
    let auto_name = session_args.name.is_none();
    let mut sessions = get_live_sessions(client, auto_name).await?;
    let mut session_name = session_args
        .name
        .clone()
        .unwrap_or_else(|| next_default_session_name(&sessions));
    let response = loop {
        let response = request(
            client,
            create_session_request(
                &session_name,
                session_args.config.as_ref(),
                session_args.session_path.as_deref(),
                session_args.continue_recent,
            ),
        )
        .await?;
        let unavailable = format!("Agent name \"{}\" is unavailable", session_name);
        let error_includes = response
            .error
            .as_deref()
            .map(|error| error.contains(&unavailable))
            .unwrap_or(false);
        if response.success || !auto_name || !error_includes {
            break response;
        }
        sessions.push(serde_json::json!({ "sessionName": session_name }));
        session_name = next_default_session_name(&sessions);
    };
    let data = require_success(&response)?;
    if !is_live_session_summary(data) {
        return Err("Daemon returned an invalid create response".to_string());
    }
    let active_session_id = data
        .get("activeSessionId")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("")
        .to_string();
    run_attach(client, &active_session_id, io).await
}

#[derive(Debug)]
pub struct ParsedSessionArgs {
    pub daemon_args: Vec<String>,
    pub name: Option<String>,
    pub config: Option<serde_json::Map<String, serde_json::Value>>,
    pub session_path: Option<String>,
    pub continue_recent: Option<bool>,
}

pub const SESSION_BOOLEAN_FLAGS: [&str; 22] = [
    "--continue",
    "-c",
    "--no-session",
    "--no-tools",
    "-nt",
    "--no-builtin-tools",
    "-nbt",
    "--no-extensions",
    "-ne",
    "--no-skills",
    "-ns",
    "--no-prompt-templates",
    "-np",
    "--no-themes",
    "--no-context-files",
    "-nc",
    "--verbose",
    "--offline",
    "--foreground",
    "--no-detach",
    "--background",
    "-d",
];

pub fn parse_session_args(args: &[String], cwd: &str) -> Result<ParsedSessionArgs, String> {
    let mut daemon_args: Vec<String> = Vec::new();
    let mut name_parts: Vec<String> = Vec::new();
    let mut config: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();
    let path_base_cwd = find_session_cwd_arg(args)?.unwrap_or_else(|| cwd.to_string());
    let mut session_path: Option<String> = None;
    let mut continue_recent: Option<bool> = None;

    let mut index = 0usize;
    while index < args.len() {
        let arg = args[index].clone();
        if arg == "--" {
            name_parts.extend(args[index + 1..].iter().cloned());
            break;
        }

        if arg == "--name" {
            let value = args.get(index + 1).cloned().filter(|value| !value.is_empty());
            let value = match value {
                Some(value) => value,
                None => return Err("--name requires a value".to_string()),
            };
            name_parts.push(value);
            index += 2;
            continue;
        }

        if arg == "--cwd" {
            let value = require_option_value(args, index, &arg)?;
            config.insert("cwd".to_string(), serde_json::json!(resolve_path(&expand_tilde_path(&value))));
            index += 2;
            continue;
        }

        if arg.starts_with('-') {
            match parse_session_option(args, index, &mut config, &path_base_cwd)? {
                None => {
                    name_parts.push(arg);
                    index += 1;
                    continue;
                }
                Some(parsed_option) => {
                    if let Some(daemon_arg) = &parsed_option.daemon_arg {
                        daemon_args.push(daemon_arg.clone());
                        if let Some(value) = &parsed_option.value {
                            daemon_args.push(value.clone());
                        }
                    }
                    if parsed_option.session_path.is_some() {
                        session_path = parsed_option.session_path.clone();
                    }
                    if parsed_option.continue_recent.is_some() {
                        continue_recent = parsed_option.continue_recent;
                    }
                    index += 1 + parsed_option.consumed;
                    continue;
                }
            }
        }

        name_parts.push(arg);
        index += 1;
    }

    let name = name_parts.join(" ").trim().to_string();
    // Validate: --goal-token-budget without --goal is an error.
    let initial_goal = config.get("initialGoal");
    let has_objective = initial_goal
        .and_then(|goal| goal.get("objective"))
        .and_then(serde_json::Value::as_str)
        .map(|objective| !objective.is_empty())
        .unwrap_or(false);
    let has_token_budget = initial_goal
        .and_then(|goal| goal.get("tokenBudget"))
        .map(|budget| !budget.is_null())
        .unwrap_or(false);
    if has_token_budget && !has_objective {
        return Err("--goal-token-budget requires --goal".to_string());
    }

    Ok(ParsedSessionArgs {
        daemon_args,
        name: if name.is_empty() { None } else { Some(name) },
        config: if config.is_empty() { None } else { Some(config.clone()) },
        session_path,
        continue_recent,
    })
}

#[derive(Debug)]
pub struct ParsedSessionOption {
    pub consumed: usize,
    pub daemon_arg: Option<String>,
    pub value: Option<String>,
    pub session_path: Option<String>,
    pub continue_recent: Option<bool>,
}fn parse_session_option(
    args: &[String],
    index: usize,
    config: &mut serde_json::Map<String, serde_json::Value>,
    path_base_cwd: &str,
) -> Result<Option<ParsedSessionOption>, String> {
    let arg = args[index].clone();
    if !arg.starts_with('-') {
        return Ok(None);
    }

    if let Some(value) = arg.strip_prefix("--resume=") {
        return session_selector_option(&arg, value.to_string(), 0, config, path_base_cwd).map(Some);
    }

    match arg.as_str() {
        "--continue" | "-c" => Ok(Some(ParsedSessionOption {
            consumed: 0,
            daemon_arg: None,
            value: None,
            session_path: None,
            continue_recent: Some(true),
        })),
        "--resume" | "-r" => {
            let value = require_option_value(args, index, &arg)?;
            session_selector_option(&arg, value, 1, config, path_base_cwd).map(Some)
        }
        "--session-dir" => {
            let value = require_option_value(args, index, &arg)?;
            config.insert("sessionDir".to_string(), serde_json::json!(expand_tilde_path(&value)));
            Ok(Some(with_value_option(&arg, value)))
        }
        "--provider" => {
            let value = require_option_value(args, index, &arg)?;
            config.insert("provider".to_string(), serde_json::json!(value));
            Ok(Some(with_value_option(&arg, value)))
        }
        "--model" => {
            let value = require_option_value(args, index, &arg)?;
            config.insert("model".to_string(), serde_json::json!(value));
            Ok(Some(with_value_option(&arg, value)))
        }
        "--api-key" => {
            let value = require_option_value(args, index, &arg)?;
            config.insert("apiKey".to_string(), serde_json::json!(value));
            Ok(Some(with_value_option(&arg, value)))
        }
        "--system-prompt" => {
            let value = require_option_value(args, index, &arg)?;
            config.insert("systemPrompt".to_string(), serde_json::json!(value));
            Ok(Some(with_value_option(&arg, value)))
        }
        "--append-system-prompt" => {
            let value = require_option_value(args, index, &arg)?;
            let mut values = config
                .get("appendSystemPrompt")
                .and_then(serde_json::Value::as_array)
                .cloned()
                .unwrap_or_default();
            values.push(serde_json::json!(value));
            config.insert("appendSystemPrompt".to_string(), serde_json::Value::Array(values));
            Ok(Some(with_value_option(&arg, value)))
        }
        "--models" => {
            let value = require_option_value(args, index, &arg)?;
            config.insert("models".to_string(), serde_json::json!(parse_csv_value(&value)));
            Ok(Some(with_value_option(&arg, value)))
        }
        "--tools" | "-t" => {
            let value = require_option_value(args, index, &arg)?;
            config.insert("tools".to_string(), serde_json::json!(parse_csv_value(&value)));
            Ok(Some(with_value_option(&arg, value)))
        }
        "--thinking" => {
            let level = require_option_value(args, index, &arg)?;
            if !is_valid_thinking_level(&level) {
                return Err(format!("Invalid thinking level \"{}\"", level));
            }
            config.insert("thinking".to_string(), serde_json::json!(level));
            Ok(Some(with_value_option(&arg, level)))
        }
        "--extension" | "-e" => {
            let value = require_option_value(args, index, &arg)?;
            let resolved = resolve_path_option(&value, config_cwd(config).unwrap_or(path_base_cwd));
            let mut values = config
                .get("extensions")
                .and_then(serde_json::Value::as_array)
                .cloned()
                .unwrap_or_default();
            values.push(serde_json::json!(resolved));
            config.insert("extensions".to_string(), serde_json::Value::Array(values));
            Ok(Some(with_value_option(&arg, value)))
        }
        "--skill" => {
            let value = require_option_value(args, index, &arg)?;
            let resolved = resolve_path_option(&value, config_cwd(config).unwrap_or(path_base_cwd));
            let mut values = config
                .get("skills")
                .and_then(serde_json::Value::as_array)
                .cloned()
                .unwrap_or_default();
            values.push(serde_json::json!(resolved));
            config.insert("skills".to_string(), serde_json::Value::Array(values));
            Ok(Some(with_value_option(&arg, value)))
        }
        "--prompt-template" => {
            let value = require_option_value(args, index, &arg)?;
            let resolved = resolve_path_option(&value, config_cwd(config).unwrap_or(path_base_cwd));
            let mut values = config
                .get("promptTemplates")
                .and_then(serde_json::Value::as_array)
                .cloned()
                .unwrap_or_default();
            values.push(serde_json::json!(resolved));
            config.insert("promptTemplates".to_string(), serde_json::Value::Array(values));
            Ok(Some(with_value_option(&arg, value)))
        }
        "--theme" => {
            let value = require_option_value(args, index, &arg)?;
            let resolved = resolve_path_option(&value, config_cwd(config).unwrap_or(path_base_cwd));
            let mut values = config
                .get("themes")
                .and_then(serde_json::Value::as_array)
                .cloned()
                .unwrap_or_default();
            values.push(serde_json::json!(resolved));
            config.insert("themes".to_string(), serde_json::Value::Array(values));
            Ok(Some(with_value_option(&arg, value)))
        }
        "--no-tools" | "-nt" => {
            config.insert("noTools".to_string(), serde_json::json!(true));
            Ok(Some(boolean_option(&arg)))
        }
        "--no-builtin-tools" | "-nbt" => {
            config.insert("noBuiltinTools".to_string(), serde_json::json!(true));
            Ok(Some(boolean_option(&arg)))
        }
        "--no-extensions" | "-ne" => {
            config.insert("noExtensions".to_string(), serde_json::json!(true));
            Ok(Some(boolean_option(&arg)))
        }
        "--no-skills" | "-ns" => {
            config.insert("noSkills".to_string(), serde_json::json!(true));
            Ok(Some(boolean_option(&arg)))
        }
        "--no-prompt-templates" | "-np" => {
            config.insert("noPromptTemplates".to_string(), serde_json::json!(true));
            Ok(Some(boolean_option(&arg)))
        }
        "--no-themes" => {
            config.insert("noThemes".to_string(), serde_json::json!(true));
            Ok(Some(boolean_option(&arg)))
        }
        "--no-context-files" | "-nc" => {
            config.insert("noContextFiles".to_string(), serde_json::json!(true));
            Ok(Some(boolean_option(&arg)))
        }
        "--goal" => {
            let value = require_option_value(args, index, &arg)?;
            if value.trim().is_empty() {
                return Err("--goal requires a non-empty objective".to_string());
            }
            let token_budget = config
                .get("initialGoal")
                .and_then(|goal| goal.get("tokenBudget"))
                .cloned()
                .unwrap_or(serde_json::Value::Null);
            config.insert(
                "initialGoal".to_string(),
                serde_json::json!({ "objective": value, "tokenBudget": token_budget }),
            );
            // Session-specific flag: do NOT propagate to daemon startup args.
            // The goal is sent per-create via the config in the daemon request,
            // so a later no-goal create is not contaminated.
            Ok(Some(ParsedSessionOption {
                consumed: 1,
                daemon_arg: None,
                value: None,
                session_path: None,
                continue_recent: None,
            }))
        }
        "--goal-token-budget" => {
            let value = require_option_value(args, index, &arg)?;
            let budget = parse_js_number(&value);
            let budget = match budget {
                Some(budget) if budget.fract() == 0.0 && budget > 0.0 => budget,
                _ => return Err("--goal-token-budget must be a positive integer".to_string()),
            };
            let objective = config
                .get("initialGoal")
                .and_then(|goal| goal.get("objective"))
                .and_then(serde_json::Value::as_str)
                .unwrap_or("")
                .to_string();
            config.insert(
                "initialGoal".to_string(),
                serde_json::json!({ "objective": objective, "tokenBudget": budget }),
            );
            // Session-specific flag: do NOT propagate to daemon startup args.
            Ok(Some(ParsedSessionOption {
                consumed: 1,
                daemon_arg: None,
                value: None,
                session_path: None,
                continue_recent: None,
            }))
        }
        "--foreground" | "--no-detach" | "--background" | "-d" | "--verbose" | "--offline" => {
            if SESSION_BOOLEAN_FLAGS.contains(&arg.as_str()) {
                Ok(Some(boolean_option(&arg)))
            } else {
                Ok(None)
            }
        }
        "--no-session" => Err(
            "--no-session is not supported for daemon sessions; daemon-owned sessions are always persisted"
                .to_string(),
        ),
        _ => {
            if !arg.starts_with("--") {
                return Ok(None);
            }
            Ok(Some(parse_extension_flag_option(&arg, config)))
        }
    }
}

fn with_value_option(daemon_arg: &str, value: String) -> ParsedSessionOption {
    ParsedSessionOption {
        consumed: 1,
        daemon_arg: Some(daemon_arg.to_string()),
        value: Some(value),
        session_path: None,
        continue_recent: None,
    }
}

fn boolean_option(daemon_arg: &str) -> ParsedSessionOption {
    ParsedSessionOption {
        consumed: 0,
        daemon_arg: Some(daemon_arg.to_string()),
        value: None,
        session_path: None,
        continue_recent: None,
    }
}

/// `withSessionSelector`: a session path when the value looks like one, otherwise
/// an id or name passed through unchanged.
fn session_selector_option(
    arg: &str,
    value: String,
    consumed: usize,
    config: &serde_json::Map<String, serde_json::Value>,
    path_base_cwd: &str,
) -> Result<ParsedSessionOption, String> {
    if value.is_empty() {
        let option = arg.split('=').next().unwrap_or(arg).to_string();
        return Err(format!("{} requires a value", option));
    }
    let session_path = if looks_like_session_path(&value) {
        resolve_path_option(&value, config_cwd(config).unwrap_or(path_base_cwd))
    } else {
        value
    };
    Ok(ParsedSessionOption {
        consumed,
        daemon_arg: None,
        value: None,
        session_path: Some(session_path),
        continue_recent: None,
    })
}

/// Local stand-in for `looksLikeSessionPath` from ../core/session-resolver.js.
fn looks_like_session_path(selector: &str) -> bool {
    crate::core::session_resolver::looks_like_session_path(selector)
}

/// Local stand-in for `parseJsNumber` from ../core/numbers.js.
fn parse_js_number(value: &str) -> Option<f64> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Some(0.0);
    }
    trimmed.parse::<f64>().ok()
}

fn config_cwd(config: &serde_json::Map<String, serde_json::Value>) -> Option<&str> {
    config.get("cwd").and_then(serde_json::Value::as_str)
}

fn parse_csv_value(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(|part| part.trim().to_string())
        .filter(|part| !part.is_empty())
        .collect()
}

fn find_session_cwd_arg(args: &[String]) -> Result<Option<String>, String> {
    let mut index = 0usize;
    while index < args.len() {
        if args[index] == "--" {
            return Ok(None);
        }
        if args[index] == "--cwd" {
            return Ok(Some(resolve_path(&expand_tilde_path(&require_option_value(
                args, index, "--cwd",
            )?))));
        }
        index += 1;
    }
    Ok(None)
}

fn parse_extension_flag_option(arg: &str, config: &mut serde_json::Map<String, serde_json::Value>) -> ParsedSessionOption {
    let mut flag_values = config
        .get("extensionFlagValues")
        .and_then(serde_json::Value::as_object)
        .cloned()
        .unwrap_or_default();
    match arg.find('=') {
        Some(eq_index) => {
            flag_values.insert(arg[2..eq_index].to_string(), serde_json::json!(arg[eq_index + 1..]));
            config.insert("extensionFlagValues".to_string(), serde_json::Value::Object(flag_values));
            ParsedSessionOption {
                consumed: 0,
                daemon_arg: Some(arg.to_string()),
                value: None,
                session_path: None,
                continue_recent: None,
            }
        }
        None => {
            flag_values.insert(arg[2..].to_string(), serde_json::json!(true));
            config.insert("extensionFlagValues".to_string(), serde_json::Value::Object(flag_values));
            ParsedSessionOption {
                consumed: 0,
                daemon_arg: Some(arg.to_string()),
                value: None,
                session_path: None,
                continue_recent: None,
            }
        }
    }
}

fn require_option_value(args: &[String], index: usize, option: &str) -> Result<String, String> {
    match args.get(index + 1) {
        Some(value) if !value.is_empty() => Ok(value.clone()),
        _ => Err(format!("{} requires a value", option)),
    }
}

fn resolve_path_option(value: &str, cwd: &str) -> String {
    let expanded = expand_tilde_path(value);
    if is_local_path(&expanded) {
        resolve_path(&Path::new(cwd).join(&expanded).to_string_lossy())
    } else {
        expanded
    }
}

async fn get_live_sessions(
    client: &Arc<DaemonClient>,
    all: bool,
) -> Result<Vec<serde_json::Value>, String> {
    let command = if all {
        serde_json::json!({ "type": "list", "all": true })
    } else {
        serde_json::json!({ "type": "list" })
    };
    let response = request(client, command).await?;
    let data = require_success(&response)?;
    get_session_summaries(data).ok_or_else(|| "Daemon returned an invalid list response".to_string())
}

fn next_default_session_name(sessions: &[serde_json::Value]) -> String {
    let existing_names: BTreeSet<String> = sessions
        .iter()
        .filter_map(|session| session.get("sessionName").and_then(serde_json::Value::as_str))
        .map(str::to_string)
        .collect();
    let numeric_names: Vec<i64> = existing_names
        .iter()
        .filter_map(|name| name.parse::<i64>().ok())
        .filter(|value| *value > 0 && *value <= 9_007_199_254_740_991)
        .collect();
    let mut next = if numeric_names.is_empty() {
        1
    } else {
        numeric_names.iter().max().copied().unwrap_or(0) + 1
    };
    while next <= 9_007_199_254_740_991 {
        if !existing_names.contains(&next.to_string()) {
            return next.to_string();
        }
        next += 1;
    }

    let mut fallback = "session".to_string();
    while existing_names.contains(&fallback) {
        fallback.push('-');
    }
    fallback
}

async fn run_start(parsed: ParsedDaemonClientCommand, io: &DaemonCommandIo<'_>) -> Result<(), String> {
    if can_connect_to_daemon(&parsed.socket_path, 250).await {
        (io.log)(&format!("Daemon already running on {}", parsed.socket_path));
        return Ok(());
    }

    let entrypoint = current_entrypoint();
    if entrypoint.is_empty() {
        return Err("Cannot determine current CLI entrypoint for daemon launch".to_string());
    }

    let session_args = parse_session_args(&parsed.positionals, &io.cwd)?;
    let mut daemon_args: Vec<String> = current_exec_args();
    daemon_args.push(entrypoint);
    daemon_args.push("--mode".to_string());
    daemon_args.push("daemon".to_string());
    daemon_args.push("--daemon-socket".to_string());
    daemon_args.push(parsed.socket_path.clone());
    for arg in session_args
        .daemon_args
        .iter()
        .filter(|arg| *arg != "--background" && *arg != "-d")
    {
        daemon_args.push(arg.clone());
    }
    let cwd = session_args
        .config
        .as_ref()
        .and_then(|config| config.get("cwd"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| io.cwd.clone());
    let env = current_process_env();
    let child = spawn_hidden_detached(&current_exec_path(), &daemon_args, &cwd, &env);
    let child_pid = child.as_ref().map(|child| child.pid);

    let deadline = now_ms() + 10_000.0;
    while now_ms() < deadline {
        if can_connect_to_daemon(&parsed.socket_path, 250).await {
            (io.log)(&format!(
                "Daemon started on {} (pid {})",
                parsed.socket_path,
                child_pid.map(|pid| pid.to_string()).unwrap_or_else(|| "unknown".to_string())
            ));
            return Ok(());
        }
        delay(25).await;
    }

    Err(format!("Timed out waiting for daemon to start on {}", parsed.socket_path))
}

async fn run_ps_command(parsed: ParsedDaemonClientCommand, io: &DaemonCommandIo<'_>) -> Result<(), String> {
    let mut reap = false;
    let mut force = false;
    for arg in &parsed.positionals {
        if arg == "--reap" || arg == "reap" {
            reap = true;
        } else if arg == "--force" || arg == "-f" {
            force = true;
        } else {
            return Err(format!("Unknown ps option: {}", arg));
        }
    }
    let ps_io = DaemonPsIo {
        log: io.log,
        error: io.error,
        stdin_is_tty: io.stdin_is_tty,
        prompt_yes_no: io.prompt_yes_no,
        set_exit_code: io.set_exit_code,
        cwd: io.cwd.clone(),
    };
    if reap {
        return run_reap(parsed.json, force, &ps_io).await;
    }
    run_ps(parsed.json, &ps_io).await
}

async fn can_connect_to_daemon(socket_path: &str, timeout_ms: u64) -> bool {
    let client = Arc::new(DaemonClient::new(socket_path));
    let connected = client.connect(timeout_ms).await.is_ok();
    client.close().await;
    connected
}

async fn run_list(
    client: &Arc<DaemonClient>,
    args: &[String],
    json: bool,
    io: &DaemonCommandIo<'_>,
) -> Result<(), String> {
    let all = parse_list_args(args)?;
    let response = request(client, serde_json::json!({ "type": "list", "all": all })).await?;
    let data = require_success(&response)?;
    if json {
        print_json(io, data);
        return Ok(());
    }

    let sessions = match get_session_summaries(data) {
        Some(sessions) => sessions,
        None => {
            print_json(io, data);
            return Ok(());
        }
    };

    if sessions.is_empty() {
        (io.log)(if all { "No agents." } else { "No active agents." });
        return Ok(());
    }

    let summaries = session_summaries_from_json(&sessions);
    (io.log)(&format_session_list_table(&summaries, now_ms()));
    Ok(())
}

fn parse_list_args(args: &[String]) -> Result<bool, String> {
    let mut all = false;
    for arg in args {
        if arg == "-a" || arg == "--all" {
            all = true;
            continue;
        }
        return Err(format!("Unknown list option: {}", arg));
    }
    Ok(all)
}

async fn run_create(
    client: &Arc<DaemonClient>,
    args: &[String],
    json: bool,
    io: &DaemonCommandIo<'_>,
) -> Result<(), String> {
    let session_args = parse_session_args(args, &io.cwd)?;
    let response = request(
        client,
        create_session_request(
            session_args.name.as_deref().unwrap_or(""),
            session_args.config.as_ref(),
            session_args.session_path.as_deref(),
            session_args.continue_recent,
        ),
    )
    .await?;
    let data = require_success(&response)?;
    if json {
        print_json(io, data);
        return Ok(());
    }

    if is_live_session_summary(data) {
        let active_session_id = data.get("activeSessionId").and_then(serde_json::Value::as_str).unwrap_or("");
        let session_name = data.get("sessionName").and_then(serde_json::Value::as_str);
        (io.log)(&format!(
            "Created {}{}",
            active_session_id,
            session_name.map(|name| format!(" ({})", name)).unwrap_or_default()
        ));
        return Ok(());
    }
    print_json(io, data);
    Ok(())
}

async fn run_attach(
    client: &Arc<DaemonClient>,
    active_session_id: &str,
    io: &DaemonCommandIo<'_>,
) -> Result<(), String> {
    let terminal = DaemonAttachTerminal::new(client.clone(), active_session_id, io);
    terminal.run().await
}

async fn run_json_attach(
    client: &Arc<DaemonClient>,
    active_session_id: &str,
    io: &DaemonCommandIo<'_>,
) -> Result<(), String> {
    let mut session_closed = SessionEndWaiter::for_session_close(active_session_id);
    require_success(&request(
        client,
        serde_json::json!({ "type": "attach", "activeSessionId": active_session_id }),
    )
    .await?)?;
    daemon_stream_until(client, &|value| print_json_line(io.log, value), &mut session_closed).await
}

async fn run_rename(
    client: &Arc<DaemonClient>,
    args: &[String],
    json: bool,
    io: &DaemonCommandIo<'_>,
) -> Result<(), String> {
    let active_session_id = require_active_session_id(args)?;
    let name = args[1.min(args.len())..].join(" ").trim().to_string();
    if name.is_empty() {
        return Err("Usage: prime-agent rename <agent> <name>".to_string());
    }
    let response = request(
        client,
        serde_json::json!({ "type": "rename", "activeSessionId": active_session_id, "name": name }),
    )
    .await?;
    let data = require_success(&response)?;
    if json {
        print_json(io, data);
        return Ok(());
    }

    if is_live_session_summary(data) {
        (io.log)(&format!(
            "Renamed {} to {}",
            data.get("activeSessionId").and_then(serde_json::Value::as_str).unwrap_or(""),
            data.get("sessionName").and_then(serde_json::Value::as_str).unwrap_or(&name)
        ));
        return Ok(());
    }
    print_json(io, data);
    Ok(())
}

async fn run_prompt(
    client: &Arc<DaemonClient>,
    args: &[String],
    io: &DaemonCommandIo<'_>,
) -> Result<(), String> {
    let active_session_id = require_active_session_id(args)?;
    let message = args[1.min(args.len())..].join(" ").trim().to_string();
    if message.is_empty() {
        return Err("Usage: daemon prompt <session> <message>".to_string());
    }

    let mut finished = SessionEndWaiter::for_session_end(&active_session_id, false);
    require_success(&request(
        client,
        serde_json::json!({ "type": "attach", "activeSessionId": active_session_id }),
    )
    .await?)?;
    require_success(&request(
        client,
        serde_json::json!({ "type": "prompt", "activeSessionId": active_session_id, "message": message }),
    )
    .await?)?;
    finished.acknowledge_prompt();
    daemon_stream_until(client, &|value| print_json_line(io.log, value), &mut finished).await
}

async fn run_agent_messages(
    client: &Arc<DaemonClient>,
    args: &[String],
    json: bool,
    io: &DaemonCommandIo<'_>,
) -> Result<(), String> {
    match args.first().map(String::as_str) {
        Some("status") => {
            require_no_extra_args(args, "daemon agent-messages status")?;
            print_response_data(client, serde_json::json!({ "type": "agent_messages_status" }), json, io).await
        }
        Some("pause") => {
            require_no_extra_args(args, "daemon agent-messages pause")?;
            print_response_data(client, serde_json::json!({ "type": "agent_messages_pause" }), json, io).await
        }
        Some("resume") => {
            require_no_extra_args(args, "daemon agent-messages resume")?;
            print_response_data(client, serde_json::json!({ "type": "agent_messages_resume" }), json, io).await
        }
        Some("clear") => {
            let active_session_id = args.get(1);
            if active_session_id.is_none() || args.len() != 2 {
                return Err("Usage: daemon agent-messages clear <session>".to_string());
            }
            print_response_data(
                client,
                serde_json::json!({
                    "type": "agent_messages_clear",
                    "activeSessionId": active_session_id.unwrap(),
                }),
                json,
                io,
            )
            .await
        }
        _ => Err("Usage: daemon agent-messages <status|pause|resume|clear>".to_string()),
    }
}

fn require_no_extra_args(args: &[String], usage: &str) -> Result<(), String> {
    if args.len() > 1 {
        return Err(format!("Usage: {}", usage));
    }
    Ok(())
}

async fn run_send(
    client: &Arc<DaemonClient>,
    args: &[String],
    json: bool,
    io: &DaemonCommandIo<'_>,
) -> Result<(), String> {
    let parsed = parse_send_args(args)?;
    let mut request_object = serde_json::Map::new();
    request_object.insert("type".to_string(), serde_json::json!("send_message"));
    request_object.insert(
        "targetActiveSessionId".to_string(),
        serde_json::json!(parsed.target_active_session_id),
    );
    if let Some(from_active_session_id) = &parsed.from_active_session_id {
        request_object.insert("fromActiveSessionId".to_string(), serde_json::json!(from_active_session_id));
    }
    request_object.insert("message".to_string(), serde_json::json!(parsed.message));
    let response = request(client, serde_json::Value::Object(request_object)).await?;
    let data = require_success(&response)?;
    if json {
        print_json(io, data);
        return Ok(());
    }
    if is_agent_message_receipt(data) {
        let target = data
            .get("target")
            .and_then(|target| target.get("sessionName"))
            .and_then(serde_json::Value::as_str)
            .map(str::to_string)
            .or_else(|| {
                data.get("target")
                    .and_then(|target| target.get("activeSessionId"))
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string)
            })
            .unwrap_or_default();
        let delivery_status = data.get("deliveryStatus").and_then(serde_json::Value::as_str);
        (io.log)(&if delivery_status == Some("queued") {
            format!("Queued for {}", target)
        } else {
            format!("Sent to {}", target)
        });
        return Ok(());
    }
    (io.log)("ok");
    Ok(())
}

#[derive(Debug)]
pub struct ParsedSendArgs {
    pub target_active_session_id: String,
    pub from_active_session_id: Option<String>,
    pub message: String,
}

pub fn parse_send_args(args: &[String]) -> Result<ParsedSendArgs, String> {
    let mut from_active_session_id: Option<String> = None;
    let mut target_active_session_id: Option<String> = None;
    let mut explicit_message: Option<String> = None;
    let mut message_parts: Vec<String> = Vec::new();
    let mut parse_options = true;

    let mut index = 0usize;
    while index < args.len() {
        let arg = args[index].clone();
        if parse_options && arg == "--" {
            parse_options = false;
            index += 1;
            continue;
        }
        if parse_options && arg == "--from" {
            let value = args.get(index + 1).cloned().filter(|value| !value.is_empty());
            let value = match value {
                Some(value) => value,
                None => return Err("--from requires a session id or name".to_string()),
            };
            from_active_session_id = Some(value);
            index += 2;
            continue;
        }
        if parse_options && arg == "--message" {
            if target_active_session_id.is_none() {
                return Err("--message must appear after the target session".to_string());
            }
            let value = args.get(index + 1).cloned().filter(|value| !value.is_empty());
            let value = match value {
                Some(value) => value,
                None => return Err("--message requires message text".to_string()),
            };
            explicit_message = Some(value);
            index += 2;
            parse_options = false;
            continue;
        }
        if parse_options && arg.starts_with("--") {
            return Err(format!(
                "Unknown option for send: {} (use -- before message text starting with --)",
                arg
            ));
        }
        if target_active_session_id.is_none() {
            target_active_session_id = Some(arg);
            index += 1;
            continue;
        }
        message_parts.push(arg);
        index += 1;
    }

    if explicit_message.is_some() && !message_parts.is_empty() {
        return Err("Usage: prime-agent send [--from <agent>] <agent> [--message <message>|<message>]".to_string());
    }
    let message = explicit_message
        .unwrap_or_else(|| message_parts.join(" "))
        .trim()
        .to_string();
    let target_active_session_id = match target_active_session_id {
        Some(target) if !message.is_empty() => target,
        _ => {
            return Err(
                "Usage: prime-agent send [--from <agent>] <agent> [--message <message>|<message>]".to_string(),
            )
        }
    };
    Ok(ParsedSendArgs { target_active_session_id, from_active_session_id, message })
}

async fn run_message_command(
    client: &Arc<DaemonClient>,
    type_: &str,
    args: &[String],
    json: bool,
    io: &DaemonCommandIo<'_>,
) -> Result<(), String> {
    let active_session_id = require_active_session_id(args)?;
    let message = args[1.min(args.len())..].join(" ").trim().to_string();
    if message.is_empty() {
        let command = if type_ == "follow_up" { "follow-up" } else { type_ };
        return Err(format!("Usage: daemon {} <session> <message>", command));
    }
    print_response_data(
        client,
        serde_json::json!({ "type": type_, "activeSessionId": active_session_id, "message": message }),
        json,
        io,
    )
    .await
}

async fn run_cron(
    client: &Arc<DaemonClient>,
    args: &[String],
    json: bool,
    io: &DaemonCommandIo<'_>,
) -> Result<(), String> {
    let subcommand = args.first().cloned().unwrap_or_else(|| "list".to_string());
    if subcommand == "list" {
        let include_inactive = args.iter().any(|arg| arg == "--all" || arg == "-a");
        let selector = args
            .iter()
            .find(|arg| !arg.starts_with('-') && *arg != "list")
            .cloned();
        let active_session_id = match selector {
            Some(selector) => Some(resolve_live_session_selector(client, &selector).await?),
            None => None,
        };
        let response = request(
            client,
            serde_json::json!({
                "type": "cron_list",
                "activeSessionId": active_session_id,
                "includeInactive": include_inactive,
            }),
        )
        .await?;
        let data = require_success(&response)?;
        if json {
            print_json(io, data);
            return Ok(());
        }
        let jobs = match get_cron_jobs(data) {
            Some(jobs) => jobs,
            None => {
                print_json(io, data);
                return Ok(());
            }
        };
        if jobs.is_empty() {
            (io.log)("No scheduled prompts.");
            return Ok(());
        }
        for job in jobs {
            (io.log)(&format_agent_cron_job(&job));
        }
        return Ok(());
    }

    if subcommand == "add" || subcommand == "schedule" {
        let separator = args.iter().position(|arg| arg == "--");
        let separator = match separator {
            Some(separator) => separator,
            None => return Err("Usage: prime-agent schedule add <agent> <schedule> -- <message>".to_string()),
        };
        let active_session_id = match args.get(1) {
            Some(active_session_id) => active_session_id.clone(),
            None => return Err("Usage: prime-agent schedule add <agent> <schedule> -- <message>".to_string()),
        };
        let schedule = args[2.min(separator)..separator].join(" ").trim().to_string();
        let message = args[separator + 1..].join(" ").trim().to_string();
        if schedule.is_empty() || message.is_empty() {
            return Err("Usage: prime-agent schedule add <agent> <schedule> -- <message>".to_string());
        }
        let response = request(
            client,
            serde_json::json!({
                "type": "cron_add",
                "activeSessionId": active_session_id,
                "schedule": schedule,
                "prompt": message,
            }),
        )
        .await?;
        let data = require_success(&response)?;
        if json {
            print_json(io, data);
            return Ok(());
        }
        match get_cron_job(data) {
            Some(job) => (io.log)(&format!(
                "Scheduled {} next={}",
                job.id,
                job.next_run_at.unwrap_or_else(|| "-".to_string())
            )),
            None => (io.log)("Scheduled prompt."),
        }
        return Ok(());
    }

    if subcommand == "cancel" || subcommand == "delete" || subcommand == "remove" {
        let job_id = match args.get(1) {
            Some(job_id) => job_id.clone(),
            None => return Err("Usage: prime-agent schedule cancel <job-id>".to_string()),
        };
        let response = request(client, serde_json::json!({ "type": "cron_cancel", "jobId": job_id })).await?;
        let data = require_success(&response)?;
        if json {
            print_json(io, data);
            return Ok(());
        }
        match get_cron_job(data) {
            Some(job) => (io.log)(&format!("Cancelled {}", job.id)),
            None => (io.log)("Cancelled cron job."),
        }
        return Ok(());
    }

    Err(format!("Unknown schedule command: {}", subcommand))
}

async fn resolve_live_session_selector(client: &Arc<DaemonClient>, selector: &str) -> Result<String, String> {
    let sessions: Vec<serde_json::Value> = get_live_sessions(client, false)
        .await?
        .into_iter()
        .filter(|session| session.get("activeSessionId").and_then(serde_json::Value::as_str).is_some())
        .collect();
    let exact: Vec<&serde_json::Value> = sessions
        .iter()
        .filter(|session| {
            session.get("activeSessionId").and_then(serde_json::Value::as_str) == Some(selector)
                || session.get("sessionId").and_then(serde_json::Value::as_str) == Some(selector)
                || session.get("sessionName").and_then(serde_json::Value::as_str) == Some(selector)
        })
        .collect();
    let suffix: Vec<&serde_json::Value> = sessions
        .iter()
        .filter(|session| {
            session
                .get("activeSessionId")
                .and_then(serde_json::Value::as_str)
                .map(|candidate| matches_session_id_suffix(candidate, selector))
                .unwrap_or(false)
                || session
                    .get("sessionId")
                    .and_then(serde_json::Value::as_str)
                    .map(|candidate| matches_session_id_suffix(candidate, selector))
                    .unwrap_or(false)
        })
        .collect();
    let matches = if exact.is_empty() { suffix } else { exact };
    if matches.len() == 1 {
        return Ok(matches[0]
            .get("activeSessionId")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("")
            .to_string());
    }
    if matches.len() > 1 {
        return Err(format!("Ambiguous active session \"{}\"", selector));
    }
    Err(format!("Unknown active session: {}", selector))
}

async fn print_response_data(
    client: &Arc<DaemonClient>,
    command: serde_json::Value,
    json: bool,
    io: &DaemonCommandIo<'_>,
) -> Result<(), String> {
    let response = request(client, command).await?;
    let data = require_success(&response)?;
    if json || !data.is_null() {
        print_json(io, data);
        return Ok(());
    }
    (io.log)("ok");
    Ok(())
}

async fn run_shutdown(
    client: &Arc<DaemonClient>,
    args: &[String],
    json: bool,
    io: &DaemonCommandIo<'_>,
) -> Result<(), String> {
    let mut force = false;
    for arg in args {
        if arg == "--force" || arg == "-f" {
            force = true;
            continue;
        }
        return Err(format!("Unknown shutdown option: {}", arg));
    }
    print_response_data(client, serde_json::json!({ "type": "shutdown", "force": force }), json, io).await
}

fn run_help(io: &DaemonCommandIo<'_>) -> Result<(), String> {
    (io.log)("Usage: prime-agent daemon <command> [args...]");
    (io.log)(&format!("Commands: {}", DAEMON_CLIENT_COMMANDS.join(", ")));
    Ok(())
}

/// `client.request(command)` for a plain JSON command body.
async fn request(client: &Arc<DaemonClient>, command: serde_json::Value) -> Result<DaemonResponse, String> {
    let command = command.as_object().cloned().ok_or_else(|| "Invalid daemon command".to_string())?;
    client
        .request(command, None, DaemonClientRequestOptions::default())
        .await
        .map_err(|error: DaemonClientError| error.message())
}

fn require_active_session_id(args: &[String]) -> Result<String, String> {
    match args.first() {
        Some(active_session_id) => Ok(active_session_id.clone()),
        None => Err("Missing agent id or name".to_string()),
    }
}

/// Stands in for the TypeScript's absent `response.data` (`undefined`).
static MISSING_DAEMON_DATA: serde_json::Value = serde_json::Value::Null;

fn require_success(response: &DaemonResponse) -> Result<&serde_json::Value, String> {
    if !response.success {
        return Err(response
            .error
            .clone()
            .unwrap_or_else(|| "Daemon request failed".to_string()));
    }
    // `"data" in response ? response.data : undefined`; the port carries an
    // absent `data` as null.
    Ok(response.data.as_ref().unwrap_or(&MISSING_DAEMON_DATA))
}

fn print_json(io: &DaemonCommandIo<'_>, value: &serde_json::Value) {
    (io.log)(&serde_json::to_string_pretty(value).unwrap_or_else(|_| "null".to_string()));
}

/// `const printJsonLine: DaemonClientMessageListener = (value) => { console.log(JSON.stringify(value)); }`.
fn print_json_line(log: &dyn Fn(&str), value: &serde_json::Value) {
    log(&serde_json::to_string(value).unwrap_or_else(|_| "null".to_string()));
}

// ---------------------------------------------------------------------------
// DaemonAttachTerminal
// ---------------------------------------------------------------------------

/// Port of `class DaemonAttachTerminal`. The TypeScript drives Node's readline
/// interface; the port reads stdin lines with tokio and writes through the
/// command `log` sink, so the interactive commands and daemon messages match.
struct DaemonAttachTerminal<'a> {
    client: Arc<DaemonClient>,
    active_session_id: String,
    io: &'a DaemonCommandIo<'a>,
    is_streaming: std::sync::atomic::AtomicBool,
    closed: std::sync::atomic::AtomicBool,
}

impl<'a> DaemonAttachTerminal<'a> {
    fn new(client: Arc<DaemonClient>, active_session_id: &str, io: &'a DaemonCommandIo<'a>) -> Self {
        Self {
            client,
            active_session_id: active_session_id.to_string(),
            io,
            is_streaming: std::sync::atomic::AtomicBool::new(false),
            closed: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Port of `run()`. The TypeScript creates a readline interface, awaits the
    /// `attach` response and then waits for readline to close. The port reads
    /// stdin lines and daemon messages in one select loop and closes on
    /// `/detach`, EOF, `session_closed` or an interrupt.
    async fn run(&self) -> Result<(), String> {
        let mut stream = DaemonMessageStream::subscribe(&self.client);
        let mut lines = {
            use tokio::io::AsyncBufReadExt;
            tokio::io::BufReader::new(tokio::io::stdin()).lines()
        };

        require_success(&request(
            &self.client,
            serde_json::json!({ "type": "attach", "activeSessionId": self.active_session_id }),
        )
        .await?)?;

        loop {
            tokio::select! {
                received = stream.receiver.recv() => match received {
                    Some(message) => {
                        self.handle_message(&message);
                        if self.closed.load(std::sync::atomic::Ordering::SeqCst) {
                            break;
                        }
                    }
                    None => break,
                },
                line = lines.next_line() => match line {
                    Ok(Some(line)) => {
                        if let Err(error) = self.handle_input(&line).await {
                            self.write_line(&red(&error));
                        }
                        if self.closed.load(std::sync::atomic::Ordering::SeqCst) {
                            break;
                        }
                    }
                    _ => break,
                },
                _ = tokio::signal::ctrl_c() => break,
            }
        }

        let _ = request(
            &self.client,
            serde_json::json!({ "type": "detach", "activeSessionId": self.active_session_id }),
        )
        .await;
        self.write_line(&dim("Detached."));
        Ok(())
    }

    async fn handle_input(&self, line: &str) -> Result<(), String> {
        let input = line.trim().to_string();
        if input.is_empty() {
            return Ok(());
        }

        if input == "/quit" || input == "/exit" || input == "/detach" {
            self.close();
            return Ok(());
        }

        if input == "/help" {
            self.print_help();
            return Ok(());
        }

        if input == "/abort" {
            require_success(&request(
                &self.client,
                serde_json::json!({ "type": "abort", "activeSessionId": self.active_session_id }),
            )
            .await?)?;
            self.write_line(&dim("Abort requested."));
            return Ok(());
        }

        if input == "/state" {
            let response = request(
                &self.client,
                serde_json::json!({ "type": "get_state", "activeSessionId": self.active_session_id }),
            )
            .await?;
            let data = require_success(&response)?;
            self.write_line(&to_pretty_json(data));
            return Ok(());
        }

        if input == "/messages" {
            let response = request(
                &self.client,
                serde_json::json!({ "type": "get_messages", "activeSessionId": self.active_session_id }),
            )
            .await?;
            let data = require_success(&response)?;
            if is_messages_data(data) {
                self.write_line(&bold("Transcript"));
                self.print_transcript(&messages_of(data));
            } else {
                self.write_line(&to_pretty_json(data));
            }
            return Ok(());
        }

        if self.is_streaming() {
            require_success(&request(
                &self.client,
                serde_json::json!({
                    "type": "follow_up",
                    "activeSessionId": self.active_session_id,
                    "message": input,
                }),
            )
            .await?)?;
            self.write_line(&dim("Queued follow-up."));
            return Ok(());
        }

        require_success(&request(
            &self.client,
            serde_json::json!({
                "type": "prompt",
                "activeSessionId": self.active_session_id,
                "message": input,
            }),
        )
        .await?)?;
        Ok(())
    }

    fn handle_message(&self, message: &serde_json::Value) {
        let message_type = message.get("type").and_then(serde_json::Value::as_str).unwrap_or("");
        match message_type {
            "daemon_hello" | "session_detached" => {}
            "session_attached" => {
                let state = message.get("state").cloned().unwrap_or(serde_json::Value::Null);
                self.set_streaming(state.get("isStreaming").and_then(serde_json::Value::as_bool) == Some(true));
                self.write_line(&bold(&format!(
                    "Attached to {}",
                    session_label(&state, message, &self.active_session_id)
                )));
                if let Some(model) = state.get("model").filter(|model| !model.is_null()) {
                    let provider = model.get("provider").and_then(serde_json::Value::as_str).unwrap_or("");
                    let id = model.get("id").and_then(serde_json::Value::as_str).unwrap_or("");
                    self.write_line(&dim(&format!("Model: {}/{}", provider, id)));
                }
                if let Some(session_file) = state.get("sessionFile").and_then(serde_json::Value::as_str) {
                    self.write_line(&dim(&format!("Session: {}", session_file)));
                }
                self.print_help();
                let messages = messages_of(message);
                if !messages.is_empty() {
                    self.write_line(&bold("Transcript"));
                    self.print_transcript(&messages);
                }
            }
            "session_event" => {
                let event = message.get("event").cloned().unwrap_or(serde_json::Value::Null);
                if event.get("type").and_then(serde_json::Value::as_str) != Some("refine_complete") {
                    self.handle_session_event(&event);
                }
            }
            "session_replaced" => {
                let state = message.get("state").cloned().unwrap_or(serde_json::Value::Null);
                self.set_streaming(state.get("isStreaming").and_then(serde_json::Value::as_bool) == Some(true));
                self.write_line(&dim(&format!(
                    "Session replaced: {}",
                    session_label(&state, message, &self.active_session_id)
                )));
                let messages = messages_of(message);
                if !messages.is_empty() {
                    self.write_line(&bold("Transcript"));
                    self.print_transcript(&messages);
                }
            }
            "session_resynced" => {
                let snapshot = message.get("snapshot").cloned().unwrap_or(serde_json::Value::Null);
                let state = snapshot.get("state").cloned().unwrap_or(serde_json::Value::Null);
                self.set_streaming(state.get("isStreaming").and_then(serde_json::Value::as_bool) == Some(true));
                self.write_line(&dim(&format!(
                    "Session resynchronized: {}",
                    session_label(&state, message, &self.active_session_id)
                )));
                let messages = messages_of(&snapshot);
                if !messages.is_empty() {
                    self.write_line(&bold("Transcript"));
                    self.print_transcript(&messages);
                }
            }
            "session_closed" => {
                let reason = message.get("reason").and_then(serde_json::Value::as_str).unwrap_or("");
                self.write_line(&yellow(&format!("Session closed: {}", reason)));
                self.close();
            }
            "extension_ui_request" => {
                let method = message.get("method").and_then(serde_json::Value::as_str).unwrap_or("");
                self.write_line(&dim(&format!("Extension UI request: {}", method)));
            }
            "extension_error" => {
                let extension_path = message
                    .get("extensionPath")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("");
                let event = message.get("event").and_then(serde_json::Value::as_str).unwrap_or("");
                let error = message.get("error").and_then(serde_json::Value::as_str).unwrap_or("");
                self.write_line(&red(&format!(
                    "Extension error ({}, {}): {}",
                    extension_path, event, error
                )));
            }
            "response" => {
                if message.get("success").and_then(serde_json::Value::as_bool) != Some(true) {
                    let error = message.get("error").and_then(serde_json::Value::as_str).unwrap_or("");
                    self.write_line(&red(error));
                }
            }
            _ => {}
        }
    }

    fn handle_session_event(&self, event: &serde_json::Value) {
        let event_type = event.get("type").and_then(serde_json::Value::as_str).unwrap_or("");
        match event_type {
            "agent_start" => {
                self.set_streaming(true);
                self.write_line(&dim("Agent started."));
            }
            "agent_end" => {
                self.set_streaming(false);
                self.write_line(&dim("Agent idle."));
            }
            "message_end" => {
                self.print_message(event.get("message").unwrap_or(&serde_json::Value::Null));
            }
            "message_update" => {
                let assistant_event = event.get("assistantMessageEvent");
                if assistant_event
                    .and_then(|assistant_event| assistant_event.get("type"))
                    .and_then(serde_json::Value::as_str)
                    == Some("toolcall_end")
                {
                    let name = assistant_event
                        .and_then(|assistant_event| assistant_event.get("toolCall"))
                        .and_then(|tool_call| tool_call.get("name"))
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("");
                    self.write_line(&dim(&format!("Tool call: {}", name)));
                }
            }
            "tool_execution_start" => {
                let tool_name = event.get("toolName").and_then(serde_json::Value::as_str).unwrap_or("");
                self.write_line(&dim(&format!("Tool started: {}", tool_name)));
            }
            "tool_execution_end" => {
                let tool_name = event.get("toolName").and_then(serde_json::Value::as_str).unwrap_or("");
                let outcome = if event.get("isError").and_then(serde_json::Value::as_bool) == Some(true) {
                    "failed"
                } else {
                    "finished"
                };
                self.write_line(&dim(&format!("Tool {}: {}", outcome, tool_name)));
            }
            "session_action_update" => {
                let actions = event.get("actions");
                let queued_count = actions
                    .and_then(|actions| actions.get("queuedCount"))
                    .and_then(serde_json::Value::as_i64)
                    .unwrap_or(0);
                if queued_count > 0 {
                    let steering = actions
                        .and_then(|actions| actions.get("steering"))
                        .and_then(serde_json::Value::as_array)
                        .map(Vec::len)
                        .unwrap_or(0);
                    let follow_ups = actions
                        .and_then(|actions| actions.get("followUps"))
                        .and_then(serde_json::Value::as_array)
                        .map(Vec::len)
                        .unwrap_or(0);
                    self.write_line(&dim(&format!(
                        "Queued: {} steering, {} follow-up",
                        steering, follow_ups
                    )));
                }
            }
            "session_info_changed" => {
                let name = event
                    .get("name")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string)
                    .unwrap_or_else(|| "(unnamed)".to_string());
                self.write_line(&dim(&format!("Session name: {}", name)));
            }
            "thinking_level_changed" => {
                let level = event.get("level").and_then(serde_json::Value::as_str).unwrap_or("");
                self.write_line(&dim(&format!("Thinking: {}", level)));
            }
            "compaction_start" => {
                let reason = event.get("reason").and_then(serde_json::Value::as_str).unwrap_or("");
                self.write_line(&dim(&format!("Compaction started: {}", reason)));
            }
            "compaction_end" => {
                let reason = event.get("reason").and_then(serde_json::Value::as_str).unwrap_or("");
                let outcome = if event.get("aborted").and_then(serde_json::Value::as_bool) == Some(true) {
                    "aborted"
                } else {
                    "finished"
                };
                self.write_line(&dim(&format!("Compaction {}: {}", outcome, reason)));
            }
            "auto_retry_start" => {
                let attempt = event.get("attempt").and_then(serde_json::Value::as_i64).unwrap_or(0);
                let max_attempts = event.get("maxAttempts").and_then(serde_json::Value::as_i64).unwrap_or(0);
                let error_message = event.get("errorMessage").and_then(serde_json::Value::as_str).unwrap_or("");
                self.write_line(&dim(&format!(
                    "Retry {}/{}: {}",
                    attempt, max_attempts, error_message
                )));
            }
            "auto_retry_end" => {
                if event.get("success").and_then(serde_json::Value::as_bool) == Some(true) {
                    self.write_line(&dim("Retry succeeded."));
                } else {
                    let final_error = event.get("finalError").and_then(serde_json::Value::as_str).unwrap_or("");
                    self.write_line(&dim(&format!("Retry failed: {}", final_error)));
                }
            }
            "rlm_child_update" => {
                let child = event.get("child");
                let label = child
                    .and_then(|child| child.get("label"))
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("");
                let status = child
                    .and_then(|child| child.get("status"))
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("");
                self.write_line(&dim(&format!("Subagent {}: {}", label, status)));
            }
            "goal_update" => {
                let status = event
                    .get("goal")
                    .and_then(|goal| goal.get("status"))
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("");
                self.write_line(&dim(&format!("Goal: {}", status)));
            }
            "refine_failed" => {
                let error = event.get("error").and_then(serde_json::Value::as_str).unwrap_or("");
                self.write_line(&red(&format!("Refinement failed: {}", error)));
            }
            "refine_complete" | "turn_start" | "turn_end" | "message_start" | "tool_execution_update"
            | "auth_stale" => {}
            _ => {}
        }
    }

    fn print_transcript(&self, messages: &[serde_json::Value]) {
        for message in messages {
            self.print_message(message);
        }
    }

    fn print_message(&self, message: &serde_json::Value) {
        let role = get_message_role(message);
        let body = get_message_text(message).trim().to_string();
        let label = format_role(&role);
        if body.is_empty() {
            self.write_line(&format!("{} {}", label, dim("[no text content]")));
        } else {
            self.write_line(&format!("{}\n{}", label, indent(&body)));
        }
    }

    fn print_help(&self) {
        self.write_line(&dim(
            "Type a message and press Enter. Commands: /help /state /messages /abort /detach",
        ));
    }

    fn write_line(&self, text: &str) {
        (self.io.log)(text);
    }

    fn is_streaming(&self) -> bool {
        self.is_streaming.load(std::sync::atomic::Ordering::SeqCst)
    }

    fn set_streaming(&self, value: bool) {
        self.is_streaming.store(value, std::sync::atomic::Ordering::SeqCst);
    }

    fn close(&self) {
        self.closed.store(true, std::sync::atomic::Ordering::SeqCst);
    }
}

fn session_label(state: &serde_json::Value, message: &serde_json::Value, fallback: &str) -> String {
    state
        .get("sessionName")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
        .or_else(|| {
            message
                .get("activeSessionId")
                .and_then(serde_json::Value::as_str)
                .map(str::to_string)
        })
        .unwrap_or_else(|| fallback.to_string())
}

fn messages_of(value: &serde_json::Value) -> Vec<serde_json::Value> {
    value
        .get("messages")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default()
}

fn is_messages_data(value: &serde_json::Value) -> bool {
    value.get("messages").map(serde_json::Value::is_array).unwrap_or(false)
}

fn get_message_role(message: &serde_json::Value) -> String {
    match message.get("role").and_then(serde_json::Value::as_str) {
        Some(role) => role.to_string(),
        None => "message".to_string(),
    }
}

fn get_message_text(message: &serde_json::Value) -> String {
    match message.get("content") {
        Some(content) => format_content(content),
        None => serde_json::to_string(message).unwrap_or_default(),
    }
}

fn format_content(content: &serde_json::Value) -> String {
    if let Some(text) = content.as_str() {
        return text.to_string();
    }
    if let Some(blocks) = content.as_array() {
        return blocks
            .iter()
            .map(format_content_block)
            .filter(|text| !text.is_empty())
            .collect::<Vec<String>>()
            .join("\n");
    }
    if content.is_null() {
        return String::new();
    }
    serde_json::to_string(content).unwrap_or_default()
}

fn format_content_block(block: &serde_json::Value) -> String {
    let block_type = match block.get("type").and_then(serde_json::Value::as_str) {
        Some(block_type) => block_type,
        None => return serde_json::to_string(block).unwrap_or_default(),
    };
    match block_type {
        "text" => get_string_property(block, "text"),
        "thinking" => {
            let thinking = get_string_property(block, "thinking");
            if thinking.is_empty() {
                "[thinking]".to_string()
            } else {
                format!("[thinking]\n{}", thinking)
            }
        }
        "image" => {
            let mime_type = get_string_property(block, "mimeType");
            if mime_type.is_empty() {
                "[image]".to_string()
            } else {
                format!("[image: {}]", mime_type)
            }
        }
        "toolCall" => {
            let name = get_string_property(block, "name");
            let name = if name.is_empty() { "unknown".to_string() } else { name };
            let args_text = match block.get("arguments") {
                Some(arguments) if !arguments.is_null() => {
                    format!(" {}", serde_json::to_string(arguments).unwrap_or_default())
                }
                _ => String::new(),
            };
            format!("[tool call: {}{}]", name, args_text)
        }
        _ => serde_json::to_string(block).unwrap_or_default(),
    }
}

fn get_string_property(value: &serde_json::Value, key: &str) -> String {
    value
        .get(key)
        .and_then(serde_json::Value::as_str)
        .unwrap_or("")
        .to_string()
}

fn format_role(role: &str) -> String {
    match role {
        "user" => blue_bold("user:"),
        "assistant" => green_bold("assistant:"),
        "toolResult" => magenta_bold("tool:"),
        other => bold(&format!("{}:", other)),
    }
}

fn indent(text: &str) -> String {
    text.split('\n').map(|line| format!("  {}", line)).collect::<Vec<String>>().join("\n")
}

// ---------------------------------------------------------------------------
// Message stream plumbing
// ---------------------------------------------------------------------------

/// `createDaemonMessageWaiter` / `waitForSessionEnd` / `waitForSessionClose` as a
/// message predicate plus the client's own on-message / on-close listeners.
#[derive(Clone)]
pub struct SessionEndWaiter {
    active_session_id: String,
    /// Set once the caller's prompt has been acknowledged (`waitForSessionEnd`).
    prompt_acknowledged: bool,
    /// `waitForSessionClose` only resolves on `session_closed`.
    close_only: bool,
    observed_agent_start: bool,
}

impl SessionEndWaiter {
    /// `waitForSessionEnd(client, activeSessionId, isPromptAcknowledged)`.
    pub fn for_session_end(active_session_id: &str, prompt_acknowledged: bool) -> Self {
        Self {
            active_session_id: active_session_id.to_string(),
            prompt_acknowledged,
            close_only: false,
            observed_agent_start: false,
        }
    }

    /// `waitForSessionClose(client, activeSessionId)`.
    pub fn for_session_close(active_session_id: &str) -> Self {
        Self {
            active_session_id: active_session_id.to_string(),
            prompt_acknowledged: false,
            close_only: true,
            observed_agent_start: false,
        }
    }

    /// The prompt is acknowledged before the waiter observes any live `agent_end`.
    pub fn acknowledge_prompt(&mut self) {
        self.prompt_acknowledged = true;
    }

    pub fn should_resolve(&mut self, message: &serde_json::Value) -> bool {
        let message_type = message.get("type").and_then(serde_json::Value::as_str).unwrap_or("");
        if message_type == "session_closed"
            && message.get("activeSessionId").and_then(serde_json::Value::as_str)
                == Some(self.active_session_id.as_str())
        {
            return true;
        }
        if self.close_only
            || message_type != "session_event"
            || message.get("activeSessionId").and_then(serde_json::Value::as_str)
                != Some(self.active_session_id.as_str())
        {
            return false;
        }
        let event_type = message
            .get("event")
            .and_then(|event| event.get("type"))
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        if event_type == "agent_start" {
            self.observed_agent_start = true;
            return false;
        }
        // Requiring agent_start guards against replayed agent_end events from the
        // attach snapshot, but an agent that was already running when we attached
        // never emits agent_start to this client. Once our prompt is acknowledged,
        // any later agent_end is live, so accept it to avoid hanging.
        event_type == "agent_end" && (self.observed_agent_start || self.prompt_acknowledged)
    }
}

/// `client.onMessage(...)` plus the receiving end of the same stream, so a
/// caller can read messages while it waits for one of them to resolve.
struct DaemonMessageStream {
    unsubscribe: Box<dyn Fn() + Send + Sync>,
    receiver: tokio::sync::mpsc::UnboundedReceiver<serde_json::Value>,
}

impl DaemonMessageStream {
    fn subscribe(client: &Arc<DaemonClient>) -> Self {
        let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();
        let unsubscribe = client.on_message(Arc::new(move |message| {
            let _ = sender.send(message.clone());
        }));
        Self { unsubscribe, receiver }
    }
}

impl Drop for DaemonMessageStream {
    fn drop(&mut self) {
        (self.unsubscribe)();
    }
}

/// Read the daemon stream until the waiter resolves, the socket closes, or the
/// process is interrupted (`waitUntilInterrupted`).
async fn daemon_stream_until(
    client: &Arc<DaemonClient>,
    output: &dyn Fn(&serde_json::Value),
    waiter: &mut SessionEndWaiter,
) -> Result<(), String> {
    let mut stream = DaemonMessageStream::subscribe(client);
    loop {
        tokio::select! {
            received = stream.receiver.recv() => match received {
                Some(message) => {
                    output(&message);
                    if waiter.should_resolve(&message) {
                        return Ok(());
                    }
                }
                None => return Ok(()),
            },
            _ = tokio::signal::ctrl_c() => return Ok(()),
        }
    }
}

// ---------------------------------------------------------------------------
// Private local stand-ins for not-yet-landed slices.
// ---------------------------------------------------------------------------

/// Local stand-in for `SessionSummary` from
/// ../modes/daemon/daemon-session-list.js, narrowed to the fields this module reads.
pub fn session_summaries_from_json(sessions: &[serde_json::Value]) -> Vec<super::daemon_list_format::SessionSummary> {
    sessions
        .iter()
        .filter_map(|session| {
            Some(super::daemon_list_format::SessionSummary {
                id: session.get("id").and_then(serde_json::Value::as_str)?.to_string(),
                lifecycle: session.get("lifecycle").and_then(serde_json::Value::as_str)?.to_string(),
                activity: session.get("activity").and_then(serde_json::Value::as_str)?.to_string(),
                session_id: session.get("sessionId").and_then(serde_json::Value::as_str)?.to_string(),
                cwd: session.get("cwd").and_then(serde_json::Value::as_str)?.to_string(),
                session_name: session
                    .get("sessionName")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string),
                model: session.get("model").and_then(|model| {
                    Some(super::daemon_list_format::SessionModel {
                        provider: model.get("provider").and_then(serde_json::Value::as_str)?.to_string(),
                        id: model.get("id").and_then(serde_json::Value::as_str)?.to_string(),
                    })
                }),
                attached_clients: session
                    .get("attachedClients")
                    .and_then(serde_json::Value::as_i64)
                    .unwrap_or(0),
                message_count: session
                    .get("messageCount")
                    .and_then(serde_json::Value::as_i64)
                    .unwrap_or(0),
                modified: session.get("modified").and_then(serde_json::Value::as_str).map(str::to_string),
            })
        })
        .collect()
}

/// `getSessionSummaries(value)`: `undefined` when any entry fails `isSessionSummary`.
pub fn get_session_summaries(value: &serde_json::Value) -> Option<Vec<serde_json::Value>> {
    let sessions = value.get("sessions")?.as_array()?;
    let mut entries = Vec::with_capacity(sessions.len());
    for session in sessions {
        if !is_session_summary(session) {
            return None;
        }
        entries.push(session.clone());
    }
    Some(entries)
}

/// `isSessionSummary(value)`.
pub fn is_session_summary(value: &serde_json::Value) -> bool {
    let candidate = match value.as_object() {
        Some(candidate) => candidate,
        None => return false,
    };
    let string_fields = ["id", "sessionId", "cwd", "lifecycle", "activity"];
    if string_fields.iter().any(|field| !candidate.get(*field).map(serde_json::Value::is_string).unwrap_or(false)) {
        return false;
    }
    let bool_fields = ["isSessionActive", "isStreaming", "isCompacting"];
    if bool_fields.iter().any(|field| !candidate.get(*field).map(serde_json::Value::is_boolean).unwrap_or(false)) {
        return false;
    }
    let number_fields = ["attachedClients", "messageCount"];
    if number_fields.iter().any(|field| !candidate.get(*field).map(serde_json::Value::is_number).unwrap_or(false)) {
        return false;
    }
    if let Some(unfinished) = candidate.get("unfinishedActionCount") {
        if !unfinished.is_null() && !unfinished.is_number() {
            return false;
        }
    }
    let actions = match candidate.get("sessionActions").and_then(serde_json::Value::as_object) {
        Some(actions) => actions,
        None => return false,
    };
    if !actions.get("queuedCount").map(serde_json::Value::is_number).unwrap_or(false) {
        return false;
    }
    if !actions.get("steering").map(serde_json::Value::is_array).unwrap_or(false)
        || !actions.get("followUps").map(serde_json::Value::is_array).unwrap_or(false)
    {
        return false;
    }
    true
}

/// `isLiveSessionSummary(value)`.
pub fn is_live_session_summary(value: &serde_json::Value) -> bool {
    is_session_summary(value)
        && value
            .get("activeSessionId")
            .map(serde_json::Value::is_string)
            .unwrap_or(false)
}

/// `getCronJobs(value)`.
pub fn get_cron_jobs(value: &serde_json::Value) -> Option<Vec<serde_json::Value>> {
    value.get("jobs")?.as_array().cloned()
}

pub struct CronJobRef {
    pub id: String,
    pub next_run_at: Option<String>,
}

/// `getCronJob(value)`.
pub fn get_cron_job(value: &serde_json::Value) -> Option<CronJobRef> {
    let job = value.get("job")?.as_object()?;
    let id = job.get("id").and_then(serde_json::Value::as_str)?.to_string();
    let next_run_at = job
        .get("nextRunAt")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string);
    Some(CronJobRef { id, next_run_at })
}

/// `isAgentMessageReceipt(value)`.
pub fn is_agent_message_receipt(value: &serde_json::Value) -> bool {
    value
        .get("target")
        .and_then(|target| target.get("activeSessionId"))
        .map(serde_json::Value::is_string)
        .unwrap_or(false)
}

/// Local stand-in for `formatAgentCronJob` from ../core/cron-jobs.js. The real
/// helper formats `Date` values with `toLocaleString()`; the port prints the raw
/// ISO-8601 string, which is the same value the daemon sends.
pub fn format_agent_cron_job(job: &serde_json::Value) -> String {
    let id = job.get("id").and_then(serde_json::Value::as_str).unwrap_or("");
    let status = job.get("status").and_then(serde_json::Value::as_str).unwrap_or("");
    let label = job
        .get("label")
        .and_then(serde_json::Value::as_str)
        .map(|label| format!(" label=\"{}\"", label))
        .unwrap_or_default();
    let next = job
        .get("nextRunAt")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| "-".to_string());
    let last = job
        .get("lastRunAt")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| "-".to_string());
    let prompt = job.get("prompt").and_then(serde_json::Value::as_str).unwrap_or("");
    let preview: String = prompt.split_whitespace().collect::<Vec<&str>>().join(" ").chars().take(80).collect();
    let error = job
        .get("lastError")
        .and_then(serde_json::Value::as_str)
        .map(|error| format!(" error={}", error))
        .unwrap_or_default();
    let skipped = job
        .get("lastSkippedAt")
        .and_then(serde_json::Value::as_str)
        .map(|skipped| format!(" skipped={}", skipped))
        .unwrap_or_default();
    let run_count = job.get("runCount").and_then(serde_json::Value::as_i64).unwrap_or(0);
    let expression = job
        .get("schedule")
        .and_then(|schedule| schedule.get("expression"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    format!(
        "{} {}{} next={} last={}{} runs={} schedule=\"{}\" prompt=\"{}\"{}",
        id, status, label, next, last, skipped, run_count, expression, preview, error
    )
}

fn now_ms() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as f64)
        .unwrap_or(0.0)
}

async fn delay(ms: u64) {
    tokio::time::sleep(std::time::Duration::from_millis(ms)).await;
}

fn to_pretty_json(value: &serde_json::Value) -> String {
    serde_json::to_string_pretty(value).unwrap_or_else(|_| "null".to_string())
}

fn resolve_path(value: &str) -> String {
    let path = Path::new(value);
    let joined = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")).join(path)
    };
    normalize_lexically(&joined)
}

fn normalize_lexically(path: &Path) -> String {
    let mut parts: Vec<String> = Vec::new();
    let mut prefix = String::new();
    for component in path.components() {
        use std::path::Component;
        match component {
            Component::Prefix(prefix_component) => {
                prefix.push_str(&prefix_component.as_os_str().to_string_lossy());
            }
            Component::RootDir => {
                // `Component::Prefix("C:")` followed by `Component::RootDir`
                // is the single drive root "C:\". Pushing the root separator
                // unconditionally keeps that root; the earlier `if prefix
                // .is_empty()` test dropped it and produced "C:Users/...",
                // a drive-relative path, so every caller failed with
                // os error 3 (and "/tmp/registry" became "C:tmp/registry").
                prefix.push('/');
            }
            Component::CurDir => {}
            Component::ParentDir => {
                if !parts.is_empty() && parts.last().map(|part| part != "..").unwrap_or(false) {
                    parts.pop();
                } else if !path.is_absolute() {
                    parts.push("..".to_string());
                }
            }
            Component::Normal(part) => parts.push(part.to_string_lossy().to_string()),
        }
    }
    if prefix.is_empty() {
        parts.join("/")
    } else if prefix == "/" {
        format!("/{}", parts.join("/"))
    } else {
        format!("{}{}", prefix, parts.join("/"))
    }
}

fn home_dir() -> String {
    std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap_or_else(|_| ".".to_string())
}

fn expand_tilde_path(path: &str) -> String {
    if path == "~" {
        return home_dir();
    }
    if let Some(rest) = path.strip_prefix("~/").or_else(|| path.strip_prefix("~\\")) {
        return format!("{}/{}", home_dir().trim_end_matches(['/', '\\']), rest);
    }
    path.to_string()
}

fn is_local_path(value: &str) -> bool {
    crate::utils::paths::is_local_path(value)
}

fn normalize_socket_path(socket_path: &str, base_dir: Option<&str>) -> String {
    crate::utils::daemon_socket_path::normalize_socket_path(socket_path, base_dir)
}

/// CLI clients and the supervisor must resolve the same per-user socket.
fn default_daemon_socket_path() -> String {
    std::env::var(super::daemon_launch::DAEMON_WORKER_SUPERVISOR_SOCKET_ENV)
        .ok()
        .filter(|path| !path.trim().is_empty())
        .unwrap_or_else(crate::modes::daemon::daemon_socket::default_daemon_socket_path)
}

fn current_entrypoint() -> String {
    super::subprocess_launch::current_entrypoint()
}

fn current_exec_path() -> String {
    super::subprocess_launch::current_exec_path()
}

fn current_exec_args() -> Vec<String> {
    super::subprocess_launch::current_exec_args()
}

fn current_process_env() -> super::subprocess_launch::ProcessEnv {
    let mut environment = super::subprocess_launch::ProcessEnv::new();
    for (key, value) in std::env::vars() {
        environment.insert(key, value);
    }
    environment
}

/// Local stand-in for `spawnHidden(command, args, { cwd, detached: true, stdio: "ignore" })`.
fn spawn_hidden_detached(
    command: &str,
    args: &[String],
    cwd: &str,
    env: &super::subprocess_launch::ProcessEnv,
) -> Option<SpawnedChild> {
    let mut process = std::process::Command::new(command);
    process
        .args(args)
        .current_dir(cwd)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .env_clear();
    for (key, value) in env {
        process.env(key, value);
    }
    crate::utils::daemon_process::detach_daemon(&mut process);
    match process.spawn() {
        Ok(child) => Some(SpawnedChild { pid: child.id() as i64 }),
        Err(_) => None,
    }
}

struct SpawnedChild {
    pid: i64,
}

/// Local stand-in for `waitUntilInterrupted()`; the stream helpers select on
/// `tokio::signal::ctrl_c()` instead, so this is only kept for callers that
/// need the same shape.
fn wait_until_interrupted() -> SessionEndWaiter {
    SessionEndWaiter::for_session_close("")
}

fn red(value: &str) -> String {
    format!("\u{1b}[31m{}\u{1b}[39m", value)
}

fn green_bold(value: &str) -> String {
    format!("\u{1b}[32m\u{1b}[1m{}\u{1b}[22m\u{1b}[39m", value)
}

fn yellow(value: &str) -> String {
    format!("\u{1b}[33m{}\u{1b}[39m", value)
}

fn blue(value: &str) -> String {
    format!("\u{1b}[34m{}\u{1b}[39m", value)
}

fn blue_bold(value: &str) -> String {
    format!("\u{1b}[34m\u{1b}[1m{}\u{1b}[22m\u{1b}[39m", value)
}

fn magenta_bold(value: &str) -> String {
    format!("\u{1b}[35m\u{1b}[1m{}\u{1b}[22m\u{1b}[39m", value)
}

fn bold(value: &str) -> String {
    format!("\u{1b}[1m{}\u{1b}[22m", value)
}

fn dim(value: &str) -> String {
    format!("\u{1b}[2m{}\u{1b}[22m", value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn logs() -> (std::sync::Arc<std::sync::Mutex<Vec<String>>>, std::sync::Arc<std::sync::Mutex<Vec<String>>>) {
        (
            std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
        )
    }

    fn summary_json(name: &str) -> serde_json::Value {
        serde_json::json!({
            "id": "0123456789abcdef",
            "lifecycle": "live",
            "activity": "idle",
            "isSessionActive": true,
            "isStreaming": false,
            "isCompacting": false,
            "attachedClients": 1,
            "messageCount": 3,
            "sessionActions": { "queuedCount": 0, "steering": [], "followUps": [] },
            "sessionId": "0123456789abcdef",
            "cwd": "/tmp",
            "sessionName": name,
            "activeSessionId": "active-1",
        })
    }

    #[test]
    fn daemon_command_is_recognised_only_for_the_daemon_verb() {
        assert_eq!(DAEMON_CLIENT_COMMANDS.len(), 22);
        assert!(DAEMON_CLIENT_COMMANDS.contains(&"agent-messages"));
        assert!(DAEMON_CLIENT_COMMANDS.contains(&"shutdown"));
    }

    #[test]
    fn parses_the_command_socket_json_and_positionals() {
        let parsed = parse_daemon_client_command(&[
            "--socket".to_string(),
            "/tmp/custom.sock".to_string(),
            "list".to_string(),
            "-a".to_string(),
        ])
        .unwrap();
        assert_eq!(parsed.command, "list");
        assert!(parsed.json == false);
        assert_eq!(parsed.positionals, vec!["-a".to_string()]);
        assert_eq!(parsed.socket_path, normalize_socket_path("/tmp/custom.sock", None));

        let parsed = parse_daemon_client_command(&["list".to_string(), "--json".to_string()]).unwrap();
        assert!(parsed.json);
        assert!(parsed.positionals.is_empty());
    }

    #[test]
    fn defaults_to_open_and_reports_a_missing_socket_value() {
        let parsed = parse_daemon_client_command(&["agent".to_string()]).unwrap();
        assert_eq!(parsed.command, "open");
        assert_eq!(parsed.positionals, vec!["agent".to_string()]);
        assert_eq!(parsed.socket_path, default_daemon_socket_path());

        let error = parse_daemon_client_command(&["--socket".to_string()]).unwrap_err();
        assert_eq!(error, "--socket requires a value");
        let error = parse_daemon_client_command(&["--daemon-socket".to_string(), String::new()]).unwrap_err();
        assert_eq!(error, "--daemon-socket requires a value");
    }

    #[test]
    fn help_flag_sets_the_command_or_appends_a_positional() {
        let parsed = parse_daemon_client_command(&["--help".to_string()]).unwrap();
        assert_eq!(parsed.command, "help");
        let parsed = parse_daemon_client_command(&["list".to_string(), "-h".to_string()]).unwrap();
        assert_eq!(parsed.command, "list");
        assert_eq!(parsed.positionals, vec!["help".to_string()]);
    }

    #[test]
    fn double_dash_ends_option_parsing_and_is_retained_for_send_and_cron() {
        let parsed = parse_daemon_client_command(&[
            "send".to_string(),
            "--".to_string(),
            "--json".to_string(),
        ])
        .unwrap();
        assert!(!parsed.json);
        assert_eq!(parsed.positionals, vec!["--".to_string(), "--json".to_string()]);

        let parsed = parse_daemon_client_command(&[
            "list".to_string(),
            "--".to_string(),
            "--json".to_string(),
        ])
        .unwrap();
        assert!(!parsed.json);
        assert_eq!(parsed.positionals, vec!["--json".to_string()]);
    }

    #[test]
    fn parses_session_args_into_a_create_config() {
        let args: Vec<String> = [
            "alpha",
            "--model",
            "m1",
            "--thinking",
            "high",
            "--tools",
            "read, write ,,",
            "--no-skills",
            "--goal",
            "ship it",
            "--goal-token-budget",
            "500",
        ]
        .iter()
        .map(|value| value.to_string())
        .collect();
        let parsed = parse_session_args(&args, "/work").unwrap();
        assert_eq!(parsed.name.as_deref(), Some("alpha"));
        let config = parsed.config.unwrap();
        assert_eq!(config.get("model").unwrap(), "m1");
        assert_eq!(config.get("thinking").unwrap(), "high");
        assert_eq!(config.get("tools").unwrap(), &serde_json::json!(["read", "write"]));
        assert_eq!(config.get("noSkills").unwrap(), true);
        assert_eq!(config.get("initialGoal").unwrap()["objective"], "ship it");
        assert_eq!(config.get("initialGoal").unwrap()["tokenBudget"], 500.0);
    }

    #[test]
    fn session_args_keep_daemon_startup_arguments() {
        let args: Vec<String> = ["--foreground", "--offline", "beta"].iter().map(|v| v.to_string()).collect();
        let parsed = parse_session_args(&args, "/work").unwrap();
        assert_eq!(parsed.daemon_args, vec!["--foreground".to_string(), "--offline".to_string()]);
        assert_eq!(parsed.name.as_deref(), Some("beta"));
        assert!(parsed.config.is_none());
    }

    #[test]
    fn resume_selectors_resolve_paths_and_ids() {
        let parsed = parse_session_args(&["--resume".to_string(), "abc123".to_string()], "/work").unwrap();
        assert_eq!(parsed.session_path.as_deref(), Some("abc123"));
        assert_eq!(parsed.continue_recent, None);

        let parsed = parse_session_args(&["-c".to_string()], "/work").unwrap();
        assert_eq!(parsed.continue_recent, Some(true));

        let parsed = parse_session_args(&["--resume".to_string(), "./sessions/a.jsonl".to_string()], "/work").unwrap();
        assert_eq!(parsed.session_path.as_deref(), Some(&resolve_path("/work/sessions/a.jsonl")[..]));
    }

    #[test]
    fn session_args_reject_invalid_values() {
        assert_eq!(
            parse_session_args(&["--name".to_string()], "/work").unwrap_err(),
            "--name requires a value"
        );
        assert_eq!(
            parse_session_args(&["--thinking".to_string(), "nope".to_string()], "/work").unwrap_err(),
            "Invalid thinking level \"nope\""
        );
        assert_eq!(
            parse_session_args(&["--no-session".to_string()], "/work").unwrap_err(),
            "--no-session is not supported for daemon sessions; daemon-owned sessions are always persisted"
        );
        assert_eq!(
            parse_session_args(&["--goal-token-budget".to_string(), "5".to_string()], "/work").unwrap_err(),
            "--goal-token-budget requires --goal"
        );
        assert_eq!(
            parse_session_args(&["--goal".to_string(), "  ".to_string()], "/work").unwrap_err(),
            "--goal requires a non-empty objective"
        );
        assert_eq!(
            parse_session_args(&["--goal".to_string(), "x".to_string(), "--goal-token-budget".to_string(), "0".to_string()], "/work")
                .unwrap_err(),
            "--goal-token-budget must be a positive integer"
        );
    }

    #[test]
    fn extension_flags_are_collected_without_consuming_arguments() {
        let parsed = parse_session_args(&["--flag=value".to_string(), "--other".to_string()], "/work").unwrap();
        let config = parsed.config.unwrap();
        assert_eq!(config.get("extensionFlagValues").unwrap()["flag"], "value");
        assert_eq!(config.get("extensionFlagValues").unwrap()["other"], true);
        assert_eq!(parsed.daemon_args, vec!["--flag=value".to_string(), "--other".to_string()]);
    }

    #[test]
    fn default_session_names_skip_taken_numbers() {
        let sessions = vec![
            serde_json::json!({ "sessionName": "1" }),
            serde_json::json!({ "sessionName": "3" }),
            serde_json::json!({ "sessionName": "agent" }),
        ];
        assert_eq!(next_default_session_name(&sessions), "4");
        assert_eq!(next_default_session_name(&[]), "1");
        let taken: Vec<serde_json::Value> = (1..=3)
            .map(|value| serde_json::json!({ "sessionName": value.to_string() }))
            .collect();
        assert_eq!(next_default_session_name(&taken), "4");
    }

    #[test]
    fn create_request_omits_absent_optional_keys() {
        let request = create_session_request("alpha", None, None, None);
        assert_eq!(request, serde_json::json!({ "type": "create", "name": "alpha" }));
        let mut config = serde_json::Map::new();
        config.insert("model".to_string(), serde_json::json!("m1"));
        let request = create_session_request("alpha", Some(&config), Some("/s.jsonl"), Some(true));
        assert_eq!(request["config"]["model"], "m1");
        assert_eq!(request["sessionPath"], "/s.jsonl");
        assert_eq!(request["continueRecent"], true);
    }

    #[test]
    fn session_summary_validation_matches_the_wire_contract() {
        let value = summary_json("alpha");
        assert!(is_session_summary(&value));
        assert!(is_live_session_summary(&value));

        let mut missing_actions = value.clone();
        missing_actions.as_object_mut().unwrap().remove("sessionActions");
        assert!(!is_session_summary(&missing_actions));

        let mut bad_queue = value.clone();
        bad_queue["sessionActions"]["queuedCount"] = serde_json::json!("0");
        assert!(!is_session_summary(&bad_queue));

        let mut not_live = value.clone();
        not_live.as_object_mut().unwrap().remove("activeSessionId");
        assert!(is_session_summary(&not_live));
        assert!(!is_live_session_summary(&not_live));

        let mut optional_unfinished = value.clone();
        optional_unfinished["unfinishedActionCount"] = serde_json::json!(2);
        assert!(is_session_summary(&optional_unfinished));
    }

    #[test]
    fn session_lists_reject_any_invalid_entry() {
        let list = serde_json::json!({ "sessions": [summary_json("alpha"), summary_json("beta")] });
        let sessions = get_session_summaries(&list).unwrap();
        assert_eq!(sessions.len(), 2);
        assert_eq!(sessions[0]["sessionName"], "alpha");

        let list = serde_json::json!({ "sessions": [summary_json("alpha"), { "id": "x" }] });
        assert!(get_session_summaries(&list).is_none());
        assert!(get_session_summaries(&serde_json::json!({})).is_none());

        let summaries = session_summaries_from_json(&sessions);
        assert_eq!(summaries.len(), 2);
        assert_eq!(summaries[0].session_name.as_deref(), Some("alpha"));
        assert_eq!(summaries[0].message_count, 3);
    }

    #[test]
    fn cron_job_accessors_validate_the_shape() {
        let list = serde_json::json!({ "jobs": [{ "id": "j1" }] });
        assert_eq!(get_cron_jobs(&list).unwrap().len(), 1);
        assert!(get_cron_jobs(&serde_json::json!({ "jobs": "x" })).is_none());
        assert!(get_cron_jobs(&serde_json::json!({})).is_none());

        let job = get_cron_job(&serde_json::json!({ "job": { "id": "j1", "nextRunAt": "2026-01-01T00:00:00.000Z" } }))
            .unwrap();
        assert_eq!(job.id, "j1");
        assert_eq!(job.next_run_at.as_deref(), Some("2026-01-01T00:00:00.000Z"));
        assert!(get_cron_job(&serde_json::json!({ "job": { "id": 4 } })).is_none());
        assert!(get_cron_job(&serde_json::json!({})).is_none());
    }

    #[test]
    fn formats_a_cron_job_like_the_typescript_line() {
        let job = serde_json::json!({
            "id": "j1",
            "status": "active",
            "label": "nightly",
            "nextRunAt": "2026-01-01T00:00:00.000Z",
            "prompt": "  do   the  thing  ",
            "runCount": 4,
            "schedule": { "expression": "0 3 * * *" },
        });
        assert_eq!(
            format_agent_cron_job(&job),
            "j1 active label=\"nightly\" next=2026-01-01T00:00:00.000Z last=- runs=4 schedule=\"0 3 * * *\" prompt=\"do the thing\""
        );
        let skipped = serde_json::json!({
            "id": "j2",
            "status": "idle",
            "lastSkippedAt": "2026-01-02T00:00:00.000Z",
            "lastError": "boom",
            "prompt": "",
            "runCount": 0,
            "schedule": { "expression": "x" },
        });
        assert_eq!(
            format_agent_cron_job(&skipped),
            "j2 idle next=- last=- skipped=2026-01-02T00:00:00.000Z runs=0 schedule=\"x\" prompt=\"\" error=boom"
        );
    }

    #[test]
    fn send_args_split_options_from_message_text() {
        let args: Vec<String> = ["--from", "a", "b", "hello", "there"].iter().map(|v| v.to_string()).collect();
        let parsed = parse_send_args(&args).unwrap();
        assert_eq!(parsed.from_active_session_id.as_deref(), Some("a"));
        assert_eq!(parsed.target_active_session_id, "b");
        assert_eq!(parsed.message, "hello there");

        let args: Vec<String> = ["b", "--message", "explicit"].iter().map(|v| v.to_string()).collect();
        let parsed = parse_send_args(&args).unwrap();
        assert_eq!(parsed.message, "explicit");

        let args: Vec<String> = ["b", "--", "--not-an-option"].iter().map(|v| v.to_string()).collect();
        assert_eq!(parse_send_args(&args).unwrap().message, "--not-an-option");
    }

    #[test]
    fn send_args_reject_malformed_invocations() {
        let args = |values: &[&str]| values.iter().map(|value| value.to_string()).collect::<Vec<String>>();
        assert_eq!(parse_send_args(&args(&["--from"])).unwrap_err(), "--from requires a session id or name");
        assert_eq!(parse_send_args(&args(&["--message"])).unwrap_err(), "--message must appear after the target session");
        assert_eq!(parse_send_args(&args(&["a", "--message"])).unwrap_err(), "--message requires message text");
        assert_eq!(
            parse_send_args(&args(&["a", "--nope"])).unwrap_err(),
            "Unknown option for send: --nope (use -- before message text starting with --)"
        );
        assert_eq!(parse_send_args(&args(&["a"])).unwrap_err(), "Usage: prime-agent send [--from <agent>] <agent> [--message <message>|<message>]");
        assert_eq!(
            parse_send_args(&args(&["a", "text", "--message", "other"])).unwrap_err(),
            "Usage: prime-agent send [--from <agent>] <agent> [--message <message>|<message>]"
        );
    }

    #[test]
    fn agent_message_receipts_require_a_target_session() {
        assert!(is_agent_message_receipt(&serde_json::json!({
            "target": { "activeSessionId": "a", "sessionName": "alpha" },
            "deliveryStatus": "queued",
        })));
        assert!(!is_agent_message_receipt(&serde_json::json!({ "target": {} })));
        assert!(!is_agent_message_receipt(&serde_json::json!(null)));
    }

    #[test]
    fn list_arguments_accept_only_the_all_flag() {
        assert!(!parse_list_args(&[]).unwrap());
        assert!(parse_list_args(&["-a".to_string()]).unwrap());
        assert!(parse_list_args(&["--all".to_string()]).unwrap());
        assert_eq!(parse_list_args(&["--nope".to_string()]).unwrap_err(), "Unknown list option: --nope");
    }

    #[test]
    fn session_end_waiter_follows_the_documented_rules() {
        let mut waiter = SessionEndWaiter::for_session_end("a", false);
        assert!(!waiter.should_resolve(&serde_json::json!({
            "type": "session_event", "activeSessionId": "other", "event": { "type": "agent_end" },
        })));
        assert!(!waiter.should_resolve(&serde_json::json!({
            "type": "session_event", "activeSessionId": "a", "event": { "type": "agent_end" },
        })));
        assert!(!waiter.should_resolve(&serde_json::json!({
            "type": "session_event", "activeSessionId": "a", "event": { "type": "agent_start" },
        })));
        assert!(waiter.should_resolve(&serde_json::json!({
            "type": "session_event", "activeSessionId": "a", "event": { "type": "agent_end" },
        })));
        assert!(waiter.should_resolve(&serde_json::json!({
            "type": "session_closed", "activeSessionId": "a",
        })));

        let mut replayed = SessionEndWaiter::for_session_end("a", true);
        assert!(replayed.should_resolve(&serde_json::json!({
            "type": "session_event", "activeSessionId": "a", "event": { "type": "agent_end" },
        })));
        replayed.acknowledge_prompt();
        assert!(replayed.prompt_acknowledged);
    }

    #[test]
    fn session_close_waiter_ignores_session_events() {
        let mut waiter = SessionEndWaiter::for_session_close("a");
        assert!(!waiter.should_resolve(&serde_json::json!({
            "type": "session_event", "activeSessionId": "a", "event": { "type": "agent_end" },
        })));
        assert!(waiter.should_resolve(&serde_json::json!({
            "type": "session_closed", "activeSessionId": "a",
        })));
    }

    #[test]
    fn message_text_helpers_render_every_content_shape() {
        assert_eq!(get_message_role(&serde_json::json!({ "role": "user" })), "user");
        assert_eq!(get_message_role(&serde_json::json!({})), "message");
        assert_eq!(get_message_text(&serde_json::json!({ "content": "hello" })), "hello");
        assert_eq!(
            get_message_text(&serde_json::json!({ "content": [{ "type": "text", "text": "a" }, { "type": "text", "text": "" }] })),
            "a"
        );
        assert_eq!(
            format_content_block(&serde_json::json!({ "type": "thinking", "thinking": "why" })),
            "[thinking]\nwhy"
        );
        assert_eq!(format_content_block(&serde_json::json!({ "type": "thinking" })), "[thinking]");
        assert_eq!(format_content_block(&serde_json::json!({ "type": "image", "mimeType": "image/png" })), "[image: image/png]");
        assert_eq!(format_content_block(&serde_json::json!({ "type": "image" })), "[image]");
        assert_eq!(
            format_content_block(&serde_json::json!({ "type": "toolCall", "name": "read", "arguments": { "a": 1 } })),
            "[tool call: read {\"a\":1}]"
        );
        assert_eq!(format_content_block(&serde_json::json!({ "type": "toolCall" })), "[tool call: unknown]");
        assert_eq!(get_message_text(&serde_json::json!({ "content": null })), "");
        assert_eq!(
            get_message_text(&serde_json::json!({ "content": 7 })),
            "7"
        );
        assert_eq!(indent("a\nb"), "  a\n  b");
    }

    #[test]
    fn role_labels_use_the_same_colours() {
        assert_eq!(format_role("user"), "\u{1b}[34m\u{1b}[1muser:\u{1b}[22m\u{1b}[39m");
        assert_eq!(format_role("assistant"), "\u{1b}[32m\u{1b}[1massistant:\u{1b}[22m\u{1b}[39m");
        assert_eq!(format_role("toolResult"), "\u{1b}[35m\u{1b}[1mtool:\u{1b}[22m\u{1b}[39m");
        assert_eq!(format_role("system"), "\u{1b}[1msystem:\u{1b}[22m");
    }

    #[test]
    fn require_success_surfaces_the_daemon_error() {
        fn response(value: serde_json::Value) -> DaemonResponse {
            serde_json::from_value(value).expect("daemon response")
        }
        let ok = response(serde_json::json!({
            "type": "response", "command": "list", "success": true, "data": { "a": 1 }
        }));
        assert_eq!(require_success(&ok).unwrap()["a"], 1);
        let failure = response(serde_json::json!({
            "type": "response", "command": "list", "success": false, "error": "nope"
        }));
        assert_eq!(require_success(&failure).unwrap_err(), "nope");
        let bare = response(serde_json::json!({
            "type": "response", "command": "list", "success": false
        }));
        assert_eq!(require_success(&bare).unwrap_err(), "Daemon request failed");
        let missing_data = response(serde_json::json!({
            "type": "response", "command": "list", "success": true
        }));
        assert!(require_success(&missing_data).unwrap().is_null());
    }

    #[test]
    fn require_active_session_id_reports_a_missing_selector() {
        assert_eq!(require_active_session_id(&[]).unwrap_err(), "Missing agent id or name");
        assert_eq!(require_active_session_id(&["a".to_string()]).unwrap(), "a");
    }

    #[test]
    fn shutdown_parses_only_the_force_flag() {
        let logged: std::cell::RefCell<Vec<String>> = std::cell::RefCell::new(Vec::new());
        let errors: std::cell::RefCell<Vec<String>> = std::cell::RefCell::new(Vec::new());
        let io = DaemonCommandIo {
            log: &|line: &str| logged.borrow_mut().push(line.to_string()),
            error: &|line: &str| errors.borrow_mut().push(line.to_string()),
            set_exit_code: &|_code: i32| {},
            stdin_is_tty: None,
            prompt_yes_no: &|_message: &str| false,
            cwd: "/work".to_string(),
        };
        assert!(run_help(&io).is_ok());
        let logged = logged.borrow();
        assert_eq!(logged[0], "Usage: prime-agent daemon <command> [args...]");
        assert!(logged[1].starts_with("Commands: start, ps, list, create"));
    }

    #[test]
    fn resolve_path_options_keep_non_local_sources() {
        assert_eq!(resolve_path_option("npm:foo", "/work"), "npm:foo");
        assert_eq!(resolve_path_option("./x", "/work"), resolve_path("/work/x"));
        assert_eq!(expand_tilde_path("~/x"), format!("{}/x", home_dir().trim_end_matches(['/', '\\'])));
        assert_eq!(expand_tilde_path("~"), home_dir());
        assert_eq!(expand_tilde_path("plain"), "plain");
    }
    /// `Component::Prefix("C:")` followed by `Component::RootDir` is the single
    /// drive root "C:\". Returns the path once the host has produced that shape
    /// (None on a POSIX host, where "C:\..." has no Prefix component at all), so
    /// the assertions below cannot silently pass on a path that never reaches the
    /// RootDir arm.
    fn drive_absolute_input(label: &str, raw: &str) -> Option<PathBuf> {
        use std::path::Component;
        let path = Path::new(raw);
        if !matches!(path.components().next(), Some(Component::Prefix(_))) {
            return None;
        }
        assert!(
            path.components().any(|component| component == Component::RootDir),
            "{raw:?} has a drive prefix but no RootDir component ({label})"
        );
        Some(path.to_path_buf())
    }

    /// The defect this pins: the old `if prefix.is_empty()` guard skipped the
    /// root separator because `Prefix("C:")` had already filled `prefix`, so a
    /// drive-absolute path collapsed to a DRIVE-RELATIVE one - "C:Users/x/registry"
    /// instead of "C:/Users/x/registry", and a drive-rooted "/tmp/x" became
    /// "C:tmp/x". Every downstream create_dir_all / open / lockfile then failed
    /// with os error 3 ("The system cannot find the path specified").
    #[test]
    fn drive_roots_survive_the_lexical_collapse() {
        // Checked on every host: a rooted path keeps its root separator.
        let rooted = normalize_lexically(Path::new("/tmp/x"));
        assert!(rooted.starts_with('/'), "root separator dropped: {rooted}");

        let registry = match drive_absolute_input("registry", "C:\\Users\\x\\registry") {
            Some(path) => path,
            None => return,
        };
        let normalised = normalize_lexically(&registry);
        assert!(normalised.starts_with("C:/"), "drive root dropped: {normalised}");
        assert_eq!(normalised, "C:/Users/x/registry");

        // `Path::join` keeps the drive when it appends the rooted POSIX spelling,
        // which is exactly how a caller's resolve("/tmp/x") reaches this function.
        let joined = drive_absolute_input("joined /tmp/x", "C:\\base")
            .expect("C:\\base is drive-absolute")
            .join("/tmp/x");
        let normalised = normalize_lexically(&joined);
        assert!(!normalised.starts_with("C:tmp"), "/tmp/x became drive-relative: {normalised}");
        assert_eq!(normalised, "C:/tmp/x");

        let dotted = drive_absolute_input("parent collapse", "C:\\Users\\x\\..\\y")
            .expect("C:\\Users\\x\\..\\y is drive-absolute");
        assert_eq!(normalize_lexically(&dotted), "C:/Users/y");

        // Only the RootDir arm changed: a drive-RELATIVE input has no RootDir, so
        // it must still come back rootless instead of gaining a "C:/" root.
        let relative = Path::new("C:registry");
        if !relative.components().any(|component| component == std::path::Component::RootDir) {
            assert_eq!(normalize_lexically(relative), "C:registry");
        }
    }
}

#[cfg(all(test, unix))]
#[test]
fn detached_daemon_survives_launcher_exit() {
    crate::utils::daemon_process::tests::assert_detached_spawn(|command, args, cwd, env| {
        assert!(spawn_hidden_detached(command, args, cwd, env).is_some());
    });
}
