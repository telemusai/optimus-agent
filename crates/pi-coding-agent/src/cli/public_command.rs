//! Port of packages/coding-agent/src/cli/public-command.ts
//!
//! blocked_on: a library crate cannot read `process.stdin`, write `console.log`
//! or set `process.exitCode`, so the port takes the same explicit host seam the
//! other CLI modules use (`DaemonCommandIo` in cli/daemon-command.ts). Every
//! callback mirrors exactly one Node call the TypeScript makes.

#[path = "shutdown_options.rs"]
mod shutdown_options;

use indexmap::IndexSet;

use crate::cli::args::{parse_args, INTERNAL_RUNTIME_COMMAND_MARKER};
use crate::cli::command_registry::{
    find_command_suggestion, format_command_help, format_top_level_help, get_child_command_specs,
    get_command_spec, is_help_command_request, public_command_names, removed_command_names,
};
use crate::cli::daemon_command::{handle_daemon_command, DaemonCommandIo};
use crate::cli::daemon_ps::{run_ps, run_reap, run_shutdown_all, DaemonPsIo};
use crate::cli::daemon_update_restart::DAEMON_UPDATE_RESTART_COORDINATOR_FLAG;
use crate::cli::dgx_auth_command::{run_dgx_auth_command, DgxAuthCommandIo};
use crate::config::{APP_COMMAND as APP_NAME, SELF_UPDATE_INTERACTIVE_CHILD_ENV};
use crate::core::auth_storage::AuthStorage;
use crate::core::mcp::mcp_command::{run_mcp_management_command, McpCredentialStore};
use crate::core::settings_manager::SettingsManager;
use crate::package_manager_cli::{handle_package_command, is_self_update_source};

/// `PublicCommandResult`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublicCommandResult {
    pub handled: bool,
    pub args: Vec<String>,
    pub explicit_agents_view: bool,
    /// `attachAgent?: string` - `None` is `undefined`.
    pub attach_agent: Option<String>,
}

/// `const HANDLED: PublicCommandResult`.
const HANDLED: PublicCommandResult = PublicCommandResult {
    handled: true,
    args: Vec::new(),
    explicit_agents_view: false,
    attach_agent: None,
};

/// Callback surface for `console.log` / `console.error` / `process.exitCode`
/// and the interactive prompt used by the nested commands.
pub struct PublicCommandIo<'a> {
    pub log: &'a dyn Fn(&str),
    pub error: &'a dyn Fn(&str),
    pub set_exit_code: &'a dyn Fn(i32),
    pub stdin_is_tty: Option<bool>,
    pub prompt_yes_no: &'a dyn Fn(&str) -> bool,
    pub cwd: String,
}

/// `chalk.red`.
fn red(message: &str) -> String {
    format!("\u{1b}[31m{message}\u{1b}[39m")
}

/// `chalk.dim`.
fn dim(message: &str) -> String {
    format!("\u{1b}[2m{message}\u{1b}[22m")
}

/// `handlePublicCommand(args)`.
pub async fn handle_public_command(args: &[String], io: &PublicCommandIo<'_>) -> PublicCommandResult {
    match run_public_command(args, io).await {
        Ok(result) => result,
        Err(error) => fail(&error, None, io),
    }
}

/// The nested commands report their own failures through the IO seam
/// (`handleDaemonCommand` and `handlePackageCommand` both set the exit code
/// themselves), so the only path that reaches `handle_public_command`'s `Err`
/// arm is the one the TypeScript's `catch` also covers: a rejected
/// `runMcpManagementCommand` promise.

/// `runPublicCommand(args)`.
async fn run_public_command(
    args: &[String],
    io: &PublicCommandIo<'_>,
) -> Result<PublicCommandResult, String> {
    let args: Vec<String> = normalize_leading_daemon_socket_option(args);
    if args.first().map(String::as_str) == Some("help")
        && is_help_command_request(&as_strs(&args[1..]))
    {
        return Ok(print_requested_help(&args[1..], io));
    }

    let command = match args.first().cloned() {
        Some(command) => command,
        None => return Ok(continue_with(args)),
    };

    if removed_command_names().contains(command.as_str()) {
        return Ok(reject_removed_command(&args, io));
    }

    if !public_command_names().contains(command.as_str()) {
        return Ok(continue_with(args));
    }
    if command == "update"
        && std::env::var(SELF_UPDATE_INTERACTIVE_CHILD_ENV).ok().as_deref() == Some("1")
    {
        let _ = handle_package_command(&args).await;
        return Ok(HANDLED);
    }
    if command == "update" && args.iter().any(|arg| arg == DAEMON_UPDATE_RESTART_COORDINATOR_FLAG) {
        let _ = handle_package_command(&args).await;
        return Ok(HANDLED);
    }

    let separator_index = args.iter().position(|arg| arg == "--").map(|index| index as i64).unwrap_or(-1);
    let help_index = args.iter().enumerate().position(|(index, arg)| {
        index > 0
            && (separator_index == -1 || (index as i64) < separator_index)
            && (arg == "--help" || arg == "-h")
    });
    if let Some(help_index) = help_index {
        return Ok(print_requested_help(&get_command_path(&args[..help_index]), io));
    }

    match command.as_str() {
        "agents" => Ok(PublicCommandResult {
            handled: false,
            args: args[1..].to_vec(),
            explicit_agents_view: true,
            attach_agent: None,
        }),
        "list" => run_internal_agent_command("list", &args[1..], io).await,
        "attach" => {
            let rest = &args[1..];
            let agent = rest.first().cloned();
            let options = rest.get(1..).unwrap_or(&[]).to_vec();
            let invalid_agent = match &agent {
                None => true,
                Some(agent) => agent.starts_with('-'),
            };
            if invalid_agent || has_positional_arguments(&options) {
                let usage = get_command_spec(&["attach"]).map(|spec| spec.usage).unwrap_or("attach");
                return Ok(fail(&format!("Usage: {APP_NAME} {usage}"), None, io));
            }
            if has_conflicting_attach_option(&options) {
                return Ok(fail(
                    "attach cannot be combined with --resume, --continue, or --fork.",
                    None,
                    io,
                ));
            }
            let agent = agent.unwrap();
            let mut attach_args = vec!["--resume".to_string(), agent.clone()];
            attach_args.extend(options);
            Ok(PublicCommandResult {
                handled: false,
                args: attach_args,
                explicit_agents_view: false,
                attach_agent: Some(agent),
            })
        }
        "stop" => {
            if !require_operand_count(&args[1..], 1, Some(1), "stop", io) {
                return Ok(HANDLED);
            }
            run_internal_agent_command("kill", &args[1..], io).await
        }
        "rename" => {
            if !require_operand_count(&args[1..], 2, None, "rename", io) {
                return Ok(HANDLED);
            }
            run_internal_agent_command("rename", &args[1..], io).await
        }
        "send" => run_internal_agent_command("send", &args[1..], io).await,
        "schedule" => run_nested_agent_command("schedule", "cron", &args[1..], io).await,
        "status" => run_status(&args[1..], io).await,
        "doctor" => run_doctor(&args[1..], io).await,
        "shutdown" => run_shutdown(&args[1..], io).await,
        "package" => run_package(&args[1..], io).await,
        "mcp" => run_mcp(&args[1..], io).await,
        "dgx-auth" => run_dgx_auth(&args[1..], io),
        "update" => {
            let rest = &args[1..];
            let has_legacy_self_target =
                rest.iter().any(|arg| arg == "--self" || is_self_update_source(arg));
            let has_legacy_package_target = rest.iter().any(|arg| {
                arg == "--extensions"
                    || arg == "--extension"
                    || (!arg.starts_with('-') && !is_self_update_source(arg))
            });
            if has_legacy_self_target && has_legacy_package_target {
                return Ok(fail(
                    "Optimus Agent and package updates are now separate.",
                    Some(&format!(
                        "Run \"{APP_NAME} update [--force]\" and \"{APP_NAME} package update [source]\" separately."
                    )),
                    io,
                ));
            }
            if has_legacy_self_target {
                return Ok(fail(
                    "An update target is no longer needed.",
                    Some(&format!("Use \"{APP_NAME} update [--force]\".")),
                    io,
                ));
            }
            if has_legacy_package_target {
                return Ok(fail(
                    "Package updates moved to the package command.",
                    Some(&format!("Use \"{APP_NAME} package update [source]\".")),
                    io,
                ));
            }
            let options = match parse_boolean_options(rest, &["--force"], "update", io) {
                Some(options) => options,
                None => return Ok(HANDLED),
            };
            let mut command_args = vec!["update".to_string(), "--self".to_string()];
            command_args.extend(options);
            let _ = handle_package_command(&command_args).await;
            Ok(HANDLED)
        }
        "model" => Ok(rewrite_nested_command("model", "list", "--list-models", &args[1..], io)),
        "session" => Ok(rewrite_nested_command("session", "export", "--export", &args[1..], io)),
        "config" => {
            if !require_argument_count(&args[1..], 0, "config", io) {
                return Ok(HANDLED);
            }
            Ok(continue_with(args))
        }
        _ => Ok(continue_with(args)),
    }
}

/// `normalizeLeadingDaemonSocketOption(args)`.
fn normalize_leading_daemon_socket_option(args: &[String]) -> Vec<String> {
    let option = args.first();
    if !matches!(option.map(String::as_str), Some("--daemon-socket" | "--socket")) {
        return args.to_vec();
    }
    let socket_path = args.get(1);
    let command = args.get(2);
    if socket_path.is_none()
        || !(command.map(String::as_str) == Some("stop") || command.map(String::as_str) == Some("rename")
            || command.map(String::as_str) == Some("shutdown"))
    {
        return args.to_vec();
    }
    let mut normalized = vec![command.unwrap().clone()];
    normalized.extend(args.get(3..).unwrap_or(&[]).iter().cloned());
    normalized.push(option.unwrap().clone());
    normalized.push(socket_path.unwrap().clone());
    normalized
}

/// `continueWith(args)`.
fn continue_with(args: Vec<String>) -> PublicCommandResult {
    PublicCommandResult {
        handled: false,
        args,
        explicit_agents_view: false,
        attach_agent: None,
    }
}

/// `printRequestedHelp(path)`.
fn print_requested_help(path: &[String], io: &PublicCommandIo<'_>) -> PublicCommandResult {
    if path.is_empty() {
        (io.log)(&format_top_level_help());
        return HANDLED;
    }
    if removed_command_names().contains(path[0].as_str()) {
        return reject_removed_command(path, io);
    }
    if let Some(help) = format_command_help(&as_strs(path)) {
        (io.log)(&help);
        return HANDLED;
    }
    let parent = &path[..path.len() - 1];
    let candidates = child_command_names(parent);
    let suggestion = find_command_suggestion(&path[path.len() - 1], &as_strs(&candidates));
    let hint = suggestion.map(|suggestion| {
        let mut full = parent.to_vec();
        full.push(suggestion);
        format!("Did you mean \"{APP_NAME} help {}\"?", full.join(" "))
    });
    fail(&format!("Unknown command: {}", path.join(" ")), hint.as_deref(), io)
}

/// `getCommandPath(args)`.
fn get_command_path(args: &[String]) -> Vec<String> {
    let mut path: Vec<String> = Vec::new();
    for arg in args {
        let mut candidate = path.clone();
        candidate.push(arg.clone());
        if get_command_spec(&as_strs(&candidate)).is_none() {
            break;
        }
        path.push(arg.clone());
    }
    path
}

/// `rejectRemovedCommand(args)`.
fn reject_removed_command(args: &[String], io: &PublicCommandIo<'_>) -> PublicCommandResult {
    let command = args.first().map(String::as_str);
    let subcommand = args.get(1).map(String::as_str);
    let replacement = match command {
        Some("daemon") => Some("Run \"optimus-agent help\" to see the agent commands.".to_string()),
        Some("app") if subcommand == Some("update") => Some("Use \"optimus-agent update\".".to_string()),
        Some("install") => Some("Use \"optimus-agent package install\".".to_string()),
        Some("remove") | Some("uninstall") => Some("Use \"optimus-agent package remove\".".to_string()),
        Some("manage") => Some("Use \"optimus-agent agents\".".to_string()),
        _ => None,
    };
    let name = args[..args.len().min(2)].join(" ");
    fail(&format!("Unknown command: {name}"), replacement.as_deref(), io)
}

/// `runInternalAgentCommand(command, args)`.
async fn run_internal_agent_command(
    command: &str,
    args: &[String],
    io: &PublicCommandIo<'_>,
) -> Result<PublicCommandResult, String> {
    let mut forwarded = vec!["daemon".to_string(), command.to_string()];
    forwarded.extend(args.iter().cloned());
    let daemon_io = daemon_command_io(io);
    handle_daemon_command(&forwarded, &daemon_io).await;
    Ok(HANDLED)
}

/// `runNestedAgentCommand(parent, internalCommand, args)`.
async fn run_nested_agent_command(
    parent: &str,
    internal_command: &str,
    args: &[String],
    io: &PublicCommandIo<'_>,
) -> Result<PublicCommandResult, String> {
    let subcommand = args.first().cloned();
    let children = child_command_names(&[parent.to_string()]);
    let known = subcommand
        .as_ref()
        .map(|subcommand| children.iter().any(|child| child == subcommand))
        .unwrap_or(false);
    if !known {
        let suggestion = subcommand
            .as_ref()
            .and_then(|subcommand| find_command_suggestion(subcommand, &as_strs(&children)));
        let message = match &subcommand {
            Some(subcommand) => format!("Unknown {parent} command: {subcommand}"),
            None => format!("Missing {parent} command."),
        };
        let hint = match suggestion {
            Some(suggestion) => format!("Did you mean \"{APP_NAME} {parent} {suggestion}\"?"),
            None => format!("Run \"{APP_NAME} help {parent}\" for usage."),
        };
        return Ok(fail(&message, Some(&hint), io));
    }
    if parent == "schedule" && !validate_schedule_args(args, io) {
        return Ok(HANDLED);
    }
    let mut forwarded = vec!["daemon".to_string(), internal_command.to_string()];
    forwarded.extend(args.iter().cloned());
    let daemon_io = daemon_command_io(io);
    handle_daemon_command(&forwarded, &daemon_io).await;
    Ok(HANDLED)
}

/// `runStatus(args)`.
async fn run_status(args: &[String], io: &PublicCommandIo<'_>) -> Result<PublicCommandResult, String> {
    let options = match parse_boolean_options(args, &["--json"], "status", io) {
        Some(options) => options,
        None => return Ok(HANDLED),
    };
    let ps_io = daemon_ps_io(io);
    run_ps(options.contains("--json"), &ps_io).await?;
    Ok(HANDLED)
}

/// `runDoctor(args)`.
async fn run_doctor(args: &[String], io: &PublicCommandIo<'_>) -> Result<PublicCommandResult, String> {
    let options = match parse_boolean_options(args, &["--fix", "--json"], "doctor", io) {
        Some(options) => options,
        None => return Ok(HANDLED),
    };
    let ps_io = daemon_ps_io(io);
    if options.contains("--fix") {
        run_reap(options.contains("--json"), false, &ps_io).await?;
    } else {
        run_ps(options.contains("--json"), &ps_io).await?;
    }
    Ok(HANDLED)
}

/// `runShutdown(args)`.
async fn run_shutdown(args: &[String], io: &PublicCommandIo<'_>) -> Result<PublicCommandResult, String> {
    let options = shutdown_options::parse(args)?;
    if let Some(socket) = options.socket {
        if !options.force {
            if options.json || io.stdin_is_tty != Some(true) {
                return Err("Targeted shutdown requires confirmation in an interactive terminal. Use --force to skip confirmation.".into());
            }
            if !(io.prompt_yes_no)(&format!("Stop the daemon at {socket}? Its active work will be interrupted.")) {
                (io.log)("Shutdown cancelled.");
                return Ok(HANDLED);
            }
        }
        let mut forwarded = vec!["--socket".to_string(), socket];
        if options.force { forwarded.push("--force".into()); }
        if options.json { forwarded.push("--json".into()); }
        return run_internal_agent_command("shutdown", &forwarded, io).await;
    }
    let ps_io = daemon_ps_io(io);
    run_shutdown_all(options.json, options.force, &ps_io).await?;
    Ok(HANDLED)
}

/// `runMcp(args)`.
async fn run_mcp(args: &[String], io: &PublicCommandIo<'_>) -> Result<PublicCommandResult, String> {
    let mut settings_manager = SettingsManager::create(&io.cwd, None);
    // `AuthStorage.create()`: wrapped in the local type below because
    // `core/auth-storage.ts` belongs to another slice and does not implement
    // `McpCredentialStore` for the Rust `AuthStorage` yet.
    let mut auth_storage = OwnedMcpCredentialStore::new(AuthStorage::create(None, None));
    let result = run_mcp_management_command(
        args,
        &mut settings_manager,
        Some(&mut auth_storage as &mut dyn McpCredentialStore),
    )
    .await?;
    (io.log)(&result.message);
    Ok(HANDLED)
}

/// `AuthStorage.create()` as the `McpCredentialStore` `runMcpManagementCommand` takes.
///
/// Internal plumbing only: it forwards to `AuthStorage::remove_verified`, the
/// same disk-verified removal the TypeScript `AuthStorage` performs.
struct OwnedMcpCredentialStore {
    auth_storage: AuthStorage,
}

impl OwnedMcpCredentialStore {
    fn new(auth_storage: AuthStorage) -> Self {
        Self { auth_storage }
    }
}

impl McpCredentialStore for OwnedMcpCredentialStore {
    fn remove_verified(&mut self, provider: &str) -> Result<(), String> {
        self.auth_storage.remove_verified(provider)
    }
}

/// `runDgxAuth(args)`: the dgx-auth command group is synchronous (the core
/// auth-file API is sync), so failures propagate through the same `Err` arm
/// the TypeScript `catch` covers for a rejected MCP promise.
fn run_dgx_auth(args: &[String], io: &PublicCommandIo<'_>) -> Result<PublicCommandResult, String> {
    let dgx_io = dgx_auth_command_io(io);
    run_dgx_auth_command(args, &dgx_io)?;
    Ok(HANDLED)
}

/// `runPackage(args)`.
async fn run_package(args: &[String], io: &PublicCommandIo<'_>) -> Result<PublicCommandResult, String> {
    let subcommand = args.first().cloned();
    if subcommand.as_deref() == Some("uninstall") {
        return Ok(fail(
            "Unknown package command: uninstall",
            Some(&format!("Use \"{APP_NAME} package remove\".")),
            io,
        ));
    }
    let children = child_command_names(&["package".to_string()]);
    let known = subcommand
        .as_ref()
        .map(|subcommand| children.iter().any(|child| child == subcommand))
        .unwrap_or(false);
    if !known {
        let suggestion = subcommand
            .as_ref()
            .and_then(|subcommand| find_command_suggestion(subcommand, &as_strs(&children)));
        let message = match &subcommand {
            Some(subcommand) => format!("Unknown package command: {subcommand}"),
            None => "Missing package command.".to_string(),
        };
        let hint = match suggestion {
            Some(suggestion) => format!("Did you mean \"{APP_NAME} package {suggestion}\"?"),
            None => format!("Run \"{APP_NAME} help package\" for usage."),
        };
        return Ok(fail(&message, Some(&hint), io));
    }
    let subcommand = subcommand.unwrap();
    let rest = args[1..].to_vec();
    if subcommand == "list" && !rest.is_empty() {
        return Ok(fail(&format!("Usage: {APP_NAME} package list"), None, io));
    }
    if subcommand == "update" {
        if rest.iter().any(|arg| {
            arg == "--self" || arg == "--extensions" || arg == "--extension" || arg == "--force"
        }) {
            return Ok(fail(
                "Package updates accept only an optional source. Use \"optimus-agent update --force\" to update Optimus Agent.",
                None,
                io,
            ));
        }
        if rest.len() > 1 {
            return Ok(fail(&format!("Usage: {APP_NAME} package update [source]"), None, io));
        }
        if let Some(source) = rest.first() {
            if is_self_update_source(source) {
                return Ok(fail(
                    "Use \"optimus-agent update\" to update Optimus Agent.",
                    None,
                    io,
                ));
            }
        }
        let mut forwarded = vec!["update".to_string()];
        if rest.is_empty() {
            forwarded.push("--extensions".to_string());
        } else {
            forwarded.extend(rest.iter().cloned());
        }
        let _ = handle_package_command(&forwarded).await;
        return Ok(HANDLED);
    }
    let mut forwarded = vec![subcommand];
    forwarded.extend(rest);
    let _ = handle_package_command(&forwarded).await;
    Ok(HANDLED)
}

/// `rewriteNestedCommand(parent, subcommand, flag, args)`.
fn rewrite_nested_command(
    parent: &str,
    subcommand: &str,
    flag: &str,
    args: &[String],
    io: &PublicCommandIo<'_>,
) -> PublicCommandResult {
    if args.first().map(String::as_str) != Some(subcommand) {
        let candidate = args.first().cloned();
        let suggestion = candidate
            .as_ref()
            .and_then(|candidate| find_command_suggestion(candidate, &[subcommand]));
        let message = match &candidate {
            Some(candidate) => format!("Unknown {parent} command: {candidate}"),
            None => format!("Missing {parent} command."),
        };
        let hint = match suggestion {
            Some(suggestion) => format!("Did you mean \"{APP_NAME} {parent} {suggestion}\"?"),
            None => format!("Run \"{APP_NAME} help {parent}\" for usage."),
        };
        return fail(&message, Some(&hint), io);
    }
    let usage = || {
        let spec = get_command_spec(&[parent, subcommand]);
        format!(
            "Usage: {APP_NAME} {}",
            spec.map(|spec| spec.usage).unwrap_or(&format!("{parent} {subcommand}"))
        )
    };
    let Some(split_args) = split_operands_and_options(&args[1..]) else {
        return fail(&usage(), None, io);
    };
    let (operands, options) = split_args;
    let valid_count = if parent == "model" {
        operands.len() <= 1
    } else {
        operands.len() >= 1 && operands.len() <= 2
    };
    if !valid_count {
        return fail(&usage(), None, io);
    }
    let mut rewritten = vec![INTERNAL_RUNTIME_COMMAND_MARKER.to_string(), flag.to_string()];
    rewritten.extend(operands);
    rewritten.extend(options);
    continue_with(rewritten)
}

/// `parseBooleanOptions(args, allowed, command)`.
fn parse_boolean_options(
    args: &[String],
    allowed: &[&str],
    command: &str,
    io: &PublicCommandIo<'_>,
) -> Option<IndexSet<String>> {
    let mut options: IndexSet<String> = IndexSet::new();
    for arg in args {
        if !allowed.contains(&arg.as_str()) {
            fail(
                &format!("Unknown option for {command}: {arg}"),
                Some(&format!("Run \"{APP_NAME} help {command}\" for usage.")),
                io,
            );
            return None;
        }
        options.insert(arg.clone());
    }
    Some(options)
}

/// `requireArgumentCount(args, count, command)`.
fn require_argument_count(args: &[String], count: usize, command: &str, io: &PublicCommandIo<'_>) -> bool {
    if args.len() == count {
        return true;
    }
    let usage = get_command_spec(&[command]).map(|spec| spec.usage).unwrap_or(command);
    fail(&format!("Usage: {APP_NAME} {usage}"), None, io);
    false
}

/// `hasPositionalArguments(args)`.
fn has_positional_arguments(args: &[String]) -> bool {
    let parsed = parse_args(args);
    !parsed.messages.is_empty() || !parsed.file_args.is_empty()
}

/// `hasConflictingAttachOption(args)`.
fn has_conflicting_attach_option(args: &[String]) -> bool {
    args.iter().any(|arg| {
        arg == "--resume"
            || arg == "-r"
            || arg.starts_with("--resume=")
            || arg == "--continue"
            || arg == "-c"
            || arg == "--fork"
    })
}

/// `splitOperandsAndOptions(args)`.
fn split_operands_and_options(args: &[String]) -> Option<(Vec<String>, Vec<String>)> {
    let options_start = args.iter().position(|arg| arg.starts_with('-'));
    let Some(options_start) = options_start else {
        return Some((args.to_vec(), Vec::new()));
    };
    let options = args[options_start..].to_vec();
    if has_positional_arguments(&options) {
        return None;
    }
    Some((args[..options_start].to_vec(), options))
}

/// `requireOperandCount(args, minimum, maximum, command)`.
fn require_operand_count(
    args: &[String],
    minimum: usize,
    maximum: Option<usize>,
    command: &str,
    io: &PublicCommandIo<'_>,
) -> bool {
    let mut operands: Vec<String> = Vec::new();
    let mut index = 0usize;
    while index < args.len() {
        let arg = &args[index];
        if arg == "--json" {
            index += 1;
            continue;
        }
        if arg == "--socket" || arg == "--daemon-socket" {
            index += 2;
            continue;
        }
        if arg.starts_with('-') {
            let usage = get_command_spec(&[command]).map(|spec| spec.usage).unwrap_or(command);
            fail(&format!("Usage: {APP_NAME} {usage}"), None, io);
            return false;
        }
        operands.push(arg.clone());
        index += 1;
    }
    if operands.len() >= minimum && maximum.map(|maximum| operands.len() <= maximum).unwrap_or(true) {
        return true;
    }
    let usage = get_command_spec(&[command]).map(|spec| spec.usage).unwrap_or(command);
    fail(&format!("Usage: {APP_NAME} {usage}"), None, io);
    false
}

/// `validateScheduleArgs(args)`.
fn validate_schedule_args(args: &[String], io: &PublicCommandIo<'_>) -> bool {
    let subcommand = args.first().map(String::as_str);
    if subcommand == Some("list") {
        let mut agent_count = 0;
        for arg in &args[1..] {
            if arg == "--all" || arg == "-a" || arg == "--json" {
                continue;
            }
            if arg.starts_with('-') || {
                agent_count += 1;
                agent_count > 1
            } {
                fail(&format!("Usage: {APP_NAME} schedule list [--all] [agent] [--json]"), None, io);
                return false;
            }
        }
        return true;
    }
    if subcommand == Some("cancel") {
        let operands: Vec<&String> = args[1..].iter().filter(|arg| *arg != "--json").collect();
        if operands.len() == 1 && !operands[0].starts_with('-') {
            return true;
        }
        let usage = get_command_spec(&["schedule", "cancel"]).map(|spec| spec.usage).unwrap_or("schedule cancel");
        fail(&format!("Usage: {APP_NAME} {usage}"), None, io);
        return false;
    }
    true
}

/// `fail(message, hint?)`.
fn fail(message: &str, hint: Option<&str>, io: &PublicCommandIo<'_>) -> PublicCommandResult {
    (io.error)(&red(&format!("Error: {message}")));
    if let Some(hint) = hint {
        (io.error)(&dim(hint));
    }
    (io.set_exit_code)(1);
    HANDLED
}

/// Borrowed view of a `Vec<String>` for the registry helpers.
fn as_strs(values: &[String]) -> Vec<&str> {
    values.iter().map(String::as_str).collect()
}

/// `getChildCommandSpecs(path).map((spec) => spec.path.at(-1)!)`.
fn child_command_names(path: &[String]) -> Vec<String> {
    get_child_command_specs(&as_strs(path))
        .iter()
        .map(|spec| spec.path[spec.path.len() - 1].to_string())
        .collect()
}

/// The `daemon-command.ts` IO surface, derived from this module's seam.
fn daemon_command_io<'a>(io: &PublicCommandIo<'a>) -> DaemonCommandIo<'a> {
    DaemonCommandIo {
        log: io.log,
        error: io.error,
        set_exit_code: io.set_exit_code,
        stdin_is_tty: io.stdin_is_tty,
        prompt_yes_no: io.prompt_yes_no,
        cwd: io.cwd.clone(),
    }
}

/// The dgx-auth command IO surface, derived from this module's seam.
fn dgx_auth_command_io<'a>(io: &PublicCommandIo<'a>) -> DgxAuthCommandIo<'a> {
    DgxAuthCommandIo { log: io.log, cwd: io.cwd.clone() }
}

/// The `daemon-ps.ts` IO surface, derived from this module's seam.
fn daemon_ps_io<'a>(io: &PublicCommandIo<'a>) -> DaemonPsIo<'a> {
    DaemonPsIo {
        log: io.log,
        error: io.error,
        stdin_is_tty: io.stdin_is_tty,
        prompt_yes_no: io.prompt_yes_no,
        set_exit_code: io.set_exit_code,
        cwd: io.cwd.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    /// The `console.log`/`console.error`/`process.exitCode` seam for one run.
    ///
    /// The callbacks are owned fields (not `&|...|` temporaries) so `io()` can
    /// hand out references that outlive the call.
    struct Harness {
        logged: Arc<Mutex<Vec<String>>>,
        errors: Arc<Mutex<Vec<String>>>,
        exit_code: Arc<Mutex<Option<i32>>>,
        log_fn: Box<dyn Fn(&str)>,
        error_fn: Box<dyn Fn(&str)>,
        exit_fn: Box<dyn Fn(i32)>,
        prompt_fn: Box<dyn Fn(&str) -> bool>,
    }

    impl Harness {
        fn new() -> Self {
            let logged = Arc::new(Mutex::new(Vec::new()));
            let errors = Arc::new(Mutex::new(Vec::new()));
            let exit_code = Arc::new(Mutex::new(None));
            let log_fn: Box<dyn Fn(&str)> = {
                let logged = logged.clone();
                Box::new(move |line: &str| logged.lock().unwrap().push(line.to_string()))
            };
            let error_fn: Box<dyn Fn(&str)> = {
                let errors = errors.clone();
                Box::new(move |line: &str| errors.lock().unwrap().push(line.to_string()))
            };
            let exit_fn: Box<dyn Fn(i32)> = {
                let exit_code = exit_code.clone();
                Box::new(move |code: i32| *exit_code.lock().unwrap() = Some(code))
            };
            let prompt_fn: Box<dyn Fn(&str) -> bool> = Box::new(|_message: &str| false);
            Self { logged, errors, exit_code, log_fn, error_fn, exit_fn, prompt_fn }
        }

        fn io(&self) -> PublicCommandIo<'_> {
            PublicCommandIo {
                log: &*self.log_fn,
                error: &*self.error_fn,
                set_exit_code: &*self.exit_fn,
                stdin_is_tty: None,
                prompt_yes_no: &*self.prompt_fn,
                cwd: "/work".to_string(),
            }
        }

        fn logged(&self) -> Vec<String> {
            self.logged.lock().unwrap().clone()
        }

        fn errors(&self) -> Vec<String> {
            self.errors.lock().unwrap().clone()
        }

        fn exit_code(&self) -> Option<i32> {
            *self.exit_code.lock().unwrap()
        }
    }

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| value.to_string()).collect()
    }

    fn run(args: &[&str]) -> (Harness, PublicCommandResult) {
        let harness = Harness::new();
        let result = futures::executor::block_on(handle_public_command(&strings(args), &harness.io()));
        (harness, result)
    }

    #[test]
    fn rewrites_attach_into_the_normal_interactive_resume_path() {
        let (_, result) = run(&["attach", "worker"]);
        assert_eq!(
            result,
            PublicCommandResult {
                handled: false,
                args: strings(&["--resume", "worker"]),
                explicit_agents_view: false,
                attach_agent: Some("worker".to_string()),
            }
        );
    }

    #[test]
    fn forwards_global_options_when_attaching() {
        let (_, result) = run(&["attach", "worker", "--verbose", "--provider", "anthropic"]);
        assert_eq!(result.args, strings(&["--resume", "worker", "--verbose", "--provider", "anthropic"]));
        assert_eq!(result.attach_agent, Some("worker".to_string()));
    }

    #[test]
    fn rejects_extra_attach_operands() {
        let (harness, result) = run(&["attach", "worker", "extra"]);
        assert!(result.handled);
        assert_eq!(harness.exit_code(), Some(1));
        assert!(!harness.errors().is_empty());
        assert!(harness.errors().iter().any(|line| line.contains("optimus-agent attach <agent>")));
    }

    #[test]
    fn rejects_conflicting_session_selectors_when_attaching() {
        let selectors: [&[&str]; 4] = [
            &["--resume", "other"],
            &["-r", "other"],
            &["--continue"],
            &["--fork", "session.jsonl"],
        ];
        for selector in selectors {
            let mut args = vec!["attach", "worker"];
            args.extend(selector.iter().copied());
            let (harness, result) = run(&args);
            assert!(result.handled);
            assert_eq!(harness.exit_code(), Some(1));
        }
    }

    #[test]
    fn forwards_global_options_when_opening_the_agents_view() {
        let (_, result) = run(&["agents", "--verbose", "--provider", "anthropic"]);
        assert_eq!(
            result,
            PublicCommandResult {
                handled: false,
                args: strings(&["--verbose", "--provider", "anthropic"]),
                explicit_agents_view: true,
                attach_agent: None,
            }
        );
    }

    #[test]
    fn leaves_natural_language_prompts_beginning_with_help_on_the_prompt_path() {
        let args = ["help", "me", "fix", "this"];
        let (_, result) = run(&args);
        assert_eq!(result, continue_with(strings(&args)));
    }

    #[test]
    fn leaves_top_level_help_flags_on_the_full_cli_help_path() {
        assert_eq!(run(&["--help"]).1, continue_with(strings(&["--help"])));
        assert_eq!(run(&["-h"]).1, continue_with(strings(&["-h"])));
    }

    #[test]
    fn rejects_invalid_paths_below_a_known_help_command() {
        let (harness, result) = run(&["help", "schedule", "nonsense"]);
        assert!(result.handled);
        assert_eq!(harness.exit_code(), Some(1));
        assert!(harness.errors()[0].contains("Unknown command: schedule nonsense"));
    }

    #[test]
    fn shows_migration_guidance_when_help_targets_removed_commands() {
        let cases: [(&[&str], &str); 6] = [
            (&["daemon"], "Run \"optimus-agent help\""),
            (&["install"], "Use \"optimus-agent package install\""),
            (&["remove"], "Use \"optimus-agent package remove\""),
            (&["uninstall"], "Use \"optimus-agent package remove\""),
            (&["manage"], "Use \"optimus-agent agents\""),
            (&["app", "update"], "Use \"optimus-agent update\""),
        ];
        for (path, hint) in cases {
            let mut args = vec!["help"];
            args.extend(path.iter().copied());
            let (harness, result) = run(&args);
            assert!(result.handled, "help {path:?} should be handled");
            assert_eq!(harness.exit_code(), Some(1));
            assert!(
                harness.errors().iter().any(|line| line.contains(hint)),
                "help {path:?} should mention {hint}"
            );
            assert!(harness.logged().is_empty());
        }
    }

    #[test]
    fn rejects_the_old_daemon_hierarchy_with_migration_guidance() {
        let (harness, result) = run(&["daemon", "list"]);
        assert!(result.handled);
        assert_eq!(harness.exit_code(), Some(1));
        assert!(harness.errors().iter().any(|line| line.contains("Run \"optimus-agent help\"")));
    }

    #[test]
    fn separates_prime_agent_updates_from_package_updates() {
        // The package command is a real dependency here (not a mock), so the
        // port asserts the routing decision rather than the recording.
        let (harness, result) = run(&["update", "--self", "--extensions"]);
        assert!(result.handled);
        assert_eq!(harness.exit_code(), Some(1));
        let printed = harness.errors().join("\n");
        assert!(printed.contains("separately"), "{printed}");

        let (harness, _) = run(&["update", "self"]);
        let printed = harness.errors().join("\n");
        assert!(printed.contains(&format!("Use \"{APP_NAME} update [--force]\".")), "{printed}");
    }

    #[test]
    fn rejects_self_update_aliases_on_the_package_update_path() {
        for source in ["self", "pi", "prime-agent"] {
            let (harness, result) = run(&["package", "update", source]);
            assert!(result.handled);
            assert!(harness.errors().iter().any(|line| line.contains("Use \"optimus-agent update\"")));
        }
    }

    #[test]
    fn directs_package_uninstall_to_package_remove() {
        let (harness, result) = run(&["package", "uninstall", "npm:@example/tools"]);
        assert!(result.handled);
        let printed = harness.errors().join("\n");
        assert!(printed.contains(&format!("Use \"{APP_NAME} package remove\".")), "{printed}");
        assert!(!printed.contains("package install"), "{printed}");
    }

    #[test]
    fn rejects_operands_for_package_list() {
        let (harness, result) = run(&["package", "list", "ignored-source"]);
        assert!(result.handled);
        assert!(harness
            .errors()
            .iter()
            .any(|line| line.contains("optimus-agent package list")));
    }

    #[test]
    fn suggests_close_nested_commands_without_executing_them() {
        let (harness, result) = run(&["schedule", "cancell", "job-1"]);
        assert!(result.handled);
        assert!(harness.errors().iter().any(|line| line.contains("schedule cancel")));
    }

    #[test]
    fn routes_dgx_auth_failures_through_the_public_error_path() {
        let (harness, result) = run(&["dgx-auth", "nonsense"]);
        assert!(result.handled);
        assert_eq!(harness.exit_code(), Some(1));
        assert!(
            harness
                .errors()
                .iter()
                .any(|line| line.contains("Usage: dgx-auth <export|import|status>."))
        );
    }

    #[test]
    fn serves_dgx_auth_help_from_the_registry_specs() {
        let (harness, result) = run(&["help", "dgx-auth"]);
        assert!(result.handled);
        let printed = harness.logged().join("\n");
        assert!(printed.contains("Usage:\n  optimus-agent dgx-auth <export|import|status>"), "{printed}");
        assert!(printed.contains("export"), "{printed}");

        let (harness, result) = run(&["dgx-auth", "export", "--help"]);
        assert!(result.handled);
        let printed = harness.logged().join("\n");
        assert!(
            printed.contains(
                "Usage:\n  optimus-agent dgx-auth export [--out <path>] [--providers <ids>] [--no-refresh]"
            ),
            "{printed}"
        );
        assert!(printed.contains("--no-refresh"), "{printed}");
    }

    #[test]
    fn maps_model_listing_and_session_export_to_the_existing_runtime_flags() {
        let (_, result) = run(&["model", "list", "sonnet"]);
        assert_eq!(
            result.args,
            strings(&[INTERNAL_RUNTIME_COMMAND_MARKER, "--list-models", "sonnet"])
        );
        let (_, result) = run(&["session", "export", "session.jsonl", "session.html"]);
        assert_eq!(
            result.args,
            strings(&[INTERNAL_RUNTIME_COMMAND_MARKER, "--export", "session.jsonl", "session.html"])
        );
    }

    #[test]
    fn preserves_trailing_global_options_for_model_listing_and_session_export() {
        let (_, result) = run(&["model", "list", "sonnet", "--offline"]);
        assert_eq!(
            result.args,
            strings(&[INTERNAL_RUNTIME_COMMAND_MARKER, "--list-models", "sonnet", "--offline"])
        );
        let (_, result) = run(&["session", "export", "session.jsonl", "session.html", "--verbose"]);
        assert_eq!(
            result.args,
            strings(&[INTERNAL_RUNTIME_COMMAND_MARKER, "--export", "session.jsonl", "session.html", "--verbose"])
        );
    }

    #[test]
    fn normalizes_a_leading_daemon_socket_option_for_stop_and_rename() {
        assert_eq!(
            normalize_leading_daemon_socket_option(&strings(&["--daemon-socket", "/tmp/d.sock", "stop", "worker"])),
            strings(&["stop", "worker", "--daemon-socket", "/tmp/d.sock"])
        );
        // Other commands keep the original order.
        assert_eq!(
            normalize_leading_daemon_socket_option(&strings(&["--daemon-socket", "/tmp/d.sock", "list"])),
            strings(&["--daemon-socket", "/tmp/d.sock", "list"])
        );
        // A missing socket path keeps the original order.
        assert_eq!(
            normalize_leading_daemon_socket_option(&strings(&["--daemon-socket"])),
            strings(&["--daemon-socket"])
        );
    }

    #[test]
    fn shutdown_socket_routing_normalizes_leading_flags_without_execution() {
        for flag in ["--socket", "--daemon-socket"] {
            assert_eq!(normalize_leading_daemon_socket_option(&strings(&[flag, "/tmp/d.sock", "shutdown", "--force"])),
                strings(&["shutdown", "--force", flag, "/tmp/d.sock"]));
        }
        let (harness, result) = run(&["shutdown", "--socket", "/never-contacted.sock"]);
        assert!(result.handled);
        assert!(harness.errors().iter().any(|line| line.contains("requires confirmation")));
        let (harness, _) = run(&["shutdown", "--socket"]);
        assert!(harness.errors().iter().any(|line| line.contains("requires a socket path")));
    }

    #[test]
    fn require_operand_count_skips_json_and_socket_options() {
        let harness = Harness::new();
        let io = harness.io();
        assert!(require_operand_count(&strings(&["worker", "--json"]), 1, Some(1), "stop", &io));
        assert!(require_operand_count(
            &strings(&["worker", "--daemon-socket", "/tmp/d.sock"]),
            1,
            Some(1),
            "stop",
            &io
        ));
        assert!(!require_operand_count(&strings(&["worker", "extra"]), 1, Some(1), "stop", &io));
        assert!(!require_operand_count(&strings(&["--force"]), 1, Some(1), "stop", &io));
        assert_eq!(harness.exit_code(), Some(1));
    }

    #[test]
    fn schedule_validation_matches_the_usage_rules() {
        let harness = Harness::new();
        let io = harness.io();
        assert!(validate_schedule_args(&strings(&["list", "--all", "--json"]), &io));
        assert!(validate_schedule_args(&strings(&["list", "worker"]), &io));
        assert!(!validate_schedule_args(&strings(&["list", "one", "two"]), &io));
        assert!(!validate_schedule_args(&strings(&["list", "--nope"]), &io));
        assert!(validate_schedule_args(&strings(&["cancel", "job-1"]), &io));
        assert!(validate_schedule_args(&strings(&["cancel", "job-1", "--json"]), &io));
        assert!(!validate_schedule_args(&strings(&["cancel"]), &io));
        assert!(validate_schedule_args(&strings(&["create", "job"]), &io));
    }

    #[test]
    fn split_operands_and_options_rejects_option_positionals() {
        assert_eq!(
            split_operands_and_options(&strings(&["sonnet", "--offline"])),
            Some((strings(&["sonnet"]), strings(&["--offline"])))
        );
        assert_eq!(
            split_operands_and_options(&strings(&["sonnet"])),
            Some((strings(&["sonnet"]), Vec::new()))
        );
        // `--offline` takes no value, so `text` stays a positional message and
        // `hasPositionalArguments(options)` rejects the split.
        assert_eq!(split_operands_and_options(&strings(&["--offline", "text"])), None);
        assert_eq!(split_operands_and_options(&strings(&["text"])), Some((strings(&["text"]), Vec::new())));
    }

    #[test]
    fn fail_sets_the_exit_code_and_prints_the_dim_hint() {
        let harness = Harness::new();
        let result = fail("something", Some("hint"), &harness.io());
        assert!(result.handled);
        assert_eq!(harness.errors().len(), 2);
        assert!(harness.errors()[0].contains("Error: something"));
        assert_eq!(harness.errors()[1], dim("hint"));
        assert_eq!(harness.exit_code(), Some(1));
    }
}
