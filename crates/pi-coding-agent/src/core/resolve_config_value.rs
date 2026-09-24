//! Port of packages/coding-agent/src/core/resolve-config-value.ts
//!
//! Resolve configuration values that may be shell commands, environment
//! variables, or literals. Used by auth-storage.ts and model-registry.ts.

use std::collections::HashMap;
use std::sync::Mutex;

use crate::utils::child_process::{wait_with_timeout, SpawnOptions};
use crate::utils::shell::get_shell_config;

/// Only successful credentials are cached; temporary helper failures must be retried.
static COMMAND_RESULT_CACHE: std::sync::LazyLock<Mutex<HashMap<String, String>>> =
    std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));

/// The Node default timeout used by both shell executions.
const CONFIG_VALUE_TIMEOUT_MS: u64 = 10_000;

/// `resolveConfigValue(config)`.
///
/// - If it starts with "!", executes the rest as a shell command and uses
///   stdout (cached)
/// - Otherwise checks the environment variable first, then treats it as a
///   literal (not cached)
pub fn resolve_config_value(config: &str) -> Option<String> {
    if config.starts_with('!') {
        return execute_command(config);
    }
    resolve_env_or_literal(config)
}

/// `resolveEnvOrLiteral(config)`.
///
/// Unset env var: fall back to the literal string. Set-but-empty: missing
/// credential, never the var name.
fn resolve_env_or_literal(config: &str) -> Option<String> {
    match std::env::var(config) {
        Ok(value) => {
            if value.is_empty() {
                None
            } else {
                Some(value)
            }
        }
        Err(_) => Some(config.to_string()),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ConfiguredShellResult {
    executed: bool,
    value: Option<String>,
}

/// The `result.error` / `result.status` / `result.stdout` handling of
/// `executeWithConfiguredShell`, split out so the timeout mapping is testable.
fn configured_shell_result(result: std::io::Result<std::process::Output>) -> ConfiguredShellResult {
    match result {
        Err(error) => {
            // `error.code === "ENOENT"` means the shell could not be found, so
            // the fallback shell is tried instead; any other error - including the
            // `spawnSync` timeout, whose `ETIMEDOUT` is not an ENOENT - is "executed".
            if error.kind() == std::io::ErrorKind::NotFound {
                ConfiguredShellResult {
                    executed: false,
                    value: None,
                }
            } else {
                ConfiguredShellResult {
                    executed: true,
                    value: None,
                }
            }
        }
        Ok(result) => {
            if result.status.code() != Some(0) {
                return ConfiguredShellResult {
                    executed: true,
                    value: None,
                };
            }
            let value = String::from_utf8_lossy(&result.stdout).trim().to_string();
            ConfiguredShellResult {
                executed: true,
                value: if value.is_empty() { None } else { Some(value) },
            }
        }
    }
}

/// `executeWithConfiguredShell(command)`.
fn execute_with_configured_shell(command: &str) -> ConfiguredShellResult {
    let config = match get_shell_config(None) {
        Ok(config) => config,
        Err(_) => {
            return ConfiguredShellResult {
                executed: false,
                value: None,
            }
        }
    };
    let mut args: Vec<String> = config.args.clone();
    args.push(command.to_string());
    // `spawnSyncHidden(shell, [...args, command], { ..., timeout: 10000, ... })`.
    // Node kills the child at the deadline and `spawnSync` then reports an error
    // that is not ENOENT, so the result is "executed" with no value.
    configured_shell_result(spawn_sync_hidden_with_timeout(
        &config.shell,
        &args,
        SpawnOptions {
            capture_stdout: true,
            ..Default::default()
        },
        CONFIG_VALUE_TIMEOUT_MS,
    ))
}

/// `executeWithDefaultShell(command)`.
fn execute_with_default_shell(command: &str) -> Option<String> {
    // `execSyncHidden(command, { ..., timeout: 10000, ... })`: a timeout throws,
    // so the credential stays unresolved.
    match exec_sync_hidden_with_timeout(
        command,
        SpawnOptions {
            capture_stdout: true,
            ..Default::default()
        },
        CONFIG_VALUE_TIMEOUT_MS,
    ) {
        Ok(output) if output.status.success() => {
            let value = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if value.is_empty() {
                None
            } else {
                Some(value)
            }
        }
        _ => None,
    }
}

/// `{ timeout, killSignal }` handling of `spawnSyncHidden` / `execSyncHidden`:
/// the child is killed once the deadline passes and the wait reports a timeout.
///
/// `child_process.rs` owns the spawn plumbing, but neither `spawn_sync_hidden`
/// nor `exec_sync_hidden` accepts a deadline yet (both call `Command::output()`
/// and block forever), so the bounded wait reuses the shared
/// `child_process::wait_with_timeout` helper, which kills the child on expiry.
/// Moving this timeout into `spawn_sync_hidden`/`exec_sync_hidden` is the
/// follow-up that removes this local plumbing once `child_process.rs` owns it.
fn run_with_config_value_timeout(
    mut command: std::process::Command,
    label: &str,
    timeout_ms: u64,
) -> std::io::Result<std::process::Output> {
    command.stdin(std::process::Stdio::null());
    let mut child = command.spawn()?;
    // Drain the pipes on their own threads: a command that writes more than the
    // pipe buffer would otherwise stall the child until the deadline killed it.
    // The readers are not joined on the timeout path, because a killed shell can
    // still hold the pipe open through its own children.
    let stdout_reader = child.stdout.take().map(|mut pipe| {
        std::thread::spawn(move || {
            use std::io::Read;
            let mut buffer = Vec::new();
            let _ = pipe.read_to_end(&mut buffer);
            buffer
        })
    });
    let stderr_reader = child.stderr.take().map(|mut pipe| {
        std::thread::spawn(move || {
            use std::io::Read;
            let mut buffer = Vec::new();
            let _ = pipe.read_to_end(&mut buffer);
            buffer
        })
    });
    let status = match wait_with_timeout(&mut child, timeout_ms) {
        Some(status) => status,
        None => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                format!("{label} ETIMEDOUT (timeout {timeout_ms}ms)"),
            ))
        }
    };
    let stdout = stdout_reader.and_then(|reader| reader.join().ok()).unwrap_or_default();
    let stderr = stderr_reader.and_then(|reader| reader.join().ok()).unwrap_or_default();
    Ok(std::process::Output { status, stdout, stderr })
}

/// The `cwd`, `env`, `stdio` and `windowsHide` handling of
/// `child_process::apply_std_options`, applied to a command built here.
fn configure_config_value_command(builder: &mut std::process::Command, options: &SpawnOptions) {
    if let Some(cwd) = &options.cwd {
        builder.current_dir(cwd);
    }
    if let Some(env) = &options.env {
        builder.envs(env.iter().map(|(key, value)| (key.clone(), value.clone())));
    }
    builder.stdout(if options.capture_stdout {
        std::process::Stdio::piped()
    } else {
        std::process::Stdio::null()
    });
    builder.stderr(if options.capture_stderr {
        std::process::Stdio::piped()
    } else {
        std::process::Stdio::null()
    });
    hide_config_value_console_window(builder);
}

#[cfg(windows)]
fn hide_config_value_console_window(builder: &mut std::process::Command) {
    use std::os::windows::process::CommandExt as _;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    builder.creation_flags(CREATE_NO_WINDOW);
}

#[cfg(not(windows))]
fn hide_config_value_console_window(_builder: &mut std::process::Command) {}

/// `spawnSyncHidden(shell, [...args, command], { ..., timeout: 10000, shell: false })`.
fn spawn_sync_hidden_with_timeout(
    command: &str,
    args: &[String],
    options: SpawnOptions,
    timeout_ms: u64,
) -> std::io::Result<std::process::Output> {
    let mut builder = std::process::Command::new(command);
    builder.args(args);
    configure_config_value_command(&mut builder, &options);
    run_with_config_value_timeout(builder, &format!("spawnSync {command}"), timeout_ms)
}

/// `execSyncHidden(command, { ..., timeout: 10000 })`: Node runs the string
/// through the shell, so `cmd /c` and `/bin/sh -c` are the equivalents.
fn exec_sync_hidden_with_timeout(
    command: &str,
    options: SpawnOptions,
    timeout_ms: u64,
) -> std::io::Result<std::process::Output> {
    let mut builder = if cfg!(windows) {
        let mut builder = std::process::Command::new("cmd");
        builder.arg("/c").arg(command);
        builder
    } else {
        let mut builder = std::process::Command::new("/bin/sh");
        builder.arg("-c").arg(command);
        builder
    };
    configure_config_value_command(&mut builder, &options);
    run_with_config_value_timeout(builder, &format!("execSync {command}"), timeout_ms)
}

/// `executeCommandUncached(commandConfig)`.
fn execute_command_uncached(command_config: &str) -> Option<String> {
    let command = &command_config[1..];
    #[cfg(windows)]
    if let Some((executable, args)) = literal_powershell_file_command(command) {
        return configured_shell_result(spawn_sync_hidden_with_timeout(
            &executable,
            &args,
            SpawnOptions {
                capture_stdout: true,
                ..Default::default()
            },
            CONFIG_VALUE_TIMEOUT_MS,
        ))
        .value;
    }
    if process_platform_is_win32() {
        let configured_result = execute_with_configured_shell(command);
        if configured_result.executed {
            configured_result.value
        } else {
            execute_with_default_shell(command)
        }
    } else {
        execute_with_default_shell(command)
    }
}

/// Avoid Git Bash's native-process descendants for literal PowerShell helpers.
/// Shell expressions and script arguments deliberately retain the shell path.
#[cfg(windows)]
fn literal_powershell_file_command(command: &str) -> Option<(String, Vec<String>)> {
    if command.contains(['\r', '\n']) {
        return None;
    }
    static PATTERN: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let pattern = PATTERN.get_or_init(|| {
        regex::Regex::new(
            r#"(?ix)^\s*
            (?:"(?P<quoted_exe>[^"$`\r\n]+)"|'(?P<single_exe>[^'\r\n]+)'|(?P<bare_exe>[^\s"'\\$`;&|<>()\[\]*?!{}]+))
            (?P<options>(?:\s+-(?:NoLogo|NoProfile|NonInteractive|ExecutionPolicy\s+(?:Bypass|RemoteSigned|AllSigned|Restricted|Unrestricted)|WindowStyle\s+Hidden))*)
            \s+-File\s+
            (?:"(?P<quoted_path>[^"$`\r\n]+)"|'(?P<single_path>[^'\r\n]+)'|(?P<bare_path>[^\s"'\\$`;&|<>()\[\]*?!{}]+))\s*$"#,
        )
        .expect("literal PowerShell helper pattern")
    });
    let captures = pattern.captures(command)?;
    let field = |names: &[&str]| {
        names.iter().find_map(|name| captures.name(name).map(|value| value.as_str()))
    };
    let executable = field(&["quoted_exe", "single_exe", "bare_exe"])?;
    let name = executable.rsplit(['/', '\\']).next()?.to_ascii_lowercase();
    if !matches!(name.as_str(), "powershell" | "powershell.exe" | "pwsh" | "pwsh.exe") {
        return None;
    }
    let script = field(&["quoted_path", "single_path", "bare_path"])?;
    if !std::path::Path::new(script).is_absolute() || !script.to_ascii_lowercase().ends_with(".ps1") {
        return None;
    }
    let mut args: Vec<String> = captures["options"].split_whitespace().map(str::to_string).collect();
    if !args.iter().any(|arg| arg.eq_ignore_ascii_case("-WindowStyle")) {
        args.extend(["-WindowStyle".to_string(), "Hidden".to_string()]);
    }
    args.extend(["-File".to_string(), script.to_string()]);
    Some((executable.to_string(), args))
}

/// `process.platform === "win32"`.
fn process_platform_is_win32() -> bool {
    cfg!(windows)
}

/// `executeCommand(commandConfig)`.
fn execute_command(command_config: &str) -> Option<String> {
    {
        let cache = COMMAND_RESULT_CACHE.lock().expect("command cache poisoned");
        if let Some(cached) = cache.get(command_config) {
            return Some(cached.clone());
        }
    }
    let result = execute_command_uncached(command_config);
    if let Some(value) = &result {
        COMMAND_RESULT_CACHE
            .lock()
            .expect("command cache poisoned")
            .insert(command_config.to_string(), value.clone());
    }
    result
}

/// `resolveConfigValueUncached(config)`.
pub fn resolve_config_value_uncached(config: &str) -> Option<String> {
    if config.starts_with('!') {
        return execute_command_uncached(config);
    }
    resolve_env_or_literal(config)
}

/// `resolveConfigValueOrThrow(config, description)`.
pub fn resolve_config_value_or_throw(config: &str, description: &str) -> Result<String, String> {
    if let Some(resolved_value) = resolve_config_value_uncached(config) {
        return Ok(resolved_value);
    }
    if config.starts_with('!') {
        return Err(format!(
            "Failed to resolve {description} from shell command: {}",
            &config[1..]
        ));
    }
    Err(format!("Failed to resolve {description}"))
}

/// `resolveHeaders(headers)`.
pub fn resolve_headers(headers: Option<&indexmap::IndexMap<String, String>>) -> Option<indexmap::IndexMap<String, String>> {
    let headers = headers?;
    let mut resolved: indexmap::IndexMap<String, String> = indexmap::IndexMap::new();
    for (key, value) in headers {
        if let Some(resolved_value) = resolve_config_value(value) {
            resolved.insert(key.clone(), resolved_value);
        }
    }
    if resolved.is_empty() {
        None
    } else {
        Some(resolved)
    }
}

/// `resolveHeadersOrThrow(headers, description)`.
pub fn resolve_headers_or_throw(
    headers: Option<&indexmap::IndexMap<String, String>>,
    description: &str,
) -> Result<Option<indexmap::IndexMap<String, String>>, String> {
    let Some(headers) = headers else {
        return Ok(None);
    };
    let mut resolved: indexmap::IndexMap<String, String> = indexmap::IndexMap::new();
    for (key, value) in headers {
        resolved.insert(
            key.clone(),
            resolve_config_value_or_throw(value, &format!("{description} header \"{key}\""))?,
        );
    }
    if resolved.is_empty() {
        Ok(None)
    } else {
        Ok(Some(resolved))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn credential_helper_retries_failure_and_empty_output_then_caches_success() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("credential.txt");
        let command = if cfg!(windows) {
            let script = directory.path().join("credential.ps1");
            std::fs::write(&script, format!(
                "$ErrorActionPreference = 'Stop'\nGet-Content -Raw -LiteralPath '{}'\n",
                path.display().to_string().replace('\'', "''"),
            )).unwrap();
            format!("!powershell.exe -NoProfile -ExecutionPolicy Bypass -File \"{}\"", script.display())
        } else {
            format!("!cat '{}'", path.display().to_string().replace('\'', "'\\''"))
        };
        assert_eq!(resolve_config_value(&command), None);
        std::fs::write(&path, " \n").unwrap();
        assert_eq!(resolve_config_value(&command), None);
        std::fs::write(&path, "synthetic-test-key\n").unwrap();
        assert_eq!(resolve_config_value(&command).as_deref(), Some("synthetic-test-key"));
        std::fs::remove_file(&path).unwrap();
        assert_eq!(resolve_config_value(&command).as_deref(), Some("synthetic-test-key"));
        COMMAND_RESULT_CACHE.lock().unwrap().remove(&command);
    }

    #[cfg(windows)]
    #[test]
    fn powershell_file_credential_runs_directly_without_a_shell_or_console() {
        let directory = tempfile::Builder::new().prefix("credential helper ").tempdir().unwrap();
        let path = directory.path().join("helper with space's.ps1");
        std::fs::write(&path, r#"$ErrorActionPreference = 'Stop'
Add-Type -TypeDefinition 'using System; using System.Runtime.InteropServices; public static class CredentialWindowProbe { [DllImport("kernel32.dll")] public static extern IntPtr GetConsoleWindow(); }'
$parentId = (Get-CimInstance Win32_Process -Filter "ProcessId = $PID").ParentProcessId
Write-Output ($parentId.ToString() + ':' + [CredentialWindowProbe]::GetConsoleWindow().ToInt64().ToString() + ':isolated-test-key')
"#).unwrap();
        let config = format!(
            "!powershell.exe -NoProfile -ExecutionPolicy Bypass -File \"{}\"",
            path.to_string_lossy().replace('\\', "/")
        );
        assert_eq!(
            resolve_config_value_uncached(&config),
            Some(format!("{}:0:isolated-test-key", std::process::id())),
            "literal PowerShell helpers must be direct children with no console, not Bash descendants"
        );
    }

    #[cfg(windows)]
    #[test]
    fn powershell_file_command_preserves_literal_paths_and_flags() {
        let command = r#""C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "C:\helper with space's\credential.ps1""#;
        let (executable, args) = literal_powershell_file_command(command).unwrap();
        assert_eq!(executable, r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe");
        assert_eq!(args, ["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-WindowStyle", "Hidden", "-File", r"C:\helper with space's\credential.ps1"]);
        let (_, args) = literal_powershell_file_command(
            "pwsh -WindowStyle Hidden -File 'C:/helper path/credential.ps1'"
        ).unwrap();
        assert_eq!(args, ["-WindowStyle", "Hidden", "-File", "C:/helper path/credential.ps1"]);
    }

    #[cfg(windows)]
    #[test]
    fn powershell_file_command_does_not_reinterpret_shell_expressions() {
        for command in [
            "powershell -File 'C:/helper.ps1' | head -1",
            "powershell -File 'C:/helper.ps1' && echo second",
            "powershell -File \"$HOME/helper.ps1\"",
            "powershell -File \"C:/$(echo name).ps1\"",
            "powershell -File relative.ps1",
            "powershell -File 'C:/helper.ps1' argument",
            "powershell -Command 'Write-Output value'",
            "echo powershell -File 'C:/helper.ps1'",
            "powershell -WindowStyle Normal -File 'C:/helper.ps1'",
            "powershell\n-File 'C:/helper.ps1'",
        ] {
            assert!(literal_powershell_file_command(command).is_none(), "{command}");
        }
    }

    const VAR: &str = "PRIME_AGENT_TEST_CREDENTIAL_VAR";

    /// A command that outlives `SLEEPER_TIMEOUT_MS` but is far shorter than
    /// `SLEEPER_LONG_MS`, so a missing deadline is visible as a slow test.
    const SLEEPER_TIMEOUT_MS: u64 = 200;
    const SLEEPER_LONG_MS: u64 = 5_000;
    const SLEEPER_WINDOWS: &str = "ping -n 6 127.0.0.1 >nul";
    const SLEEPER_UNIX: &str = "sleep 4";

    #[test]
    fn env_var_value_is_used_when_set() {
        std::env::set_var(VAR, "secret-value");
        assert_eq!(resolve_config_value(VAR).as_deref(), Some("secret-value"));
        assert_eq!(resolve_config_value_uncached(VAR).as_deref(), Some("secret-value"));
        std::env::remove_var(VAR);
    }

    #[test]
    fn unset_env_var_falls_back_to_the_literal() {
        std::env::remove_var(VAR);
        assert_eq!(resolve_config_value(VAR).as_deref(), Some(VAR));
        assert_eq!(resolve_config_value("sk-literal-key").as_deref(), Some("sk-literal-key"));
    }

    #[test]
    fn set_but_empty_env_var_is_a_missing_credential() {
        std::env::set_var(VAR, "");
        assert_eq!(resolve_config_value(VAR), None);
        assert_eq!(resolve_config_value_uncached(VAR), None);
        assert_eq!(
            resolve_config_value_or_throw(VAR, "test credential").unwrap_err(),
            "Failed to resolve test credential"
        );
        std::env::remove_var(VAR);
    }

    #[test]
    fn shell_command_failures_report_the_command_text() {
        let error = resolve_config_value_or_throw("!definitely-not-a-real-binary-xyz", "test credential")
            .unwrap_err();
        assert!(error.starts_with("Failed to resolve test credential"));
    }

    #[test]
    fn default_shell_nonzero_exit_does_not_accept_stdout_as_a_key() {
        let command = if cfg!(windows) {
            "echo not-a-valid-key & exit /b 9"
        } else {
            "printf not-a-valid-key; exit 9"
        };
        assert_eq!(execute_with_default_shell(command), None);
    }

    #[test]
    fn resolve_headers_drops_unresolvable_values_and_missing_input() {
        std::env::remove_var(VAR);
        assert!(resolve_headers(None).is_none());
        let mut headers: indexmap::IndexMap<String, String> = indexmap::IndexMap::new();
        headers.insert("x-literal".to_string(), "value".to_string());
        let resolved = resolve_headers(Some(&headers)).unwrap();
        assert_eq!(resolved.get("x-literal").map(String::as_str), Some("value"));

        let mut empty_var_headers: indexmap::IndexMap<String, String> = indexmap::IndexMap::new();
        empty_var_headers.insert("x-empty".to_string(), VAR.to_string());
        std::env::set_var(VAR, "");
        assert!(resolve_headers(Some(&empty_var_headers)).is_none());
        std::env::remove_var(VAR);
    }

    #[test]
    fn resolve_headers_or_throw_names_the_header() {
        std::env::set_var(VAR, "secret");
        let mut headers: indexmap::IndexMap<String, String> = indexmap::IndexMap::new();
        headers.insert("authorization".to_string(), VAR.to_string());
        let resolved = resolve_headers_or_throw(Some(&headers), "provider").unwrap().unwrap();
        assert_eq!(resolved.get("authorization").map(String::as_str), Some("secret"));
        std::env::remove_var(VAR);

        std::env::set_var(VAR, "");
        let error = resolve_headers_or_throw(Some(&headers), "provider").unwrap_err();
        assert_eq!(error, "Failed to resolve provider header \"authorization\"");
        std::env::remove_var(VAR);
    }

    #[test]
    fn resolve_headers_or_throw_returns_none_without_headers() {
        assert!(resolve_headers_or_throw(None, "provider").unwrap().is_none());
    }

    /// `spawnSyncHidden(shell, args, { timeout })` kills the child at the deadline.
    #[test]
    fn configured_shell_command_is_killed_at_the_deadline() {
        let (shell, mut args) = if cfg!(windows) {
            ("cmd", vec!["/c".to_string()])
        } else {
            ("/bin/sh", vec!["-c".to_string()])
        };
        let sleeper = if cfg!(windows) { SLEEPER_WINDOWS } else { SLEEPER_UNIX };
        args.push(sleeper.to_string());
        let started = std::time::Instant::now();
        let result = spawn_sync_hidden_with_timeout(
            shell,
            &args,
            SpawnOptions {
                capture_stdout: true,
                ..Default::default()
            },
            SLEEPER_TIMEOUT_MS,
        );
        let error = result.expect_err("a hung command must time out instead of blocking");
        assert_eq!(error.kind(), std::io::ErrorKind::TimedOut, "{error}");
        assert!(
            started.elapsed() < std::time::Duration::from_millis(SLEEPER_LONG_MS),
            "the wait must return at the deadline, took {:?}",
            started.elapsed()
        );
    }

    /// A timeout is not an ENOENT: the configured shell counts as "executed".
    #[test]
    fn configured_shell_timeout_is_executed_not_enoent() {
        let timed_out = std::io::Error::new(std::io::ErrorKind::TimedOut, "ETIMEDOUT");
        assert_eq!(
            configured_shell_result(Err(timed_out)),
            ConfiguredShellResult {
                executed: true,
                value: None,
            }
        );
        let missing = std::io::Error::new(std::io::ErrorKind::NotFound, "ENOENT");
        assert_eq!(
            configured_shell_result(Err(missing)),
            ConfiguredShellResult {
                executed: false,
                value: None,
            }
        );
    }
}
