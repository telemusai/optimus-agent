//! Port of packages/coding-agent/src/utils/shell.ts

use std::collections::HashSet;
use std::path::Path;
use std::sync::Mutex;

use super::child_process::{spawn_sync_hidden, SpawnOptions};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShellConfig {
    pub shell: String,
    pub args: Vec<String>,
}

/// System32\\bash.exe is the WSL launcher (runs Linux-side), so %SystemRoot% matches are only a last resort.
pub fn order_windows_bash_candidates(matches: &[String], system_root: Option<&str>) -> Vec<String> {
    let Some(system_root) = system_root else {
        return matches.to_vec();
    };
    let prefix = format!("{}\\", system_root.trim_end_matches('\\')).to_lowercase();
    let under_system_root = |candidate: &String| candidate.to_lowercase().starts_with(&prefix);
    let mut ordered: Vec<String> = matches
        .iter()
        .filter(|candidate| !under_system_root(candidate))
        .cloned()
        .collect();
    ordered.extend(matches.iter().filter(|candidate| under_system_root(candidate)).cloned());
    ordered
}

/// Find bash executable on PATH (cross-platform)
fn find_bash_on_path() -> Option<String> {
    if super::pi_user_agent::process_platform() == "win32" {
        // Windows: Use 'where' and verify file exists (where can return non-existent paths)
        // TS shell.ts:27: spawnSyncHidden(..., { timeout: 5000 }).
        let result = crate::utils::child_process::spawn_sync_hidden_with_timeout(
            "where",
            &["bash.exe".to_string()],
            SpawnOptions {
                capture_stdout: true,
                capture_stderr: true,
                ..Default::default()
            },
            5000,
        );
        if let Ok(output) = result {
            if output.status.success() && !output.stdout.is_empty() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let matches: Vec<String> = stdout
                    .trim()
                    .split('\n')
                    .map(|line| line.trim_end_matches('\r').to_string())
                    .filter(|line| !line.is_empty())
                    .collect();
                let system_root = std::env::var("SystemRoot").ok();
                for candidate in order_windows_bash_candidates(&matches, system_root.as_deref()) {
                    if Path::new(&candidate).exists() {
                        return Some(candidate);
                    }
                }
            }
        }
        return None;
    }

    // Unix: Use 'which' and trust its output (handles Termux and special filesystems)
    // TS shell.ts:44: spawnSyncHidden(..., { timeout: 5000 }).
    let result = crate::utils::child_process::spawn_sync_hidden_with_timeout(
        "which",
        &["bash".to_string()],
        SpawnOptions {
            capture_stdout: true,
            capture_stderr: true,
            ..Default::default()
        },
        5000,
    );
    if let Ok(output) = result {
        if output.status.success() && !output.stdout.is_empty() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            if let Some(first_match) = stdout.trim().split('\n').next() {
                if !first_match.is_empty() {
                    return Some(first_match.trim_end_matches('\r').to_string());
                }
            }
        }
    }
    None
}

/// Resolve shell configuration based on platform and an optional explicit shell path.
/// Resolution order:
/// 1. User-specified shellPath
/// 2. On Windows: Git Bash in known locations, then bash on PATH
/// 3. On Unix: /bin/bash, then bash on PATH, then fallback to sh
pub fn get_shell_config(custom_shell_path: Option<&str>) -> Result<ShellConfig, String> {
    // 1. Check user-specified shell path
    if let Some(custom_shell_path) = custom_shell_path {
        if !custom_shell_path.is_empty() {
            if Path::new(custom_shell_path).exists() {
                return Ok(ShellConfig {
                    shell: custom_shell_path.to_string(),
                    args: vec!["-c".to_string()],
                });
            }
            return Err(format!("Custom shell path not found: {}", custom_shell_path));
        }
    }

    if super::pi_user_agent::process_platform() == "win32" {
        // 2. Try Git Bash in known locations
        let mut paths: Vec<String> = Vec::new();
        if let Ok(program_files) = std::env::var("ProgramFiles") {
            if !program_files.is_empty() {
                paths.push(format!("{}\\Git\\bin\\bash.exe", program_files));
            }
        }
        if let Ok(program_files_x86) = std::env::var("ProgramFiles(x86)") {
            if !program_files_x86.is_empty() {
                paths.push(format!("{}\\Git\\bin\\bash.exe", program_files_x86));
            }
        }

        for path in &paths {
            if Path::new(path).exists() {
                return Ok(ShellConfig {
                    shell: path.clone(),
                    args: vec!["-c".to_string()],
                });
            }
        }

        // 3. Fallback: search bash.exe on PATH (Cygwin, MSYS2, WSL, etc.)
        if let Some(bash_on_path) = find_bash_on_path() {
            return Ok(ShellConfig {
                shell: bash_on_path,
                args: vec!["-c".to_string()],
            });
        }

        let searched = paths
            .iter()
            .map(|path| format!("  {}", path))
            .collect::<Vec<String>>()
            .join("\n");
        return Err(format!(
            "No bash shell found. Options:\n  1. Install Git for Windows: https://git-scm.com/download/win\n  2. Add your bash to PATH (Cygwin, MSYS2, etc.)\n  3. Set shellPath in settings.json\n\nSearched Git Bash in:\n{}",
            searched
        ));
    }

    // Unix: try /bin/bash, then bash on PATH, then fallback to sh
    if Path::new("/bin/bash").exists() {
        return Ok(ShellConfig {
            shell: "/bin/bash".to_string(),
            args: vec!["-c".to_string()],
        });
    }

    if let Some(bash_on_path) = find_bash_on_path() {
        return Ok(ShellConfig {
            shell: bash_on_path,
            args: vec!["-c".to_string()],
        });
    }

    Ok(ShellConfig {
        shell: "sh".to_string(),
        args: vec!["-c".to_string()],
    })
}

// Hardcoded literals: ProgramFiles env vars are ambient attacker-influenceable
// input, the same trust-laundering class as PATH.
const WINDOWS_GIT_BASH_PATHS: [&str; 2] = [
    "C:\\Program Files\\Git\\bin\\bash.exe",
    "C:\\Program Files (x86)\\Git\\bin\\bash.exe",
];

/// Absolute default shell for the kernel's bash(): explicit shellPath wins; POSIX
/// uses /bin/bash else /bin/sh (absolute, never PATH); win32 uses only the
/// canonical Git Bash install paths, never PATH.
/// None = no shell found: kernel startup must not fail, bash() raises its
/// teaching error.
pub fn resolve_kernel_bash_shell(custom_shell_path: Option<&str>) -> Option<String> {
    if let Some(explicit) = custom_shell_path.map(|value| value.trim()).filter(|value| !value.is_empty()) {
        return Some(explicit.to_string());
    }
    if super::pi_user_agent::process_platform() != "win32" {
        return Some(if Path::new("/bin/bash").exists() {
            "/bin/bash".to_string()
        } else {
            "/bin/sh".to_string()
        });
    }
    for path in WINDOWS_GIT_BASH_PATHS {
        if Path::new(path).exists() {
            return Some(path.to_string());
        }
    }
    None
}

pub fn get_shell_env() -> Vec<(String, String)> {
    let bin_dir = super::tools_manager::get_bin_dir().to_string_lossy().to_string();
    shell_env_with_bin_dir(std::env::vars().collect(), &bin_dir, cfg!(windows))
}

fn shell_env_with_bin_dir(mut env: Vec<(String, String)>, bin_dir: &str, windows: bool) -> Vec<(String, String)> {
    let is_path = |key: &str| if windows { key.eq_ignore_ascii_case("PATH") } else { key == "PATH" };
    let path_index = env.iter().position(|(key, _)| is_path(key));
    let current_path = path_index.map(|index| env[index].1.as_str()).unwrap_or_default();
    let separator = if windows { ';' } else { ':' };
    // The managed bin dir must lead PATH so managed tools win command
    // resolution, and it must appear exactly once so repeated calls stay
    // idempotent. An inherited PATH that already lists it mid-list (for
    // example a launcher that exports its own PATH prefixes ahead of it)
    // would otherwise leave it shadowed, so existing copies are dropped
    // instead of skipping the prepend. Empty entries are dropped too: they
    // resolve to the current directory and only enable binary planting.
    let updated_path = std::iter::once(bin_dir)
        .chain(
            current_path
                .split(separator)
                .filter(|entry| !entry.is_empty() && *entry != bin_dir),
        )
        .collect::<Vec<_>>()
        .join(&separator.to_string());
    if let Some(index) = path_index {
        env[index].1 = updated_path;
    } else {
        env.push(("PATH".to_string(), updated_path));
    }
    env
}

/// Sanitize binary output for display/storage.
/// Removes characters that crash string-width or cause display issues:
/// - Control characters (except tab, newline, carriage return)
/// - Lone surrogates
/// - Unicode Format characters (crash string-width due to a bug)
/// - Characters with undefined code points
pub fn sanitize_binary_output(value: &str) -> String {
    value
        .chars()
        .filter(|character| {
            let code = *character as u32;

            // Allow tab, newline, carriage return
            if code == 0x09 || code == 0x0a || code == 0x0d {
                return true;
            }

            // Filter out control characters (0x00-0x1F, except 0x09, 0x0a, 0x0d)
            if code <= 0x1f {
                return false;
            }

            // Filter out Unicode format characters
            if (0xfff9..=0xfffb).contains(&code) {
                return false;
            }

            true
        })
        .collect()
}

/// Detached child processes must be tracked so they can be killed on parent
/// shutdown signals (SIGHUP/SIGTERM).
static TRACKED_DETACHED_CHILD_PIDS: Mutex<Option<HashSet<i32>>> = Mutex::new(None);

fn tracked_pids() -> std::sync::MutexGuard<'static, Option<HashSet<i32>>> {
    let mut guard = TRACKED_DETACHED_CHILD_PIDS.lock().expect("tracked pids");
    if guard.is_none() {
        *guard = Some(HashSet::new());
    }
    guard
}

pub fn track_detached_child_pid(pid: i32) {
    if let Some(set) = tracked_pids().as_mut() {
        set.insert(pid);
    }
    record_orphan_process_state(pid, true);
}

pub fn untrack_detached_child_pid(pid: i32) {
    if let Some(set) = tracked_pids().as_mut() {
        set.remove(&pid);
    }
    record_orphan_process_state(pid, false);
}

pub fn kill_tracked_detached_children() {
    kill_tracked_detached_children_with(kill_process_tree);
}

fn kill_tracked_detached_children_with(mut kill: impl FnMut(i32) -> bool) {
    let mut failed = Vec::new();
    // One initial pending snapshot and one settling pass. Continuous
    // registration cannot keep process shutdown in an unbounded loop; entries
    // arriving after the second snapshot remain pending for external cleanup.
    for _ in 0..2 {
        let pids: Vec<i32> = {
            let mut guard = tracked_pids();
            guard
                .as_mut()
                .map(|set| set.drain().collect())
                .unwrap_or_default()
        };
        for pid in pids {
            if kill(pid) {
                record_orphan_process_state(pid, false);
            } else {
                failed.push(pid);
            }
        }
    }
    // Keep active evidence for failed kills so a later recovered teardown
    // retries them without repeating an immutable failure forever here.
    if !failed.is_empty() {
        let mut guard = tracked_pids();
        if let Some(set) = guard.as_mut() {
            set.extend(failed.iter().copied());
        }
    }
    for pid in failed {
        record_orphan_process_state(pid, true);
    }
}

/// `recordOrphanProcessState` from core/orphan-process-journal.ts.
///
/// Delegates to the journal module (same crate) so detached bash() children
/// and autonomous spawns surface to the recovered supervisor's reaper, and a
/// failed kill keeps its truthful active evidence.
fn record_orphan_process_state(pid: i32, tracked: bool) {
    crate::core::orphan_process_journal::record_orphan_process_state(pid as i64, tracked);
}

/// Absolute System32 taskkill for Windows tree kills. A bare "taskkill" name
/// would resolve through PATH and could pick up a planted CWD taskkill.exe.
pub fn windows_taskkill_program() -> String {
    windows_taskkill_program_for(std::env::var("SystemRoot").ok().as_deref())
}

fn windows_taskkill_program_for(system_root: Option<&str>) -> String {
    let system_root = system_root.unwrap_or("C:\\Windows");
    format!("{}\\System32\\taskkill.exe", system_root.trim_end_matches('\\'))
}

const PROCESS_EXIT_VERIFY_MS: u64 = 1_000;
const WINDOWS_TASKKILL_TIMEOUT_MS: u64 = 3_000;

fn wait_for_process_exit_with(mut alive: impl FnMut() -> bool, timeout: std::time::Duration) -> bool {
    let deadline = std::time::Instant::now() + timeout;
    loop {
        if !alive() {
            return true;
        }
        if std::time::Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(std::time::Duration::from_millis(10).min(
            deadline.saturating_duration_since(std::time::Instant::now()),
        ));
    }
}

#[cfg(windows)]
const STILL_ACTIVE: u32 = 259;

#[cfg(windows)]
fn windows_exit_code_is_running(exit_code: u32) -> bool {
    exit_code == STILL_ACTIVE
}

/// Windows OpenProcess success alone is not liveness: another handle can keep
/// an exited process object open. Query the real status and fail conservative.
#[cfg(windows)]
fn process_target_running_status(pid: i32) -> Option<bool> {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Threading::{
        GetExitCodeProcess, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
    };
    unsafe {
        let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid as u32);
        if handle.is_null() {
            return match std::io::Error::last_os_error().raw_os_error() {
                // Windows reports a nonexistent PID as ERROR_INVALID_PARAMETER.
                Some(87) => Some(false),
                _ => None,
            };
        }
        let mut exit_code = 0;
        let queried = GetExitCodeProcess(handle, &mut exit_code);
        CloseHandle(handle);
        (queried != 0).then(|| windows_exit_code_is_running(exit_code))
    }
}

#[cfg(not(windows))]
fn process_target_running_status(pid: i32) -> Option<bool> {
    Some(super::child_process::is_process_alive(pid))
}

fn process_target_running(pid: i32) -> bool {
    process_target_running_status(pid).unwrap_or(true)
}

fn process_exited(pid: i32) -> bool {
    process_exited_with(
        || process_target_running_status(pid),
        std::time::Duration::from_millis(PROCESS_EXIT_VERIFY_MS),
    )
}

fn process_exited_with(
    mut running: impl FnMut() -> Option<bool>,
    timeout: std::time::Duration,
) -> bool {
    wait_for_process_exit_with(|| running().unwrap_or(true), timeout)
}

fn taskkill_delivered(result: std::io::Result<std::process::Output>) -> bool {
    result.is_ok_and(|output| output.status.success())
}

fn checked_signal_kill(pid: i32, signalled: bool) -> bool {
    signalled && process_exited(pid)
}

/// Kill a process and all its children. `true` means the target was already
/// gone or was observed exited inside the verification bound. False keeps
/// detached tracking and journal evidence active for a later retry.
pub fn kill_process_tree(pid: i32) -> bool {
    if !process_target_running(pid) {
        return true;
    }
    if super::pi_user_agent::process_platform() == "win32" {
        // Capture and drain taskkill under the same bounded sync helper as
        // other children; a failed spawn/status or a live target fails closed.
        let result = super::child_process::spawn_sync_hidden_with_timeout(
            &windows_taskkill_program(),
            &[
                "/F".to_string(),
                "/T".to_string(),
                "/PID".to_string(),
                pid.to_string(),
            ],
            SpawnOptions {
                env: Some(vec![("NoDefaultCurrentDirectoryInExePath".to_string(), "1".to_string())]),
                capture_stdout: true,
                capture_stderr: true,
                ..Default::default()
            },
            WINDOWS_TASKKILL_TIMEOUT_MS,
        );
        taskkill_delivered(result) && process_exited(pid)
    } else {
        // Use SIGKILL on Unix/Linux/Mac: the helper already falls back to a
        // single-pid kill when the process group is unavailable.
        let signalled = super::child_process::signal_process_group_or_process(
            pid,
            super::child_process::Signal::Kill,
        );
        checked_signal_kill(pid, signalled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    static DETACHED_TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn sanitize_binary_output_keeps_printable_characters() {
        assert_eq!(sanitize_binary_output("hello\tworld\n"), "hello\tworld\n");
        assert_eq!(sanitize_binary_output("a\u{0}b"), "ab");
        assert_eq!(sanitize_binary_output("a\u{1f}b"), "ab");
        assert_eq!(sanitize_binary_output("a\u{fff9}b"), "ab");
        assert_eq!(sanitize_binary_output("a\u{fffb}b"), "ab");
        assert_eq!(sanitize_binary_output("emoji \u{1f600}"), "emoji \u{1f600}");
    }

    #[test]
    fn windows_bash_candidates_put_system_root_last() {
        let matches = vec![
            "C:\\Windows\\System32\\bash.exe".to_string(),
            "C:\\Program Files\\Git\\bin\\bash.exe".to_string(),
        ];
        let ordered = order_windows_bash_candidates(&matches, Some("C:\\Windows"));
        assert_eq!(ordered[0], "C:\\Program Files\\Git\\bin\\bash.exe");
        assert_eq!(ordered[1], "C:\\Windows\\System32\\bash.exe");
        assert_eq!(order_windows_bash_candidates(&matches, None), matches);
    }

    #[test]
    fn explicit_shell_path_wins_and_missing_paths_fail() {
        let existing = std::env::current_exe().unwrap().to_string_lossy().to_string();
        let config = get_shell_config(Some(&existing)).unwrap();
        assert_eq!(config.shell, existing);
        assert_eq!(config.args, vec!["-c".to_string()]);

        let error = get_shell_config(Some("no-such-shell-xyz")).unwrap_err();
        assert_eq!(error, "Custom shell path not found: no-such-shell-xyz");
    }

    #[test]
    fn windows_taskkill_program_is_the_absolute_system32_path() {
        assert_eq!(
            windows_taskkill_program_for(None),
            "C:\\Windows\\System32\\taskkill.exe"
        );
        assert_eq!(
            windows_taskkill_program_for(Some("D:\\WinTest")),
            "D:\\WinTest\\System32\\taskkill.exe"
        );
        assert_eq!(
            windows_taskkill_program_for(Some("D:\\WinTest\\")),
            "D:\\WinTest\\System32\\taskkill.exe"
        );
    }

    #[test]
    fn kernel_bash_shell_prefers_the_explicit_path() {
        assert_eq!(
            resolve_kernel_bash_shell(Some("  /custom/bash  ")),
            Some("/custom/bash".to_string())
        );
        if cfg!(windows) {
            let resolved = resolve_kernel_bash_shell(None);
            if let Some(resolved) = resolved {
                assert!(WINDOWS_GIT_BASH_PATHS.contains(&resolved.as_str()));
            }
        } else {
            assert!(resolve_kernel_bash_shell(None).is_some());
        }
    }

    #[test]
    fn managed_path_uses_platform_keys_and_preserves_existing_entries() {
        let env = vec![("Path".into(), "unrelated".into()), ("PATH".into(), ":/usr/bin:".into())];
        let posix = shell_env_with_bin_dir(env, "/managed", false);
        assert_eq!(posix[0].1, "unrelated");
        assert_eq!(posix[1].1, "/managed:/usr/bin");
        assert_eq!(shell_env_with_bin_dir(posix.clone(), "/managed", false), posix);
        let windows = shell_env_with_bin_dir(vec![("Path".into(), r"C:\System;D:\Tools".into())], r"C:\Managed", true);
        assert_eq!(windows, vec![("Path".into(), r"C:\Managed;C:\System;D:\Tools".into())]);
        assert_eq!(shell_env_with_bin_dir(windows.clone(), r"C:\Managed", true), windows);
        for env in [vec![], vec![("PATH".into(), String::new())]] {
            assert_eq!(shell_env_with_bin_dir(env, "/managed", false), vec![("PATH".into(), "/managed".into())]);
        }
    }

    #[test]
    fn shell_env_prepends_the_bin_dir_once() {
        let bin_dir = super::super::tools_manager::get_bin_dir().to_string_lossy().to_string();
        let env = get_shell_env();
        let path_entry = env
            .iter()
            .find(|(key, _)| key.to_lowercase() == "path")
            .map(|(_, value)| value.clone())
            .unwrap();
        // The managed bin dir must lead PATH exactly once, even when the
        // inherited PATH already lists it mid-list behind launcher-exported
        // prefixes.
        let separator = if cfg!(windows) { ';' } else { ':' };
        let entries: Vec<&str> = path_entry.split(separator).collect();
        assert_eq!(
            entries.first(),
            Some(&bin_dir.as_str()),
            "PATH must start with the managed bin dir: {path_entry}"
        );
        assert_eq!(
            entries.iter().filter(|entry| **entry == bin_dir).count(),
            1,
            "managed bin dir must appear exactly once: {path_entry}"
        );

        let second = get_shell_env();
        let second_path = second
            .iter()
            .find(|(key, _)| key.to_lowercase() == "path")
            .map(|(_, value)| value.clone())
            .unwrap();
        assert_eq!(path_entry, second_path);
    }

    // Regression (product): an inherited PATH that already contains the
    // managed bin dir mid-list must get it moved to the front exactly once,
    // not left in place. Host launchers export their own PATH prefixes ahead
    // of the agent bin dir, which used to leave the managed tools shadowed
    // (observed on Windows where the test harness prepends toolchain dirs).
    #[test]
    fn shell_env_moves_an_inherited_mid_list_bin_dir_to_the_front() {
        let posix = shell_env_with_bin_dir(
            vec![("PATH".into(), "/usr/bin:/managed:/extra".into())],
            "/managed",
            false,
        );
        assert_eq!(posix, vec![("PATH".into(), "/managed:/usr/bin:/extra".into())]);

        let windows = shell_env_with_bin_dir(
            vec![("Path".into(), r"C:\Toolchain;C:\Managed;C:\Tools".into())],
            r"C:\Managed",
            true,
        );
        assert_eq!(
            windows,
            vec![("Path".into(), r"C:\Managed;C:\Toolchain;C:\Tools".into())]
        );
    }

    #[test]
    fn detached_child_tracking_is_idempotent() {
        let _serial = DETACHED_TEST_LOCK.lock().unwrap();
        untrack_detached_child_pid(999_991);
        track_detached_child_pid(999_991);
        track_detached_child_pid(999_991);
        untrack_detached_child_pid(999_991);
        assert!(tracked_pids().as_ref().map(|set| set.is_empty()).unwrap_or(true));
    }

    #[test]
    fn failed_kill_and_spawn_results_are_not_delivered() {
        assert!(!checked_signal_kill(999_990, false));
        let spawn_failure: std::io::Result<std::process::Output> = Err(
            std::io::Error::new(std::io::ErrorKind::PermissionDenied, "injected spawn failure"),
        );
        assert!(!taskkill_delivered(spawn_failure));
    }

    #[test]
    fn process_exit_verification_is_bounded_and_observes_late_exit() {
        let mut probes = 0;
        assert!(wait_for_process_exit_with(
            || {
                probes += 1;
                probes < 3
            },
            std::time::Duration::from_secs(1),
        ));
        assert_eq!(probes, 3);
        let started = std::time::Instant::now();
        assert!(!wait_for_process_exit_with(
            || true,
            std::time::Duration::from_millis(15),
        ));
        assert!(started.elapsed() < std::time::Duration::from_secs(1));
    }

    #[test]
    fn failed_detached_kill_stays_registered_for_retry() {
        let _serial = DETACHED_TEST_LOCK.lock().unwrap();
        let first = 999_970;
        let retry = 999_971;
        untrack_detached_child_pid(first);
        untrack_detached_child_pid(retry);
        track_detached_child_pid(first);
        track_detached_child_pid(retry);
        kill_tracked_detached_children_with(|pid| pid != retry);
        let guard = tracked_pids();
        let set = guard.as_ref().unwrap();
        assert!(!set.contains(&first));
        assert!(set.contains(&retry));
        drop(guard);
        untrack_detached_child_pid(retry);
    }

    #[test]
    fn registration_during_shutdown_is_processed_in_the_next_round() {
        let _serial = DETACHED_TEST_LOCK.lock().unwrap();
        let first = 999_960;
        let second = 999_961;
        untrack_detached_child_pid(first);
        untrack_detached_child_pid(second);
        track_detached_child_pid(first);
        let (entered_sender, entered_receiver) = std::sync::mpsc::channel();
        let (continue_sender, continue_receiver) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            let mut killed = Vec::new();
            kill_tracked_detached_children_with(|pid| {
                killed.push(pid);
                if pid == first {
                    entered_sender.send(()).unwrap();
                    continue_receiver.recv().unwrap();
                }
                true
            });
            killed
        });
        entered_receiver.recv().unwrap();
        track_detached_child_pid(second);
        continue_sender.send(()).unwrap();
        let killed = worker.join().unwrap();
        assert!(killed.contains(&first));
        assert!(killed.contains(&second));
        let guard = tracked_pids();
        let set = guard.as_ref().unwrap();
        assert!(!set.contains(&first));
        assert!(!set.contains(&second));
    }


    #[test]
    fn continuous_registration_is_left_pending_after_two_bounded_passes() {
        let _serial = DETACHED_TEST_LOCK.lock().unwrap();
        let first = 999_950;
        untrack_detached_child_pid(first);
        track_detached_child_pid(first);
        let mut killed = Vec::new();
        kill_tracked_detached_children_with(|pid| {
            killed.push(pid);
            // Register exactly one successor per pass, forever in principle.
            // The second registration must survive the bounded shutdown call.
            track_detached_child_pid(pid + 1);
            true
        });
        assert_eq!(killed, vec![first, first + 1]);
        let guard = tracked_pids();
        let set = guard.as_ref().unwrap();
        assert!(!set.contains(&first));
        assert!(!set.contains(&(first + 1)));
        assert!(set.contains(&(first + 2)));
        drop(guard);
        untrack_detached_child_pid(first + 2);
    }


    #[test]
    fn process_exit_probe_honors_status_and_conserves_query_errors() {
        assert!(process_exited_with(
            || Some(false),
            std::time::Duration::ZERO,
        ));
        assert!(!process_exited_with(
            || None,
            std::time::Duration::ZERO,
        ));
    }

    #[cfg(windows)]
    #[test]
    fn windows_exit_status_distinguishes_running_from_exited_handles() {
        assert!(windows_exit_code_is_running(STILL_ACTIVE));
        assert!(!windows_exit_code_is_running(0));
        assert!(!windows_exit_code_is_running(1));
    }

}
