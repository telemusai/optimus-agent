//! Port of packages/coding-agent/src/utils/shell.ts

use std::collections::HashSet;
use std::path::Path;
use std::sync::Mutex;

use super::child_process::{spawn_hidden, spawn_sync_hidden, SpawnOptions};

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
    let mut env: Vec<(String, String)> = std::env::vars().collect();
    let path_key = env
        .iter()
        .map(|(key, _)| key.clone())
        .find(|key| key.to_lowercase() == "path")
        .unwrap_or_else(|| "PATH".to_string());
    let current_path = env
        .iter()
        .find(|(key, _)| key == &path_key)
        .map(|(_, value)| value.clone())
        .unwrap_or_default();
    let separator = if cfg!(windows) { ';' } else { ':' };
    let path_entries: Vec<&str> = current_path.split(separator).filter(|entry| !entry.is_empty()).collect();
    let has_bin_dir = path_entries.contains(&bin_dir.as_str());
    let updated_path = if has_bin_dir {
        current_path
    } else {
        let mut entries: Vec<String> = vec![bin_dir];
        entries.extend(path_entries.iter().map(|entry| entry.to_string()));
        entries
            .into_iter()
            .filter(|entry| !entry.is_empty())
            .collect::<Vec<String>>()
            .join(&separator.to_string())
    };

    if let Some(entry) = env.iter_mut().find(|(key, _)| key == &path_key) {
        entry.1 = updated_path;
    } else {
        env.push((path_key, updated_path));
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
    let pids: Vec<i32> = {
        let guard = tracked_pids();
        guard
            .as_ref()
            .map(|set| set.iter().copied().collect())
            .unwrap_or_default()
    };
    for pid in pids {
        kill_process_tree(pid);
        record_orphan_process_state(pid, false);
    }
    if let Some(set) = tracked_pids().as_mut() {
        set.clear();
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

/// Kill a process and all its children (cross-platform)
pub fn kill_process_tree(pid: i32) {
    if super::pi_user_agent::process_platform() == "win32" {
        // Use the absolute System32 taskkill on Windows to kill the process tree
        // (audit A4: the bare-name spawn was the one unhardened sibling site).
        let _ = spawn_hidden(
            &windows_taskkill_program(),
            &[
                "/F".to_string(),
                "/T".to_string(),
                "/PID".to_string(),
                pid.to_string(),
            ],
            SpawnOptions {
                detached: true,
                env: Some(vec![("NoDefaultCurrentDirectoryInExePath".to_string(), "1".to_string())]),
                ..Default::default()
            },
        );
    } else {
        // Use SIGKILL on Unix/Linux/Mac: the helper already falls back to a
        // single-pid kill when the process group is unavailable.
        super::child_process::signal_process_group_or_process(pid, super::child_process::Signal::Kill);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn shell_env_prepends_the_bin_dir_once() {
        let bin_dir = super::super::tools_manager::get_bin_dir().to_string_lossy().to_string();
        let env = get_shell_env();
        let path_entry = env
            .iter()
            .find(|(key, _)| key.to_lowercase() == "path")
            .map(|(_, value)| value.clone())
            .unwrap();
        assert!(path_entry.starts_with(&bin_dir));

        let second = get_shell_env();
        let second_path = second
            .iter()
            .find(|(key, _)| key.to_lowercase() == "path")
            .map(|(_, value)| value.clone())
            .unwrap();
        assert_eq!(path_entry, second_path);
    }

    #[test]
    fn detached_child_tracking_is_idempotent() {
        untrack_detached_child_pid(999_991);
        track_detached_child_pid(999_991);
        track_detached_child_pid(999_991);
        untrack_detached_child_pid(999_991);
        assert!(tracked_pids().as_ref().map(|set| set.is_empty()).unwrap_or(true));
    }
}
