//! Port of packages/coding-agent/src/core/orphan-process-journal.ts

use std::collections::HashMap;
use std::io::Write;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::core::session_lease::get_process_start_id;
use crate::utils::child_process::{spawn_sync_hidden, SpawnOptions};

pub const ORPHAN_PROCESS_JOURNAL_ENV: &str = "PRIME_AGENT_INTERNAL_ORPHAN_PROCESS_JOURNAL";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrphanProcessRecord {
    pub version: i64,
    pub pid: i64,
    #[serde(rename = "ownerPid")]
    pub owner_pid: i64,
    /// Set on records written by a kernel (e.g. bash() children) so the host can reap per kernel.
    #[serde(rename = "kernelPid", skip_serializing_if = "Option::is_none")]
    pub kernel_pid: Option<i64>,
    #[serde(rename = "processStartId", skip_serializing_if = "Option::is_none")]
    pub process_start_id: Option<String>,
    pub active: bool,
    #[serde(rename = "recordedAt")]
    pub recorded_at: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ActiveOrphanProcess {
    pub pid: i64,
    pub kernel_pid: Option<i64>,
    /// Missing on identity-free records: old journals or host writes whose start-id query failed
    /// (kernels no longer write pid-only records).
    pub process_start_id: Option<String>,
}

pub fn record_orphan_process_state(pid: i64, active: bool) {
    let Some(path) = std::env::var_os(ORPHAN_PROCESS_JOURNAL_ENV) else {
        return;
    };
    let path = path.to_string_lossy().to_string();
    if pid <= 0 {
        return;
    }
    let process_start_id = if active { get_process_start_id(pid) } else { None };
    let record = OrphanProcessRecord {
        version: 1,
        pid,
        owner_pid: std::process::id() as i64,
        kernel_pid: None,
        process_start_id: process_start_id.clone(),
        active,
        recorded_at: now_iso_string(),
    };
    let line = match serde_json::to_string(&record) {
        Ok(line) => line,
        Err(_) => return,
    };
    // Process tracking must not make a successfully spawned command fail.
    let Ok(mut file) = std::fs::OpenOptions::new().append(true).create(true).open(&path) else {
        return;
    };
    let _ = file.write_all(format!("{line}\n").as_bytes());
    let _ = file.sync_all();
}

pub fn read_active_orphan_processes(path: &str, owner_pid: i64) -> Result<Vec<ActiveOrphanProcess>, std::io::Error> {
    let contents = match std::fs::read_to_string(path) {
        Ok(contents) => contents,
        Err(error) => {
            if error.kind() == std::io::ErrorKind::NotFound {
                return Ok(Vec::new());
            }
            return Err(error);
        }
    };
    let mut latest: Vec<OrphanProcessRecord> = Vec::new();
    let mut index_by_pid: HashMap<i64, usize> = HashMap::new();
    for line in contents.split('\n') {
        if line.is_empty() {
            continue;
        }
        // A crash can truncate only the final append.
        let Ok(record) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        let Some(object) = record.as_object() else {
            continue;
        };
        if object.get("version").and_then(Value::as_i64) != Some(1) {
            continue;
        }
        let Some(pid) = object.get("pid").and_then(Value::as_i64) else {
            continue;
        };
        if pid <= 0 || object.get("ownerPid").and_then(Value::as_i64) != Some(owner_pid) {
            continue;
        }
        if !object.get("active").map(Value::is_boolean).unwrap_or(false) {
            continue;
        }
        if !object.get("recordedAt").map(Value::is_string).unwrap_or(false) {
            continue;
        }
        let Ok(record) = serde_json::from_value::<OrphanProcessRecord>(record) else {
            continue;
        };
        match index_by_pid.get(&pid) {
            Some(index) => latest[*index] = record,
            None => {
                index_by_pid.insert(pid, latest.len());
                latest.push(record);
            }
        }
    }
    // Pid-only actives (no processStartId) still surface from old journals or
    // host writes whose start-id query failed; reapers decide per-platform.
    Ok(latest
        .into_iter()
        .filter(|record| record.active)
        .map(|record| ActiveOrphanProcess {
            pid: record.pid,
            kernel_pid: record.kernel_pid,
            process_start_id: record.process_start_id,
        })
        .collect())
}

pub fn is_orphan_process_identity_current(orphan: &ActiveOrphanProcess) -> bool {
    // Pid-only records can never claim identity (undefined === undefined must not match).
    match &orphan.process_start_id {
        Some(process_start_id) => get_process_start_id(orphan.pid).as_deref() == Some(process_start_id.as_str()),
        None => false,
    }
}

/// Identity-free records cannot prove the pid still names the journaled process.
/// On win32 the kernel's kill-on-close job already reaped its tree when it died,
/// so a bare-pid taskkill only risks killing a reused pid. POSIX keeps the
/// best-effort kill (group-scoped, and the spawn gate makes pid-only actives
/// host-written rarities there).
pub fn should_reap_orphan_process(orphan: &ActiveOrphanProcess) -> bool {
    if orphan.process_start_id.is_none() {
        return !cfg!(windows);
    }
    is_orphan_process_identity_current(orphan)
}

pub fn clear_orphan_process_journal(path: &str) {
    let _ = std::fs::remove_file(path);
}

/// Kills still-active bash() children journaled by the given kernel pid; sibling
/// kernels' records are untouched.
pub fn reap_kernel_orphan_processes(kernel_pid: i64) {
    let Some(path) = std::env::var_os(ORPHAN_PROCESS_JOURNAL_ENV) else {
        return;
    };
    let path = path.to_string_lossy().to_string();
    if kernel_pid <= 0 {
        return;
    }
    let Ok(orphans) = read_active_orphan_processes(&path, std::process::id() as i64) else {
        return;
    };
    for orphan in orphans {
        if orphan.kernel_pid != Some(kernel_pid) || orphan.pid == kernel_pid {
            continue;
        }
        if !should_reap_orphan_process(&orphan) {
            continue;
        }
        // Inactive only after a delivered signal; a stale record is neutralized by the startId check.
        if kill_orphan_process(orphan.pid) {
            record_orphan_process_state(orphan.pid, false);
        }
    }
}

/// Hardened cross-platform tree kill for journaled orphans: absolute System32
/// taskkill /T on win32 (a bare name could resolve a planted CWD taskkill.exe),
/// process-group then pid SIGKILL elsewhere.
pub fn kill_orphan_process(pid: i64) -> bool {
    if cfg!(windows) {
        // In-kernel bash() kill paths use taskkill /T; the reaper must kill the same tree,
        // not just the shell pid.
        let taskkill = crate::utils::shell::windows_taskkill_program();
        let mut env = SpawnOptions::default();
        env.env = Some(vec![("NoDefaultCurrentDirectoryInExePath".to_string(), "1".to_string())]);
        let result = spawn_sync_hidden(
            &taskkill,
            &["/F".to_string(), "/T".to_string(), "/PID".to_string(), pid.to_string()],
            env,
        );
        return matches!(result, Ok(output) if output.status.success());
    }
    #[cfg(unix)]
    {
        unsafe {
            if libc::kill(-(pid as i32), libc::SIGKILL) == 0 {
                return true;
            }
        }
        unsafe {
            if libc::kill(pid as i32, libc::SIGKILL) == 0 {
                return true;
            }
        }
    }
    // The orphan may already have exited.
    false
}

/// `new Date().toISOString()`.
fn now_iso_string() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let seconds = now.as_secs() as i64;
    let millis = now.subsec_millis();
    let days = seconds.div_euclid(86_400);
    let secs_of_day = seconds.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{millis:03}Z",
        secs_of_day / 3600,
        (secs_of_day % 3600) / 60,
        secs_of_day % 60
    )
}

/// Howard Hinnant's `civil_from_days` (the proleptic Gregorian calendar).
fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_round_trips_and_keeps_the_latest_state_per_pid() {
        let dir = std::env::temp_dir().join(format!("orphan-journal-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("journal.jsonl");
        let path_text = path.to_string_lossy().to_string();
        let owner = std::process::id() as i64;

        std::fs::write(
            &path,
            format!(
                "{}\n{}\n",
                serde_json::json!({"version": 1, "pid": 42, "ownerPid": owner, "active": true, "recordedAt": "t"}),
                serde_json::json!({"version": 1, "pid": 42, "ownerPid": owner, "active": false, "recordedAt": "t2"})
            ),
        )
        .unwrap();
        let active = read_active_orphan_processes(&path_text, owner).unwrap();
        assert!(active.is_empty());

        std::fs::write(
            &path,
            format!(
                "{}\nnot json\n",
                serde_json::json!({"version": 1, "pid": 42, "ownerPid": owner, "kernelPid": 7, "active": true, "recordedAt": "t"})
            ),
        )
        .unwrap();
        let active = read_active_orphan_processes(&path_text, owner).unwrap();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].pid, 42);
        assert_eq!(active[0].kernel_pid, Some(7));

        clear_orphan_process_journal(&path_text);
        assert!(!path.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_journal_reads_as_empty() {
        assert!(read_active_orphan_processes("/definitely/not/here.jsonl", 1)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn other_owners_are_ignored() {
        let dir = std::env::temp_dir().join(format!("orphan-owner-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("journal.jsonl");
        std::fs::write(
            &path,
            format!(
                "{}\n",
                serde_json::json!({"version": 1, "pid": 42, "ownerPid": 999999, "active": true, "recordedAt": "t"})
            ),
        )
        .unwrap();
        let active = read_active_orphan_processes(&path.to_string_lossy(), 1).unwrap();
        assert!(active.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn identity_free_records_are_not_current_but_are_reaped_off_windows() {
        let orphan = ActiveOrphanProcess {
            pid: std::process::id() as i64,
            kernel_pid: None,
            process_start_id: None,
        };
        assert!(!is_orphan_process_identity_current(&orphan));
        assert_eq!(should_reap_orphan_process(&orphan), !cfg!(windows));
    }

    #[test]
    fn identity_check_compares_the_start_id() {
        let pid = std::process::id() as i64;
        let Some(start_id) = get_process_start_id(pid) else {
            return;
        };
        let orphan = ActiveOrphanProcess {
            pid,
            kernel_pid: None,
            process_start_id: Some(start_id),
        };
        assert!(is_orphan_process_identity_current(&orphan));
        let stale = ActiveOrphanProcess {
            process_start_id: Some("win:0".to_string()),
            ..orphan
        };
        assert!(!is_orphan_process_identity_current(&stale));
    }

    #[test]
    fn iso_timestamp_has_the_javascript_shape() {
        let stamp = now_iso_string();
        assert_eq!(stamp.len(), 24);
        assert!(stamp.ends_with('Z'));
        assert_eq!(&stamp[4..5], "-");
    }

    #[test]
    fn civil_from_days_matches_known_dates() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(19_723), (2024, 1, 1));
    }
}
