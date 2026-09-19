//! T04 — process cleanup, journal and kernel error paths (owner: sessionkernel).
//! Spec: TEST-SPEC.md T04. Findings: G-19, G-20, G-23, G-30, G2-01, G2-02, G2-03, G2-08.
//!
//! Isolation (V00): every writable artifact lives under
//! work/state-roots/sessionkernel/<case>/. Only test-owned pids (recorded from
//! our own spawn calls) are killed. No production pipes, no ports 43119/43120,
//! no Prime/Optimus venv.

#![cfg(windows)]

use indexmap::IndexMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use pi_coding_agent::core::orphan_process_journal::{
    read_active_orphan_processes, reap_kernel_orphan_processes, should_reap_orphan_process,
    OrphanProcessRecord, ORPHAN_PROCESS_JOURNAL_ENV,
};
use pi_coding_agent::core::session_lease::{SESSION_LEASES_ENABLED_ENV, SESSION_LEASE_OWNER_ID_ENV};
use pi_coding_agent::core::tools::bash::{
    create_local_bash_operations, BashExecOptions, BashExecResult,
};
use pi_coding_agent::modes::daemon::daemon_mode::daemon_supervisor_launch_env;
use pi_coding_agent::modes::daemon::daemon_worker_protocol::{
    DAEMON_WORKER_ACTIVE_SESSION_ID_ENV, DAEMON_WORKER_RECOVERY_JOURNAL_ENV, DAEMON_WORKER_ROLE_ENV,
    DAEMON_WORKER_SUPERVISOR_SOCKET_ENV, DAEMON_WORKER_TOKEN_ENV,
};
use pi_coding_agent::utils::child_process::{
    set_sync_spawn_override_for_tests, spawn_hidden, SpawnOptions,
};
use pi_coding_agent::utils::shell::{get_shell_config, kill_process_tree};

/// Ambient-env tests serialize on one process-wide lock.
static TEST_ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    TEST_ENV_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn work_root() -> PathBuf {
    static ROOT: OnceLock<tempfile::TempDir> = OnceLock::new();
    ROOT.get_or_init(|| tempfile::Builder::new().prefix("optimus-sessionkernel-").tempdir().expect("private test root"))
        .path()
        .to_path_buf()
}

fn state_root(case: &str) -> PathBuf {
    let root = work_root().join("state-roots").join("sessionkernel").join(case);
    std::fs::create_dir_all(&root).expect("state root");
    root
}

fn read_journal_records(path: &Path) -> Vec<OrphanProcessRecord> {
    match std::fs::read_to_string(path) {
        Ok(contents) => contents
            .lines()
            .filter_map(|line| serde_json::from_str::<OrphanProcessRecord>(line).ok())
            .collect(),
        Err(_) => Vec::new(),
    }
}

/// Test liveness oracle: `process_id_exists` keeps counting a terminated
/// process whose handle is still open (tokio reaper), so probe the exit code
/// like the OS task list does.
fn probe_process_running(pid: i32) -> bool {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Threading::{
        GetExitCodeProcess, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
    };
    const STILL_ACTIVE: u32 = 259;
    unsafe {
        let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid as u32);
        if handle.is_null() {
            return false;
        }
        let mut exit_code: u32 = 0;
        let ok = GetExitCodeProcess(handle, &mut exit_code) != 0;
        CloseHandle(handle);
        ok && exit_code == STILL_ACTIVE
    }
}

fn wait_process_gone(pid: i32, deadline: Duration) -> bool {
    let start = std::time::Instant::now();
    while start.elapsed() < deadline {
        if !probe_process_running(pid) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    !probe_process_running(pid)
}

/// Simulates a hung synchronous runner: hands back a long-lived child so the
/// unbounded baseline path waits forever and the bounded fix kills it at the
/// timeout. Used for `where` (G-23).
fn hang_on_where(
    command: &str,
    _args: &[String],
    _options: &SpawnOptions,
) -> Option<std::io::Result<std::process::Child>> {
    if command.eq_ignore_ascii_case("where") {
        let mut stand_in = std::process::Command::new("ping");
        stand_in.args(["-n", "60", "127.0.0.1"]);
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            stand_in.creation_flags(0x08000000);
        }
        stand_in
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());
        Some(stand_in.spawn())
    } else {
        None
    }
}

/// G-19: detached bash() children must be journaled (active at spawn,
/// inactive after settle) and the recovered supervisor must reap exactly the
/// journaled helper, leaving an unrelated process alone.
/// TS contract: core/tools/bash.ts:80 (track), :102/:116 (untrack),
/// core/orphan-process-journal.ts reaper.
#[tokio::test(flavor = "multi_thread")]
async fn detached_bash_is_journaled_and_reaped() {
    let _guard = env_lock();
    let root = state_root("t04-journal-reap");
    let journal_path = root.join("orphans.jsonl");
    let _ = std::fs::remove_file(&journal_path);
    std::env::set_var(ORPHAN_PROCESS_JOURNAL_ENV, &journal_path);

    let ops = create_local_bash_operations(None);
    let cwd = root.join("bash-cwd");
    std::fs::create_dir_all(&cwd).unwrap();
    let result: Result<BashExecResult, String> = ops
        .exec(
            "echo started; sleep 25 & echo done",
            cwd.to_str().unwrap(),
            BashExecOptions {
                on_data: Arc::new(|_chunk| {}),
                signal: None,
                timeout: Some(10.0),
                env: None,
            },
        )
        .await;
    let result = result.expect("bash exec must succeed");
    assert_eq!(result.exit_code, Some(0), "bash command must run");

    // The settled bash() shell was journaled and then untracked.
    let records = read_journal_records(&journal_path);
    assert!(
        !records.is_empty(),
        "bash() child was never journaled: record_orphan_process_state is a no-op stub"
    );
    assert!(
        records.iter().all(|record| record.owner_pid == std::process::id() as i64),
        "journal records must carry the owner pid"
    );
    assert!(
        records.iter().any(|record| !record.active),
        "settled bash() child was never untracked"
    );

    // Crash simulation: a journaled live helper, then the recovered supervisor.
    let helper = spawn_hidden(
        "ping",
        &["-n".to_string(), "20".to_string(), "127.0.0.1".to_string()],
        SpawnOptions::default(),
    )
    .expect("helper spawn");
    let helper_pid = helper.child.id().expect("helper pid") as i32;
    let control = spawn_hidden(
        "ping",
        &["-n".to_string(), "30".to_string(), "127.0.0.1".to_string()],
        SpawnOptions::default(),
    )
    .expect("control spawn");
    let control_pid = control.child.id().expect("control pid") as i32;
    // Journal the helper the way the kernel's bash.py does: a kernelPid record
    // under this process as the "kernel", so the recovered supervisor's reaper
    // (journal.ts:122-145) owns it. track_detached_child_pid's pid-only record
    // is intentionally not a reaper target (win32 relies on the kill-on-close
    // job for those).
    let helper_record = OrphanProcessRecord {
        version: 1,
        pid: helper_pid as i64,
        owner_pid: std::process::id() as i64,
        kernel_pid: Some(std::process::id() as i64),
        process_start_id: Some(
            pi_coding_agent::core::session_lease::get_process_start_id(helper_pid as i64)
                .expect("helper start id"),
        ),
        active: true,
        recorded_at: "parity-test".to_string(),
    };
    {
        use std::io::Write as _;
        let mut file = std::fs::OpenOptions::new().append(true).create(true).open(&journal_path).unwrap();
        file.write_all(format!("{}\n", serde_json::to_string(&helper_record).unwrap()).as_bytes()).unwrap();
    }

    let orphans = read_active_orphan_processes(journal_path.to_str().unwrap(), std::process::id() as i64)
        .expect("journal read");
    let target = orphans
        .iter()
        .find(|orphan| orphan.pid == helper_pid as i64)
        .expect("journaled helper pid missing from the active set");
    assert!(should_reap_orphan_process(target), "journaled identity must be current");
    // TS journal.ts:122-145: the reaper loop kills and marks the record
    // inactive only after a delivered kill; killOrphanProcess itself does not
    // touch the journal.
    reap_kernel_orphan_processes(std::process::id() as i64);
    assert!(
        wait_process_gone(helper_pid, Duration::from_secs(10)),
        "reaped helper survived"
    );
    assert!(
        probe_process_running(control_pid),
        "the unrelated control process was killed by the reaper"
    );
    let records = read_journal_records(&journal_path);
    assert!(
        records
            .iter()
            .any(|record| record.pid == helper_pid as i64 && !record.active),
        "delivered reaper kill must mark the record inactive"
    );

    // Test-owned cleanup.
    kill_process_tree(control_pid);
    let _ = wait_process_gone(control_pid, Duration::from_secs(10));
    std::env::remove_var(ORPHAN_PROCESS_JOURNAL_ENV);
}

/// A4 regression: kill_process_tree must spawn the hardened absolute System32
/// taskkill (a bare-name spawn would resolve a planted CWD taskkill.exe via
/// PATH) and still kill the whole tree.
#[test]
fn kill_process_tree_uses_the_hardened_absolute_taskkill() {
    let helper = spawn_hidden(
        "ping",
        &["-n".to_string(), "30".to_string(), "127.0.0.1".to_string()],
        SpawnOptions::default(),
    )
    .expect("helper spawn");
    let helper_pid = helper.child.id().expect("helper pid") as i32;
    kill_process_tree(helper_pid);
    assert!(
        wait_process_gone(helper_pid, Duration::from_secs(10)),
        "the hardened taskkill tree kill did not terminate the helper"
    );
}

/// G-20: the daemon supervisor launch environment must strip inherited
/// worker role/token/lease variables and must NOT mark the orphan journal
/// with a literal "1" (daemon-mode.ts:925-934 deletes all of them).
#[tokio::test]
async fn supervisor_env_strips_worker_role() {
    let root = state_root("t04-env-strips");
    let cwd = root.join("cwd");
    std::fs::create_dir_all(&cwd).unwrap();
    let source: IndexMap<String, String> = [
        (DAEMON_WORKER_ROLE_ENV.to_string(), "1".to_string()),
        (DAEMON_WORKER_TOKEN_ENV.to_string(), "parity-token".to_string()),
        (DAEMON_WORKER_ACTIVE_SESSION_ID_ENV.to_string(), "parity-session".to_string()),
        (DAEMON_WORKER_RECOVERY_JOURNAL_ENV.to_string(), "parity-recovery.jsonl".to_string()),
        (
            DAEMON_WORKER_SUPERVISOR_SOCKET_ENV.to_string(),
            r"\\.\pipe\parity-supervisor".to_string(),
        ),
        (ORPHAN_PROCESS_JOURNAL_ENV.to_string(), "parity-stale-journal.jsonl".to_string()),
        (SESSION_LEASES_ENABLED_ENV.to_string(), "1".to_string()),
        (SESSION_LEASE_OWNER_ID_ENV.to_string(), "parity-owner".to_string()),
    ]
    .into_iter()
    .collect();
    let agent_dir = "Z:\\nonexistent-parity-agent-dir".to_string();

    let env = daemon_supervisor_launch_env(&source, Some(&agent_dir));
    for key in [
        DAEMON_WORKER_ROLE_ENV,
        DAEMON_WORKER_TOKEN_ENV,
        DAEMON_WORKER_ACTIVE_SESSION_ID_ENV,
        DAEMON_WORKER_RECOVERY_JOURNAL_ENV,
        DAEMON_WORKER_SUPERVISOR_SOCKET_ENV,
        ORPHAN_PROCESS_JOURNAL_ENV,
        SESSION_LEASES_ENABLED_ENV,
        SESSION_LEASE_OWNER_ID_ENV,
    ] {
        assert!(
            !env.contains_key(key),
            "supervisor env leaks {key}: a worker-role child can launch in worker mode or journal to a file named \"1\""
        );
    }
    assert_eq!(
        env.get("PRIME_AGENT_AGENT_DIR").map(String::as_str),
        Some(agent_dir.as_str()),
        "agent dir injection must survive"
    );

    // Real spawn probe: the built env, verbatim, must be what a supervisor child sees.
    let mut command = tokio::process::Command::new("cmd");
    command.args(["/c", "set"]);
    command.current_dir(&cwd);
    command.env_clear();
    command.envs(env.iter().map(|(key, value)| (key.clone(), value.clone())));
    command.stdin(Stdio::null());
    command.stdout(Stdio::piped());
    command.stderr(Stdio::null());
    command.creation_flags(0x0800_0000);
    let mut child = command.spawn().expect("probe spawn");
    let mut stdout = child.stdout.take().expect("probe stdout");
    let reader = tokio::spawn(async move {
        use tokio::io::AsyncReadExt;
        let mut buffer = Vec::new();
        let _ = stdout.read_to_end(&mut buffer).await;
        buffer
    });
    let status = tokio::time::timeout(Duration::from_secs(20), child.wait())
        .await
        .expect("probe child hung")
        .expect("probe wait");
    assert!(status.success(), "probe child failed");
    let probe_text = String::from_utf8_lossy(&reader.await.unwrap_or_default()).to_string();
    for key in [
        DAEMON_WORKER_ROLE_ENV,
        DAEMON_WORKER_TOKEN_ENV,
        ORPHAN_PROCESS_JOURNAL_ENV,
        SESSION_LEASES_ENABLED_ENV,
    ] {
        assert!(
            !probe_text.contains(key),
            "spawned supervisor child still sees {key} in its environment"
        );
    }
    // A literal journal path "1" would materialize as a file in the child cwd.
    let listing: Vec<String> = std::fs::read_dir(&cwd)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert!(
        !listing.iter().any(|name| name == "1"),
        "journal lines would go to a file named \"1\" in cwd: {listing:?}"
    );
}

/// G-23: bash resolution via `where` must be bounded (TS shell.ts uses a
/// 5000ms timeout). The seam hangs only the `where` call; the resolution
/// path must still return.
#[tokio::test(flavor = "multi_thread")]
async fn bash_resolution_where_call_is_bounded() {
    let _guard = env_lock();
    let original_program_files = std::env::var("ProgramFiles").ok();
    let original_program_files_x86 = std::env::var("ProgramFiles(x86)").ok();
    std::env::set_var("ProgramFiles", r"Z:\nonexistent-parity-probe");
    std::env::set_var("ProgramFiles(x86)", r"Z:\nonexistent-parity-probe");

    set_sync_spawn_override_for_tests(Some(hang_on_where));

    let (sender, receiver) =
        std::sync::mpsc::channel::<(Result<pi_coding_agent::utils::shell::ShellConfig, String>, Duration)>();
    std::thread::spawn(move || {
        let start = std::time::Instant::now();
        let result = get_shell_config(None);
        let _ = sender.send((result, start.elapsed()));
    });
    let outcome = receiver.recv_timeout(Duration::from_secs(20));
    set_sync_spawn_override_for_tests(None);
    match original_program_files {
        Some(value) => std::env::set_var("ProgramFiles", value),
        None => std::env::remove_var("ProgramFiles"),
    }
    match original_program_files_x86 {
        Some(value) => std::env::set_var("ProgramFiles(x86)", value),
        None => std::env::remove_var("ProgramFiles(x86)"),
    }
    let (result, elapsed) = outcome.expect("bash resolution hung: the where.exe call is unbounded");
    assert!(
        elapsed < Duration::from_secs(15),
        "bash resolution took {elapsed:?}: the where.exe timeout is not honored"
    );
    // With Git Bash hidden and `where` failing, resolution must fail with the
    // teaching error instead of hanging or panicking.
    assert!(result.is_err(), "with no bash available, resolution must fail explicitly");
}

/// G2-02: a failed signal must not report a delivered kill (TS uses the
/// `child.kill()` result; repl-manager.ts:1430-1435). The fallback-signal
/// journal divergence is reachable on POSIX; the Windows tree-kill path
/// checks its own status already.
#[cfg(unix)]
#[test]
fn failed_signal_keeps_journal_active() {
    let _guard = env_lock();
    let root = state_root("t04-journal-truth");
    let journal_path = root.join("orphans.jsonl");
    let _ = std::fs::remove_file(&journal_path);
    std::env::set_var(ORPHAN_PROCESS_JOURNAL_ENV, &journal_path);
    pi_coding_agent::core::orphan_process_journal::record_orphan_process_state(999_991, true);
    assert!(!kill_orphan_process(999_991), "a dead pid must not report a delivered signal");
    let records = read_journal_records(&journal_path);
    let record = records
        .iter()
        .find(|record| record.pid == 999_991)
        .expect("journaled pid missing");
    assert!(record.active, "a failed signal must leave the record active");
    std::env::remove_var(ORPHAN_PROCESS_JOURNAL_ENV);
}
