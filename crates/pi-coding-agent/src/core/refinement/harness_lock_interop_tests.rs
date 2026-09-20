//! Offline Windows proof of the production Rust/Python checked-save lock contract.
use super::{
    empty_harness_state, load_harness_state_details, lock_harness_state,
    save_harness_state_checked, HarnessEntry, HarnessScope, HarnessState,
};
use crate::utils::dir_lock::{stat_identity, StatIdentity};
use std::fs::{self, File, FileTimes};
use std::os::windows::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const WAIT: Duration = Duration::from_secs(15);
const LOCK_BYTES: &[u8] = b"synthetic stable harness lock\r\n";
const PYTHON_ENV: &str = "OPTIMUS_COMPAT_PYTHON";

struct OwnedChild {
    child: Child,
    log: PathBuf,
}

impl OwnedChild {
    fn spawn(mut command: Command, root: &Path, name: &str) -> Self {
        let log = root.join(format!("{name}.log"));
        let output = File::create(&log).unwrap();
        command
            .creation_flags(0x0800_0000) // CREATE_NO_WINDOW
            .stdin(Stdio::null())
            .stdout(output.try_clone().unwrap())
            .stderr(output);
        Self {
            child: command.spawn().expect("test-owned hidden child"),
            log,
        }
    }

    fn wait_marker(&mut self, marker: &Path) {
        let deadline = Instant::now() + WAIT;
        while !marker.exists() {
            assert!(
                self.child.try_wait().unwrap().is_none(),
                "child exited before {marker:?}: {}",
                self.diagnostics()
            );
            assert!(
                Instant::now() < deadline,
                "child handshake timed out: {}",
                self.diagnostics()
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    fn finish(&mut self) {
        let deadline = Instant::now() + WAIT;
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                assert!(
                    status.success(),
                    "test child failed: {}",
                    self.diagnostics()
                );
                return;
            }
            assert!(
                Instant::now() < deadline,
                "test child did not exit: {}",
                self.diagnostics()
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    fn diagnostics(&self) -> String {
        fs::read_to_string(&self.log)
            .unwrap_or_default()
            .chars()
            .take(4000)
            .collect()
    }
}

impl Drop for OwnedChild {
    fn drop(&mut self) {
        if !matches!(self.child.try_wait(), Ok(Some(_))) {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

fn isolated_command(executable: &Path, root: &Path) -> Command {
    let mut command = Command::new(executable);
    command.env_clear();
    for key in ["SystemRoot", "WINDIR"] {
        if let Some(value) = std::env::var_os(key) {
            command.env(key, value);
        }
    }
    for key in [
        "HOME",
        "USERPROFILE",
        "TEMP",
        "TMP",
        "PI_CODING_AGENT_DIR",
        "PRIME_AGENT_CODING_AGENT_DIR",
        "RLM_HARNESS_STATE_DIR",
        "RLM_GLOBAL_HARNESS_STATE_DIR",
        "RLM_SESSION_DIR",
    ] {
        command.env(key, root);
    }
    command.current_dir(root);
    command
}

fn python(root: &Path, mode: &str) -> OwnedChild {
    let executable = PathBuf::from(
        std::env::var_os(PYTHON_ENV)
            .expect("set OPTIMUS_COMPAT_PYTHON to the fresh candidate Python 3.11.15 executable"),
    );
    assert!(
        executable.is_absolute() && executable.is_file(),
        "explicit candidate Python executable required"
    );
    let repository = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let script = repository.join("crates/pi-coding-agent/tests/fixtures/harness_lock_interop.py");
    let mut command = isolated_command(&executable, root);
    command
        .arg("-I")
        .arg("-B")
        .arg(script)
        .arg(repository)
        .arg(root)
        .arg(mode);
    OwnedChild::spawn(command, root, mode)
}

fn state_path(root: &Path) -> PathBuf {
    root.join("harness_state.json")
}
fn lock_path(root: &Path) -> PathBuf {
    root.join("harness_state.json.lock")
}
fn dir(root: &Path) -> String {
    root.to_string_lossy().into_owned()
}

fn add_marker(state: &mut HarnessState, id: &str) {
    let entry: HarnessEntry = serde_json::from_value(serde_json::json!({
        "id": id, "kind": "memory", "title": id, "content": "synthetic interoperability marker",
        "path": "test-only", "scope": "local", "source": "test", "created_at": "2000-01-01",
        "updated_at": "2000-01-01", "version": 1
    }))
    .unwrap();
    state
        .entries
        .get_mut("memory")
        .unwrap()
        .insert(id.to_string(), entry);
}

fn seed(root: &Path) -> Vec<u8> {
    let baseline = load_harness_state_details(&dir(root), HarnessScope::Local);
    let mut state = empty_harness_state();
    add_marker(&mut state, "seed");
    save_harness_state_checked(&dir(root), &state, &baseline).unwrap();
    fs::write(lock_path(root), LOCK_BYTES).unwrap();
    assert_eq!(fs::read(lock_path(root)).unwrap(), LOCK_BYTES);
    fs::read(state_path(root)).unwrap()
}

fn assert_lock_stable(root: &Path, identity: StatIdentity) {
    // Windows byte-range locks forbid payload reads, but not file-identity queries.
    assert_eq!(stat_identity(&lock_path(root)), Some(identity));
    let locks = fs::read_dir(root)
        .unwrap()
        .flatten()
        .filter(|entry| entry.file_name().to_string_lossy().ends_with(".lock"))
        .count();
    assert_eq!(locks, 1, "writers must keep the same stable side file");
}

// Only the exact subprocess command below supplies this variable.
#[test]
fn rust_holder_process() {
    let Some(root) = std::env::var_os("OPTIMUS_COMPAT_RUST_HOLDER") else {
        return;
    };
    let root = PathBuf::from(root);
    assert!(root.join("synthetic-owner").exists());
    let _lock = lock_harness_state(&state_path(&root).to_string_lossy()).unwrap();
    fs::write(root.join("rust-ready"), b"ready").unwrap();
    let deadline = Instant::now() + WAIT;
    while !root.join("release-rust").exists() {
        assert!(Instant::now() < deadline, "holder release timed out");
        std::thread::sleep(Duration::from_millis(10));
    }
    // Exit releases the production OS lock. No application/daemon process is involved.
}

#[test]
fn windows_harness_rust_holder_python_checked_save() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    let original = seed(root);
    let identity = stat_identity(&lock_path(root)).expect("stable lock identity");
    let mtime = fs::metadata(state_path(root)).unwrap().modified().unwrap();
    fs::write(root.join("synthetic-owner"), b"test").unwrap();
    let mut command = isolated_command(&std::env::current_exe().unwrap(), root);
    command
        .env("OPTIMUS_COMPAT_RUST_HOLDER", root)
        .arg("--exact")
        .arg("core::refinement::refinement::harness_lock_interop_tests::rust_holder_process")
        .arg("--test-threads=1")
        .arg("--nocapture");
    let mut holder = OwnedChild::spawn(command, root, "rust-holder");
    holder.wait_marker(&root.join("rust-ready"));
    let mut contender = python(root, "python-contender");
    contender.wait_marker(&root.join("python-refused"));
    assert_eq!(fs::read(state_path(root)).unwrap(), original);
    assert_lock_stable(root, identity);
    fs::write(root.join("release-rust"), b"release").unwrap();
    holder.finish();

    let baseline = load_harness_state_details(&dir(root), HarnessScope::Local);
    let mut state = baseline.state.clone();
    add_marker(&mut state, "rust-peer");
    save_harness_state_checked(&dir(root), &state, &baseline).unwrap();
    File::options()
        .write(true)
        .open(state_path(root))
        .unwrap()
        .set_times(FileTimes::new().set_modified(mtime))
        .unwrap();
    assert_eq!(
        fs::metadata(state_path(root)).unwrap().modified().unwrap(),
        mtime
    );
    fs::write(root.join("rust-exited-and-saved"), b"reload").unwrap();
    contender.finish();
    let result = load_harness_state_details(&dir(root), HarnessScope::Local);
    for id in ["seed", "rust-peer", "python-success"] {
        assert!(
            result.state.entries["memory"].contains_key(id),
            "missing {id}"
        );
    }
    assert_lock_stable(root, identity);
    assert_eq!(fs::read(lock_path(root)).unwrap(), LOCK_BYTES);
}

#[test]
fn windows_harness_python_holder_rust_checked_save() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    let original = seed(root);
    let identity = stat_identity(&lock_path(root)).expect("stable lock identity");
    let mtime = fs::metadata(state_path(root)).unwrap().modified().unwrap();
    let baseline = load_harness_state_details(&dir(root), HarnessScope::Local);
    let mut proposed = baseline.state.clone();
    add_marker(&mut proposed, "rust-success");
    let mut holder = python(root, "python-holder");
    holder.wait_marker(&root.join("python-ready"));
    let started = Instant::now();
    let error = save_harness_state_checked(&dir(root), &proposed, &baseline).unwrap_err();
    assert!(error.contains("another writer"), "{error}");
    assert!((Duration::from_millis(800)..Duration::from_secs(5)).contains(&started.elapsed()));
    assert_eq!(fs::read(state_path(root)).unwrap(), original);
    assert_lock_stable(root, identity);
    fs::write(root.join("release-python"), b"release").unwrap();
    holder.finish();
    let newer = fs::read(state_path(root)).unwrap();
    assert_ne!(newer, original);
    assert_eq!(
        fs::metadata(state_path(root)).unwrap().modified().unwrap(),
        mtime
    );
    let error = save_harness_state_checked(&dir(root), &proposed, &baseline).unwrap_err();
    assert!(error.contains("changed since it was loaded"), "{error}");
    assert_eq!(fs::read(state_path(root)).unwrap(), newer);
    let fresh = load_harness_state_details(&dir(root), HarnessScope::Local);
    let mut state = fresh.state.clone();
    add_marker(&mut state, "rust-success");
    save_harness_state_checked(&dir(root), &state, &fresh).unwrap();
    let result = load_harness_state_details(&dir(root), HarnessScope::Local);
    for id in ["seed", "python-peer", "rust-success"] {
        assert!(
            result.state.entries["memory"].contains_key(id),
            "missing {id}"
        );
    }
    assert_lock_stable(root, identity);
    assert_eq!(fs::read(lock_path(root)).unwrap(), LOCK_BYTES);
}
