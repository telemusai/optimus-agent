use crate::cli::subprocess_launch::ProcessEnv;
use std::io::{Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const ROLE: &str = "OPTIMUS_TEST_DAEMON_DETACH_ROLE";
const ROOT: &str = "OPTIMUS_TEST_DAEMON_DETACH_ROOT";

/// Re-exec the current test as a short-lived launcher and a detached responder.
/// Each caller exercises its production spawn path, with no user profile loaded.
pub(crate) fn assert_detached_spawn(
    spawn: impl Fn(&str, &[String], &str, &ProcessEnv),
) {
    let name = std::thread::current().name().unwrap().to_owned();
    let executable = std::env::current_exe().unwrap();
    let args = vec!["--exact".into(), name.clone(), "--nocapture".into()];
    if let Ok(role) = std::env::var(ROLE) {
        let root = std::env::var(ROOT).unwrap();
        if role == "launcher" {
            let env = ProcessEnv::from([
                (ROLE.into(), "daemon".into()),
                (ROOT.into(), root.clone()),
            ]);
            spawn(executable.to_str().unwrap(), &args, &root, &env);
            return;
        }
        assert_eq!(role, "daemon");
        let listener = UnixListener::bind(format!("{root}/probe.sock")).unwrap();
        std::fs::write(format!("{root}/pid"), std::process::id().to_string()).unwrap();
        listener.set_nonblocking(true).unwrap();
        let deadline = Instant::now() + Duration::from_secs(15);
        while Instant::now() < deadline {
            if let Ok((mut stream, _)) = listener.accept() {
                stream.write_all(b"alive").unwrap();
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        return;
    }

    let dir = tempfile::tempdir().unwrap();
    let mut command = Command::new(executable);
    command.args(args).env_clear().env(ROLE, "launcher").env(ROOT, dir.path())
        .stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null());
    super::detach_daemon(&mut command);
    let mut launcher = command.spawn().unwrap();
    let launcher_pid = launcher.id() as i32;
    let mut cleanup = Cleanup { launcher: Some(launcher_pid), daemon: None };
    let deadline = Instant::now() + Duration::from_secs(10);
    let status = loop {
        if let Some(status) = launcher.try_wait().unwrap() { break status; }
        assert!(Instant::now() < deadline, "launcher did not exit");
        std::thread::sleep(Duration::from_millis(10));
    };
    cleanup.launcher = None;
    assert!(status.success());
    let pid_path = dir.path().join("pid");
    while !pid_path.exists() {
        assert!(Instant::now() < deadline, "daemon did not become ready after launcher exit");
        std::thread::sleep(Duration::from_millis(10));
    }
    let pid: i32 = std::fs::read_to_string(pid_path).unwrap().parse().unwrap();
    cleanup.daemon = Some(pid);
    assert_eq!(unsafe { libc::getsid(pid) }, pid, "daemon retained launcher's terminal session");
    assert_eq!(unsafe { libc::getpgid(pid) }, pid);
    // The terminal would send SIGHUP to the original session's process group.
    unsafe { libc::kill(-launcher_pid, libc::SIGHUP); }
    let mut stream = UnixStream::connect(dir.path().join("probe.sock")).unwrap();
    stream.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
    let mut response = [0; 5];
    stream.read_exact(&mut response).unwrap();
    assert_eq!(&response, b"alive");
}

struct Cleanup { launcher: Option<i32>, daemon: Option<i32> }
impl Drop for Cleanup {
    fn drop(&mut self) {
        for pid in [self.daemon, self.launcher].into_iter().flatten() {
            unsafe { libc::kill(pid, libc::SIGKILL); }
        }
    }
}
