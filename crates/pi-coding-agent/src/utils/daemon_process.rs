/// Detach a long-lived daemon from the terminal session that launched it.
pub(crate) fn detach_daemon(command: &mut std::process::Command) {
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        // A separate process group alone still belongs to the terminal session.
        // setsid is async-signal-safe; do not allocate or take locks after fork.
        unsafe {
            command.pre_exec(|| {
                if libc::setsid() < 0 {
                    Err(std::io::Error::last_os_error())
                } else {
                    Ok(())
                }
            });
        }
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
        command.creation_flags(CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP);
    }
}

#[cfg(all(test, unix))]
#[path = "daemon_process_tests.rs"]
pub(crate) mod tests;
