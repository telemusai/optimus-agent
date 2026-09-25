//! Stable OS-owned locks for settings and credential transactions.
//!
//! Never unlink these side files: waiters must keep locking the same file object.
use std::fs::{File, OpenOptions, TryLockError};
use std::io;
use std::path::Path;
use std::time::Duration;

pub(crate) fn open_store_lock(path: &str) -> io::Result<File> {
    let lock_path = format!("{path}.lock");
    match std::fs::symlink_metadata(&lock_path) {
        Ok(metadata) if !metadata.is_file() || metadata.file_type().is_symlink() => {
            return Err(io::Error::other(
                "Store lock is a legacy directory or non-regular file; stop all writers before manual recovery",
            ));
        }
        Ok(_) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error),
    }
    let mut options = OpenOptions::new();
    options.read(true).write(true).create(true).truncate(false);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600).custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        use windows_sys::Win32::Storage::FileSystem::FILE_FLAG_OPEN_REPARSE_POINT;
        options.custom_flags(FILE_FLAG_OPEN_REPARSE_POINT);
    }
    let file = options.open(Path::new(&lock_path))?;
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.file_type().is_symlink() {
        return Err(io::Error::other("Store lock is not a regular file"));
    }
    Ok(file)
}

pub(crate) fn lock_store_sync(path: &str) -> io::Result<File> {
    let file = open_store_lock(path)?;
    for attempt in 0..10 {
        match file.try_lock() {
            Ok(()) => return Ok(file),
            Err(TryLockError::WouldBlock) if attempt < 9 => {
                std::thread::sleep(Duration::from_millis(20));
            }
            Err(TryLockError::WouldBlock) => {
                return Err(io::Error::new(io::ErrorKind::WouldBlock, "Store is locked by another writer; retry"));
            }
            Err(TryLockError::Error(error)) => return Err(error),
        }
    }
    unreachable!("bounded lock loop returns on its final attempt")
}

pub(crate) fn read_store(path: &str) -> io::Result<Option<String>> {
    match std::fs::read_to_string(path) {
        Ok(content) => Ok(Some(content)),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn process_death_releases_store_lock() {
        use std::io::BufRead;
        use std::process::{Command, Stdio};
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("synthetic.json");
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "utils::store_lock::tests::crash_lock_fixture", "--ignored", "--nocapture"])
            .env("OPTIMUS_STORE_LOCK_TEST_PATH", &path)
            .stdin(Stdio::piped()).stdout(Stdio::piped()).spawn().unwrap();
        let mut output = std::io::BufReader::new(child.stdout.take().unwrap());
        loop {
            let mut line = String::new();
            assert_ne!(output.read_line(&mut line).unwrap(), 0, "child failed before lock acquisition");
            if line.trim() == "store-lock-ready" { break; }
        }
        let contender = open_store_lock(path.to_str().unwrap()).unwrap();
        assert!(matches!(contender.try_lock(), Err(TryLockError::WouldBlock)));
        child.kill().unwrap();
        child.wait().unwrap();
        contender.try_lock().unwrap();
        assert!(dir.path().join("synthetic.json.lock").is_file());
    }

    #[test]
    #[ignore = "subprocess fixture; only synthetic path supplied by parent test"]
    fn crash_lock_fixture() {
        use std::io::{Read, Write};
        let path = std::env::var("OPTIMUS_STORE_LOCK_TEST_PATH").unwrap();
        let _guard = lock_store_sync(&path).unwrap();
        println!("store-lock-ready");
        std::io::stdout().flush().unwrap();
        let mut byte = [0];
        let _ = std::io::stdin().read(&mut byte);
    }

    #[cfg(unix)]
    #[test]
    fn lock_symlink_is_rejected_without_touching_target() {
        use std::os::unix::fs::{symlink, PermissionsExt};
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("synthetic.json");
        let target = dir.path().join("target");
        std::fs::write(&target, "preserve").unwrap();
        symlink(&target, dir.path().join("synthetic.json.lock")).unwrap();
        assert!(open_store_lock(path.to_str().unwrap()).is_err());
        assert_eq!(std::fs::read_to_string(target).unwrap(), "preserve");
        let safe = dir.path().join("safe.json");
        let file = open_store_lock(safe.to_str().unwrap()).unwrap();
        assert_eq!(file.metadata().unwrap().permissions().mode() & 0o777, 0o600);
    }
}
