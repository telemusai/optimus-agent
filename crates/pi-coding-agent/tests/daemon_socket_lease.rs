use std::path::Path;
use std::time::Duration;

use pi_coding_agent::modes::daemon::daemon_socket::acquire_daemon_socket_path_lease;

#[tokio::test]
async fn lease_heartbeat_blocks_a_second_acquire_until_release() {
    if utils_platform_is_windows() {
        assert!(acquire_daemon_socket_path_lease("/tmp/whatever.sock")
            .await
            .is_none());
        return;
    }
    let root = tempfile::tempdir().expect("test directory");
    let socket_path = root
        .path()
        .join("daemon.sock")
        .to_string_lossy()
        .to_string();
    let lock_path = format!("{socket_path}.lock");
    let lease = acquire_daemon_socket_path_lease(&socket_path)
        .await
        .expect("lease");
    assert!(Path::new(&lock_path).exists());

    let second_path = socket_path.clone();
    let second = tokio::spawn(async move { acquire_daemon_socket_path_lease(&second_path).await });
    tokio::time::sleep(Duration::from_millis(6000)).await;
    assert!(
        !second.is_finished(),
        "heartbeat kept the lease past the stale window"
    );

    lease.release().await;
    let second = tokio::time::timeout(Duration::from_secs(2), second)
        .await
        .expect("second acquire finishes after release")
        .expect("second acquire task")
        .expect("second lease");
    second.release().await;
    assert!(!Path::new(&lock_path).exists());
}

#[tokio::test]
async fn replaced_lock_compromises_the_original_lease() {
    if utils_platform_is_windows() {
        return;
    }
    let root = tempfile::tempdir().expect("test directory");
    let socket_path = root
        .path()
        .join("daemon.sock")
        .to_string_lossy()
        .to_string();
    let lock_path = root.path().join("daemon.sock.lock");
    let lease = acquire_daemon_socket_path_lease(&socket_path)
        .await
        .expect("lease");

    let compromised = std::sync::Arc::new(tokio::sync::Notify::new());
    let listener_compromised = std::sync::Arc::clone(&compromised);
    let _unsubscribe = lease.on_compromised(std::sync::Arc::new(move |_error| {
        listener_compromised.notify_one();
    }));

    let staged_replacement = root.path().join("replacement.lock");
    std::fs::create_dir(&staged_replacement).expect("staged replacement lock");
    std::fs::remove_dir(&lock_path).expect("remove original lock");
    std::fs::rename(&staged_replacement, &lock_path).expect("replacement lock");

    tokio::time::timeout(Duration::from_millis(2500), compromised.notified())
        .await
        .expect("heartbeat detects replacement");
    assert!(lease.compromise().is_some(), "recorded compromise");
    lease.release().await;
    assert!(lock_path.is_dir(), "release preserved the replacement lock");
}

fn utils_platform_is_windows() -> bool {
    pi_coding_agent::utils::pi_user_agent::process_platform() == "win32"
}
