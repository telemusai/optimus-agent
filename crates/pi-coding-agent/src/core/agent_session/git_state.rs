//! Probe Git off the executor and outside the session lock.
use crate::core::session_manager::{capture_git_context, GitContext, SessionManager};
use std::sync::{Arc, Mutex};

pub(super) async fn record(manager: Arc<Mutex<SessionManager>>) {
    record_with(manager, |cwd| capture_git_context(&cwd)).await;
}

async fn record_with(
    manager: Arc<Mutex<SessionManager>>,
    capture: impl FnOnce(String) -> Option<GitContext> + Send + 'static,
) {
    let identity = {
        let manager = manager.lock().unwrap();
        if !manager.is_persisted() {
            return;
        }
        (
            manager.get_cwd(),
            manager.get_session_id(),
            manager.get_session_file(),
            manager.get_leaf_id(),
        )
    };
    let cwd = identity.0.clone();
    let Ok(Some(git)) = tokio::task::spawn_blocking(move || capture(cwd)).await else {
        return;
    };
    let mut manager = manager.lock().unwrap();
    // A delayed probe must not append metadata to a replaced session or branch.
    if identity
        == (
            manager.get_cwd(),
            manager.get_session_id(),
            manager.get_session_file(),
            manager.get_leaf_id(),
        )
    {
        let _ = manager.record_captured_git_state(&git);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn git_probe_releases_session_lock_and_discards_result_after_session_switch() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().to_str().unwrap();
        let manager = Arc::new(Mutex::new(
            SessionManager::create(path, Some(path)).unwrap(),
        ));
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let worker = tokio::spawn(record_with(manager.clone(), move |_| {
            started_tx.send(()).unwrap();
            release_rx
                .recv_timeout(std::time::Duration::from_secs(5))
                .unwrap();
            Some(GitContext {
                commit: Some("stale".into()),
                ..Default::default()
            })
        }));
        started_rx.await.unwrap();
        {
            let mut session = manager
                .try_lock()
                .expect("Git must not hold the session lock");
            session.new_session(None).unwrap();
        }
        release_tx.send(()).unwrap();
        worker.await.unwrap();
        assert!(manager.lock().unwrap().get_leaf_id().is_none());
        let git = GitContext {
            commit: Some("current".into()),
            ..Default::default()
        };
        for _ in 0..2 {
            let git = git.clone();
            record_with(manager.clone(), move |_| Some(git)).await;
        }
        let mut session = manager.lock().unwrap();
        assert!(session.get_leaf_id().is_some());
        assert_eq!(
            session.record_captured_git_state(&git).unwrap(),
            None,
            "capture is deduplicated"
        );
    }

    #[tokio::test]
    async fn transient_sessions_never_spawn_git_probes() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().to_str().unwrap();
        let manager = Arc::new(Mutex::new(
            SessionManager::in_memory(Some(path), Some(path)).unwrap(),
        ));
        record_with(manager, |_| panic!("in-memory session must not probe Git")).await;
    }
}
