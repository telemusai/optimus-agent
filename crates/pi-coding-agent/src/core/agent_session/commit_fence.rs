//! Cancellation-safe serialized admission. A dropped waiter cannot unlink its predecessor.
use super::*;

impl AgentSession {
    pub(super) async fn acquire_commit_fence(self: &Arc<Self>, _owner_id: bool) -> Result<CommitFence, String> {
        self.pending_session_action_fence_waiters.fetch_add(1, Ordering::SeqCst);
        let (release_tx, release_rx) = tokio::sync::oneshot::channel::<()>();
        let (ready_tx, ready_rx) = tokio::sync::oneshot::channel::<Result<(), String>>();
        let (done_tx, done_rx) = tokio::sync::oneshot::channel::<()>();
        let previous = {
            let mut tail = self.session_action_commit_tail.lock().unwrap();
            tail.replace(Box::pin(async move { let _ = done_rx.await; Ok(()) }))
        };
        let owner = uuid::Uuid::new_v4().to_string();
        let session = self.clone();
        let owner_for_driver = owner.clone();
        // The driver owns the predecessor even if its admission caller disappears.
        tokio::spawn(async move {
            let result = match previous { Some(previous) => previous.await, None => Ok(()) };
            session.pending_session_action_fence_waiters.fetch_sub(1, Ordering::SeqCst);
            if result.is_ok() {
                *session.session_action_commit_owner.lock().unwrap() = Some(owner_for_driver.clone());
            }
            session.notify_session_input_checkpoint_change();
            let _ = ready_tx.send(result);
            let _ = release_rx.await;
            let mut active = session.session_action_commit_owner.lock().unwrap();
            if active.as_deref() == Some(owner_for_driver.as_str()) { *active = None; }
            drop(active);
            let _ = done_tx.send(());
            session.notify_session_input_checkpoint_change();
        });
        // Dropping the acquisition future drops release_tx, but the driver still joins its predecessor.
        let result = ready_rx.await.map_err(|_| "Admission fence owner failed".to_string())?;
        result?;
        let sender = Mutex::new(Some(release_tx));
        Ok(CommitFence {
            owner: Some(owner),
            release: Some(Arc::new(move || { if let Some(sender) = sender.lock().unwrap().take() { let _ = sender.send(()); } })),
        })
    }
}

impl Drop for CommitFence {
    fn drop(&mut self) { self.release(); }
}
