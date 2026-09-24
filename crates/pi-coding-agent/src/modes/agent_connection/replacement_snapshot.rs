use super::*;

pub(super) struct ReplacementSnapshot {
    session_file: std::path::PathBuf,
    result: tokio::sync::watch::Sender<Option<Result<(), String>>>,
}

impl ReplacementSnapshot {
    fn new(session_file: std::path::PathBuf) -> Self {
        Self {
            session_file,
            result: tokio::sync::watch::channel(None).0,
        }
    }

    pub(super) fn finish(&self, result: Result<(), String>) {
        self.result.send_if_modified(|current| {
            if current.is_some() {
                return false;
            }
            *current = Some(result);
            true
        });
    }

    fn matches(&self, session_file: Option<&str>) -> bool {
        let Some(actual) = session_file else {
            return false;
        };
        let canonical =
            |path: &std::path::Path| path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        canonical(&self.session_file) == canonical(std::path::Path::new(actual))
    }

    async fn wait(&self) -> Result<(), String> {
        let mut receiver = self.result.subscribe();
        loop {
            let result = receiver.borrow_and_update().clone();
            if let Some(result) = result {
                return result;
            }
            receiver
                .changed()
                .await
                .map_err(|_| "Session replacement was abandoned".to_string())?;
        }
    }
}

impl DaemonAgentConnection {
    pub(super) fn begin_replacement_snapshot(&self, path: &str) -> Arc<ReplacementSnapshot> {
        let mut path = std::path::PathBuf::from(path);
        if !path.is_absolute() {
            if let Some(snapshot) = self.latest_snapshot.lock().unwrap().as_ref() {
                path = std::path::Path::new(&snapshot.state.cwd).join(path);
            }
        }
        let pending = Arc::new(ReplacementSnapshot::new(path));
        if let Some(previous) = self
            .pending_replacement_snapshot
            .lock()
            .unwrap()
            .replace(pending.clone())
        {
            previous.finish(Err(
                "Session switch superseded by a newer switch".to_string()
            ));
        }
        pending
    }

    pub(super) fn clear_replacement_snapshot(&self, pending: &Arc<ReplacementSnapshot>) {
        let mut current = self.pending_replacement_snapshot.lock().unwrap();
        if current
            .as_ref()
            .is_some_and(|current| Arc::ptr_eq(current, pending))
        {
            current.take();
        }
    }

    pub(super) fn fail_replacement_snapshot(&self, error: String) {
        if let Some(pending) = self.pending_replacement_snapshot.lock().unwrap().take() {
            pending.finish(Err(error));
        }
    }

    pub(super) fn observe_replacement_snapshot(&self, event: &AgentConnectionEvent) {
        let session_file = match event {
            AgentConnectionEvent::SessionReplaced { state, .. } => state.session_file.as_deref(),
            AgentConnectionEvent::SessionResynced { snapshot } => {
                snapshot.state.session_file.as_deref()
            }
            AgentConnectionEvent::Closed { error } => {
                self.fail_replacement_snapshot(error.clone().unwrap_or_else(|| {
                    "Daemon connection closed during session switch".to_string()
                }));
                return;
            }
            _ => return,
        };
        if let Some(pending) = self.pending_replacement_snapshot.lock().unwrap().clone() {
            if pending.matches(session_file) {
                pending.finish(Ok(()));
            }
        }
    }

    pub(super) async fn wait_for_replacement_snapshot(
        &self,
        pending: &Arc<ReplacementSnapshot>,
    ) -> Result<(), String> {
        let timeout = self
            .options
            .lock()
            .unwrap()
            .snapshot_timeout_ms
            .unwrap_or(DAEMON_SNAPSHOT_TIMEOUT_MS);
        match tokio::time::timeout(Duration::from_millis(timeout), pending.wait()).await {
            Ok(result) => result,
            Err(_) => {
                let error =
                    "Timed out waiting for the selected session's replacement snapshot".to_string();
                pending.finish(Err(error.clone()));
                self.clear_replacement_snapshot(pending);
                Err(error)
            }
        }
    }
}

pub(super) struct ReplacementSnapshotGuard {
    connection: DaemonAgentConnection,
    pending: Arc<ReplacementSnapshot>,
}

impl ReplacementSnapshotGuard {
    pub(super) fn new(
        connection: &DaemonAgentConnection,
        pending: Arc<ReplacementSnapshot>,
    ) -> Self {
        Self {
            connection: connection.clone(),
            pending,
        }
    }
}

impl Drop for ReplacementSnapshotGuard {
    fn drop(&mut self) {
        self.pending.finish(Err(
            "Session switch cancelled before replacement completed".to_string()
        ));
        self.connection.clear_replacement_snapshot(&self.pending);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path(name: &str) -> String {
        std::env::temp_dir()
            .join(format!("optimus-replacement-{name}.jsonl"))
            .to_string_lossy()
            .into_owned()
    }

    fn snapshot(name: &str, sequence: i64) -> DaemonSessionSnapshot {
        DaemonSessionSnapshot {
            state: AgentConnectionState {
                session_id: name.into(),
                active_session_id: Some("active".into()),
                session_file: Some(path(name)),
                ..Default::default()
            },
            last_event_sequence: Some(sequence),
            last_event_cursor: Some(DaemonEventCursor {
                generation: "generation".into(),
                sequence,
            }),
            ..Default::default()
        }
    }

    fn connection(timeout: u64) -> DaemonAgentConnection {
        let client = crate::modes::daemon::daemon_client::DaemonClient::create(
            "unused-replacement-test-socket",
        );
        let transport = Arc::new(crate::main_entry::MainEntryDaemonTransport::new(client));
        let connection = DaemonAgentConnection::new(
            transport,
            "active".into(),
            DaemonAgentConnectionOptions {
                snapshot_timeout_ms: Some(timeout),
                defer_session_events: false,
                ..Default::default()
            },
        );
        connection.apply_replacement_snapshot(&snapshot("old", 0), None);
        connection
    }

    #[tokio::test]
    async fn snapshot_reader_waits_for_complete_replacement_without_fetching_history_again() {
        let connection = connection(1000);
        let pending = connection.begin_replacement_snapshot(&path("target"));
        let reader = connection.clone();
        let mut reading = tokio::spawn(async move { reader.get_initial_snapshot().await });
        assert!(
            tokio::time::timeout(Duration::from_millis(10), &mut reading)
                .await
                .is_err()
        );
        connection
            .handle_daemon_message(DaemonOutbound::SessionSnapshotBegin {
                active_session_id: "active".into(),
                snapshot_id: "replacement".into(),
                snapshot: snapshot("target", 1),
                message_count: 1,
                purpose: Some("replacement".into()),
            })
            .await
            .unwrap();
        connection
            .handle_daemon_message(DaemonOutbound::SessionSnapshotChunk {
                active_session_id: "active".into(),
                snapshot_id: "replacement".into(),
                index: 0,
                messages: vec![serde_json::from_value(
                    json!({"role":"user","content":"restored chat","timestamp":1}),
                )
                .unwrap()],
            })
            .await
            .unwrap();
        assert!(
            pending.result.borrow().is_none(),
            "partial chunks cannot complete a switch"
        );
        connection
            .handle_daemon_message(DaemonOutbound::SessionSnapshotEnd {
                active_session_id: "active".into(),
                snapshot_id: "replacement".into(),
                chunk_count: 1,
                last_event_sequence: 1,
                last_event_cursor: Some(DaemonEventCursor {
                    generation: "generation".into(),
                    sequence: 1,
                }),
            })
            .await
            .unwrap();
        let result = tokio::time::timeout(Duration::from_secs(2), reading)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(result.state.session_id, "target");
        assert_eq!(result.messages.len(), 1);
        connection
            .wait_for_replacement_snapshot(&pending)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn superseded_switch_cannot_complete_or_clear_the_new_target() {
        let connection = connection(1000);
        let old = connection.begin_replacement_snapshot(&path("old"));
        let new = connection.begin_replacement_snapshot(&path("new"));
        assert!(old.wait().await.unwrap_err().contains("superseded"));
        connection.clear_replacement_snapshot(&old);
        connection
            .emit(AgentConnectionEvent::SessionReplaced {
                state: snapshot("old", 1).state,
                messages: Vec::new(),
            })
            .await;
        assert!(new.result.borrow().is_none());
        assert!(Arc::ptr_eq(
            connection
                .pending_replacement_snapshot
                .lock()
                .unwrap()
                .as_ref()
                .unwrap(),
            &new
        ));
        connection
            .emit(AgentConnectionEvent::SessionReplaced {
                state: snapshot("new", 2).state,
                messages: Vec::new(),
            })
            .await;
        connection
            .wait_for_replacement_snapshot(&new)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn reconnect_resync_can_finish_an_interrupted_replacement() {
        let connection = connection(1000);
        let pending = connection.begin_replacement_snapshot(&path("target"));
        connection.reject_snapshot_assemblies("transport interrupted".to_string());
        assert!(pending.result.borrow().is_none());
        let snapshot = map_daemon_session_snapshot(&snapshot("target", 2), None).unwrap();
        connection
            .emit(AgentConnectionEvent::SessionResynced { snapshot })
            .await;
        connection
            .wait_for_replacement_snapshot(&pending)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn timeout_close_and_dropped_switch_release_waiters() {
        let connection = connection(5);
        let timed = connection.begin_replacement_snapshot(&path("target"));
        assert!(connection
            .wait_for_replacement_snapshot(&timed)
            .await
            .unwrap_err()
            .contains("Timed out"));
        assert!(connection
            .pending_replacement_snapshot
            .lock()
            .unwrap()
            .is_none());
        let closed = connection.begin_replacement_snapshot(&path("target"));
        connection
            .emit(AgentConnectionEvent::Closed {
                error: Some("transport closed".into()),
            })
            .await;
        assert_eq!(closed.wait().await.unwrap_err(), "transport closed");
        let dropped = connection.begin_replacement_snapshot(&path("target"));
        drop(ReplacementSnapshotGuard::new(&connection, dropped.clone()));
        assert!(dropped.wait().await.unwrap_err().contains("cancelled"));
        assert!(connection
            .pending_replacement_snapshot
            .lock()
            .unwrap()
            .is_none());
    }
}
