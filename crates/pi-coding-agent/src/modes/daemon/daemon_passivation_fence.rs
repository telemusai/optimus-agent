use super::*;

impl AgentDaemon {
    pub(super) async fn fenced_passivation_snapshot(
        self: &Arc<Self>,
        state: &Arc<StdMutex<ActiveSessionState>>,
        active_session_id: &str,
        snapshot: impl std::future::Future<Output = SessionPassivationSnapshot>,
    ) -> Option<SessionPassivationSnapshot> {
        let current = || {
            !self.shutting_down.load(Ordering::SeqCst)
                && self
                    .update_restart
                    .lock()
                    .expect("update restart poisoned")
                    .is_none()
                && self
                    .sessions
                    .lock()
                    .expect("sessions poisoned")
                    .get(active_session_id)
                    .is_some_and(|entry| Arc::ptr_eq(&entry.state, state))
        };
        if !current() {
            return None;
        }
        let snapshot = snapshot.await;
        // Collecting descendants may yield to shutdown, restart or session replacement.
        current().then_some(snapshot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn passivation_discards_delayed_snapshot_after_shutdown_restart_or_replacement() {
        for change in ["none", "shutdown", "restart", "replacement"] {
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().to_string_lossy().into_owned();
            let socket = directory
                .path()
                .join("daemon.sock")
                .to_string_lossy()
                .into_owned();
            let daemon = AgentDaemon::new(
                socket.clone(),
                DaemonModeOptions {
                    socket_path: Some(socket),
                    default_session_config: AgentSessionRuntimeConfig {
                        cwd: Some(path.clone()),
                        agent_dir: Some(path),
                        ..Default::default()
                    },
                    create_runtime: Arc::new(|_| Box::pin(async { panic!("offline fixture") })),
                    worker: None,
                },
            );
            let state = Arc::new(StdMutex::new(ActiveSessionState::new(
                "child",
                Default::default(),
            )));
            let entry = |state| {
                Arc::new(DaemonSessionState {
                    state,
                    session: Arc::new(MissingSession::new("child")),
                    runtime_metadata: Default::default(),
                    snapshot_boundary: StdMutex::new(None),
                })
            };
            daemon
                .sessions
                .lock()
                .unwrap()
                .insert("child".into(), entry(state.clone()));
            let (release, ready) = oneshot::channel();
            let snapshot = async {
                ready.await.unwrap();
                SessionPassivationSnapshot {
                    eviction: SessionEvictionSnapshot {
                        is_session_active: false,
                        attached_clients: 0,
                        has_registered_cron_job: false,
                        last_activity_at: 0.0,
                    },
                    has_parent: true,
                    has_non_passive_descendants: false,
                    is_hydrating: false,
                }
            };
            let pending = daemon.fenced_passivation_snapshot(&state, "child", snapshot);
            tokio::pin!(pending);
            assert!(futures::poll!(&mut pending).is_pending());
            match change {
                "shutdown" => daemon.shutting_down.store(true, Ordering::SeqCst),
                "restart" => {
                    *daemon.update_restart.lock().unwrap() = Some(UpdateRestartTransaction {
                        id: 1,
                        owner: None,
                        abort: tokio_util::sync::CancellationToken::new(),
                        deadline_expired: Arc::new(AtomicBool::new(false)),
                        phase: "preparing".into(),
                        manifest: None,
                        deferred_client_env: Vec::new(),
                    })
                }
                "replacement" => {
                    daemon.sessions.lock().unwrap().insert(
                        "child".into(),
                        entry(Arc::new(StdMutex::new(ActiveSessionState::new(
                            "child",
                            Default::default(),
                        )))),
                    );
                }
                _ => {}
            }
            release.send(()).unwrap();
            assert_eq!(pending.await.is_some(), change == "none", "{change}");
        }
    }
}
