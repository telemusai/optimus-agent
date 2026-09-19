//! Server-side compatibility checks with isolated settings and no running service.

use super::*;
use pi_jev::config::{JevFeature, JevMode};

#[test]
fn jev_old_client_modes_and_combined_mode_preserve_independent_settings() {
    let directory = tempfile::tempdir().unwrap();
    let agent_dir = directory.path().join("agent");
    std::fs::create_dir_all(&agent_dir).unwrap();
    let socket_path = directory
        .path()
        .join("unused.sock")
        .to_string_lossy()
        .into_owned();
    let daemon = AgentDaemon::new(
        socket_path.clone(),
        DaemonModeOptions {
            socket_path: Some(socket_path),
            default_session_config: AgentSessionRuntimeConfig {
                cwd: Some(directory.path().to_string_lossy().into_owned()),
                agent_dir: Some(agent_dir.to_string_lossy().into_owned()),
                ..Default::default()
            },
            create_runtime: Arc::new(|_| {
                Box::pin(async { panic!("settings must not create a runtime") })
            }),
            worker: None,
        },
    );
    let store = daemon.jev_settings_store();
    let mut settings = store.load();
    settings.set_session_compaction_enabled("session-a", true);
    settings.set_session_feature("session-a", JevFeature::Verification, true);
    store.save(&settings).unwrap();

    for (raw, expected) in [
        ("off", JevMode::Off),
        ("on", JevMode::Compare),
        ("compare", JevMode::Compare),
        ("active", JevMode::Active),
        ("compare-active", JevMode::CompareAndActive),
    ] {
        let requested = JevMode::parse(raw).unwrap();
        let (applied, effective) = daemon
            .jev_apply_session_mode("session-a", requested)
            .unwrap();
        assert!(applied);
        assert_eq!(effective, expected);
        assert_eq!(
            jev_mode_change_message(applied, effective),
            format!("Jev mode: {} (scope: this chat)", expected.as_str())
        );
        let saved = store.load();
        assert_eq!(saved.effective_mode("session-a"), expected);
        assert!(saved.effective_compaction_enabled("session-a"));
        assert!(saved.effective_features("session-a").verification);
    }
    assert!(!directory.path().join("unused.sock").exists());
}
