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
        // jev_apply_session_mode also reports whether the write went through
        // the full-jev emergency-exit path; fresh store -> false here.
        let (applied, effective, _full_jev_exit) = daemon
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

#[test]
fn jev_full_jev_overlay_persists_and_masks_saved_modes_through_the_daemon_store() {
    // The overlay lives in the same settings file the daemon serves, so a
    // daemon-side load resolves it exactly like the interactive path: above
    // every saved session, global and inherited override.
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
    settings.set_session_mode("session-a", JevMode::Off);
    settings.set_session_compaction_enabled("session-a", false);
    settings.set_session_feature("session-a", JevFeature::Verification, true);
    store.save(&settings).unwrap();

    // Install through the same store the interactive command uses.
    let mut settings = store.load();
    assert!(settings.full_jev_install());
    store.save(&settings).unwrap();

    let masked = store.load();
    assert!(masked.full_jev_active());
    assert_eq!(
        masked.effective_mode("session-a"),
        JevMode::CompareAndActive
    );
    assert!(masked.effective_compaction_enabled("session-a"));
    assert!(masked.effective_features("session-a").verification);
    assert!(masked.effective_features("session-a").code_search_reranking);
    assert!(masked.effective_features("session-a").line_find);
    // The saved decisions are still on disk, only masked.
    let saved = masked.sessions.get("session-a").unwrap().clone();
    assert_eq!(saved.mode, Some(JevMode::Off));
    assert_eq!(saved.compaction_enabled, Some(false));

    // Remove: the saved decisions resolve again, unchanged.
    let mut settings = store.load();
    assert!(settings.full_jev_remove());
    store.save(&settings).unwrap();
    let restored = store.load();
    assert!(!restored.full_jev_active());
    assert_eq!(restored.effective_mode("session-a"), JevMode::Off);
    assert!(!restored.effective_compaction_enabled("session-a"));
    assert!(restored.effective_features("session-a").verification);
    assert!(
        !restored
            .effective_features("session-a")
            .code_search_reranking
    );
    assert_eq!(
        restored.sessions.get("session-a").unwrap().mode,
        Some(JevMode::Off)
    );
    assert!(!directory.path().join("unused.sock").exists());
}
