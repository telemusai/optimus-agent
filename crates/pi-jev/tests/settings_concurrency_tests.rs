use pi_jev::{JevMode, JevSettingsStore};

#[test]
fn stale_settings_cannot_overwrite_another_sessions_change() {
    let dir = tempfile::tempdir().unwrap();
    let first = JevSettingsStore::new(dir.path());
    let second = JevSettingsStore::new(dir.path());
    let mut a = first.load();
    let mut b = second.load();
    a.set_session_mode("session-a", JevMode::Compare);
    b.set_session_mode("session-b", JevMode::Compare);
    first.save(&a).unwrap();
    assert!(second.save(&b).unwrap_err().to_string().contains("reload"));
    let mut b = second.load();
    b.set_session_mode("session-b", JevMode::Compare);
    second.save(&b).unwrap();
    let settings = first.load();
    assert_eq!(settings.session_mode("session-a"), Some(JevMode::Compare));
    assert_eq!(settings.session_mode("session-b"), Some(JevMode::Compare));
    assert!(!std::fs::read_to_string(first.path()).unwrap().contains("loaded_generation"));
}

#[test]
fn corrupt_settings_remain_untouched_on_save() {
    let dir = tempfile::tempdir().unwrap();
    let store = JevSettingsStore::new(dir.path());
    std::fs::create_dir_all(store.path().parent().unwrap()).unwrap();
    std::fs::write(store.path(), b"broken settings").unwrap();
    let mut settings = store.load();
    assert_eq!(settings.effective_mode("session-a"), JevMode::Off);
    settings.set_session_mode("session-a", JevMode::Compare);
    assert!(store.save(&settings).is_err());
    assert_eq!(std::fs::read(store.path()).unwrap(), b"broken settings");
}

#[test]
fn settings_writer_child() {
    let Ok(path) = std::env::var("JEV_TEST_SETTINGS_DIR") else { return; };
    let prefix = std::env::var("JEV_TEST_WRITER_ID").unwrap();
    let store = JevSettingsStore::new(path);
    for index in 0..10 {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            let mut settings = store.load();
            settings.set_session_mode(&format!("{prefix}-{index}"), JevMode::Compare);
            if store.save(&settings).is_ok() { break; }
            assert!(std::time::Instant::now() < deadline, "writer failed to settle");
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }
}

#[test]
fn separate_processes_preserve_all_updates_after_conflict_reload() {
    let dir = tempfile::tempdir().unwrap();
    let mut children = Vec::new();
    for index in 0..4 {
        children.push(std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "settings_writer_child"])
            .env("JEV_TEST_SETTINGS_DIR", dir.path())
            .env("JEV_TEST_WRITER_ID", format!("writer-{index}"))
            .stdout(std::process::Stdio::null()).spawn().unwrap());
    }
    for mut child in children { assert!(child.wait().unwrap().success()); }
    let store = JevSettingsStore::new(dir.path());
    assert_eq!(store.load().sessions.len(), 40);
    assert!(std::fs::read_dir(store.path().parent().unwrap()).unwrap()
        .all(|entry| !entry.unwrap().path().extension().is_some_and(|extension| extension == "tmp")));
}
