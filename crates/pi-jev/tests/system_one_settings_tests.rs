use pi_jev::config::{inherit_mode, JevFeature, JevFeatures, JevMode, JevSettings, JevSettingsStore, ModeScope};

#[test]
fn all_modes_round_trip_and_on_remains_compare() {
    for (mode, name, compare, active) in [
        (JevMode::Off, "off", false, false),
        (JevMode::Compare, "compare", true, false),
        (JevMode::Active, "active", false, true),
        (JevMode::CompareAndActive, "compare-active", true, true),
    ] {
        assert_eq!(mode.as_str(), name);
        assert_eq!(mode.allows_compare(), compare);
        assert_eq!(mode.allows_active(), active);
        assert_eq!(mode.is_enabled(), compare || active);
        assert_eq!(JevMode::parse(name), Some(mode));
        assert_eq!(serde_json::to_string(&mode).unwrap(), format!("\"{name}\""));
        assert_eq!(serde_json::from_str::<JevMode>(&format!("\"{name}\"")).unwrap(), mode);
    }
    assert_eq!(JevMode::parse("on"), Some(JevMode::Compare));
    for name in ["both", "compare_and_active", "compare-and-active"] {
        assert_eq!(JevMode::parse(name), Some(JevMode::CompareAndActive));
    }
    assert!(JevMode::parse("active --force").is_none());
}

#[test]
fn defaults_are_explicit_and_credentials_grant_nothing() {
    let mut settings = JevSettings::default();
    settings.credential_configured = true;
    assert_eq!(settings.effective_mode("s"), JevMode::Off);
    assert!(!settings.effective_compaction_enabled("s"));
    for feature in JevFeature::ALL {
        assert_eq!(settings.features.enabled(feature), matches!(feature, JevFeature::ToolRequirement | JevFeature::Complexity));
    }
    settings.validate().unwrap();
    assert!(!settings.wants_observer());
    settings.global_default = Some(JevMode::CompareAndActive);
    assert!(settings.wants_observer());
}

#[test]
fn optional_fields_load_without_arming_new_features() {
    let settings: JevSettings = serde_json::from_str(r#"{"schema_version":1,"global_default":"active","sessions":{"s":{"mode":"compare"}}}"#).unwrap();
    assert_eq!(settings.effective_mode("s"), JevMode::Compare);
    assert_eq!(settings.effective_mode("other"), JevMode::Active);
    assert_eq!(settings.effective_features("s"), JevFeatures::default());
    assert!(!settings.effective_compaction_enabled("s"));
    settings.validate().unwrap();
}

#[test]
fn mode_feature_and_compaction_controls_are_independent() {
    let mut settings = JevSettings::with_global_default(JevMode::CompareAndActive);
    settings.compaction_enabled = true;
    settings.set_session_compaction_enabled("s", false);
    settings.set_session_feature("s", JevFeature::MemoryRelevance, true);
    assert_eq!(settings.effective_mode("s"), JevMode::CompareAndActive);
    assert_eq!(settings.effective_compaction_with_scope("s"), (false, ModeScope::Session));
    assert_eq!(settings.effective_compaction_with_scope("other"), (true, ModeScope::GlobalDefault));
    settings.set_session_mode("s", JevMode::Off);
    assert!(settings.effective_features("s").memory_relevance);
    assert!(!settings.effective_compaction_enabled("s"));
    settings.set_session_mode("s", JevMode::Compare);
    assert!(settings.effective_features("s").memory_relevance);
    assert!(!settings.effective_compaction_enabled("s"));
    assert!(!settings.effective_features("other").memory_relevance);
}

#[test]
fn every_mode_obeys_explicit_session_precedence() {
    for mode in [JevMode::Off, JevMode::Compare, JevMode::Active, JevMode::CompareAndActive] {
        for global in [JevMode::Off, JevMode::Compare, JevMode::Active, JevMode::CompareAndActive] {
            let mut settings = JevSettings::with_global_default(global);
            settings.set_session_mode("s", mode);
            assert_eq!(settings.effective_mode("s"), mode);
            assert_eq!(settings.effective_mode("other"), global);
        }
    }
}

#[test]
fn inheritance_snapshots_independent_controls_and_preserves_child_overrides() {
    let mut settings = JevSettings::with_global_default(JevMode::Off);
    settings.set_session_mode("parent", JevMode::CompareAndActive);
    settings.set_session_compaction_enabled("parent", true);
    settings.set_session_feature("parent", JevFeature::Verification, true);
    inherit_mode(&mut settings, "child", "parent", None);
    assert_eq!(settings.effective_mode("child"), JevMode::CompareAndActive);
    assert!(settings.effective_features("child").verification);
    assert!(settings.effective_compaction_enabled("child"));
    settings.set_session_compaction_enabled("override", false);
    settings.set_session_feature("override", JevFeature::Verification, false);
    inherit_mode(&mut settings, "override", "parent", Some(JevMode::Compare));
    assert_eq!(settings.effective_mode("override"), JevMode::Compare);
    assert!(!settings.effective_features("override").verification);
    assert!(!settings.effective_compaction_enabled("override"));
    settings.set_session_feature("parent", JevFeature::Verification, false);
    assert!(settings.effective_features("child").verification);
}

#[test]
fn policy_and_session_controls_survive_store_reload() {
    let dir = tempfile::tempdir().unwrap();
    let store = JevSettingsStore::new(dir.path());
    let mut settings = store.load();
    settings.set_session_mode("s", JevMode::CompareAndActive);
    settings.set_session_compaction_enabled("s", true);
    settings.set_session_feature("s", JevFeature::ToolCandidates, true);
    settings.compaction.keep_threshold = 0.7;
    settings.compaction.preserve_recent_messages = 10;
    settings.filtering.optional_tool_names.push("read_file".into());
    store.save(&settings).unwrap();
    let reloaded = store.load();
    assert_eq!(reloaded.effective_mode("s"), JevMode::CompareAndActive);
    assert!(reloaded.effective_features("s").tool_candidates);
    assert!(reloaded.effective_compaction_enabled("s"));
    assert_eq!(reloaded.compaction.keep_threshold, 0.7);
    assert_eq!(reloaded.compaction.preserve_recent_messages, 10);
    assert_eq!(reloaded.filtering.optional_tool_names, ["read_file"]);
    assert!(!reloaded.looks_like_it_contains_a_secret());
}

#[test]
fn invalid_numeric_policies_fail_closed_without_overwriting_the_file() {
    let dir = tempfile::tempdir().unwrap();
    let store = JevSettingsStore::new(dir.path());
    let mut settings = store.load();
    settings.compaction.keep_threshold = f64::NAN;
    assert!(store.save(&settings).is_err());
    assert!(!store.path().exists());
    let raw = r#"{"schema_version":1,"global_default":"active","compaction":{"keep_threshold":2.0}}"#;
    std::fs::create_dir_all(store.path().parent().unwrap()).unwrap();
    std::fs::write(store.path(), raw).unwrap();
    let mut loaded = store.load();
    assert_eq!(loaded.effective_mode("s"), JevMode::Off);
    loaded.set_session_mode("s", JevMode::Active);
    assert!(store.save(&loaded).is_err());
    assert_eq!(std::fs::read_to_string(store.path()).unwrap(), raw);
}

#[test]
fn features_are_individually_addressable_and_reject_unknown_names() {
    let mut features = JevFeatures::default();
    for feature in JevFeature::ALL {
        assert_eq!(JevFeature::parse(feature.as_str()), Some(feature));
        let before = features.enabled(feature);
        features.set(feature, !before);
        assert_eq!(features.enabled(feature), !before);
        features.set(feature, before);
    }
    assert_eq!(features, JevFeatures::default());
    assert!(JevFeature::parse("model").is_none());
    assert!(JevFeature::parse("permissions").is_none());
}
