//! Session-local OCC preferences. Subagents inherit dynamically; independent chats do not.

use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};

const SCHEMA: &str = "optimus.occ-session.v1";
static WRITE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Preference {
    schema: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    occ2_enabled: Option<bool>,
    // Each lane inherits independently. An absent latent override leaves the
    // already-verified global opt-in authoritative; it is not an implicit veto.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    occ3_enabled: Option<bool>,
}

pub fn preference_path(session_file: &str) -> PathBuf {
    PathBuf::from(crate::core::session_manager::get_session_artifact_path_for_file(session_file, None)).join("occ.json")
}

/// A chat's resolved preference with provenance: which chat's `occ.json`
/// decided (`None` for the default), so daemon-owned reads can report whether
/// the value was inherited from a subagent ancestry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resolution {
    pub occ2_enabled: bool,
    pub occ3_enabled: bool,
    pub source_session_id: Option<String>,
    pub occ3_source_session_id: Option<String>,
}

impl Default for Resolution {
    fn default() -> Self {
        Resolution { occ2_enabled: true, occ3_enabled: false, source_session_id: None, occ3_source_session_id: None }
    }
}

fn session_id_of(session_file: &str) -> String {
    // Mirrors the id derivation in `get_session_artifact_path_for_file`
    // (session_manager.rs): basename minus the `.jsonl` suffix.
    let name = Path::new(session_file)
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_default();
    name.strip_suffix(".jsonl").unwrap_or(&name).to_string()
}

/// Missing preferences are on. A child's nearest explicit preference wins.
/// Only bounded headers are read, never the chat transcript; no process cache
/// means existing workers see the next toggle without being restarted.
pub fn load(session_file: &str) -> Result<bool, String> {
    resolve_inner(session_file, true).map(|resolution| resolution.occ2_enabled)
}

/// `load` with provenance for the daemon-owned `occ_get_preferences`
/// response: the effective value plus the chat whose preference decided.
pub fn resolve(session_file: &str) -> Result<Resolution, String> {
    resolve_inner(session_file, false)
}

fn resolve_inner(session_file: &str, occ2_only: bool) -> Result<Resolution, String> {
    let mut file = PathBuf::from(session_file);
    let mut visited = BTreeSet::new();
    let mut resolution = Resolution::default();
    for _ in 0..64 {
        if !visited.insert(file.clone()) { return Err("OCC session ancestry contains a cycle".into()); }
        if let Some(value) = read_preference(&file.to_string_lossy())? {
            let id = session_id_of(&file.to_string_lossy());
            if resolution.source_session_id.is_none() {
                if let Some(enabled) = value.occ2_enabled {
                    resolution.occ2_enabled = enabled;
                    resolution.source_session_id = Some(id.clone());
                }
            }
            if resolution.occ3_source_session_id.is_none() {
                if let Some(enabled) = value.occ3_enabled {
                    resolution.occ3_enabled = enabled;
                    resolution.occ3_source_session_id = Some(id);
                }
            }
            if resolution.source_session_id.is_some() && (occ2_only || resolution.occ3_source_session_id.is_some()) {
                return Ok(resolution);
            }
        }
        let reader = match File::open(&file) {
            Ok(reader) => reader,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(resolution),
            Err(error) => return Err(error.to_string()),
        };
        let mut header = Vec::new();
        BufReader::new(reader.take(65537)).read_until(b'\n', &mut header).map_err(|e| e.to_string())?;
        if header.len() > 65536 { return Err("OCC session header is too large".into()); }
        let header: serde_json::Value = serde_json::from_slice(&header).map_err(|e| e.to_string())?;
        // Depth-zero forks are independent chats even when parentSession exists.
        if header["rlmDepth"].as_u64().unwrap_or(0) == 0 { return Ok(resolution); }
        let Some(parent) = header["parentSession"].as_str().filter(|s| !s.is_empty()) else { return Ok(resolution); };
        let parent = Path::new(parent);
        file = if parent.is_absolute() { parent.to_path_buf() }
            else { file.parent().unwrap_or_else(|| Path::new(".")).join(parent) };
    }
    Err("OCC session ancestry exceeds the supported depth".into())
}

fn read_preference(session_file: &str) -> Result<Option<Preference>, String> {
    let reader = match File::open(preference_path(session_file)) {
        Ok(reader) => reader,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.to_string()),
    };
    let mut bytes = Vec::new();
    reader.take(4097).read_to_end(&mut bytes).map_err(|e| e.to_string())?;
    if bytes.len() > 4096 { return Err("OCC session preference is too large".into()); }
    let value: Preference = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    if value.schema != SCHEMA { return Err("Unsupported OCC session preference schema".into()); }
    Ok(Some(value))
}

/// The `occ_get_preferences` response payload, built where the preference
/// lives so the daemon and an in-process host report identical shapes:
/// `occ2Enabled`, `inherited` (decided by an ancestor chat) and
/// `sourceSessionId` (the deciding chat, or null for the default).
pub fn preferences(session_file: &str) -> Result<serde_json::Value, String> {
    let resolution = resolve_inner(session_file, true)?;
    let inherited = resolution
        .source_session_id
        .as_deref()
        .is_some_and(|source| source != session_id_of(session_file));
    Ok(serde_json::json!({
        "occ2Enabled": resolution.occ2_enabled,
        "inherited": inherited,
        "sourceSessionId": resolution.source_session_id,
    }))
}

/// The `occ_set_preferences` response payload after persisting `enabled`.
/// The daemon-authoritative echo comes from a fresh resolve, so a client can
/// never report a value the storage did not accept.
pub fn set_preferences(session_file: &str, enabled: bool) -> Result<serde_json::Value, String> {
    save(session_file, enabled)?;
    let resolution = resolve_inner(session_file, true)?;
    Ok(serde_json::json!({
        "occ2Enabled": resolution.occ2_enabled,
        "applied": resolution.occ2_enabled == enabled,
        "scope": "this chat",
        "effectiveForNewMessages": true,
        "sourceSessionId": resolution.source_session_id,
    }))
}
/// Runtime sends fail closed; interactive commands surface load errors.
pub fn enabled(session_file: Option<&str>) -> bool {
    session_file.and_then(|file| load(file).ok()).unwrap_or(false)
}

pub fn save(session_file: &str, enabled: bool) -> Result<(), String> {
    let _guard = WRITE_LOCK.lock().map_err(|_| "OCC preference writer unavailable".to_string())?;
    // Preserve the historical heal-on-set behavior, but never turn a damaged
    // latent override into a global opt-in. Valid files keep the other lane.
    let mut value = preference_for_write(session_file);
    value.occ2_enabled = Some(enabled);
    write_preference(session_file, &value)
}

pub fn save_latent(session_file: &str, enabled: bool) -> Result<(), String> {
    let _guard = WRITE_LOCK.lock().map_err(|_| "OCC preference writer unavailable".to_string())?;
    // Unlike the legacy OCC2 healing contract, latent-only writes must not
    // silently replace an unreadable OCC2 override.
    let mut value = read_preference(session_file)?.unwrap_or(Preference {
        schema: SCHEMA.into(), occ2_enabled: None, occ3_enabled: None,
    });
    value.occ3_enabled = Some(enabled);
    write_preference(session_file, &value)
}

/// Host-owned latent setter echo; the legacy OCC2 JSON contract stays unchanged.
pub fn set_latent_preferences(session_file: &str, enabled: bool) -> Result<serde_json::Value, String> {
    save_latent(session_file, enabled)?;
    let resolution = resolve(session_file)?;
    Ok(serde_json::json!({
        "occ32Enabled": resolution.occ3_enabled,
        "applied": resolution.occ3_enabled == enabled,
        "scope": "this chat",
        "effectiveForNewOperations": true,
    }))
}

fn preference_for_write(session_file: &str) -> Preference {
    match read_preference(session_file) {
        Ok(Some(value)) => value,
        Ok(None) => Preference { schema: SCHEMA.into(), occ2_enabled: None, occ3_enabled: None },
        Err(_) => Preference { schema: SCHEMA.into(), occ2_enabled: None, occ3_enabled: Some(false) },
    }
}

/// Explicitly saves both existing lanes; single-lane setters preserve inheritance.
pub fn save_occ3(session_file: &str, occ2_enabled: bool, occ3_enabled: bool) -> Result<(), String> {
    let _guard = WRITE_LOCK.lock().map_err(|_| "OCC preference writer unavailable".to_string())?;
    write_preference(session_file, &Preference {
        schema: SCHEMA.into(), occ2_enabled: Some(occ2_enabled), occ3_enabled: Some(occ3_enabled),
    })
}

fn write_preference(session_file: &str, value: &Preference) -> Result<(), String> {
    if session_file.is_empty() { return Err("This chat has no saved session identity".into()); }
    let path = preference_path(session_file);
    let parent = path.parent().ok_or("Invalid OCC session preference path")?;
    fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    let mut pending = tempfile::NamedTempFile::new_in(parent).map_err(|e| e.to_string())?;
    serde_json::to_writer(pending.as_file_mut(), value).map_err(|e| e.to_string())?;
    pending.write_all(b"\n").map_err(|e| e.to_string())?;
    pending.as_file().sync_all().map_err(|e| e.to_string())?;
    pending.persist(path).map_err(|e| e.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn session(root: &Path, id: &str, parent: Option<&str>, depth: usize) -> String {
        let path = root.join(format!("{id}.jsonl"));
        fs::write(&path, format!("{}\n", json!({"type":"session","id":id,"rlmDepth":depth,"parentSession":parent}))).unwrap();
        path.to_str().unwrap().to_string()
    }

    #[test]
    fn occ_sessions_isolate_chats_resume_and_dynamically_inherit_for_existing_children() {
        let root = tempfile::tempdir().unwrap();
        let sessions = root.path().join("sessions");
        fs::create_dir(&sessions).unwrap();
        let a = session(&sessions, "a", None, 0);
        let b = session(&sessions, "b", None, 0);
        let child = session(&sessions, "child", Some(&a), 1);
        let nested = session(&sessions, "nested", Some(&child), 2);
        let fork = session(&sessions, "fork", Some(&a), 0);
        for file in [&a, &b, &child, &nested, &fork] { assert!(load(file).unwrap()); }
        save(&a, false).unwrap();
        assert!(!load(&a).unwrap());
        assert!(!load(&child).unwrap());
        assert!(!load(&nested).unwrap());
        assert!(load(&b).unwrap());
        assert!(load(&fork).unwrap());
        save(&a, true).unwrap();
        assert!(load(&a).unwrap()); // reopening reads the saved preference
        assert!(load(&b).unwrap());
        assert!(load(&fork).unwrap());
        assert!(load(&child).unwrap());
        assert!(load(&nested).unwrap());
        save(&child, false).unwrap();
        assert!(load(&a).unwrap());
        assert!(!load(&child).unwrap());
        assert!(!load(&nested).unwrap());
        save(&b, true).unwrap();
        save(&a, false).unwrap();
        assert!(load(&b).unwrap());
        assert!(!load(&a).unwrap());
        let new_chat = session(&sessions, "new", None, 0);
        assert!(load(&new_chat).unwrap());
        // Removing a child override re-inherits the parent, dynamically.
        assert!(!load(&child).unwrap());
        fs::remove_file(preference_path(&child)).unwrap();
        assert!(!load(&child).unwrap());
        save(&a, true).unwrap();
        assert!(load(&child).unwrap());
        assert!(load(&nested).unwrap());
    }

    #[test]
    fn occ2_and_latent_overrides_preserve_each_other_and_inherit_independently() {
        let root = tempfile::tempdir().unwrap();
        let sessions = root.path().join("sessions");
        fs::create_dir(&sessions).unwrap();
        let a = session(&sessions, "a", None, 0);
        let child = session(&sessions, "child", Some("a.jsonl"), 1);
        save(&a, false).unwrap();
        assert_eq!(resolve(&a).unwrap().occ3_source_session_id, None);
        save_latent(&a, true).unwrap();
        save(&child, true).unwrap();
        let inherited = resolve(&child).unwrap();
        assert!(inherited.occ2_enabled && inherited.occ3_enabled);
        assert_eq!(inherited.source_session_id.as_deref(), Some("child"));
        assert_eq!(inherited.occ3_source_session_id.as_deref(), Some("a"));
        save_latent(&a, false).unwrap();
        assert!(!resolve(&child).unwrap().occ3_enabled);
        save_latent(&child, true).unwrap();
        save(&child, false).unwrap();
        assert!(resolve(&child).unwrap().occ3_enabled);
        save_latent(&child, false).unwrap();
        save(&child, true).unwrap();
        assert!(!resolve(&child).unwrap().occ3_enabled);
        assert_eq!(resolve(&child).unwrap().occ3_source_session_id.as_deref(), Some("child"));
        // Previously written explicit false remains a veto, never an ambient opt-in.
        fs::write(preference_path(&a), json!({"schema":SCHEMA,"occ2Enabled":true,"occ3Enabled":false}).to_string()).unwrap();
        save(&a, false).unwrap();
        assert!(!resolve(&a).unwrap().occ3_enabled);
        assert_eq!(resolve(&a).unwrap().occ3_source_session_id.as_deref(), Some("a"));
        // An old OCC2-only record has no explicit latent preference.
        fs::write(preference_path(&a), json!({"schema":SCHEMA,"occ2Enabled":true}).to_string()).unwrap();
        assert_eq!(resolve(&a).unwrap().occ3_source_session_id, None);
    }

    #[test]
    fn occ_session_resolution_reports_provenance_and_round_trips() {
        let root = tempfile::tempdir().unwrap();
        let sessions = root.path().join("sessions");
        fs::create_dir(&sessions).unwrap();
        let a = session(&sessions, "a", None, 0);
        let child = session(&sessions, "child", Some("a.jsonl"), 1);
        // Default: on, no source.
        let resolution = resolve(&a).unwrap();
        assert_eq!(resolution, Resolution::default());
        assert_eq!(preferences(&a).unwrap(), json!({"occ2Enabled": true, "inherited": false, "sourceSessionId": null}));
        // The chat that owns the toggle is not inherited.
        save(&a, true).unwrap();
        let resolution = resolve(&a).unwrap();
        assert_eq!(resolution.occ2_enabled, true);
        assert_eq!(resolution.source_session_id.as_deref(), Some("a"));
        assert_eq!(preferences(&a).unwrap(), json!({"occ2Enabled": true, "inherited": false, "sourceSessionId": "a"}));
        // A subagent reports the ancestor as its source.
        assert_eq!(preferences(&child).unwrap(), json!({"occ2Enabled": true, "inherited": true, "sourceSessionId": "a"}));
        // The set response echoes the daemon-authoritative value.
        let set = set_preferences(&a, false).unwrap();
        assert_eq!(set["occ2Enabled"], json!(false));
        assert_eq!(set["applied"], json!(true));
        assert_eq!(set["scope"], json!("this chat"));
        assert_eq!(set["effectiveForNewMessages"], json!(true));
        assert_eq!(set["sourceSessionId"], json!("a"));
        assert!(!load(&a).unwrap());
        // A child override becomes the child's own source for its subtree.
        set_preferences(&child, true).unwrap();
        assert_eq!(preferences(&child).unwrap()["sourceSessionId"], json!("child"));
        // Read failures propagate instead of degrading to a silent value.
        fs::write(preference_path(&a), "not json").unwrap();
        assert!(preferences(&a).is_err());
        assert!(load(&a).is_err());
        // A set heals corrupt storage: it overwrites the bad file atomically
        // and echoes the stored value, never a silent default (the live
        // daemon test pins the same heal-on-set semantics).
        let healed = set_preferences(&a, true).unwrap();
        assert_eq!(healed["occ2Enabled"], json!(true));
        assert_eq!(healed["applied"], json!(true));
        assert!(load(&a).unwrap());
        // A write that cannot land fails loudly: the preference path is a
        // directory, so the atomic rename in `save` cannot complete.
        fs::remove_file(preference_path(&a)).unwrap();
        fs::create_dir(preference_path(&a)).unwrap();
        assert!(set_preferences(&a, false).is_err());
        fs::remove_dir(preference_path(&a)).unwrap();
        fs::write(preference_path(&a), "x".repeat(4097)).unwrap();
        assert!(preferences(&a).is_err());
    }

    #[test]
    fn occ_sessions_bound_reads_fail_closed_and_support_empty_new_chats() {
        let root = tempfile::tempdir().unwrap();
        let sessions = root.path().join("sessions");
        fs::create_dir(&sessions).unwrap();
        let a = session(&sessions, "a", Some("b.jsonl"), 1);
        let b = session(&sessions, "b", Some("a.jsonl"), 2);
        assert!(load(&a).is_err());
        assert!(!enabled(Some(&a)));
        save(&b, true).unwrap();
        assert!(load(&a).unwrap());
        fs::write(preference_path(&b), "invalid").unwrap();
        assert!(!enabled(Some(&a)));
        assert!(load(&a).is_err());
        fs::write(preference_path(&b), "x".repeat(4097)).unwrap();
        assert!(load(&b).is_err());
        let empty = sessions.join("new.jsonl").to_str().unwrap().to_string();
        assert!(load(&empty).unwrap());
        save(&empty, false).unwrap();
        assert!(!load(&empty).unwrap());
        assert!(!enabled(None));
    }
}
