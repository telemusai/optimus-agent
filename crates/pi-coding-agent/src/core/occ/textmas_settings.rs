//! Independent, session-local OCC3.1 TextMAS mode. No provider configuration is read.

use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};
use super::textmas::TextMasMode;

const SCHEMA: &str = "optimus.occ31-session.v1";

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Preference {
    schema: String,
    mode: TextMasMode,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Resolution {
    pub mode: TextMasMode,
    pub source_session_id: Option<String>,
}

pub fn preference_path(session_file: &str) -> PathBuf {
    PathBuf::from(crate::core::session_manager::get_session_artifact_path_for_file(session_file, None))
        .join("occ31.json")
}

fn session_id(session_file: &str) -> String {
    let name = Path::new(session_file).file_name().map(|name| name.to_string_lossy().into_owned()).unwrap_or_default();
    name.strip_suffix(".jsonl").unwrap_or(&name).to_string()
}

/// Resolve only explicit TextMAS modes. Depth-zero forks never inherit.
pub fn resolve(session_file: &str) -> Result<Resolution, String> {
    if session_file.is_empty() { return Err("This chat has no saved session identity".into()); }
    let mut file = PathBuf::from(session_file);
    let mut visited = BTreeSet::new();
    for _ in 0..64 {
        if !visited.insert(file.clone()) { return Err("OCC3.1 session ancestry contains a cycle".into()); }
        match File::open(preference_path(&file.to_string_lossy())) {
            Ok(reader) => {
                let mut bytes = Vec::new();
                reader.take(4097).read_to_end(&mut bytes).map_err(|error| error.to_string())?;
                if bytes.len() > 4096 { return Err("OCC3.1 session preference is too large".into()); }
                let preference: Preference = serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
                if preference.schema != SCHEMA { return Err("Unsupported OCC3.1 session preference schema".into()); }
                return Ok(Resolution { mode: preference.mode, source_session_id: Some(session_id(&file.to_string_lossy())) });
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.to_string()),
        }
        // An unreadable or replaced session file must degrade to the default
        // mode, not fail the turn: OCC3.1 is opt-in and fail-open here.
        let reader = match File::open(&file) {
            Ok(reader) => reader,
            Err(_) => return Ok(Resolution::default()),
        };
        let mut header = Vec::new();
        match BufReader::new(reader.take(65537)).read_until(b'\n', &mut header) {
            // An unreadable session file (replaced by a directory, EACCES) must
            // degrade to the default mode, not fail the turn: OCC3.1 is opt-in.
            Err(_) => return Ok(Resolution::default()),
            Ok(_) => {}
        }
        if header.len() > 65536 { return Err("OCC3.1 session header is too large".into()); }
        let header: serde_json::Value = serde_json::from_slice(&header).map_err(|error| error.to_string())?;
        if header["rlmDepth"].as_u64().unwrap_or(0) == 0 { return Ok(Resolution::default()); }
        let Some(parent) = header["parentSession"].as_str().filter(|parent| !parent.is_empty()) else { return Ok(Resolution::default()); };
        let parent = Path::new(parent);
        file = if parent.is_absolute() { parent.to_path_buf() }
            else { file.parent().unwrap_or_else(|| Path::new(".")).join(parent) };
    }
    Err("OCC3.1 session ancestry exceeds the supported depth".into())
}

/// Non-interactive fallback only. Runtime operations should use resolve to surface errors.
pub fn mode(session_file: Option<&str>) -> TextMasMode {
    session_file.and_then(|file| resolve(file).ok()).map(|value| value.mode).unwrap_or_default()
}

pub fn preferences(session_file: &str) -> Result<serde_json::Value, String> {
    let resolution = resolve(session_file)?;
    let inherited = resolution.source_session_id.as_deref().is_some_and(|source| source != session_id(session_file));
    Ok(serde_json::json!({
        "occ31Mode": resolution.mode,
        "inherited": inherited,
        "sourceSessionId": resolution.source_session_id,
    }))
}

/// Atomic replacement owns only occ31.json; neither existing OCC lane is touched.
pub fn set_mode(session_file: &str, mode: TextMasMode) -> Result<serde_json::Value, String> {
    if session_file.is_empty() { return Err("This chat has no saved session identity".into()); }
    let path = preference_path(session_file);
    let parent = path.parent().ok_or("Invalid OCC3.1 session preference path")?;
    fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    let mut pending = tempfile::NamedTempFile::new_in(parent).map_err(|error| error.to_string())?;
    serde_json::to_writer(pending.as_file_mut(), &Preference { schema: SCHEMA.into(), mode }).map_err(|error| error.to_string())?;
    pending.write_all(b"\n").map_err(|error| error.to_string())?;
    pending.as_file().sync_all().map_err(|error| error.to_string())?;
    pending.persist(path).map_err(|error| error.to_string())?;
    let mut value = preferences(session_file)?;
    value["applied"] = serde_json::json!(value["occ31Mode"] == serde_json::json!(mode));
    value["scope"] = serde_json::json!("this chat");
    value["effectiveForNewOperations"] = serde_json::json!(true);
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn session(root: &Path, name: &str, parent: Option<&str>, depth: usize) -> String {
        let sessions = root.join("sessions");
        fs::create_dir_all(&sessions).unwrap();
        let path = sessions.join(format!("{name}.jsonl"));
        fs::write(&path, format!("{}\n", json!({"id":name,"rlmDepth":depth,"parentSession":parent}))).unwrap();
        path.to_string_lossy().into_owned()
    }

    #[test]
    fn modes_roundtrip_and_inherit_dynamically_without_crossing_forks_or_lanes() {
        let root = tempfile::tempdir().unwrap();
        let a = session(root.path(), "a", None, 0);
        let b = session(root.path(), "b", None, 0);
        let child = session(root.path(), "child", Some("a.jsonl"), 1);
        let nested = session(root.path(), "nested", Some("child.jsonl"), 2);
        let fork = session(root.path(), "fork", Some("a.jsonl"), 0);
        assert_eq!(resolve(&a).unwrap(), Resolution::default());
        super::super::session_settings::save_occ3(&a, false, true).unwrap();
        let old_bytes = fs::read(super::super::session_settings::preference_path(&a)).unwrap();
        for selected in [TextMasMode::Sequential, TextMasMode::Hierarchical, TextMasMode::Off] {
            let echo = set_mode(&a, selected).unwrap();
            assert_eq!(echo["occ31Mode"], json!(selected));
            assert_eq!(echo["applied"], true);
            assert_eq!(mode(Some(&child)), selected);
            assert_eq!(mode(Some(&nested)), selected);
            assert_eq!(preferences(&child).unwrap()["sourceSessionId"], "a");
            assert_eq!(preferences(&child).unwrap()["inherited"], true);
            assert_eq!(mode(Some(&b)), TextMasMode::Off);
            assert_eq!(mode(Some(&fork)), TextMasMode::Off);
            assert_eq!(fs::read(super::super::session_settings::preference_path(&a)).unwrap(), old_bytes);
        }
        set_mode(&child, TextMasMode::Off).unwrap();
        set_mode(&a, TextMasMode::Hierarchical).unwrap();
        assert_eq!(mode(Some(&nested)), TextMasMode::Off);
        fs::remove_file(preference_path(&child)).unwrap();
        assert_eq!(mode(Some(&nested)), TextMasMode::Hierarchical);
        assert!(!super::super::session_settings::load(&a).unwrap());
        assert!(super::super::session_settings::resolve(&a).unwrap().occ3_enabled);
    }

    #[test]
    fn corrupt_oversize_unknown_and_unwritable_modes_fail_visibly() {
        let root = tempfile::tempdir().unwrap();
        let a = session(root.path(), "a", None, 0);
        set_mode(&a, TextMasMode::Sequential).unwrap();
        for body in ["not json".into(), "x".repeat(4097), json!({"schema":SCHEMA,"mode":"on"}).to_string(),
            json!({"schema":"unknown","mode":"off"}).to_string(), json!({"schema":SCHEMA,"mode":"off","extra":true}).to_string()] {
            fs::write(preference_path(&a), body).unwrap();
            assert!(preferences(&a).is_err());
            assert_eq!(mode(Some(&a)), TextMasMode::Off);
        }
        assert_eq!(set_mode(&a, TextMasMode::Off).unwrap()["occ31Mode"], "off");
        fs::remove_file(preference_path(&a)).unwrap();
        fs::create_dir(preference_path(&a)).unwrap();
        assert!(set_mode(&a, TextMasMode::Sequential).is_err());
        assert!(set_mode("", TextMasMode::Off).is_err());
        assert!(resolve("").is_err());
        assert_eq!(mode(None), TextMasMode::Off);
    }

    #[test]
    fn ancestry_cycles_and_large_headers_are_bounded() {
        let root = tempfile::tempdir().unwrap();
        let a = session(root.path(), "a", Some("b.jsonl"), 1);
        let _b = session(root.path(), "b", Some("a.jsonl"), 2);
        assert!(resolve(&a).unwrap_err().contains("cycle"));
        fs::write(&a, "x".repeat(65537)).unwrap();
        assert!(resolve(&a).unwrap_err().contains("header is too large"));
        for depth in 0..65 {
            session(root.path(), &format!("d{depth}"), Some(&format!("d{}.jsonl", depth + 1)), 1);
        }
        assert!(resolve(&root.path().join("sessions/d0.jsonl").to_string_lossy()).unwrap_err().contains("depth"));
    }
}
