//! Isolated damaged-transcript regressions; no providers or user profiles.
use super::*;

fn fixture_header() -> Value {
    serde_json::json!({"type":"session","id":"fixture","version":3,
        "timestamp":"2026-01-01T00:00:00Z","cwd":"/fixture","rlmDepth":0})
}

fn fixture_message(id: &str, parent: Option<&str>) -> Value {
    serde_json::json!({"type":"message","id":id,"parentId":parent,
        "timestamp":"2026-01-01T00:00:01Z",
        "message":{"role":"user","content":id,"timestamp":1}})
}

fn write_fixture(path: &Path, rows: &[Value]) -> Vec<u8> {
    let content = rows.iter().map(Value::to_string).collect::<Vec<_>>().join("\n") + "\n";
    std::fs::write(path, &content).unwrap();
    content.into_bytes()
}

#[test]
fn session_safety_parent_cycles_are_bounded_without_rewriting_entries() {
    let dir = tempfile::tempdir().unwrap();
    for two_nodes in [false, true] {
        let file = dir.path().join("cycle.jsonl");
        let mut rows = vec![fixture_header(), fixture_message("a", Some(if two_nodes { "b" } else { "a" }))];
        if two_nodes { rows.push(fixture_message("b", Some("a"))); }
        let before = write_fixture(&file, &rows);
        let manager = SessionManager::open(&file.to_string_lossy(), None, None).unwrap();
        let expected = if two_nodes { vec!["a", "b"] } else { vec!["a"] };
        assert_eq!(manager.get_branch(None).iter().map(entry_id).collect::<Vec<_>>(), expected);
        let mut visited = Vec::new();
        manager.visit_branch(None, |entry| visited.push(entry_id(entry)));
        assert_eq!(visited, expected);
        let context = manager.build_session_context_with_entry_ids(None);
        assert_eq!(context.entry_ids, expected);
        assert_eq!(context.messages.len(), expected.len());
        let history = manager.build_session_history(None, None).unwrap();
        assert_eq!(history.entry_ids, expected);
        assert_eq!(history.tip_entry_id, manager.get_leaf_id());
        assert!(manager.get_latest_agent_status().is_none());
        assert!(manager.get_active_git_context().is_none());
        assert_eq!(std::fs::read(&file).unwrap(), before);
    }
}

#[test]
fn session_safety_valid_ancestry_and_unknown_entries_remain_intact() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("branches.jsonl");
    let rows = vec![fixture_header(), fixture_message("root", None),
        serde_json::json!({"type":"future_entry","id":"opaque","parentId":"root","payload":{"keep":[1,2]}}),
        fixture_message("sibling", Some("root")), fixture_message("tip", Some("opaque"))];
    let before = write_fixture(&file, &rows);
    let manager = SessionManager::open(&file.to_string_lossy(), None, None).unwrap();
    assert_eq!(manager.get_branch(None).iter().map(entry_id).collect::<Vec<_>>(), ["root", "opaque", "tip"]);
    assert_eq!(manager.build_session_context_with_entry_ids(None).entry_ids, ["root", "tip"]);
    assert_eq!(manager.get_branch(Some("sibling")).iter().map(entry_id).collect::<Vec<_>>(), ["root", "sibling"]);
    assert_eq!(manager.get_entry("opaque").unwrap(), rows[2].as_object().unwrap().clone());
    assert_eq!(std::fs::read(&file).unwrap(), before);
}

#[test]
fn session_safety_fork_bounds_cycles_in_removed_git_entries() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("source.jsonl");
    let before = write_fixture(&file, &[fixture_header(),
        serde_json::json!({"type":"git_state","id":"g1","parentId":"g2"}),
        serde_json::json!({"type":"git_state","id":"g2","parentId":"g1"}),
        fixture_message("tip", Some("g1"))]);
    let fork = SessionManager::fork_from(&file.to_string_lossy(), &dir.path().to_string_lossy(), Some(&dir.path().to_string_lossy())).unwrap();
    assert_eq!(fork.get_branch(None).iter().map(entry_id).collect::<Vec<_>>(), ["tip"]);
    assert_eq!(fork.get_entry("tip").unwrap()["parentId"], Value::Null);
    assert_eq!(std::fs::read(&file).unwrap(), before);
}

#[tokio::test]
async fn session_safety_tail_repair_preserves_prefix_and_allows_distinct_append() {
    use std::io::Write;
    let dir = tempfile::tempdir().unwrap();
    for (index, tail) in [b"{\"unfinished\":".as_slice(), b"{\"text\":\"\xf0\x9f", b"\0\0", b""].iter().enumerate() {
        let file = dir.path().join(format!("repair-{index}.jsonl"));
        let prefix = write_fixture(&file, &[fixture_header(), fixture_message("first", None),
            serde_json::json!({"type":"future_entry","id":"opaque","parentId":"first","opaque":[1,2]})]);
        let complete = fixture_message("last", Some("opaque")).to_string();
        let mut writer = std::fs::OpenOptions::new().append(true).open(&file).unwrap();
        if tail.is_empty() { writer.write_all(complete.as_bytes()).unwrap(); }
        else { writer.write_all(tail).unwrap(); }
        drop(writer);
        let mut manager = SessionManager::open_async(&file.to_string_lossy(), None, None).await.unwrap();
        let parent = manager.get_leaf_id().unwrap();
        let added = manager.append_session_info("after repair").unwrap();
        let bytes = std::fs::read(&file).unwrap();
        assert!(bytes.starts_with(&prefix));
        assert_eq!(bytes.last(), Some(&b'\n'));
        for line in String::from_utf8(bytes).unwrap().lines() {
            assert!(serde_json::from_str::<Value>(line).is_ok());
        }
        let reopened = SessionManager::open(&file.to_string_lossy(), None, None).unwrap();
        assert_eq!(reopened.get_entry(&added).unwrap()["parentId"], parent);
        assert_eq!(reopened.get_leaf_id(), Some(added));
        assert_eq!(reopened.get_entry("last").is_some(), tail.is_empty());
    }
}

// The destination remains appendable, but a sibling atomic temp filename exceeds
// NAME_MAX. Unlike chmod tests this produces a real repair failure even as root.
#[cfg(unix)]
#[tokio::test]
async fn session_safety_required_repair_failure_blocks_open_and_preserves_current_manager() {
    let dir = tempfile::tempdir().unwrap();
    let damaged = dir.path().join(format!("{}.jsonl", "x".repeat(240)));
    let contents = format!("{}\n{}", fixture_header(), fixture_message("tail", None));
    std::fs::write(&damaged, contents.as_bytes()).unwrap();
    let path = damaged.to_string_lossy();
    assert!(std::fs::OpenOptions::new().append(true).open(&damaged).is_ok());
    let error = SessionManager::open(&path, None, None).err().expect("required repair must fail");
    assert!(error.contains("Session repair failed"), "{error}");
    assert!(SessionManager::open_async(&path, None, None).await.is_err());
    assert_eq!(std::fs::read(&damaged).unwrap(), contents.as_bytes());

    let safe = dir.path().join("safe.jsonl");
    write_fixture(&safe, &[fixture_header(), fixture_message("safe", None)]);
    let mut manager = SessionManager::open(&safe.to_string_lossy(), None, None).unwrap();
    let old_file = manager.get_session_file();
    let old_entries = manager.get_entries();
    let old_leaf = manager.get_leaf_id();
    let old_observation = manager.get_load_observation();
    for preloaded in [None, Some(load_entries_from_file(&path))] {
        assert!(manager.set_session_file(&path, preloaded, None).is_err());
        assert_eq!(manager.get_session_file(), old_file);
        assert_eq!(manager.get_entries(), old_entries);
        assert_eq!(manager.get_leaf_id(), old_leaf);
        assert_eq!(manager.get_load_observation(), old_observation);
    }
    manager.append_session_info("still at safe path").unwrap();
    assert_eq!(std::fs::read(&damaged).unwrap(), contents.as_bytes());
}
