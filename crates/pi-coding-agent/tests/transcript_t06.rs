//! T06 — transcript durability and format compatibility (owner: transcript).
//! Spec: TEST-SPEC.md §T06. Findings B-01, B-03, B-05, B-08, B-11.
//!
//! Layer: unit/integration through the real `SessionManager` production entry
//! points (create/open/append/flush/set_session_file/build_session_history).
//! No provider or network. All writable state under OPTIMUS_PARITY_STATE_ROOT
//! (V00 isolation).
//!
//! Source-derived oracles (documented per TEST-SPEC "Common proof standard"):
//! - inline tool text reference contract = TypeScript session-manager.ts
//!   sha256Text:277-281, inlineToolTextReference:293-317 (start/length in JS
//!   UTF-16 code units), decodeInlineToolTextReference:378-400 (slices
//!   source.length in UTF-16 units).
//! - transcript order oracle = orderSessionContextForTranscript
//!   (session-manager.ts:802-822): boundary = retainedMessageCount when a safe
//!   non-negative integer, else the legacy timestamp fallback; model-facing
//!   order stays summary-first.
//! - missing-tip history oracle = buildSessionHistory default parameter
//!   `tipEntryId = this.leafId` (session-manager.ts:2426).
//! - failure contract = session-manager.ts _rewriteFile:1899 propagates write
//!   errors; persistence listeners observe committed writes only.

use std::path::{Path, PathBuf};

use pi_ai::types::{
    ImageOrTextContent, TextContent, ToolResultMessage, UserContent, UserMessage,
};
use pi_coding_agent::core::session_manager::{load_entries_from_file, SessionManager};
use serde_json::{json, Value};

// ---------------------------------------------------------------------------
// V00 harness
// ---------------------------------------------------------------------------

fn state_root() -> PathBuf {
    static ROOT: std::sync::OnceLock<tempfile::TempDir> = std::sync::OnceLock::new();
    let root = std::env::var_os("OPTIMUS_PARITY_STATE_ROOT")
        .map(|raw| {
            assert!(!raw.is_empty(), "OPTIMUS_PARITY_STATE_ROOT must not be empty");
            std::path::absolute(raw).expect("absolute state root")
        })
        .unwrap_or_else(|| {
            ROOT.get_or_init(|| tempfile::Builder::new().prefix("optimus-transcript-t06-").tempdir().expect("private state root"))
                .path().to_path_buf()
        });
    let lowered = root.to_string_lossy().to_lowercase();
    assert!(!lowered.contains(".prime"), "state root must not touch .prime");
    std::fs::create_dir_all(&root).expect("create state root");
    assert!(!std::fs::canonicalize(&root).expect("canonical state root").to_string_lossy().to_lowercase().contains(".prime"), "state root must not resolve into .prime");
    root
}

fn case_dir(tag: &str) -> PathBuf {
    let dir = state_root()
        .join(tag)
        .join(format!("{}-{}", std::process::id(), unique_tag()));
    std::fs::create_dir_all(&dir).expect("create case dir");
    dir
}

fn unique_tag() -> String {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};
    static TAG_SEQ: AtomicUsize = AtomicUsize::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    format!(
        "{}-{}",
        nanos,
        TAG_SEQ.fetch_add(1, Ordering::Relaxed)
    )
}

/// Write a session JSONL fixture with controlled entries.
fn write_jsonl(path: &Path, entries: &[Value]) {
    let mut body = String::new();
    for entry in entries {
        body.push_str(&serde_json::to_string(entry).unwrap());
        body.push('\n');
    }
    std::fs::write(path, body).expect("write fixture");
}

fn session_header(id: &str, timestamp: &str, cwd: &str) -> Value {
    json!({
        "type": "session",
        "version": 3,
        "id": id,
        "timestamp": timestamp,
        "cwd": cwd,
        "rlmDepth": 0,
    })
}

fn message_entry(id: &str, parent: Value, timestamp: &str, text: &str) -> Value {
    json!({
        "type": "message",
        "id": id,
        "parentId": parent,
        "timestamp": timestamp,
        "message": {"role": "user", "content": text, "timestamp": 0},
    })
}

/// sha256 over the JSON-escaped string form, exactly like sha256Text.
fn sha256_text_hex(value: &str) -> String {
    use sha2::{Digest, Sha256};
    let escaped = serde_json::to_string(&Value::String(value.to_string())).unwrap();
    let mut hasher = Sha256::new();
    hasher.update(escaped.as_bytes());
    let digest = hasher.finalize();
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Independent TS-contract decoder for inline tool text references.
/// Oracle: decodeInlineToolTextReference (session-manager.ts:378-400) with
/// UTF-16 code-unit slicing. Source-derived, marked per the proof standard.
mod ts_contract {
    use serde_json::Value;

    fn sha256_text(value: &str) -> String {
        super::sha256_text_hex(value)
    }

    fn utf16_units(source: &str) -> Vec<u16> {
        source.encode_utf16().collect()
    }

    pub fn units_len_of(source: &str) -> usize {
        utf16_units(source).len()
    }

    fn utf16_slice(units: &[u16], start: usize, length: usize) -> String {
        let end = (start + length).min(units.len());
        String::from_utf16_lossy(&units[start..end])
    }

    /// Decode `details[key]` under the TypeScript UTF-16 contract.
    pub fn decode(entry: &Value, key: &str) -> Option<String> {
        let message = entry.get("message")?;
        let value = message.get("details")?.get(key)?;
        let candidate = value.get("$primeToolText")?;
        let version = candidate.get("version")?.as_i64()?;
        if version != 1 || candidate.get("source")?.as_str()? != "content" {
            return None;
        }
        let content_index = candidate.get("contentIndex")?.as_u64()? as usize;
        let start = candidate.get("start")?.as_u64()? as usize;
        let length = candidate.get("length")?.as_u64()? as usize;
        let sha = candidate.get("sha256")?.as_str()?.to_string();
        if sha.len() != 64 {
            return None;
        }
        let content = message.get("content")?;
        let source = match content {
            Value::String(text) if content_index == 0 => text.clone(),
            Value::Array(parts) => {
                let part = parts.get(content_index)?;
                if part.get("type")?.as_str()? != "text" {
                    return None;
                }
                part.get("text")?.as_str()?.to_string()
            }
            _ => return None,
        };
        let units = utf16_units(&source);
        if length > units.len() || start > units.len() - length {
            return None;
        }
        let decoded = utf16_slice(&units, start, length);
        if sha256_text(&decoded) == sha {
            Some(decoded)
        } else {
            None
        }
    }
}

const STDOUT_LEN: usize = 400;

fn ascii_body() -> String {
    "x".repeat(STDOUT_LEN)
}

fn read_details_stdout(entry: &Value) -> (Option<String>, bool, bool) {
    let message = entry.get("message").cloned().unwrap_or(Value::Null);
    let stdout = message
        .get("details")
        .and_then(|details| details.get("stdout"))
        .and_then(|value| value.as_str())
        .map(str::to_string);
    let is_error = message
        .get("isError")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let diagnostic = message
        .get("content")
        .and_then(|content| content.as_array())
        .map(|parts| {
            parts.iter().any(|part| {
                part.get("text")
                    .and_then(Value::as_str)
                    .map(|text| text.contains("session recovery error"))
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false);
    (stdout, is_error, diagnostic)
}

fn append_tool_result(
    manager: &mut SessionManager,
    prefix: &str,
    stdout: &str,
) -> Result<String, String> {
    let mut tool_result = ToolResultMessage::new(
        "call-1",
        "bash",
        vec![ImageOrTextContent::Text(TextContent::new(format!(
            "{prefix}{stdout}"
        )))],
        false,
        1,
    );
    tool_result.details = Some(json!({"stdout": stdout, "exitCode": 0}));
    manager.append_message(AgentMessage::from(tool_result))
}

use pi_agent_core::types::AgentMessage;

// ---------------------------------------------------------------------------
// unicode_inline_tool_refs_roundtrip (B-01)
// ---------------------------------------------------------------------------

#[test]
fn unicode_inline_tool_refs_roundtrip() {
    let dir = case_dir("t06-unicode");
    let sessions = dir.join("sessions");
    std::fs::create_dir_all(&sessions).unwrap();
    let stdout = ascii_body();
    // All sub-cases run to completion on both lanes; oracle deviations are
    // collected so one baseline run carries the full B-01 evidence set.
    let mut failures: Vec<String> = Vec::new();

    // (a) ASCII control: the pipeline is unit-coherent for pure ASCII.
    {
        let case_dir = dir.join("rt-ascii");
        std::fs::create_dir_all(&case_dir).unwrap();
        let mut manager = SessionManager::create(
            case_dir.to_string_lossy().as_ref(),
            Some(sessions.to_string_lossy().as_ref()),
        )
        .expect("create");
        let file = manager.get_session_file().expect("session file").clone();
        // V00: the transcript must live inside the per-run private dir tree.
        // The sessions dir is a sibling of the per-case dir here (shared by all
        // sub-cases), so the correct containment parent is `sessions`.
        assert!(
            Path::new(&file).starts_with(&sessions),
            "V00: private session file, got {file} (sessions dir {})",
            sessions.display()
        );
        append_tool_result(&mut manager, "", &stdout).expect("append ascii toolResult");
        manager.flush_now();
        assert!(Path::new(&file).is_file(), "flush must materialize the file");

        let entries = load_entries_from_file(file.as_str());
        let tool_entry = entries
            .iter()
            .find(|entry| entry.get("type").and_then(Value::as_str) == Some("message"))
            .expect("toolResult entry");
        let (stored, is_error, diagnostic) = read_details_stdout(&Value::Object(tool_entry.clone()));
        if stored.as_deref() != Some(stdout.as_str()) {
            failures.push("ascii: control round-trip must be exact".to_string());
        }
        if is_error || diagnostic {
            failures.push("ascii: control must not raise recovery errors".to_string());
        }
        drop(manager);
    }

    // (b)-(d) Rust-written round-trips with non-ASCII prefixes. Expected
    // contract (TS, UTF-16 units): decoded stdout == original, no isError, for
    // every prefix. Baseline mixes byte start with char length.
    let cases: Vec<(&str, &str)> = vec![
        ("cyrillic", "\u{0416}"),   // Ж: BMP, 2 UTF-8 bytes, 1 UTF-16 unit
        ("cjk", "\u{4E2D}"),        // 中: BMP, 3 UTF-8 bytes, 1 UTF-16 unit
        ("combining", "e\u{0301}"), // 3 UTF-8 bytes, 2 UTF-16 units
        ("emoji", "\u{1F600}"),     // astral: 4 UTF-8 bytes, 2 UTF-16 units
    ];
    for (tag, prefix) in &cases {
        let case_dir = dir.join(format!("rt-{tag}"));
        std::fs::create_dir_all(&case_dir).unwrap();
        let mut manager = SessionManager::create(
            case_dir.to_string_lossy().as_ref(),
            Some(sessions.to_string_lossy().as_ref()),
        )
        .expect("create");
        let file = manager.get_session_file().expect("session file").clone();
        append_tool_result(&mut manager, prefix, &stdout).expect("append toolResult");
        manager.flush_now();

        // The versioned reference envelope must be on disk (encode engaged).
        let raw = std::fs::read_to_string(&file).unwrap();
        assert!(
            raw.contains("$primeSessionEncoding") && raw.contains("$primeToolText"),
            "{tag}: reference envelope must be on disk"
        );

        // Reopen through the production loader and require exact survival.
        let entries = load_entries_from_file(file.as_str());
        let tool_entry = entries
            .iter()
            .find(|entry| entry.get("type").and_then(Value::as_str) == Some("message"))
            .expect("entry present");
        let (stored, is_error, diagnostic) = read_details_stdout(&Value::Object(tool_entry.clone()));
        if stored.as_deref() != Some(stdout.as_str()) {
            failures.push(format!(
                "{tag}: stdout must survive the disk round-trip under the UTF-16 contract (got {stored:?})"
            ));
        }
        if is_error {
            failures.push(format!("{tag}: isError must not be raised"));
        }
        if diagnostic {
            failures.push(format!("{tag}: no recovery diagnostic may be appended"));
        }

        // Cross-check the Rust-written offsets with the TS contract decoder:
        // Rust-written files must stay readable by TypeScript.
        let raw_entry: Value = serde_json::from_str(
            raw.lines()
                .find(|line| line.contains("toolResult"))
                .expect("raw toolResult line"),
        )
        .unwrap();
        let decoded_via_ts = ts_contract::decode(&raw_entry, "stdout");
        if decoded_via_ts.as_deref() != Some(stdout.as_str()) {
            failures.push(format!(
                "{tag}: Rust-written reference must decode under the TS UTF-16 contract (got {decoded_via_ts:?})"
            ));
        }
    }

    // (e) TS-format fixtures (UTF-16 semantics, incl. astral) written directly
    // from the TS contract; Rust rehydrate must decode every case.
    for (tag, prefix) in &cases {
        let case_dir = dir.join(format!("tsfix-{tag}"));
        std::fs::create_dir_all(&case_dir).unwrap();
        let content_text = format!("{prefix}{stdout} tail");
        let start = ts_contract::units_len_of(prefix);
        let entry = json!({
            "type": "message",
            "id": "t1",
            "parentId": Value::Null,
            "timestamp": "2026-09-16T00:00:00.000Z",
            "$primeSessionEncoding": {"version": 1, "kind": "inline_tool_text", "fields": ["stdout"]},
            "message": {
                "role": "toolResult",
                "toolCallId": "call-1",
                "toolName": "bash",
                "content": [{"type": "text", "text": content_text}],
                "details": {
                    "stdout": {"$primeToolText": {
                        "version": 1, "source": "content", "contentIndex": 0,
                        "start": start,
                        "length": stdout.encode_utf16().count(),
                        "sha256": sha256_text_hex(&stdout),
                    }},
                    "exitCode": 0,
                },
                "isError": false,
                "timestamp": 1,
            },
        });
        let file = case_dir.join("ts-encoded.jsonl");
        write_jsonl(
            file.as_path(),
            &[session_header("s1", "2026-09-16T00:00:00.000Z", "C:/fixture"), entry],
        );
        let manager = SessionManager::open(file.to_string_lossy().as_ref(), None, None)
            .expect("open TS fixture");
        let entries = manager.get_entries();
        let tool_entry = entries
            .iter()
            .find(|entry| entry.get("type").and_then(Value::as_str) == Some("message"))
            .expect("toolResult entry");
        let (stored, is_error, diagnostic) = read_details_stdout(&Value::Object(tool_entry.clone()));
        if stored.as_deref() != Some(stdout.as_str()) {
            failures.push(format!(
                "{tag}: TS-written UTF-16 reference must decode exactly (got {stored:?})"
            ));
        }
        if is_error || diagnostic {
            failures.push(format!("{tag}: no recovery error for TS files"));
        }
    }
    assert!(
        failures.is_empty(),
        "B-01 evidence (baseline deviations from the TS UTF-16 contract):\n{}",
        failures.join("\n")
    )
}

// ---------------------------------------------------------------------------
// failed_rewrite_never_acknowledges_persistence (B-05)
// ---------------------------------------------------------------------------

#[test]
fn failed_rewrite_never_acknowledges_persistence() {
    let dir = case_dir("t06-failed-rewrite");
    // Oracle deviations are collected across both parts so one baseline run
    // carries the complete B-05 evidence set.
    let mut failures: Vec<String> = Vec::new();

    // --- Part A: first creation ---
    {
        let case_dir = dir.join("first-creation");
        std::fs::create_dir_all(&case_dir).unwrap();
        let sessions = case_dir.join("sessions");
        std::fs::create_dir_all(&sessions).unwrap();
        let mut manager = SessionManager::create(
            case_dir.to_string_lossy().as_ref(),
            Some(sessions.to_string_lossy().as_ref()),
        )
        .expect("create");
        let file = manager.get_session_file().expect("deferred session file").clone();
        assert!(Path::new(&file).starts_with(&case_dir), "V00: private target");

        // Inject rename failure for the first write: a directory occupies the target.
        std::fs::create_dir_all(&file).expect("occupy target path with a directory");

        let listener_calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        {
            let counter = std::sync::Arc::clone(&listener_calls);
            let _unsub = manager.on_persist(Box::new(move |_file| {
                counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            }));
        }

        // User message alone must not touch the disk (pre-assistant guard).
        let user = AgentMessage::from(UserMessage::new(UserContent::Text("hello".to_string()), 1));
        manager
            .append_message(user)
            .expect("pre-assistant append stays unwritten");

        // The assistant message triggers the first materializing rewrite.
        let assistant = AgentMessage::from(
            pi_ai::types::AssistantMessage {
                content: vec![],
                timestamp: 2,
                ..Default::default()
            },
        );
        let result = manager.append_message(assistant);
        if result.is_ok() {
            failures.push(
                "PART A: append_message returned Ok although the first rewrite failed (TS: caller sees the error)"
                    .to_string(),
            );
        }
        if Path::new(&file).is_file() {
            failures.push("PART A: bytes were committed behind the caller's back".to_string());
        }
        let listener_count = listener_calls.load(std::sync::atomic::Ordering::SeqCst);
        if listener_count != 0 {
            failures.push(format!(
                "PART A: persistence listeners fired for a failed rewrite: {listener_count}"
            ));
        }
        drop(manager);
    }

    // --- Part B: replacement of an existing transcript (header migration rewrite) ---
    // Windows denies replacement while this handle is open; Unix permits it.
    #[cfg(windows)]
    {
        let case_dir = dir.join("replacement");
        std::fs::create_dir_all(&case_dir).unwrap();
        let file = case_dir.join("legacy-v1.jsonl");
        let header = json!({
            "type": "session",
            "id": "legacy-1",
            "timestamp": "2026-09-16T00:00:00.000Z",
            "cwd": case_dir.to_string_lossy(),
        });
        let message = message_entry("m1", Value::Null, "2026-09-16T00:00:00.000Z", "acknowledged history");
        write_jsonl(file.as_path(), &[header, message]);
        let before = std::fs::read(&file).unwrap();

        // Deterministic rename failure: hold the target open without
        // write/delete sharing (MoveFileEx needs DELETE on the destination).
        let lock = {
            use std::os::windows::fs::OpenOptionsExt;
            std::fs::OpenOptions::new()
                .read(true)
                .share_mode(1) // FILE_SHARE_READ only
                .open(&file)
                .expect("hold exclusive lock on previous file")
        };

        let listener_calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        // On the fixed tree the migration rewrite failure surfaces at open()
        // (TS: setSessionFile throws out of the constructor path). On the
        // baseline open() swallows it, so the same still-pending rewrite is
        // forced again via the public set_session_file switch with a listener
        // attached, and both lanes must see the failure either way.
        let rewrite_result: Option<Result<(), String>>;
        let mut baseline_open_ok = false;
        match SessionManager::open(file.to_string_lossy().as_ref(), None, None) {
            Ok(mut manager) => {
                baseline_open_ok = true;
                let counter = std::sync::Arc::clone(&listener_calls);
                let _unsub = manager.on_persist(Box::new(move |_| {
                    counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                }));
                rewrite_result =
                    Some(manager.set_session_file(file.to_string_lossy().as_ref(), None, None));
            }
            Err(error) => {
                // The fixed tree surfaces the failed rewrite from open().
                rewrite_result = None;
                assert!(
                    error.contains("session") || error.contains("write") || !error.is_empty(),
                    "open error must be a typed write failure: {error}"
                );
            }
        }
        let _ = baseline_open_ok;
        drop(lock);

        let surfaced = match &rewrite_result {
            Some(result) => result.is_err(),
            None => true,
        };
        if !surfaced {
            failures.push(
                "PART B: a failed replacement rewrite must surface as an error, got Ok".to_string(),
            );
        }
        if baseline_open_ok {
            let listener_count = listener_calls.load(std::sync::atomic::Ordering::SeqCst);
            if listener_count != 0 {
                failures.push(format!(
                    "PART B: listeners imply committed state after a failed rewrite: {listener_count}"
                ));
            }
        }

        // Previous file remains parseable and authoritative.
        assert!(Path::new(&file).is_file(), "previous file must survive");
        let entries = load_entries_from_file(file.to_string_lossy().as_ref());
        // load_entries_from_file includes the session header; the acknowledged
        // transcript must hold exactly header + one message entry.
        let non_header = entries
            .iter()
            .filter(|entry| entry.get("type").and_then(Value::as_str) != Some("session"))
            .count();
        if non_header != 1 {
            failures.push(format!(
                "PART B: previous file must keep its acknowledged entry (header intact), got {non_header} non-header entries of {} lines",
                entries.len()
            ));
        }
        // The header stays first and authoritative (control, both lanes).
        assert_eq!(
            entries[0].get("type").and_then(Value::as_str),
            Some("session"),
            "previous transcript must keep its session header first"
        );

        // Retry/reopen after releasing the lock must not lose acknowledged history.
        let after = std::fs::read(&file).unwrap();
        if before != after {
            failures.push("PART B: failed rewrite mutated acknowledged bytes".to_string());
        }
        let reopened = SessionManager::open(file.to_string_lossy().as_ref(), None, None)
            .expect("previous file must stay openable");
        assert_eq!(
            reopened.get_entries().len(),
            1,
            "acknowledged history survives"
        );
    }
    assert!(
        failures.is_empty(),
        "B-05 evidence (baseline deviations from the TS failure contract):\n{}",
        failures.join("\n")
    )
}

// ---------------------------------------------------------------------------
// history_boundary_and_order (B-03 + B-08)
// ---------------------------------------------------------------------------

fn compaction_fixture(dir: &Path, m1_ts: &str, summary_ts: &str) -> PathBuf {
    let file = dir.join("history.jsonl");
    let header = session_header("s1", "2026-09-16T00:00:00.000Z", "C:/fixture");
    let m1 = json!({
        "type": "message", "id": "m1", "parentId": Value::Null,
        "timestamp": m1_ts,
        "message": {"role": "user", "content": "before", "timestamp": 1},
    });
    let compaction = json!({
        "type": "compaction", "id": "c1", "parentId": "m1",
        "timestamp": summary_ts,
        "summary": "SUMMARY", "firstKeptEntryId": "m1", "tokensBefore": 100,
    });
    let m2 = json!({
        "type": "message", "id": "m2", "parentId": "c1",
        "timestamp": "2026-09-16T00:00:05.000Z",
        "message": {"role": "user", "content": "after", "timestamp": 2},
    });
    write_jsonl(file.as_path(), &[header, m1, compaction, m2]);
    file
}

#[test]
fn history_boundary_and_order() {
    let dir = case_dir("t06-history");
    let tie_ts = "2026-09-16T00:00:02.000Z";
    let mut failures: Vec<String> = Vec::new();

    // B-03: tied timestamps around the compaction summary.
    {
        let case_dir = dir.join("tie");
        std::fs::create_dir_all(&case_dir).unwrap();
        let file = compaction_fixture(case_dir.as_path(), tie_ts, tie_ts);
        let manager = SessionManager::open(file.to_string_lossy().as_ref(), None, None).expect("open");

        // Model-facing order is summary-first in both trees (must not change).
        let context = manager.build_session_context(None);
        let roles: Vec<String> = context
            .messages
            .iter()
            .map(|message| message.role().to_string())
            .collect();
        assert_eq!(
            roles,
            vec![
                "compactionSummary".to_string(),
                "user".to_string(),
                "user".to_string()
            ],
            "model order stays summary-first"
        );

        // Presentation order: TS places the summary after the retained prefix
        // (boundary = retainedMessageCount derived from firstKeptEntryId = 1),
        // even on timestamp ties.
        let snapshot = manager
            .build_session_history(Some("m2"), None)
            .expect("explicit tip history");
        if snapshot.entry_ids != vec!["m1".to_string(), "c1".to_string(), "m2".to_string()] {
            failures.push(format!(
                "B-03: tied timestamps must preserve the TS presentation boundary, got {:?}",
                snapshot.entry_ids
            ));
        }

        // B-08: a missing tip falls back to the current leaf (TS default parameter).
        match manager.build_session_history(None, None) {
            Err(error) => failures.push(format!(
                "B-08: missing tip must return the documented current-leaf history, got error {error}"
            )),
            Ok(missing_tip) => {
                if missing_tip.messages.is_empty() {
                    failures.push("B-08: missing tip returned an empty snapshot".to_string());
                }
                if missing_tip.entry_ids != snapshot.entry_ids {
                    failures.push(format!(
                        "B-08: missing tip must match the explicit current-leaf history, got {:?} vs {:?}",
                        missing_tip.entry_ids, snapshot.entry_ids
                    ));
                }
            }
        }

        // Negative control: an unknown tip stays an error.
        assert!(manager.build_session_history(Some("nope"), None).is_err());
    }

    // Control: legacy fallback agreement when no tie exists.
    {
        let untied_dir = dir.join("untied");
        std::fs::create_dir_all(&untied_dir).unwrap();
        let untied_file = compaction_fixture(untied_dir.as_path(), "2026-09-16T00:00:01.000Z", tie_ts);
        let untied =
            SessionManager::open(untied_file.to_string_lossy().as_ref(), None, None).expect("open");
        let untied_history = untied.build_session_history(Some("m2"), None).expect("history");
        if untied_history.entry_ids != vec!["m1".to_string(), "c1".to_string(), "m2".to_string()] {
            failures.push(format!(
                "B-03: untied fallback must agree with the count boundary, got {:?}",
                untied_history.entry_ids
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "B-03/B-08 evidence (baseline deviations from the TS order contract):\n{}",
        failures.join("\n")
    )
}

// ---------------------------------------------------------------------------
// unique_name_exhaustion_is_catchable (B-11)
// ---------------------------------------------------------------------------

#[test]
fn unique_name_exhaustion_is_catchable() {
    let dir = case_dir("t06-unique-names");
    let sessions = dir.join("sessions");
    std::fs::create_dir_all(&sessions).unwrap();

    // Control: normal operation creates distinct files and never aborts the
    // process. Twenty rapid creations exercise the loop's happy path.
    for index in 0..20 {
        let manager = SessionManager::create(
            dir.to_string_lossy().as_ref(),
            Some(sessions.to_string_lossy().as_ref()),
        )
        .unwrap_or_else(|error| panic!("creation {index} must not fail: {error}"));
        let file = manager.get_session_file().expect("session file");
        assert!(!Path::new(&file).exists(), "fresh target must be unique");
    }

    // Exhaustion contract: the injectable exists seam makes the otherwise
    // unreachable UUID collision path a mandatory regression assertion.
    let exhausted = unique_name_exhaustion_via_seam(&sessions);
    assert!(
        exhausted.is_err(),
        "exhaustion must return a catchable error, got {:?}",
        exhausted
    );
    assert!(
        exhausted
            .unwrap_err()
            .contains("Unable to create a unique session file"),
        "typed error text must match the TS message"
    );
}

fn unique_name_exhaustion_via_seam(sessions: &Path) -> Result<String, String> {
    match pi_coding_agent::core::session_manager::create_unique_session_file_target_with(
        sessions.to_string_lossy().as_ref(),
        &|_candidate| true, // every probe reports "exists": exhaust the loop
    ) {
        Ok((session_id, session_file)) => Err(format!(
            "exhaustion seam did not fire: {session_id}:{session_file}"
        )),
        Err(error) => Err(error),
    }
}
