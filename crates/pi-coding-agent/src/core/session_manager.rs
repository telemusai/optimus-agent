//! Port of packages/coding-agent/src/core/session-manager.ts
//!
//! Cross-slice dependencies are provided as PRIVATE plumbing in this module so
//! no other slice's file is touched:
//!   - core/messages.ts        (createCustomMessage/createBranchSummaryMessage/createCompactionSummaryMessage)
//!   - core/usage.ts           (emptyUsage/addAssistantUsage/subtractAssistantUsage/cloneUsage/sessionUsageSummaryFrom)
//!   - core/compaction/checkpoint.ts (hasProviderCheckpoint/getProviderCheckpoint)
//!   - utils/atomic-file.ts    (writeFileAtomicSync/realpathIfPresentSync)
//!   - utils/git.ts             (captureGitContext/GitContext/gitContextsEqual)
//!   - config.ts                (getAgentDir/getSessionsDir)
//! All of these are recorded in blocked_on.
//!
//! `FileEntry` is kept as `serde_json::Value` on purpose: the TypeScript parses a
//! JSONL line, mutates a few fields during migration, and writes every unknown
//! field back verbatim on rewrite. A typed enum would drop unknown fields.

#![allow(clippy::too_many_arguments)]

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use pi_agent_core::types::{AgentMessage, CustomAgentMessage, CustomMessageContent};
use pi_ai::compaction::ProviderCompactionCheckpoint;
use pi_ai::types::{Message, ServiceTier, Usage};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::utils::file_lines::{
    read_bytes_sync, read_first_line_sync, read_lines_as_buffers, ReadLinesRange,
};

pub const CURRENT_SESSION_VERSION: i64 = 3;
const SESSION_LIST_SEARCH_TEXT_MAX_CHARS: usize = 64 * 1024;
const SESSION_LIST_PARSE_MAX_LINE_CHARS: usize = 1024 * 1024;
const SESSION_LIST_LARGE_MESSAGE_PREVIEW_MAX_CHARS: usize = 256;
const SESSION_STREAMING_LOAD_THRESHOLD_BYTES: u64 = 128 * 1024 * 1024;
const SESSION_ASYNC_PARSE_YIELD_BYTES: usize = 4 * 1024 * 1024;

/// The exact `CONTENT_ENTRY_TYPES` set (deduplicated).
fn content_entry_types() -> &'static HashSet<&'static str> {
    static TYPES: OnceLock<HashSet<&'static str>> = OnceLock::new();
    TYPES.get_or_init(|| {
        [
            "message",
            "custom_message",
            "custom",
            "model_change",
            "thinking_level_change",
            "service_tier_change",
            "session_info",
            "label",
            "compaction",
            "branch_summary",
        ]
        .into_iter()
        .collect()
    })
}

// ---------------------------------------------------------------------------
// PRIVATE stand-in: utils/atomic-file.ts
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Default)]
struct WriteFileAtomicOptions {
    /// `mode?` - applied exactly (umask-masked bits are re-applied).
    mode: Option<u32>,
    fsync: bool,
}

/// Port of `writeFileAtomicSync` for the options used by the session manager.
fn write_file_atomic_sync(
    path: &str,
    data: &str,
    options: WriteFileAtomicOptions,
    before_rename: Option<&dyn Fn(&str) -> Result<(), String>>,
) -> Result<(), String> {
    let temp_path = format!("{path}.{}.{}.tmp", std::process::id(), uuid::Uuid::new_v4());
    let result = (|| -> Result<(), String> {
        std::fs::write(&temp_path, data.as_bytes()).map_err(|error| error.to_string())?;
        if let Some(mode) = options.mode {
            set_file_mode(&temp_path, mode);
        }
        if let Some(before_rename) = before_rename {
            before_rename(&temp_path)?;
        }
        rename_onto_sync(&temp_path, path).map_err(|error| error.to_string())
    })();
    let _ = std::fs::remove_file(&temp_path);
    result
}

fn set_file_mode(path: &str, mode: u32) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode));
    }
    #[cfg(not(unix))]
    {
        let _ = (path, mode);
    }
}

const WIN32_RENAME_ATTEMPTS: usize = 5;

fn is_transient_windows_rename_error(error: &std::io::Error) -> bool {
    if !cfg!(windows) {
        return false;
    }
    matches!(
        error.kind(),
        std::io::ErrorKind::PermissionDenied | std::io::ErrorKind::ResourceBusy
    )
}

// Windows raises transient EPERM/EACCES when the destination is held open (antivirus, indexer).
fn rename_onto_sync(from: &str, to: &str) -> std::io::Result<()> {
    let mut attempt = 1usize;
    loop {
        match std::fs::rename(from, to) {
            Ok(()) => return Ok(()),
            Err(error) => {
                if !is_transient_windows_rename_error(&error) || attempt >= WIN32_RENAME_ATTEMPTS {
                    return Err(error);
                }
                std::thread::sleep(std::time::Duration::from_millis(10 * attempt as u64));
                attempt += 1;
            }
        }
    }
}

/// Port of `realpathIfPresentSync`.
fn realpath_if_present_sync(path: &str) -> String {
    match std::fs::canonicalize(path) {
        Ok(canonical) => canonical.to_string_lossy().to_string(),
        Err(_) => path.to_string(),
    }
}

// ---------------------------------------------------------------------------
// PRIVATE stand-in: utils/git.ts
// ---------------------------------------------------------------------------

/// Port of `interface GitContext` from utils/git.ts.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GitContext {
    #[serde(rename = "repoUrl", skip_serializing_if = "Option::is_none")]
    pub repo_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub commit: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub branch: Option<String>,
}

pub fn git_contexts_equal(a: &GitContext, b: &GitContext) -> bool {
    a.repo_url == b.repo_url && a.commit == b.commit && a.branch == b.branch
}

fn run_git(cwd: &str, args: &[&str]) -> Option<String> {
    let mut full_args: Vec<String> = vec!["--no-optional-locks".to_string()];
    full_args.extend(args.iter().map(|arg| arg.to_string()));
    let mut process = std::process::Command::new("git");
    process.args(&full_args);
    process.current_dir(cwd);
    process.stdin(std::process::Stdio::null());
    process.stdout(std::process::Stdio::piped());
    process.stderr(std::process::Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        process.creation_flags(CREATE_NO_WINDOW);
    }
    let output = process.output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

/// Port of `parseGitUrl(remote)?.repo` for URL-shaped and scp-like remotes; the
/// hosted-git-info shorthand resolution is not ported (see blocked_on).
fn git_repo_url_from_remote(remote: &str) -> String {
    let split = split_git_ref(remote);
    let repo_without_ref = split.0;
    if repo_without_ref.starts_with("https://")
        || repo_without_ref.starts_with("http://")
        || repo_without_ref.starts_with("ssh://")
        || repo_without_ref.starts_with("git://")
        || repo_without_ref.starts_with("git@")
    {
        return repo_without_ref;
    }
    remote.to_string()
}

fn split_git_ref(url: &str) -> (String, Option<String>) {
    if let Some(rest) = url.strip_prefix("git@") {
        if let Some(colon) = rest.find(':') {
            let path_with_maybe_ref = &rest[colon + 1..];
            if let Some(separator) = path_with_maybe_ref.find('@') {
                let repo_path = &path_with_maybe_ref[..separator];
                let reference = &path_with_maybe_ref[separator + 1..];
                if !repo_path.is_empty() && !reference.is_empty() {
                    return (
                        format!("git@{}:{repo_path}", &rest[..colon]),
                        Some(reference.to_string()),
                    );
                }
            }
        }
        return (url.to_string(), None);
    }
    if url.contains("://") {
        if let Some(scheme_end) = url.find("://") {
            let after_scheme = &url[scheme_end + 3..];
            if let Some(slash) = after_scheme.find('/') {
                let authority = &after_scheme[..slash];
                let path_with_maybe_ref = after_scheme[slash + 1..].trim_start_matches('/');
                if let Some(separator) = path_with_maybe_ref.find('@') {
                    let repo_path = &path_with_maybe_ref[..separator];
                    let reference = &path_with_maybe_ref[separator + 1..];
                    if !repo_path.is_empty() && !reference.is_empty() {
                        let repo = format!("{}://{authority}/{repo_path}", &url[..scheme_end]);
                        return (
                            repo.trim_end_matches('/').to_string(),
                            Some(reference.to_string()),
                        );
                    }
                }
            }
        }
        return (url.to_string(), None);
    }
    (url.to_string(), None)
}

pub fn capture_git_context(cwd: &str) -> Option<GitContext> {
    let commit = run_git(cwd, &["rev-parse", "HEAD"]);
    let branch = run_git(cwd, &["branch", "--show-current"]);
    let remote = run_git(cwd, &["remote", "get-url", "origin"]);
    if commit.is_none() && branch.is_none() && remote.is_none() {
        return None;
    }

    let mut context = GitContext::default();
    if let Some(remote) = remote {
        context.repo_url = Some(git_repo_url_from_remote(&remote));
    }
    if let Some(commit) = commit {
        context.commit = Some(commit);
    }
    if let Some(branch) = branch {
        context.branch = Some(branch);
    }
    Some(context)
}

// ---------------------------------------------------------------------------
// PRIVATE stand-in: core/usage.ts
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SessionUsageSummary {
    #[serde(rename = "inputTokens")]
    pub input_tokens: f64,
    #[serde(rename = "outputTokens")]
    pub output_tokens: f64,
    pub cost: f64,
}

pub fn session_usage_summary_from(usage: &Usage) -> Option<SessionUsageSummary> {
    let input_tokens = usage.input + usage.cache_read + usage.cache_write;
    if input_tokens == 0.0 && usage.output == 0.0 && usage.cost.total == 0.0 {
        return None;
    }
    Some(SessionUsageSummary {
        input_tokens,
        output_tokens: usage.output,
        cost: usage.cost.total,
    })
}

pub fn empty_usage() -> Usage {
    Usage::default()
}

pub fn add_assistant_usage(total: &mut Usage, usage: &Usage) {
    total.input += usage.input;
    total.output += usage.output;
    total.cache_read += usage.cache_read;
    total.cache_write += usage.cache_write;
    total.total_tokens += usage.total_tokens;
    total.cost.input += usage.cost.input;
    total.cost.output += usage.cost.output;
    total.cost.cache_read += usage.cost.cache_read;
    total.cost.cache_write += usage.cost.cache_write;
    total.cost.total += usage.cost.total;
}

/// Remove a previously added usage, clamping at zero to absorb attribution drift.
pub fn subtract_assistant_usage(total: &mut Usage, usage: &Usage) {
    total.input = (total.input - usage.input).max(0.0);
    total.output = (total.output - usage.output).max(0.0);
    total.cache_read = (total.cache_read - usage.cache_read).max(0.0);
    total.cache_write = (total.cache_write - usage.cache_write).max(0.0);
    total.total_tokens = (total.total_tokens - usage.total_tokens).max(0.0);
    total.cost.input = (total.cost.input - usage.cost.input).max(0.0);
    total.cost.output = (total.cost.output - usage.cost.output).max(0.0);
    total.cost.cache_read = (total.cost.cache_read - usage.cost.cache_read).max(0.0);
    total.cost.cache_write = (total.cost.cache_write - usage.cost.cache_write).max(0.0);
    total.cost.total = (total.cost.total - usage.cost.total).max(0.0);
}

pub fn clone_usage(usage: &Usage) -> Usage {
    usage.clone()
}

// ---------------------------------------------------------------------------
// PRIVATE stand-in: config.ts
// ---------------------------------------------------------------------------

// Session paths follow the user-config contract (config.ts getAgentDir /
// getSessionsDir): PRIME_AGENT_* env names with tilde expansion, PRIME_AGENT
// precedence over the legacy alias, default <home>/.prime/agent.
fn get_default_agent_dir() -> String {
    crate::config::get_agent_dir()
}

fn get_sessions_dir(agent_dir: &str) -> String {
    crate::config::get_sessions_dir(Some(agent_dir))
}

// ---------------------------------------------------------------------------
// PRIVATE stand-in: core/compaction/checkpoint.ts
// ---------------------------------------------------------------------------

fn is_legacy_checkpoint(details: &Map<String, Value>) -> bool {
    details.get("strategy").and_then(Value::as_str) == Some("openai-responses-compaction-v2")
}

pub fn has_provider_checkpoint(details: &Value) -> bool {
    match details.as_object() {
        Some(details) => {
            details.contains_key("providerCheckpoint") || is_legacy_checkpoint(details)
        }
        None => false,
    }
}

pub fn get_provider_checkpoint(details: &Value) -> Option<ProviderCompactionCheckpoint> {
    let details = details.as_object()?;
    if let Some(checkpoint) = details.get("providerCheckpoint") {
        if pi_ai::compaction::is_compaction_checkpoint(checkpoint) {
            return serde_json::from_value(checkpoint.clone()).ok();
        }
        return None;
    }
    // Read older extension checkpoints without rewriting their opaque provider items.
    if !is_legacy_checkpoint(details) {
        return None;
    }
    let items = details.get("compactedWindow")?.as_array()?;
    let has_encrypted = items.iter().any(|item| {
        item.as_object().map_or(false, |item| {
            item.get("type").and_then(Value::as_str) == Some("compaction")
                && item
                    .get("encrypted_content")
                    .and_then(Value::as_str)
                    .map_or(false, |value| !value.is_empty())
        })
    });
    if !has_encrypted {
        return None;
    }
    let mut images = 0usize;
    let serialized = stringify_with_image_placeholders(&Value::Array(items.clone()), &mut images);
    let checkpoint = serde_json::json!({
        "version": 1,
        "provider": details.get("provider").cloned().unwrap_or(Value::Null),
        "api": details.get("api").cloned().unwrap_or(Value::Null),
        "model": details.get("model").cloned().unwrap_or(Value::Null),
        "baseUrl": details.get("baseUrl").cloned().unwrap_or(Value::Null),
        "items": Value::Array(items.clone()),
        "estimatedTokens": ((serialized.chars().count() as f64) / 4.0).ceil() + (images as f64) * 1200.0,
    });
    if pi_ai::compaction::is_compaction_checkpoint(&checkpoint) {
        return serde_json::from_value(checkpoint).ok();
    }
    None
}

/// `JSON.stringify(value, replacer)` that replaces data-image URLs with "(image)".
fn stringify_with_image_placeholders(value: &Value, images: &mut usize) -> String {
    let replaced = replace_data_images(value, images);
    serde_json::to_string(&replaced).unwrap_or_default()
}

fn replace_data_images(value: &Value, images: &mut usize) -> Value {
    match value {
        Value::Object(map) => {
            let mut out = Map::new();
            for (key, entry) in map {
                if (key == "image_url" || key == "url")
                    && entry
                        .as_str()
                        .map_or(false, |text| text.starts_with("data:image/"))
                {
                    *images += 1;
                    out.insert(key.clone(), Value::String("(image)".to_string()));
                } else {
                    out.insert(key.clone(), replace_data_images(entry, images));
                }
            }
            Value::Object(out)
        }
        Value::Array(items) => Value::Array(
            items
                .iter()
                .map(|item| replace_data_images(item, images))
                .collect(),
        ),
        other => other.clone(),
    }
}

// ---------------------------------------------------------------------------
// Entry helpers
//
// Entries stay `serde_json::Map` so unknown fields survive a load -> rewrite
// round trip exactly like a JavaScript object does.
// ---------------------------------------------------------------------------

pub type SessionEntry = Map<String, Value>;
pub type FileEntry = Map<String, Value>;

fn entry_type(entry: &SessionEntry) -> &str {
    entry.get("type").and_then(Value::as_str).unwrap_or("")
}

fn entry_id(entry: &SessionEntry) -> String {
    entry
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

fn entry_parent_id(entry: &SessionEntry) -> Option<String> {
    match entry.get("parentId") {
        Some(Value::String(value)) => Some(value.clone()),
        _ => None,
    }
}

fn entry_timestamp(entry: &SessionEntry) -> String {
    entry
        .get("timestamp")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

fn message_of(entry: &SessionEntry) -> Option<&Map<String, Value>> {
    entry.get("message").and_then(Value::as_object)
}

fn message_role(entry: &SessionEntry) -> &str {
    message_of(entry)
        .and_then(|message| message.get("role"))
        .and_then(Value::as_str)
        .unwrap_or("")
}

fn is_session_header(entry: &SessionEntry) -> bool {
    entry_type(entry) == "session"
}

fn json_stringify(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "null".to_string())
}

fn json_byte_len(value: &Value) -> usize {
    json_stringify(value).len()
}

fn is_safe_integer(value: &Value) -> Option<i64> {
    let number = value.as_f64()?;
    if number.is_finite() && number.fract() == 0.0 && number.abs() <= 9_007_199_254_740_991.0 {
        Some(number as i64)
    } else {
        None
    }
}

fn is_nonnegative_safe_integer(value: &Value) -> Option<i64> {
    let integer = is_safe_integer(value)?;
    if integer >= 0 {
        Some(integer)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Inline tool-text encoding
// ---------------------------------------------------------------------------

const INLINE_TOOL_TEXT_REFERENCE_KEY: &str = "$primeToolText";
const INLINE_TOOL_TEXT_REFERENCE_VERSION: i64 = 1;
const INLINE_TOOL_TEXT_ENCODING_KEY: &str = "$primeSessionEncoding";
const INLINE_TOOL_TEXT_ENCODING_KIND: &str = "inline_tool_text";
const TOOL_DETAIL_TEXT_KEYS: [&str; 4] = ["stdout", "stderr", "result", "backgroundOutput"];

fn sha256_text(value: &str) -> String {
    // JSON's escaped string form is lossless for every JavaScript UTF-16 code unit,
    // including lone surrogates that raw UTF-8 would normalize to U+FFFD.
    let mut hasher = Sha256::new();
    hasher.update(json_stringify(&Value::String(value.to_string())).as_bytes());
    let digest = hasher.finalize();
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

fn tool_result_content_text(message: &Map<String, Value>, content_index: usize) -> Option<String> {
    match message.get("content") {
        Some(Value::String(content)) => {
            if content_index == 0 {
                Some(content.clone())
            } else {
                None
            }
        }
        Some(Value::Array(parts)) => {
            let part = parts.get(content_index)?;
            let candidate = part.as_object()?;
            if candidate.get("type").and_then(Value::as_str) == Some("text") {
                candidate
                    .get("text")
                    .and_then(Value::as_str)
                    .map(str::to_string)
            } else {
                None
            }
        }
        _ => None,
    }
}

fn inline_tool_text_reference(message: &Map<String, Value>, value: &str) -> Option<Value> {
    if value.is_empty() {
        return None;
    }
    let content_count = match message.get("content") {
        Some(Value::Array(parts)) => parts.len(),
        Some(Value::String(_)) => 1,
        _ => 0,
    };
    for content_index in 0..content_count {
        let text = tool_result_content_text(message, content_index);
        // TypeScript stores start/length in UTF-16 code units (JS string
        // indices), not bytes and not chars (session-manager.ts:300-308).
        let start = match &text {
            Some(text) => match text.find(value) {
                Some(index) => text[..index].encode_utf16().count() as i64,
                None => continue,
            },
            None => continue,
        };
        let reference = serde_json::json!({
            INLINE_TOOL_TEXT_REFERENCE_KEY: {
                "version": INLINE_TOOL_TEXT_REFERENCE_VERSION,
                "source": "content",
                "contentIndex": content_index,
                "start": start,
                "length": value.encode_utf16().count(),
                "sha256": sha256_text(value),
            }
        });
        // Small values cost more as references and are intentionally left inline.
        if json_byte_len(&reference) < json_byte_len(&Value::String(value.to_string())) {
            return Some(reference);
        }
    }
    None
}

fn is_tool_detail_text_key(value: &str) -> bool {
    TOOL_DETAIL_TEXT_KEYS.contains(&value)
}

fn encoded_inline_tool_text_fields(entry: &FileEntry) -> Option<Vec<String>> {
    let envelope = entry.get(INLINE_TOOL_TEXT_ENCODING_KEY)?.as_object()?;
    let fields = envelope.get("fields")?.as_array()?;
    if envelope.get("version").and_then(Value::as_i64) != Some(INLINE_TOOL_TEXT_REFERENCE_VERSION)
        || envelope.get("kind").and_then(Value::as_str) != Some(INLINE_TOOL_TEXT_ENCODING_KIND)
        || fields.is_empty()
    {
        return None;
    }
    let mut names: Vec<String> = Vec::with_capacity(fields.len());
    for field in fields {
        let name = field.as_str()?;
        if !is_tool_detail_text_key(name) {
            return None;
        }
        names.push(name.to_string());
    }
    let unique: BTreeSet<&String> = names.iter().collect();
    if unique.len() != names.len() {
        return None;
    }
    Some(names)
}

/// Encode exact tool detail/content duplication within one JSONL entry.
/// The source text remains inline and the reference is versioned, hashed and reconstructible without external state.
pub fn serialize_session_file_entry(entry: &FileEntry) -> String {
    let original = json_stringify(&Value::Object(entry.clone()));
    if entry_type(entry) != "message" || message_role(entry) != "toolResult" {
        return original;
    }
    let message = match message_of(entry) {
        Some(message) => message.clone(),
        None => return original,
    };
    let details = match message.get("details").and_then(Value::as_object) {
        Some(details) => details.clone(),
        None => return original,
    };
    let mut encoded_details: Option<Map<String, Value>> = None;
    let mut encoded_fields: Vec<String> = Vec::new();
    for key in TOOL_DETAIL_TEXT_KEYS {
        let value = match details.get(key).and_then(Value::as_str) {
            Some(value) => value.to_string(),
            None => continue,
        };
        let reference = match inline_tool_text_reference(&message, &value) {
            Some(reference) => reference,
            None => continue,
        };
        let target = encoded_details.get_or_insert_with(|| details.clone());
        target.insert(key.to_string(), reference);
        encoded_fields.push(key.to_string());
    }
    let encoded_details = match encoded_details {
        Some(encoded_details) => encoded_details,
        None => return original,
    };
    let mut candidate_entry = entry.clone();
    candidate_entry.insert(
        INLINE_TOOL_TEXT_ENCODING_KEY.to_string(),
        serde_json::json!({
            "version": INLINE_TOOL_TEXT_REFERENCE_VERSION,
            "kind": INLINE_TOOL_TEXT_ENCODING_KIND,
            "fields": encoded_fields,
        }),
    );
    let mut candidate_message = message.clone();
    candidate_message.insert("details".to_string(), Value::Object(encoded_details));
    candidate_entry.insert("message".to_string(), Value::Object(candidate_message));
    let candidate = json_stringify(&Value::Object(candidate_entry));
    // The disk-only discriminator has a fixed cost; never expand a row merely to encode references.
    if candidate.len() < original.len() {
        candidate
    } else {
        original
    }
}

fn decode_inline_tool_text_reference(
    message: &Map<String, Value>,
    value: &Value,
) -> Option<String> {
    let candidate = value.as_object()?.get(INLINE_TOOL_TEXT_REFERENCE_KEY)?;
    let reference = candidate.as_object()?;
    if reference.get("version").and_then(Value::as_i64) != Some(INLINE_TOOL_TEXT_REFERENCE_VERSION)
        || reference.get("source").and_then(Value::as_str) != Some("content")
    {
        return None;
    }
    let content_index = is_nonnegative_safe_integer(reference.get("contentIndex")?)? as usize;
    let start = is_nonnegative_safe_integer(reference.get("start")?)? as usize;
    let length = is_nonnegative_safe_integer(reference.get("length")?)? as usize;
    let sha = reference.get("sha256").and_then(Value::as_str)?;
    if sha.len() != 64
        || !sha
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return None;
    }
    let source = tool_result_content_text(message, content_index)?;
    // TS slices the source by UTF-16 code units (session-manager.ts:395-399).
    let source_units: Vec<u16> = source.encode_utf16().collect();
    if length > source_units.len() || start > source_units.len() - length {
        return None;
    }
    let decoded: String = String::from_utf16_lossy(&source_units[start..start + length]);
    if sha256_text(&decoded) == sha {
        Some(decoded)
    } else {
        None
    }
}

/// Rehydrate explicitly tagged disk-only references before entries reach model, UI, export or observation callers.
pub fn rehydrate_session_file_entry(entry: FileEntry) -> FileEntry {
    let encoded_fields = match encoded_inline_tool_text_fields(&entry) {
        Some(fields) => fields,
        None => return entry,
    };
    if entry_type(&entry) != "message" || message_role(&entry) != "toolResult" {
        return entry;
    }
    let mut public_entry = entry.clone();
    public_entry.shift_remove(INLINE_TOOL_TEXT_ENCODING_KEY);
    let message = match message_of(&entry) {
        Some(message) => message.clone(),
        None => return entry,
    };
    let source_details = match message.get("details").and_then(Value::as_object) {
        Some(details) => details.clone(),
        None => Map::new(),
    };
    let mut decoded_details = source_details.clone();
    let mut failed_keys: Vec<String> = Vec::new();
    for key in &encoded_fields {
        match decode_inline_tool_text_reference(
            &message,
            source_details.get(key).unwrap_or(&Value::Null),
        ) {
            Some(decoded) => {
                decoded_details.insert(key.clone(), Value::String(decoded));
            }
            None => {
                failed_keys.push(key.clone());
                decoded_details.insert(
                    key.clone(),
                    Value::String(format!(
                        "[session recovery error: invalid inline {key} reference]"
                    )),
                );
            }
        }
    }
    let mut out_message = message.clone();
    out_message.insert("details".to_string(), Value::Object(decoded_details));
    if failed_keys.is_empty() {
        public_entry.insert("message".to_string(), Value::Object(out_message));
        return public_entry;
    }
    let diagnostic = format!(
        "[session recovery error: could not reconstruct {} from tool-result content]",
        failed_keys.join(", ")
    );
    let content = match message.get("content") {
        Some(Value::Array(parts)) => {
            let mut parts = parts.clone();
            parts.push(serde_json::json!({ "type": "text", "text": diagnostic }));
            Value::Array(parts)
        }
        Some(Value::String(text)) => Value::String(format!("{text}\n{diagnostic}")),
        _ => Value::Array(vec![
            serde_json::json!({ "type": "text", "text": diagnostic }),
        ]),
    };
    out_message.insert("content".to_string(), content);
    out_message.insert("isError".to_string(), Value::Bool(true));
    public_entry.insert("message".to_string(), Value::Object(out_message));
    public_entry
}

// ---------------------------------------------------------------------------
// Session context building
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct SessionContext {
    pub messages: Vec<AgentMessage>,
    pub thinking_level: String,
    pub service_tier: ServiceTier,
    pub model: Option<SessionContextModel>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionContextModel {
    pub provider: String,
    pub model_id: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SessionContextWithEntryIds {
    pub messages: Vec<AgentMessage>,
    /// Stable session-entry id aligned by index with messages.
    pub entry_ids: Vec<String>,
    pub thinking_level: String,
    pub service_tier: ServiceTier,
    pub model: Option<SessionContextModel>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SessionHistorySnapshot {
    /// Chronological presentation order; model-facing context order is unchanged.
    pub messages: Vec<AgentMessage>,
    pub entry_ids: Vec<String>,
    pub tip_entry_id: Option<String>,
}

struct SessionContextMessage {
    entry_id: String,
    message: AgentMessage,
}

/// Private checkpoint/model comparison mirroring `compactionMatchesModel`, whose
/// `model` argument may omit `api`/`baseUrl`/`nativeCompaction`.
struct CheckpointModelTarget {
    provider: String,
    id: String,
    api: Option<String>,
    base_url: Option<String>,
    native_endpoint: Option<String>,
}

fn trim_trailing_slashes(value: &str) -> &str {
    value.trim_end_matches('/')
}

fn compaction_matches_target(
    checkpoint: &ProviderCompactionCheckpoint,
    model: &CheckpointModelTarget,
) -> bool {
    if checkpoint.provider != model.provider || checkpoint.model != model.id {
        return false;
    }
    if let Some(api) = &model.api {
        if &checkpoint.api != api {
            return false;
        }
    }
    if let Some(base_url) = &model.base_url {
        if trim_trailing_slashes(&checkpoint.base_url) != trim_trailing_slashes(base_url) {
            return false;
        }
    }
    match (&checkpoint.endpoint, &model.native_endpoint) {
        (None, None) => true,
        (Some(endpoint), Some(native_endpoint)) => {
            trim_trailing_slashes(endpoint) == trim_trailing_slashes(native_endpoint)
        }
        _ => false,
    }
}

pub fn iso_to_millis(timestamp: &str) -> f64 {
    match chrono::DateTime::parse_from_rfc3339(timestamp) {
        Ok(parsed) => parsed.timestamp_millis() as f64,
        Err(_) => f64::NAN,
    }
}

/// `new Date().toISOString()`.
fn iso_now() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    let millis = now.subsec_millis();
    let days = (secs / 86_400) as i64;
    let seconds_of_day = secs % 86_400;
    let (year, month, day) = civil_from_days(days);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{millis:03}Z",
        seconds_of_day / 3600,
        (seconds_of_day % 3600) / 60,
        seconds_of_day % 60
    )
}

fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

fn create_custom_message(
    custom_type: &str,
    content: Value,
    display: bool,
    details: Option<Value>,
    timestamp: &str,
) -> AgentMessage {
    let content = serde_json::from_value::<CustomMessageContent>(content)
        .unwrap_or_else(|_| CustomMessageContent::Text(String::new()));
    AgentMessage::Custom(CustomAgentMessage::Custom {
        custom_type: custom_type.to_string(),
        content,
        display,
        details,
        timestamp: iso_to_millis(timestamp) as i64,
    })
}

fn create_branch_summary_message(summary: &str, from_id: &str, timestamp: &str) -> AgentMessage {
    AgentMessage::Custom(CustomAgentMessage::BranchSummary {
        summary: summary.to_string(),
        from_id: from_id.to_string(),
        timestamp: iso_to_millis(timestamp) as i64,
    })
}

fn create_compaction_summary_message(
    summary: &str,
    tokens_before: f64,
    timestamp: &str,
    custom_instructions: Option<String>,
    retained_message_count: Option<f64>,
    provider_context: Option<ProviderCompactionCheckpoint>,
    harness_digest: Option<String>,
) -> AgentMessage {
    AgentMessage::Custom(CustomAgentMessage::CompactionSummary {
        summary: summary.to_string(),
        provider_context,
        tokens_before,
        retained_message_count,
        custom_instructions,
        harness_digest,
        timestamp: iso_to_millis(timestamp) as i64,
    })
}

pub fn get_latest_compaction_entry(entries: &[SessionEntry]) -> Option<SessionEntry> {
    for entry in entries.iter().rev() {
        if entry_type(entry) == "compaction" {
            return Some(entry.clone());
        }
    }
    None
}

pub fn build_session_context_with_entry_ids(
    entries: &[SessionEntry],
    leaf_id: Option<Option<&str>>,
    by_id: Option<&BTreeMap<String, SessionEntry>>,
    target_model: Option<&pi_ai::types::Model>,
) -> SessionContextWithEntryIds {
    let owned_index;
    let by_id = match by_id {
        Some(by_id) => by_id,
        None => {
            let mut map: BTreeMap<String, SessionEntry> = BTreeMap::new();
            for entry in entries {
                map.insert(entry_id(entry), entry.clone());
            }
            owned_index = map;
            &owned_index
        }
    };

    let empty = SessionContextWithEntryIds {
        messages: Vec::new(),
        entry_ids: Vec::new(),
        thinking_level: "off".to_string(),
        service_tier: Some(Some("default".to_string())),
        model: None,
    };

    let resolved_leaf: Option<SessionEntry> = match leaf_id {
        // `leafId === null` resolves no context at all.
        Some(None) => return empty,
        Some(Some(id)) => match by_id.get(id) {
            Some(entry) => Some(entry.clone()),
            None => entries.last().cloned(),
        },
        None => entries.last().cloned(),
    };
    let leaf = match resolved_leaf {
        Some(leaf) => leaf,
        None => return empty,
    };

    // push+reverse, not unshift-per-entry: unshift is O(n), making this O(n^2) on long sessions.
    let mut path: Vec<SessionEntry> = Vec::new();
    let mut current = Some(leaf);
    let mut visited = HashSet::new();
    while let Some(entry) = current {
        // Damaged ancestry must not repeat entries or grow context without bound.
        if !visited.insert(entry_id(&entry)) { break; }
        current = entry_parent_id(&entry).and_then(|parent| by_id.get(&parent).cloned());
        path.push(entry);
    }
    path.reverse();

    let mut thinking_level = "off".to_string();
    let mut service_tier: ServiceTier = Some(Some("default".to_string()));
    let mut model: Option<SessionContextModel> = None;
    let mut compaction: Option<SessionEntry> = None;

    for entry in &path {
        match entry_type(entry) {
            "thinking_level_change" => {
                if let Some(level) = entry.get("thinkingLevel").and_then(Value::as_str) {
                    thinking_level = level.to_string();
                }
            }
            "service_tier_change" => {
                service_tier = entry
                    .get("serviceTier")
                    .map(|value| serde_json::from_value(value.clone()).unwrap_or(None))
                    .unwrap_or(None);
            }
            "model_change" => {
                model = Some(SessionContextModel {
                    provider: entry
                        .get("provider")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    model_id: entry
                        .get("modelId")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                });
            }
            "message" if message_role(entry) == "assistant" => {
                if let Some(message) = message_of(entry) {
                    model = Some(SessionContextModel {
                        provider: message
                            .get("provider")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string(),
                        model_id: message
                            .get("model")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string(),
                    });
                }
            }
            _ => {}
        }
    }

    // Opaque windows belong to one provider/model/endpoint. A switch or an unknown
    // checkpoint version reconstructs context from the durable original transcript.
    let checkpoint_model: Option<CheckpointModelTarget> = match target_model {
        Some(target) => Some(CheckpointModelTarget {
            provider: target.provider.clone(),
            id: target.id.clone(),
            api: Some(target.api.clone()),
            base_url: Some(target.base_url.clone()),
            native_endpoint: target
                .native_compaction
                .as_ref()
                .map(|native| native.endpoint.clone()),
        }),
        None => model.as_ref().map(|model| CheckpointModelTarget {
            provider: model.provider.clone(),
            id: model.model_id.clone(),
            api: None,
            base_url: None,
            native_endpoint: None,
        }),
    };
    for entry in &path {
        if entry_type(entry) != "compaction" {
            continue;
        }
        let details = entry.get("details").cloned().unwrap_or(Value::Null);
        if has_provider_checkpoint(&details) {
            let checkpoint = get_provider_checkpoint(&details);
            let Some(checkpoint) = checkpoint else {
                continue;
            };
            let Some(checkpoint_model) = checkpoint_model.as_ref() else {
                continue;
            };
            if !compaction_matches_target(&checkpoint, checkpoint_model) {
                continue;
            }
        }
        compaction = Some(entry.clone());
    }

    // Build messages and collect corresponding entries
    // When there's a compaction, model context remains summary-first while the
    // summary records where clients should present it among retained messages.
    let mut message_entries: Vec<SessionContextMessage> = Vec::new();

    fn append_message(entry: &SessionEntry, target: &mut Vec<SessionContextMessage>) {
        match entry_type(entry) {
            "message" => {
                if let Some(message) = entry.get("message") {
                    if let Ok(message) = serde_json::from_value::<AgentMessage>(message.clone()) {
                        target.push(SessionContextMessage {
                            entry_id: entry_id(entry),
                            message,
                        });
                    }
                }
            }
            "custom_message" => {
                let custom_type = entry
                    .get("customType")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let content = entry.get("content").cloned().unwrap_or(Value::Null);
                let display = entry
                    .get("display")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                let details = entry.get("details").cloned();
                let timestamp = entry_timestamp(entry);
                target.push(SessionContextMessage {
                    entry_id: entry_id(entry),
                    message: create_custom_message(
                        &custom_type,
                        content,
                        display,
                        details,
                        &timestamp,
                    ),
                });
            }
            "branch_summary" => {
                let summary = entry
                    .get("summary")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                if !summary.is_empty() {
                    let from_id = entry
                        .get("fromId")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string();
                    let timestamp = entry_timestamp(entry);
                    target.push(SessionContextMessage {
                        entry_id: entry_id(entry),
                        message: create_branch_summary_message(&summary, &from_id, &timestamp),
                    });
                }
            }
            _ => {}
        }
    }

    match compaction {
        Some(compaction) => {
            let compaction_id = entry_id(&compaction);
            let compaction_idx = path
                .iter()
                .position(|entry| {
                    entry_type(entry) == "compaction" && entry_id(entry) == compaction_id
                })
                .unwrap_or(0);
            let provider_context =
                get_provider_checkpoint(compaction.get("details").unwrap_or(&Value::Null));

            // Collect kept messages (before compaction, starting from firstKeptEntryId).
            // The context remains summary-first for the model; retainedMessageCount records
            // the exact chronological presentation boundary for clients.
            let mut retained_messages: Vec<SessionContextMessage> = Vec::new();
            let mut found_first_kept = false;
            for index in 0..compaction_idx {
                if provider_context.is_some() {
                    break;
                }
                let entry = &path[index];
                if entry_id(entry)
                    == compaction
                        .get("firstKeptEntryId")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                {
                    found_first_kept = true;
                }
                if found_first_kept {
                    append_message(entry, &mut retained_messages);
                }
            }

            message_entries.push(SessionContextMessage {
                entry_id: compaction_id.clone(),
                message: create_compaction_summary_message(
                    compaction
                        .get("summary")
                        .and_then(Value::as_str)
                        .unwrap_or_default(),
                    compaction
                        .get("tokensBefore")
                        .and_then(Value::as_f64)
                        .unwrap_or(0.0),
                    &entry_timestamp(&compaction),
                    compaction
                        .get("customInstructions")
                        .and_then(Value::as_str)
                        .map(str::to_string),
                    // TS derives the boundary from retainedMessages.length at read
                    // time (session-manager.ts:766); the entry field is never written
                    // by TS writers, so reading it always fell back to the timestamp.
                    Some(retained_messages.len() as f64),
                    provider_context,
                    compaction
                        .get("harnessDigest")
                        .and_then(Value::as_str)
                        .map(str::to_string),
                ),
            });
            message_entries.extend(retained_messages);

            for entry in path.iter().skip(compaction_idx + 1) {
                append_message(entry, &mut message_entries);
            }
        }
        None => {
            for entry in &path {
                append_message(entry, &mut message_entries);
            }
        }
    }

    SessionContextWithEntryIds {
        messages: message_entries
            .iter()
            .map(|item| item.message.clone())
            .collect(),
        entry_ids: message_entries
            .iter()
            .map(|item| item.entry_id.clone())
            .collect(),
        thinking_level,
        service_tier,
        model,
    }
}

pub fn build_session_context(
    entries: &[SessionEntry],
    leaf_id: Option<Option<&str>>,
    by_id: Option<&BTreeMap<String, SessionEntry>>,
    target_model: Option<&pi_ai::types::Model>,
) -> SessionContext {
    let context = build_session_context_with_entry_ids(entries, leaf_id, by_id, target_model);
    SessionContext {
        messages: context.messages,
        thinking_level: context.thinking_level,
        service_tier: context.service_tier,
        model: context.model,
    }
}

pub fn order_session_context_for_transcript(
    context: &SessionContextWithEntryIds,
) -> SessionHistorySnapshot {
    let mut pairs: Vec<(AgentMessage, String)> = context
        .messages
        .iter()
        .cloned()
        .zip(context.entry_ids.iter().cloned())
        .collect();
    let summary_index = pairs
        .iter()
        .position(|(message, _)| message.role() == "compactionSummary");
    if let Some(summary_index) = summary_index {
        let summary_pair = pairs[summary_index].clone();
        let mut remaining: Vec<(AgentMessage, String)> = pairs
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != summary_index)
            .map(|(_, pair)| pair.clone())
            .collect();
        let (summary_message, _) = &summary_pair;
        let boundary = match summary_message {
            AgentMessage::Custom(CustomAgentMessage::CompactionSummary {
                retained_message_count,
                timestamp,
                ..
            }) => match retained_message_count {
                Some(count) if *count >= 0.0 && count.fract() == 0.0 => {
                    (*count as usize).min(remaining.len())
                }
                _ => remaining
                    .iter()
                    .filter(|(message, _)| message_timestamp(message) < *timestamp)
                    .count(),
            },
            _ => 0,
        };
        let mut ordered: Vec<(AgentMessage, String)> = Vec::with_capacity(pairs.len());
        ordered.extend(remaining.drain(..boundary.min(remaining.len())));
        ordered.push(summary_pair);
        ordered.extend(remaining);
        pairs = ordered;
    }
    SessionHistorySnapshot {
        messages: pairs.iter().map(|(message, _)| message.clone()).collect(),
        entry_ids: pairs.iter().map(|(_, entry_id)| entry_id.clone()).collect(),
        tip_entry_id: None,
    }
}

/// `message.timestamp` for every AgentMessage shape.
fn message_timestamp(message: &AgentMessage) -> i64 {
    match message {
        AgentMessage::Message(Message::User(user)) => user.timestamp,
        AgentMessage::Message(Message::Assistant(assistant)) => assistant.timestamp,
        AgentMessage::Message(Message::ToolResult(result)) => result.timestamp,
        AgentMessage::Custom(CustomAgentMessage::BashExecution { timestamp, .. }) => *timestamp,
        AgentMessage::Custom(CustomAgentMessage::Custom { timestamp, .. }) => *timestamp,
        AgentMessage::Custom(CustomAgentMessage::BranchSummary { timestamp, .. }) => *timestamp,
        AgentMessage::Custom(CustomAgentMessage::CompactionSummary { timestamp, .. }) => *timestamp,
    }
}

pub fn get_default_session_dir(_cwd: &str, agent_dir: Option<&str>) -> String {
    let agent_dir = agent_dir
        .map(str::to_string)
        .unwrap_or_else(get_default_agent_dir);
    let session_dir = get_sessions_dir(&agent_dir);
    if !Path::new(&session_dir).exists() {
        let _ = std::fs::create_dir_all(&session_dir);
    }
    session_dir
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

fn append_entry_from_buffer(entries: &mut Vec<FileEntry>, buffer: &[u8], start: usize, end: usize) {
    if end <= start {
        return;
    }
    let text = String::from_utf8_lossy(&buffer[start..end]).to_string();
    match serde_json::from_str::<Value>(&text) {
        Ok(Value::Object(entry)) => entries.push(rehydrate_session_file_entry(entry)),
        // Skip malformed or blank lines.
        _ => {}
    }
}

fn parse_entries_from_buffer(buffer: &[u8]) -> Vec<FileEntry> {
    let mut entries: Vec<FileEntry> = Vec::new();
    let mut start = 0usize;
    while start < buffer.len() {
        let end = match buffer[start..].iter().position(|byte| *byte == 0x0a) {
            Some(offset) => start + offset,
            None => buffer.len(),
        };
        append_entry_from_buffer(&mut entries, buffer, start, end);
        start = end + 1;
    }
    entries
}

async fn parse_entries_from_buffer_async(buffer: &[u8]) -> Vec<FileEntry> {
    let mut entries: Vec<FileEntry> = Vec::new();
    let mut start = 0usize;
    let mut bytes_since_yield = 0usize;
    while start < buffer.len() {
        let end = match buffer[start..].iter().position(|byte| *byte == 0x0a) {
            Some(offset) => start + offset,
            None => buffer.len(),
        };
        append_entry_from_buffer(&mut entries, buffer, start, end);
        bytes_since_yield += end - start + 1;
        start = end + 1;
        if bytes_since_yield >= SESSION_ASYNC_PARSE_YIELD_BYTES {
            bytes_since_yield = 0;
            tokio::task::yield_now().await;
        }
    }
    entries
}

// Crash damage (torn tail, zero-filled append) poisons the NEXT append into the
// same physical line, so the file is repaired once at open, not tolerated in memory.
const REPAIR_SUSPICION_WINDOW_BYTES: u64 = 1024 * 1024;

fn parses_as_json(line: &[u8]) -> bool {
    serde_json::from_str::<Value>(&String::from_utf8_lossy(line)).is_ok()
}

/// A bounded tail read gates the full repair scan: clean opens stay O(window).
fn tail_looks_damaged(target_path: &str) -> Result<bool, String> {
    let mut file = match std::fs::File::open(target_path) {
        Ok(file) => file,
        Err(error) => return Err(error.to_string()),
    };
    let size = match file.metadata() {
        Ok(metadata) => metadata.len(),
        Err(error) => return Err(error.to_string()),
    };
    if size == 0 {
        return Ok(false);
    }
    use std::io::{Read, Seek, SeekFrom};
    let window_bytes = size.min(REPAIR_SUSPICION_WINDOW_BYTES) as usize;
    let mut window = vec![0u8; window_bytes];
    if file
        .seek(SeekFrom::Start(size - window_bytes as u64))
        .is_err()
    {
        return Ok(true);
    }
    let mut read = 0usize;
    while read < window_bytes {
        match file.read(&mut window[read..]) {
            Ok(0) => break,
            Ok(bytes) => read += bytes,
            Err(error) => return Err(error.to_string()),
        }
    }
    window.truncate(read);
    if window.contains(&0) {
        return Ok(true);
    }
    if window.last() != Some(&0x0a) {
        return Ok(true);
    }
    let previous_newline = window[..window.len().saturating_sub(1)]
        .iter()
        .rposition(|byte| *byte == 0x0a);
    let previous_newline = match previous_newline {
        // No boundary inside the window: the final line exceeds it; scan to be sure.
        None => {
            if (window_bytes as u64) < size {
                return Ok(true);
            }
            None
        }
        Some(index) => Some(index),
    };
    let last_line = &window[previous_newline.map(|i| i + 1).unwrap_or(0)..window.len() - 1];
    // A blank final line is benign (the loader skips it) and appends stay safe.
    Ok(!last_line.is_empty() && !parses_as_json(last_line))
}

/// A required repair must commit before a writable manager can adopt this path.
/// `true` invalidates any entries preloaded before the repair.
fn repair_jsonl_damage(file_path: &str) -> Result<bool, String> {
    repair_jsonl_damage_checked(file_path)
        .map_err(|error| format!("Session repair failed for {file_path}: {error}"))
}

fn repair_jsonl_damage_checked(file_path: &str) -> Result<bool, String> {
    let target_path = realpath_if_present_sync(file_path);
    if !tail_looks_damaged(&target_path)? {
        return Ok(false);
    }
    let buffer = match std::fs::read(&target_path) {
        Ok(buffer) => buffer,
        Err(error) => return Err(error.to_string()),
    };
    let snapshot = match std::fs::metadata(&target_path) {
        Ok(metadata) => (metadata.len(), metadata.modified().ok()),
        Err(error) => return Err(error.to_string()),
    };
    if buffer.is_empty() || snapshot.0 != buffer.len() as u64 {
        return Err("session changed during required repair; retry opening".to_string());
    }
    let mut kept_lines: Vec<Vec<u8>> = Vec::new();
    let mut recovered_nul_lines = 0usize;
    let mut dropped_lines = 0usize;
    let mut repaired_tail = false;
    let mut dirty = false;
    let mut start = 0usize;
    while start < buffer.len() {
        let newline = buffer[start..].iter().position(|byte| *byte == 0x0a);
        let terminated = newline.is_some();
        let end = match newline {
            Some(offset) => start + offset,
            None => buffer.len(),
        };
        let mut line_start = start;
        while line_start < end && buffer[line_start] == 0 {
            line_start += 1;
        }
        let line = &buffer[line_start..end];
        if line_start > start {
            dirty = true;
            if !line.is_empty() && parses_as_json(line) {
                kept_lines.push(line.to_vec());
                recovered_nul_lines += 1;
            } else {
                dropped_lines += 1;
            }
        } else if !terminated {
            // An unterminated tail merges with the next append: re-terminate or truncate.
            dirty = true;
            if !line.is_empty() && parses_as_json(line) {
                kept_lines.push(line.to_vec());
                repaired_tail = true;
            } else {
                dropped_lines += 1;
            }
        } else if end + 1 >= buffer.len() && !line.is_empty() && !parses_as_json(line) {
            dirty = true;
            dropped_lines += 1;
        } else {
            kept_lines.push(line.to_vec());
        }
        start = end + 1;
    }
    if !dirty {
        return Ok(false);
    }
    let metadata = stat_metadata_if_present(&target_path);
    let content = if kept_lines.is_empty() {
        String::new()
    } else {
        let joined = kept_lines
            .iter()
            .map(|line| String::from_utf8_lossy(line).to_string())
            .collect::<Vec<_>>()
            .join("\n");
        format!("{joined}\n")
    };
    let mode = metadata.as_ref().map(|metadata| metadata.mode);
    let before_rename = |_temp_path: &str| -> Result<(), String> {
        // A concurrent appender wins; this opener must fail rather than append to an unrepaired tail.
        match std::fs::metadata(&target_path) {
            Ok(current) => {
                if current.len() != snapshot.0 || current.modified().ok() != snapshot.1 {
                    return Err("session changed during required repair; retry opening".to_string());
                }
                Ok(())
            }
            Err(error) => Err(error.to_string()),
        }
    };
    write_file_atomic_sync(
        &target_path,
        &content,
        WriteFileAtomicOptions { mode, fsync: false },
        Some(&before_rename),
    )?;
    eprintln!(
        "Repaired crash damage in {target_path}: recovered {recovered_nul_lines} zero-filled line(s), dropped {dropped_lines} unrecoverable line(s){}",
        if repaired_tail { ", restored the trailing newline" } else { "" }
    );
    Ok(true)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StatMetadata {
    mode: u32,
    uid: u32,
    gid: u32,
}

fn stat_metadata_if_present(path: &str) -> Option<StatMetadata> {
    let metadata = std::fs::metadata(path).ok()?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        use std::os::unix::fs::PermissionsExt;
        return Some(StatMetadata {
            mode: metadata.permissions().mode() & 0o777,
            uid: metadata.uid(),
            gid: metadata.gid(),
        });
    }
    #[cfg(not(unix))]
    {
        Some(StatMetadata {
            mode: 0o600,
            uid: 0,
            gid: 0,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionLoadObservation {
    /// Bytes returned by the primary transcript read that populated this manager.
    pub read_bytes: u64,
}

struct LoadedSessionEntries {
    entries: Vec<FileEntry>,
    observation: Option<SessionLoadObservation>,
}

fn finalize_loaded_entries(mut entries: Vec<FileEntry>) -> Vec<FileEntry> {
    if entries.is_empty() {
        return entries;
    }
    let header = &entries[0];
    if entry_type(header) != "session" || header.get("id").and_then(Value::as_str).is_none() {
        return Vec::new();
    }
    apply_child_usage_attributions(&mut entries);
    entries
}

fn apply_child_usage_attributions(entries: &mut [FileEntry]) {
    let mut assistant_entries_by_id: HashMap<String, usize> = HashMap::new();
    for (index, entry) in entries.iter().enumerate() {
        if entry_type(entry) == "message" && message_role(entry) == "assistant" {
            assistant_entries_by_id.insert(entry_id(entry), index);
        }
    }

    let attributions: Vec<(String, Usage)> = entries
        .iter()
        .filter(|entry| entry_type(entry) == "child_usage_attributed")
        .filter_map(|entry| {
            let target_id = entry.get("targetId").and_then(Value::as_str)?.to_string();
            let aggregate = entry.get("aggregateUsage")?;
            let usage = serde_json::from_value::<Usage>(aggregate.clone()).ok()?;
            Some((target_id, usage))
        })
        .collect();

    for (target_id, usage) in attributions {
        if let Some(index) = assistant_entries_by_id.get(&target_id).copied() {
            if let Some(message) = entries[index]
                .get_mut("message")
                .and_then(Value::as_object_mut)
            {
                message.insert(
                    "usage".to_string(),
                    serde_json::to_value(clone_usage(&usage)).unwrap_or(Value::Null),
                );
            }
        }
    }
}

fn load_entries_from_file_observed(file_path: &str) -> LoadedSessionEntries {
    if !Path::new(file_path).exists() {
        return LoadedSessionEntries {
            entries: Vec::new(),
            observation: None,
        };
    }
    let buffer = match std::fs::read(file_path) {
        Ok(buffer) => buffer,
        Err(_) => {
            return LoadedSessionEntries {
                entries: Vec::new(),
                observation: None,
            }
        }
    };
    let entries = finalize_loaded_entries(parse_entries_from_buffer(&buffer));
    if entries.is_empty() {
        LoadedSessionEntries {
            entries,
            observation: None,
        }
    } else {
        LoadedSessionEntries {
            entries,
            observation: Some(SessionLoadObservation {
                read_bytes: buffer.len() as u64,
            }),
        }
    }
}

pub fn load_entries_from_file(file_path: &str) -> Vec<FileEntry> {
    load_entries_from_file_observed(file_path).entries
}

/// Async loader for the daemon: reads off the event loop and yields while parsing so a
/// large load doesn't freeze other sessions. Large files stream to avoid retaining both
/// the full input Buffer and the parsed entry graph at the same time.
async fn load_entries_from_file_async_observed(
    file_path: &str,
    stream_threshold_bytes: Option<u64>,
) -> LoadedSessionEntries {
    if !Path::new(file_path).exists() {
        return LoadedSessionEntries {
            entries: Vec::new(),
            observation: None,
        };
    }
    let stream_threshold_bytes =
        stream_threshold_bytes.unwrap_or(SESSION_STREAMING_LOAD_THRESHOLD_BYTES);
    let size_at_open = match std::fs::metadata(file_path) {
        Ok(metadata) => metadata.len(),
        Err(_) => {
            return LoadedSessionEntries {
                entries: Vec::new(),
                observation: None,
            }
        }
    };
    if size_at_open < stream_threshold_bytes {
        let buffer = match std::fs::read(file_path) {
            Ok(buffer) => buffer,
            Err(_) => {
                return LoadedSessionEntries {
                    entries: Vec::new(),
                    observation: None,
                }
            }
        };
        let entries = finalize_loaded_entries(parse_entries_from_buffer_async(&buffer).await);
        return if entries.is_empty() {
            LoadedSessionEntries {
                entries,
                observation: None,
            }
        } else {
            LoadedSessionEntries {
                entries,
                observation: Some(SessionLoadObservation {
                    read_bytes: buffer.len() as u64,
                }),
            }
        };
    }

    let mut entries: Vec<FileEntry> = Vec::new();
    let mut bytes_since_yield = 0usize;
    let mut read_bytes = 0u64;
    // Pin the stream to the measured boundary. A concurrent append belongs to the next reload,
    // and onBytesRead counts the chunks actually returned rather than treating stat size as I/O.
    let range = ReadLinesRange {
        start: None,
        end: if size_at_open > 0 {
            Some(size_at_open as i64 - 1)
        } else {
            None
        },
    };
    // `onBytesRead` counts the chunks actually returned. The counter is an atomic rather than a
    // `Cell` because the closure's borrow must not be held across the parse loop's awaits below.
    let observed_bytes = std::sync::atomic::AtomicU64::new(0);
    let lines = {
        let mut on_bytes_read = |bytes: usize| {
            observed_bytes.fetch_add(bytes as u64, std::sync::atomic::Ordering::Relaxed);
        };
        read_lines_as_buffers(file_path, Some(&range), Some(&mut on_bytes_read)).unwrap_or_default()
    };
    read_bytes += observed_bytes.load(std::sync::atomic::Ordering::Relaxed);
    for line in lines {
        append_entry_from_buffer(&mut entries, &line, 0, line.len());
        bytes_since_yield += line.len() + 1;
        if bytes_since_yield >= SESSION_ASYNC_PARSE_YIELD_BYTES {
            bytes_since_yield = 0;
            tokio::task::yield_now().await;
        }
    }
    let finalized = finalize_loaded_entries(entries);
    if finalized.is_empty() {
        LoadedSessionEntries {
            entries: finalized,
            observation: None,
        }
    } else {
        LoadedSessionEntries {
            entries: finalized,
            observation: Some(SessionLoadObservation { read_bytes }),
        }
    }
}

pub async fn load_entries_from_file_async(
    file_path: &str,
    stream_threshold_bytes: Option<u64>,
) -> Vec<FileEntry> {
    load_entries_from_file_async_observed(file_path, stream_threshold_bytes)
        .await
        .entries
}

fn read_session_header(file_path: &str) -> Option<Map<String, Value>> {
    let first_line = read_first_line_sync(file_path, 0)?;
    let parsed = serde_json::from_str::<Value>(&first_line).ok()?;
    parsed.as_object().cloned()
}

fn header_rlm_depth(header: &Map<String, Value>) -> Option<i64> {
    header
        .get("rlmDepth")
        .and_then(is_safe_integer)
        .filter(|value| *value >= 0)
}

fn header_parent_session(header: &Map<String, Value>) -> Option<String> {
    header
        .get("parentSession")
        .and_then(Value::as_str)
        .map(str::to_string)
}

pub fn resolve_session_rlm_depth(header: &Map<String, Value>, session_path: &str) -> i64 {
    resolve_legacy_session_rlm_depth(header, session_path, &mut BTreeSet::new())
        .unwrap_or_else(|| legacy_child_depth_from_path(session_path))
}

fn resolve_legacy_session_rlm_depth(
    header: &Map<String, Value>,
    session_path: &str,
    visited_paths: &mut BTreeSet<String>,
) -> Option<i64> {
    if let Some(depth) = header_rlm_depth(header) {
        return Some(depth);
    }
    let parent_session = header_parent_session(header)?;

    let resolved_session_path = resolve_path(session_path);
    if !visited_paths.insert(resolved_session_path.clone()) {
        return None;
    }

    let path_depth = legacy_child_depth_from_path(session_path);
    let parent_session_path = Path::new(&dirname(session_path))
        .join(&parent_session)
        .to_string_lossy()
        .to_string();
    let mut result = None;
    if let Some(parent_header) = read_session_header(&parent_session_path) {
        if let Some(parent_depth) =
            resolve_legacy_session_rlm_depth(&parent_header, &parent_session_path, visited_paths)
        {
            result = Some(if path_depth > 0 {
                parent_depth + 1
            } else {
                parent_depth
            });
        }
    }
    visited_paths.remove(&resolved_session_path);
    match result {
        Some(value) => Some(value),
        // Fall back to artifact ancestry for unavailable or invalid legacy parents.
        None => Some(path_depth),
    }
}

fn legacy_child_depth_from_path(session_path: &str) -> i64 {
    let mut depth = 0i64;
    let dir = dirname(session_path);
    let segments: Vec<&str> = dir
        .split(['\\', '/'])
        .filter(|segment| !segment.is_empty())
        .collect();
    for segment in segments.iter().rev() {
        if !is_sub_session_segment(segment) {
            break;
        }
        depth += 1;
    }
    depth
}

/// `/^sub-[0-9a-f]{8}$/`
fn is_sub_session_segment(segment: &str) -> bool {
    match segment.strip_prefix("sub-") {
        Some(rest) => {
            rest.len() == 8
                && rest
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        }
        None => false,
    }
}

fn derive_child_rlm_depth(parent_header: Option<&Map<String, Value>>) -> Option<i64> {
    let depth = parent_header.and_then(header_rlm_depth)?;
    if depth < i64::MAX {
        Some(depth + 1)
    } else {
        None
    }
}

fn root_rlm_depth_from_env() -> Result<i64, String> {
    let value = match std::env::var("RLM_DEPTH") {
        Ok(value) => value,
        Err(_) => return Ok(0),
    };
    if value.is_empty() {
        return Ok(0);
    }
    let parsed = value.parse::<i64>().ok();
    let digits_only = !value.is_empty() && value.bytes().all(|b| b.is_ascii_digit());
    match (digits_only, parsed) {
        (true, Some(parsed)) if parsed >= 0 => Ok(parsed),
        _ => Err("RLM_DEPTH must be a non-negative integer".to_string()),
    }
}

fn is_valid_session_file(file_path: &str) -> bool {
    match read_session_header(file_path) {
        Some(header) => {
            header.get("type").and_then(Value::as_str) == Some("session")
                && header.get("id").and_then(Value::as_str).is_some()
        }
        None => false,
    }
}

fn session_has_conversation_history(file_path: &str) -> bool {
    use std::io::{BufRead, BufReader, Read};

    let Ok(file) = std::fs::File::open(file_path) else {
        return false;
    };
    let Ok(metadata) = file.metadata() else {
        return false;
    };
    // Stop at the first conversation entry and at the captured file boundary.
    // Opening a draft can persist settings/status without starting a conversation.
    for line in BufReader::new(file.take(metadata.len())).split(b'\n') {
        let Ok(line) = line else {
            return false;
        };
        let Ok(Value::Object(entry)) =
            serde_json::from_str::<Value>(&String::from_utf8_lossy(&line))
        else {
            continue;
        };
        match entry_type(&entry) {
            "message" if !message_role(&entry).is_empty() => return true,
            "custom_message" if entry.get("content").is_some_and(|content| {
                content.is_string() || content.is_array()
            }) => return true,
            "compaction" | "branch_summary"
                if entry
                    .get("summary")
                    .and_then(Value::as_str)
                    .is_some_and(|summary| !summary.is_empty()) =>
            {
                return true;
            }
            "compaction"
                if entry
                    .get("details")
                    .and_then(get_provider_checkpoint)
                    .is_some() =>
            {
                return true;
            }
            _ => {}
        }
    }
    false
}

pub fn find_most_recent_session(session_dir: &str) -> Option<String> {
    let dir_entries = std::fs::read_dir(session_dir).ok()?;
    let mut files: Vec<(String, std::time::SystemTime)> = Vec::new();
    for entry in dir_entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if !name.ends_with(".jsonl") {
            continue;
        }
        let path = Path::new(session_dir)
            .join(&name)
            .to_string_lossy()
            .to_string();
        if !is_valid_session_file(&path) {
            continue;
        }
        if let Ok(metadata) = std::fs::metadata(&path) {
            if let Ok(modified) = metadata.modified() {
                files.push((path, modified));
            }
        }
    }
    files.sort_by(|a, b| b.1.cmp(&a.1));
    files
        .into_iter()
        .find(|(path, _)| session_has_conversation_history(path))
        .map(|(path, _)| path)
}

// TS normalizeCwd resolves through path.resolve, which lexically collapses
// dot segments and trailing separators (session-manager.ts:1169-1171); a
// recorded '../' cwd must match the plain query.
fn normalize_cwd(cwd: &str) -> String {
    lexical_resolve(cwd)
}

fn lexical_resolve(path: &str) -> String {
    let candidate = PathBuf::from(path);
    let base = if candidate.is_absolute() {
        candidate
    } else {
        match std::env::current_dir() {
            Ok(cwd) => cwd.join(&candidate),
            Err(_) => candidate,
        }
    };
    let mut result = PathBuf::new();
    for component in base.components() {
        match component {
            std::path::Component::ParentDir => {
                result.pop();
            }
            std::path::Component::CurDir => {}
            other => result.push(other.as_os_str()),
        }
    }
    result.to_string_lossy().to_string()
}

fn session_info_matches_cwd(session: &SessionInfo, cwd: &str) -> bool {
    !session.cwd.is_empty() && normalize_cwd(&session.cwd) == normalize_cwd(cwd)
}

fn session_header_matches_cwd(header: Option<&Map<String, Value>>, cwd: &str) -> bool {
    match header {
        Some(header) => {
            header.get("type").and_then(Value::as_str) == Some("session")
                && header.get("id").and_then(Value::as_str).is_some()
                && header
                    .get("cwd")
                    .and_then(Value::as_str)
                    .map(|header_cwd| normalize_cwd(header_cwd) == normalize_cwd(cwd))
                    .unwrap_or(false)
        }
        None => false,
    }
}

pub fn find_most_recent_session_for_cwd(session_dir: &str, cwd: &str) -> Option<String> {
    let dir_entries = std::fs::read_dir(session_dir).ok()?;
    let mut files: Vec<(String, std::time::SystemTime)> = Vec::new();
    for entry in dir_entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if !name.ends_with(".jsonl") {
            continue;
        }
        let path = Path::new(session_dir)
            .join(&name)
            .to_string_lossy()
            .to_string();
        let header = read_session_header(&path);
        if !session_header_matches_cwd(header.as_ref(), cwd) {
            continue;
        }
        if let Ok(metadata) = std::fs::metadata(&path) {
            if let Ok(modified) = metadata.modified() {
                files.push((path, modified));
            }
        }
    }
    files.sort_by(|a, b| b.1.cmp(&a.1));
    files
        .into_iter()
        .find(|(path, _)| session_has_conversation_history(path))
        .map(|(path, _)| path)
}

fn resolve_path(path: &str) -> String {
    let candidate = PathBuf::from(path);
    if candidate.is_absolute() {
        return candidate.to_string_lossy().to_string();
    }
    match std::env::current_dir() {
        Ok(cwd) => cwd.join(candidate).to_string_lossy().to_string(),
        Err(_) => candidate.to_string_lossy().to_string(),
    }
}

fn basename(value: &str) -> String {
    Path::new(value)
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_default()
}

fn dirname(value: &str) -> String {
    Path::new(value)
        .parent()
        .map(|parent| {
            let text = parent.to_string_lossy().to_string();
            if text.is_empty() {
                ".".to_string()
            } else {
                text
            }
        })
        .unwrap_or_else(|| ".".to_string())
}

// ---------------------------------------------------------------------------
// Session info scanning
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionState {
    pub status: SessionStateStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SessionStateStatus {
    Active,
    Archived,
    Crash,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentTaskState {
    NeedsInput,
    Completed,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentStatus {
    pub summary: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_state: Option<AgentTaskState>,
    pub based_on_message_count: i64,
}

/// `SessionInfo` - `created`/`modified` are millisecond epochs so ordering and
/// formatting stay identical to `Date.getTime()`.
#[derive(Debug, Clone, PartialEq)]
pub struct SessionInfo {
    pub path: String,
    pub id: String,
    pub cwd: String,
    pub name: Option<String>,
    pub state: Option<SessionState>,
    pub parent_session_path: Option<String>,
    pub rlm_depth: i64,
    pub created: f64,
    pub modified: f64,
    pub message_count: i64,
    pub first_message: String,
    pub all_messages_text: String,
    pub agent_status: Option<AgentStatus>,
    pub usage: Option<SessionUsageSummary>,
}

pub type SessionListProgress = dyn Fn(i64, i64) + Send + Sync;
pub type SessionListItem = dyn Fn(&SessionInfo) + Send + Sync;

#[derive(Default)]
pub struct SessionListCallbacks {
    pub on_progress: Option<Box<SessionListProgress>>,
    pub on_session: Option<Box<SessionListItem>>,
}

struct SessionScanAccumulator {
    header: Option<Map<String, Value>>,
    /// The first parsed entry was not a session header; appends cannot repair this.
    invalid: bool,
    message_count: i64,
    first_message: String,
    all_messages_text: String,
    name: Option<String>,
    state: Option<SessionState>,
    agent_status: Option<AgentStatus>,
    last_activity_time: Option<f64>,
    // Fold attribution aggregates like the loader: either disk representation cancels to the same own spend.
    assistant_usage_by_id: BTreeMap<String, Usage>,
    attributed_child_usage: Usage,
    summarization_usage: Usage,
}

struct SessionScanState {
    file_size: u64,
    mtime: Option<std::time::SystemTime>,
    /// Identity of the scanned file: a rename rewrite replaces the inode and invalidates the resume state.
    dev: u64,
    ino: u64,
    /// Bytes consumed as complete newline-terminated lines; the resume point for the next scan.
    offset: u64,
    /// Last bytes of the consumed prefix; a resume only proceeds while the file still starts with them.
    tail: Vec<u8>,
    acc: SessionScanAccumulator,
    info: Option<SessionInfo>,
    /// Usage entries counted against the retained bound at the last store.
    accounted_usage_entries: usize,
}

const SESSION_SCAN_RESUME_TAIL_BYTES: usize = 16;
// Memory bound (~tens of MB): LRU whole-state eviction only while over it, so
// small states never thrash and an evicted file just pays one full rescan.
const SESSION_SCAN_MAX_RETAINED_USAGE_ENTRIES: usize = 100_000;

struct SessionScanStore {
    states: BTreeMap<String, SessionScanState>,
    retained_usage_entries: usize,
}

fn session_scan_store() -> &'static Mutex<SessionScanStore> {
    static STORE: OnceLock<Mutex<SessionScanStore>> = OnceLock::new();
    STORE.get_or_init(|| {
        Mutex::new(SessionScanStore {
            states: BTreeMap::new(),
            retained_usage_entries: 0,
        })
    })
}

/// Session scans of a path run one at a time; a later caller chains its own pass.
fn session_scan_queue() -> &'static Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>> {
    static QUEUE: OnceLock<Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>> = OnceLock::new();
    QUEUE.get_or_init(|| Mutex::new(HashMap::new()))
}

pub async fn read_session_info(file_path: &str) -> Option<SessionInfo> {
    let lock = {
        let mut queue = session_scan_queue()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        queue
            .entry(file_path.to_string())
            .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
            .clone()
    };
    let _guard = lock.lock().await;
    scan_session_info(file_path, true).await
}

fn drop_session_scan_state(file_path: &str) {
    let mut store = session_scan_store()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if let Some(state) = store.states.remove(file_path) {
        store.retained_usage_entries = store
            .retained_usage_entries
            .saturating_sub(state.accounted_usage_entries);
    }
}

/// Refresh LRU recency and enforce the retained-usage bound.
fn store_session_scan_state(file_path: &str, mut state: SessionScanState) {
    drop_session_scan_state(file_path);
    state.accounted_usage_entries = state.acc.assistant_usage_by_id.len();
    let accounted = state.accounted_usage_entries;
    let mut store = session_scan_store()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    store.retained_usage_entries += accounted;
    store.states.insert(file_path.to_string(), state);
    while store.retained_usage_entries > SESSION_SCAN_MAX_RETAINED_USAGE_ENTRIES {
        let key = match store.states.keys().next() {
            Some(key) => key.clone(),
            None => break,
        };
        if let Some(dropped) = store.states.remove(&key) {
            store.retained_usage_entries = store
                .retained_usage_entries
                .saturating_sub(dropped.accounted_usage_entries);
        }
    }
}

fn create_session_scan_accumulator() -> SessionScanAccumulator {
    SessionScanAccumulator {
        header: None,
        invalid: false,
        message_count: 0,
        first_message: String::new(),
        all_messages_text: String::new(),
        name: None,
        state: None,
        agent_status: None,
        last_activity_time: None,
        assistant_usage_by_id: BTreeMap::new(),
        attributed_child_usage: empty_usage(),
        summarization_usage: empty_usage(),
    }
}

fn scanned_prefix_intact(file_path: &str, state: &SessionScanState) -> bool {
    if state.offset == 0 {
        return true;
    }
    let start = state
        .offset
        .saturating_sub(SESSION_SCAN_RESUME_TAIL_BYTES as u64) as i64;
    read_bytes_sync(file_path, start, state.offset as i64) == state.tail
}

/// Last bytes of the consumed prefix after appending one line and its newline, copied out of the stream chunk.
fn advance_scan_tail(tail: &[u8], line: &[u8]) -> Vec<u8> {
    if line.len() >= SESSION_SCAN_RESUME_TAIL_BYTES - 1 {
        let mut out = line[line.len() - (SESSION_SCAN_RESUME_TAIL_BYTES - 1)..].to_vec();
        out.push(0x0a);
        return out;
    }
    let mut combined = tail.to_vec();
    combined.extend_from_slice(line);
    combined.push(0x0a);
    if combined.len() <= SESSION_SCAN_RESUME_TAIL_BYTES {
        combined
    } else {
        combined[combined.len() - SESSION_SCAN_RESUME_TAIL_BYTES..].to_vec()
    }
}

async fn scan_session_info(file_path: &str, retry_on_replacement: bool) -> Option<SessionInfo> {
    let (stats, (dev, ino)) = match session_scan_metadata(file_path) {
        Ok(stats) => stats,
        Err(_) => {
            drop_session_scan_state(file_path);
            return None;
        }
    };
    let size = stats.len();
    let mtime = stats.modified().ok();

    let previous = {
        let store = session_scan_store()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        store.states.get(file_path).map(|state| SessionScanState {
            file_size: state.file_size,
            mtime: state.mtime,
            dev: state.dev,
            ino: state.ino,
            offset: state.offset,
            tail: state.tail.clone(),
            acc: SessionScanAccumulator {
                header: state.acc.header.clone(),
                invalid: state.acc.invalid,
                message_count: state.acc.message_count,
                first_message: state.acc.first_message.clone(),
                all_messages_text: state.acc.all_messages_text.clone(),
                name: state.acc.name.clone(),
                state: state.acc.state.clone(),
                agent_status: state.acc.agent_status.clone(),
                last_activity_time: state.acc.last_activity_time,
                assistant_usage_by_id: state.acc.assistant_usage_by_id.clone(),
                attributed_child_usage: state.acc.attributed_child_usage.clone(),
                summarization_usage: state.acc.summarization_usage.clone(),
            },
            info: state.info.clone(),
            accounted_usage_entries: state.accounted_usage_entries,
        })
    };

    // Match TS scanSessionInfo: unchanged identity/size/mtime reuses the
    // snapshot, and append-only growth resumes from the scanned prefix.
    let same_file = previous
        .as_ref()
        .map(|previous| same_file_identity(previous.dev, previous.ino, dev, ino))
        .unwrap_or(false);
    if let Some(previous) = previous.as_ref() {
        if same_file && previous.file_size == size && previous.mtime == mtime {
            store_session_scan_state(file_path, clone_scan_state(previous));
            return previous.info.clone();
        }
    }
    let resume = same_file
        && previous
            .as_ref()
            .map(|previous| size > previous.file_size && scanned_prefix_intact(file_path, previous))
            .unwrap_or(false);
    let mut state = if resume {
        clone_scan_state(previous.as_ref().unwrap())
    } else {
        SessionScanState {
            file_size: 0,
            mtime: None,
            dev,
            ino,
            offset: 0,
            tail: Vec::new(),
            acc: create_session_scan_accumulator(),
            info: None,
            accounted_usage_entries: 0,
        }
    };

    let torn_tail = match scan_session_lines(file_path, &mut state, size) {
        Ok(torn_tail) => torn_tail,
        Err(_) => {
            drop_session_scan_state(file_path);
            return None;
        }
    };
    state.info = snapshot_session_info(&state.acc, torn_tail.as_deref(), file_path, size, mtime);

    // A rename rewrite racing the scan can mix two files' bytes into one
    // accumulator: a changed inode afterwards discards the state and rescans.
    let after_identity = session_scan_metadata(file_path).ok().map(|(_, identity)| identity);
    if after_identity != Some((dev, ino)) {
        drop_session_scan_state(file_path);
        if retry_on_replacement {
            return Box::pin(scan_session_info(file_path, false)).await;
        }
        return None;
    }
    state.file_size = size;
    state.mtime = mtime;
    store_session_scan_state(file_path, state);
    let store = session_scan_store()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    store
        .states
        .get(file_path)
        .and_then(|state| state.info.clone())
}

fn clone_scan_state(state: &SessionScanState) -> SessionScanState {
    SessionScanState {
        file_size: state.file_size,
        mtime: state.mtime,
        dev: state.dev,
        ino: state.ino,
        offset: state.offset,
        tail: state.tail.clone(),
        acc: SessionScanAccumulator {
            header: state.acc.header.clone(),
            invalid: state.acc.invalid,
            message_count: state.acc.message_count,
            first_message: state.acc.first_message.clone(),
            all_messages_text: state.acc.all_messages_text.clone(),
            name: state.acc.name.clone(),
            state: state.acc.state.clone(),
            agent_status: state.acc.agent_status.clone(),
            last_activity_time: state.acc.last_activity_time,
            assistant_usage_by_id: state.acc.assistant_usage_by_id.clone(),
            attributed_child_usage: state.acc.attributed_child_usage.clone(),
            summarization_usage: state.acc.summarization_usage.clone(),
        },
        info: state.info.clone(),
        accounted_usage_entries: state.accounted_usage_entries,
    }
}

fn session_scan_metadata(file_path: &str) -> std::io::Result<(std::fs::Metadata, (u64, u64))> {
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        use windows_sys::Win32::Storage::FileSystem::FILE_READ_ATTRIBUTES;

        // Windows metadata from path stat does not expose a stable file ID.
        // Query the existing handle-based identity helper; metadata and ID
        // must describe the same opened file even when the path is replaced.
        let file = std::fs::OpenOptions::new()
            .access_mode(FILE_READ_ATTRIBUTES)
            .open(file_path)?;
        let metadata = file.metadata()?;
        let identity = crate::utils::dir_lock::file_identity(&file)
            .map(|identity| (identity.dev, identity.ino))
            .unwrap_or((0, 0));
        Ok((metadata, identity))
    }
    #[cfg(not(windows))]
    {
        let metadata = std::fs::metadata(file_path)?;
        #[cfg(unix)]
        let identity = {
            use std::os::unix::fs::MetadataExt;
            (metadata.dev(), metadata.ino())
        };
        #[cfg(not(unix))]
        let identity = (0, 0);
        Ok((metadata, identity))
    }
}

#[cfg(unix)]
fn same_file_identity(a_dev: u64, a_ino: u64, b_dev: u64, b_ino: u64) -> bool {
    a_dev == b_dev && a_ino == b_ino
}

#[cfg(windows)]
fn same_file_identity(a_dev: u64, a_ino: u64, b_dev: u64, b_ino: u64) -> bool {
    // An unavailable file index must never claim an unchanged file.
    (a_dev, a_ino) != (0, 0) && a_dev == b_dev && a_ino == b_ino
}

#[cfg(not(any(unix, windows)))]
fn same_file_identity(_a_dev: u64, _a_ino: u64, _b_dev: u64, _b_ino: u64) -> bool {
    false
}

/// Fold the complete lines in [state.offset, size) into the accumulator. An
/// unterminated final line may be an in-progress append: it is returned for
/// snapshot-only folding, never consumed into the resumable accumulator.
fn scan_session_lines(
    file_path: &str,
    state: &mut SessionScanState,
    size: u64,
) -> std::io::Result<Option<Vec<u8>>> {
    if state.acc.invalid || state.offset >= size {
        return Ok(None);
    }
    use std::io::{BufRead, Read, Seek};
    let mut file = std::fs::File::open(file_path)?;
    file.seek(std::io::SeekFrom::Start(state.offset))?;
    // Preserve the stat boundary and append-resume semantics without holding
    // every line of a potentially hundreds-of-MB transcript at once.
    let mut reader = std::io::BufReader::with_capacity(64 * 1024, file.take(size - state.offset));
    let mut line_buffer = Vec::new();
    loop {
        line_buffer.clear();
        let bytes_read = reader.read_until(b'\n', &mut line_buffer)?;
        if bytes_read == 0 { break; }
        if line_buffer.last() != Some(&b'\n') { return Ok(Some(line_buffer)); }
        line_buffer.pop();
        fold_session_scan_line(&mut state.acc, &line_buffer);
        state.tail = advance_scan_tail(&state.tail, &line_buffer);
        state.offset += bytes_read as u64;
        if state.acc.invalid {
            break;
        }
    }
    Ok(None)
}

fn fold_session_scan_line(acc: &mut SessionScanAccumulator, line_buffer: &[u8]) {
    let line = String::from_utf8_lossy(line_buffer);
    if line.trim().is_empty() {
        return;
    }

    // Large tool-result entries can be many MB. They do not carry the
    // session-list metadata we need, and parsing them during every refresh
    // can exhaust the daemon heap.
    if line.len() > SESSION_LIST_PARSE_MAX_LINE_CHARS {
        if looks_like_message_entry(&line) {
            acc.message_count += 1;
            let summary = extract_oversized_message_summary(&line);
            if let Some(timestamp) = summary.timestamp {
                if summary.role.as_deref() == Some("user")
                    || summary.role.as_deref() == Some("assistant")
                {
                    acc.last_activity_time =
                        Some(acc.last_activity_time.unwrap_or(0.0).max(timestamp));
                }
            }
            if summary.role.as_deref() == Some("user") && acc.first_message.is_empty() {
                acc.first_message = summary
                    .text_preview
                    .clone()
                    .filter(|preview| !preview.is_empty())
                    .unwrap_or_else(|| "(large message)".to_string());
            }
        }
        return;
    }

    let trimmed = line.trim();
    let entry: Map<String, Value> = match serde_json::from_str::<Value>(trimmed) {
        Ok(Value::Object(entry)) => entry,
        _ => return,
    };

    if entry_type(&entry) == "session_info" {
        let name = entry
            .get("name")
            .and_then(Value::as_str)
            .map(|name| name.trim().to_string())
            .filter(|name| !name.is_empty());
        acc.name = name;
    }
    if entry_type(&entry) == "session_state" {
        let status = entry
            .get("state")
            .and_then(Value::as_object)
            .and_then(|state| state.get("status"))
            .and_then(normalize_session_state_status);
        if let Some(status) = status {
            acc.state = Some(SessionState { status });
        }
    }
    // Keep the latest recap/verdict so off-daemon sessions don't all show as
    // unjudged in the agents view. Append-only, so last seen wins.
    if entry_type(&entry) == "agent_status" {
        acc.agent_status = entry
            .get("status")
            .and_then(|status| serde_json::from_value::<AgentStatus>(status.clone()).ok());
    }
    if entry_type(&entry) == "child_usage_attributed" {
        let target_id = entry
            .get("targetId")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        if acc.assistant_usage_by_id.contains_key(&target_id) {
            if let Some(aggregate) = entry
                .get("aggregateUsage")
                .and_then(|value| serde_json::from_value::<Usage>(value.clone()).ok())
            {
                acc.assistant_usage_by_id.insert(target_id, aggregate);
            }
            if let Some(child_usage) = entry
                .get("childUsage")
                .and_then(|value| serde_json::from_value::<Usage>(value.clone()).ok())
            {
                add_assistant_usage(&mut acc.attributed_child_usage, &child_usage);
            }
        }
    }
    if entry_type(&entry) == "compaction" || entry_type(&entry) == "branch_summary" {
        if let Some(summarization_usage) = entry
            .get("usage")
            .and_then(|value| serde_json::from_value::<Usage>(value.clone()).ok())
        {
            add_assistant_usage(&mut acc.summarization_usage, &summarization_usage);
        }
    }
    if acc.header.is_none() {
        if entry_type(&entry) != "session" {
            acc.invalid = true;
            return;
        }
        acc.header = Some(entry.clone());
    }

    acc.last_activity_time = update_last_activity_time(acc.last_activity_time, &entry);

    if entry_type(&entry) != "message" {
        return;
    }
    acc.message_count += 1;

    let message = match message_of(&entry) {
        Some(message) => message,
        None => return,
    };
    if message.get("role").and_then(Value::as_str) == Some("assistant") {
        if let Some(usage) = message
            .get("usage")
            .and_then(|value| serde_json::from_value::<Usage>(value.clone()).ok())
        {
            acc.assistant_usage_by_id.insert(entry_id(&entry), usage);
        }
    }
    if !is_message_with_content(&message) {
        return;
    }
    let role = message.get("role").and_then(Value::as_str).unwrap_or("");
    if role != "user" && role != "assistant" {
        return;
    }

    let text_content = extract_text_content(&message);
    if text_content.is_empty() {
        return;
    }

    acc.all_messages_text = append_capped_search_text(&acc.all_messages_text, &text_content);
    if acc.first_message.is_empty() && role == "user" {
        acc.first_message = text_content;
    }
}

fn snapshot_session_info(
    persistent: &SessionScanAccumulator,
    torn_tail: Option<&[u8]>,
    file_path: &str,
    size: u64,
    mtime: Option<std::time::SystemTime>,
) -> Option<SessionInfo> {
    let mut acc = clone_accumulator(persistent);
    if let Some(torn_tail) = torn_tail {
        if !torn_tail.is_empty() && !acc.invalid {
            fold_session_scan_line(&mut acc, torn_tail);
        }
    }
    if acc.invalid {
        return None;
    }
    let header = acc.header.clone()?;
    let mut usage_total = empty_usage();
    for usage in acc.assistant_usage_by_id.values() {
        add_assistant_usage(&mut usage_total, usage);
    }
    add_assistant_usage(&mut usage_total, &acc.summarization_usage);
    subtract_assistant_usage(&mut usage_total, &acc.attributed_child_usage);
    let cwd = header
        .get("cwd")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let parent_session_path = header_parent_session(&header);
    let rlm_depth = resolve_session_rlm_depth(&header, file_path);
    let modified =
        get_session_modified_date_from_last_activity(acc.last_activity_time, &header, mtime);

    Some(SessionInfo {
        path: file_path.to_string(),
        id: header
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        cwd,
        name: acc.name.clone(),
        state: acc.state.clone(),
        parent_session_path,
        rlm_depth,
        created: header
            .get("timestamp")
            .and_then(Value::as_str)
            .map(iso_to_millis)
            .unwrap_or(f64::NAN),
        modified,
        message_count: acc.message_count,
        first_message: if acc.first_message.is_empty() {
            "(no messages)".to_string()
        } else {
            acc.first_message.clone()
        },
        all_messages_text: acc.all_messages_text.clone(),
        agent_status: acc.agent_status.clone(),
        usage: session_usage_summary_from(&usage_total),
    })
}

fn clone_accumulator(acc: &SessionScanAccumulator) -> SessionScanAccumulator {
    SessionScanAccumulator {
        header: acc.header.clone(),
        invalid: acc.invalid,
        message_count: acc.message_count,
        first_message: acc.first_message.clone(),
        all_messages_text: acc.all_messages_text.clone(),
        name: acc.name.clone(),
        state: acc.state.clone(),
        agent_status: acc.agent_status.clone(),
        last_activity_time: acc.last_activity_time,
        assistant_usage_by_id: acc.assistant_usage_by_id.clone(),
        attributed_child_usage: acc.attributed_child_usage.clone(),
        summarization_usage: acc.summarization_usage.clone(),
    }
}

fn is_message_with_content(message: &Map<String, Value>) -> bool {
    message.get("role").map(Value::is_string).unwrap_or(false) && message.contains_key("content")
}

fn extract_text_content(message: &Map<String, Value>) -> String {
    match message.get("content") {
        Some(Value::String(content)) => content.clone(),
        Some(Value::Array(blocks)) => blocks
            .iter()
            .filter_map(|block| {
                let block = block.as_object()?;
                if block.get("type").and_then(Value::as_str) == Some("text") {
                    block
                        .get("text")
                        .and_then(Value::as_str)
                        .map(str::to_string)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join(" "),
        _ => String::new(),
    }
}

fn normalize_session_state_status(value: &Value) -> Option<SessionStateStatus> {
    match value.as_str() {
        Some("active") => Some(SessionStateStatus::Active),
        Some("archived") => Some(SessionStateStatus::Archived),
        Some("crash") => Some(SessionStateStatus::Crash),
        // Legacy statuses map onto archived.
        Some("hidden") | Some("sleep") => Some(SessionStateStatus::Archived),
        _ => None,
    }
}

fn update_last_activity_time(
    last_activity_time: Option<f64>,
    entry: &Map<String, Value>,
) -> Option<f64> {
    if entry_type(entry) != "message" {
        return last_activity_time;
    }
    let message = match message_of(entry) {
        Some(message) => message,
        None => return last_activity_time,
    };
    if !is_message_with_content(message) {
        return last_activity_time;
    }
    let role = message.get("role").and_then(Value::as_str).unwrap_or("");
    if role != "user" && role != "assistant" {
        return last_activity_time;
    }

    if let Some(timestamp) = message.get("timestamp").and_then(Value::as_f64) {
        return Some(last_activity_time.unwrap_or(0.0).max(timestamp));
    }

    if let Some(entry_timestamp) = entry.get("timestamp").and_then(Value::as_str) {
        let parsed = iso_to_millis(entry_timestamp);
        if !parsed.is_nan() {
            return Some(last_activity_time.unwrap_or(0.0).max(parsed));
        }
    }

    last_activity_time
}

fn get_session_modified_date_from_last_activity(
    last_activity_time: Option<f64>,
    header: &Map<String, Value>,
    stats_mtime: Option<std::time::SystemTime>,
) -> f64 {
    if let Some(last_activity_time) = last_activity_time {
        if last_activity_time > 0.0 {
            return last_activity_time;
        }
    }

    let header_time = header
        .get("timestamp")
        .and_then(Value::as_str)
        .map(iso_to_millis)
        .unwrap_or(f64::NAN);
    if !header_time.is_nan() {
        return header_time;
    }
    stats_mtime
        .and_then(|mtime| mtime.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_millis() as f64)
        .unwrap_or(f64::NAN)
}

fn append_capped_search_text(current: &str, text: &str) -> String {
    if text.is_empty() || current.len() >= SESSION_LIST_SEARCH_TEXT_MAX_CHARS {
        return current.to_string();
    }
    let next = if current.is_empty() {
        text.to_string()
    } else {
        format!(" {text}")
    };
    let remaining = SESSION_LIST_SEARCH_TEXT_MAX_CHARS - current.len();
    format!(
        "{current}{}",
        next.chars().take(remaining).collect::<String>()
    )
}

fn looks_like_message_entry(line: &str) -> bool {
    line.contains("\"type\":\"message\"") || line.contains("\"type\": \"message\"")
}

fn extract_json_string_property_prefix(
    text: &str,
    property_name: &str,
    max_chars: usize,
    start_index: usize,
) -> Option<String> {
    let needle = format!("\"{property_name}\"");
    let property_index =
        text[start_index.min(text.len())..].find(&needle)? + start_index.min(text.len());
    let mut chars: Vec<char> = text.chars().collect();
    let byte_to_char = |byte_index: usize| text[..byte_index.min(text.len())].chars().count();
    let mut index = byte_to_char(property_index + property_name.len() + 2);
    while index < chars.len() && chars[index].is_whitespace() {
        index += 1;
    }
    if index >= chars.len() || chars[index] != ':' {
        return None;
    }
    index += 1;
    while index < chars.len() && chars[index].is_whitespace() {
        index += 1;
    }
    if index >= chars.len() || chars[index] != '"' {
        return None;
    }
    index += 1;

    let mut result = String::new();
    let mut escaped = false;
    while index < chars.len() && result.chars().count() < max_chars {
        let character = chars[index];
        index += 1;
        if escaped {
            result.push(character);
            escaped = false;
            continue;
        }
        if character == '\\' {
            escaped = true;
            continue;
        }
        if character == '"' {
            break;
        }
        result.push(character);
    }
    let _ = &mut chars;
    Some(result)
}

#[derive(Default)]
struct OversizedMessageSummary {
    role: Option<String>,
    timestamp: Option<f64>,
    text_preview: Option<String>,
}

fn extract_oversized_message_summary(line: &str) -> OversizedMessageSummary {
    let timestamp_text = extract_json_string_property_prefix(line, "timestamp", 64, 0);
    let timestamp = timestamp_text.map(|text| iso_to_millis(&text));
    let message_index = line.find("\"message\"");
    let role = match message_index {
        Some(index) => extract_json_string_property_prefix(line, "role", 64, index),
        None => extract_json_string_property_prefix(line, "role", 64, 0),
    };
    let mut text_preview = None;
    if let Some(index) = message_index {
        text_preview = extract_json_string_property_prefix(
            line,
            "content",
            SESSION_LIST_LARGE_MESSAGE_PREVIEW_MAX_CHARS,
            index,
        )
        .or_else(|| {
            extract_json_string_property_prefix(
                line,
                "text",
                SESSION_LIST_LARGE_MESSAGE_PREVIEW_MAX_CHARS,
                index,
            )
        });
    }
    OversizedMessageSummary {
        role,
        timestamp: timestamp.filter(|value| !value.is_nan()),
        text_preview: text_preview.filter(|preview| !preview.is_empty()),
    }
}

async fn list_sessions_from_dir(
    dir: &str,
    callbacks: Option<&SessionListCallbacks>,
    progress_offset: i64,
    progress_total: Option<i64>,
) -> Vec<SessionInfo> {
    let mut sessions: Vec<SessionInfo> = Vec::new();
    if !Path::new(dir).exists() {
        let keys: Vec<String> = {
            let store = session_scan_store()
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            store.states.keys().cloned().collect()
        };
        for key in keys {
            if dirname(&key) == dir {
                drop_session_scan_state(&key);
            }
        }
        return sessions;
    }

    let dir_entries = match std::fs::read_dir(dir) {
        Ok(dir_entries) => dir_entries,
        // Return no sessions when the directory cannot be read.
        Err(_) => return sessions,
    };
    let mut files: Vec<String> = Vec::new();
    for entry in dir_entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.ends_with(".jsonl") {
            files.push(Path::new(dir).join(&name).to_string_lossy().to_string());
        }
    }
    let total = progress_total.unwrap_or(files.len() as i64);

    let present: HashSet<String> = files.iter().cloned().collect();
    let keys: Vec<String> = {
        let store = session_scan_store()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        store.states.keys().cloned().collect()
    };
    for key in keys {
        if dirname(&key) == dir && !present.contains(&key) {
            drop_session_scan_state(&key);
        }
    }

    let mut loaded = 0i64;
    for file in files {
        let info = read_session_info(&file).await;
        loaded += 1;
        if let Some(callbacks) = callbacks {
            if let Some(on_progress) = callbacks.on_progress.as_ref() {
                on_progress(progress_offset + loaded, total);
            }
        }
        if let Some(info) = info {
            if let Some(callbacks) = callbacks {
                if let Some(on_session) = callbacks.on_session.as_ref() {
                    on_session(&info);
                }
            }
            sessions.push(info);
        }
    }

    sessions
}

// ---------------------------------------------------------------------------
// Session file paths and migration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionHeader {
    #[serde(rename = "type")]
    pub type_: String,
    /// v1 sessions don't have this.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<i64>,
    pub id: String,
    pub timestamp: String,
    pub cwd: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_session: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rlm_depth: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub git: Option<GitContext>,
}

#[derive(Debug, Clone, Default)]
pub struct NewSessionOptions {
    pub id: Option<String>,
    pub parent_session: Option<String>,
    pub rlm_depth: Option<i64>,
}

impl NewSessionOptions {
    /// `Object.hasOwn(options, "rlmDepth")` - an explicit `null`/absent distinction.
    pub fn has_explicit_rlm_depth(&self) -> bool {
        self.rlm_depth.is_some()
    }
}

pub type SessionPersistListener = Box<dyn Fn(&str) + Send + Sync>;

fn create_session_id() -> String {
    uuid::Uuid::now_v7().to_string()
}

fn get_session_file_path(session_dir: &str, session_id: &str) -> String {
    Path::new(session_dir)
        .join(format!("{session_id}.jsonl"))
        .to_string_lossy()
        .to_string()
}

fn create_unique_session_file_target(session_dir: &str) -> Result<(String, String), String> {
    create_unique_session_file_target_with(session_dir, &|session_file| {
        Path::new(session_file).exists()
    })
}

/// TS `createUniqueSessionFileTarget` throws a catchable Error when the retry
/// loop exhausts (session-manager.ts:514-523); the panic made exhaustion
/// unobservable. The existence probe is injectable for parity tests.
pub fn create_unique_session_file_target_with(
    session_dir: &str,
    exists: &dyn Fn(&str) -> bool,
) -> Result<(String, String), String> {
    for _ in 0..100 {
        let session_id = create_session_id();
        let session_file = get_session_file_path(session_dir, &session_id);
        if !exists(&session_file) {
            return Ok((session_id, session_file));
        }
    }
    Err("Unable to create a unique session file".to_string())
}

pub fn get_session_artifacts_root(session_dir: &str) -> String {
    // `session-manager.ts:525`: `join(dirname(sessionDir), "session-artifacts")`,
    // so this is a plain host join. Only the full artifact path below is
    // POSIX-normalised (see `join_posix`).
    Path::new(&dirname(session_dir))
        .join("session-artifacts")
        .to_string_lossy()
        .to_string()
}

/// Forward-slash join for the artifact path, mirroring the port's existing
/// POSIX-normalisation convention (`to_posix_path`, `package_manager.rs:246`,
/// and `core/refinement/refinement.rs` for the harness-state paths that live
/// under this same artifact directory).
///
/// `session-file-actions.ts:16` documents the shape as
/// `<dirname(sessionDir)>/session-artifacts/<id>`, and session artifact paths
/// are exported to the RLM kernel and recorded in the session JSON, so a
/// host-style separator must not leak into them.
fn to_posix_path(value: &str) -> String {
    value.replace(std::path::MAIN_SEPARATOR, "/")
}

/// `join(base, leaf)` after POSIX normalisation; an empty base stays empty so
/// the daemon's `artifact_dir.is_empty()` guard still works.
fn join_posix(base: &str, leaf: &str) -> String {
    let base = to_posix_path(base);
    if base.is_empty() {
        return leaf.to_string();
    }
    if base.ends_with('/') || leaf.is_empty() {
        format!("{base}{leaf}")
    } else {
        format!("{base}/{leaf}")
    }
}

pub fn get_session_artifact_path(session_dir: &str, session_id: &str) -> String {
    join_posix(&get_session_artifacts_root(session_dir), session_id)
}

pub fn get_session_artifact_path_for_file(session_file: &str, session_id: Option<&str>) -> String {
    let session_id = session_id.map(str::to_string).unwrap_or_else(|| {
        let name = basename(session_file);
        name.strip_suffix(".jsonl").unwrap_or(&name).to_string()
    });
    get_session_artifact_path(&dirname(session_file), &session_id)
}

fn generate_id(by_id: &dyn Fn(&str) -> bool) -> String {
    for _ in 0..100 {
        let id: String = uuid::Uuid::new_v4().to_string().chars().take(8).collect();
        if !by_id(&id) {
            return id;
        }
    }
    uuid::Uuid::new_v4().to_string()
}

fn migrate_v1_to_v2(entries: &mut Vec<FileEntry>) {
    // The TypeScript passes a Set that is never populated, so the id is always fresh.
    let ids: BTreeSet<String> = BTreeSet::new();
    let mut prev_id: Option<String> = None;

    for index in 0..entries.len() {
        if entry_type(&entries[index]) == "session" {
            entries[index].insert("version".to_string(), Value::Number(2.into()));
            continue;
        }

        let id = generate_id(&|candidate| ids.contains(candidate));
        entries[index].insert("id".to_string(), Value::String(id.clone()));
        entries[index].insert(
            "parentId".to_string(),
            match &prev_id {
                Some(prev) => Value::String(prev.clone()),
                None => Value::Null,
            },
        );
        prev_id = Some(id);

        if entry_type(&entries[index]) == "compaction" {
            let first_kept_index = entries[index]
                .get("firstKeptEntryIndex")
                .and_then(is_safe_integer);
            if let Some(first_kept_index) = first_kept_index {
                if let Some(target) = entries.get(first_kept_index.max(0) as usize) {
                    if entry_type(target) != "session" {
                        let target_id = entry_id(target);
                        entries[index]
                            .insert("firstKeptEntryId".to_string(), Value::String(target_id));
                    }
                }
                entries[index].shift_remove("firstKeptEntryIndex");
            }
        }
    }
}

fn migrate_v2_to_v3(entries: &mut Vec<FileEntry>) {
    for entry in entries.iter_mut() {
        if entry_type(entry) == "session" {
            entry.insert("version".to_string(), Value::Number(3.into()));
            continue;
        }

        if entry_type(entry) == "message" {
            if let Some(message) = entry.get_mut("message").and_then(Value::as_object_mut) {
                if message.get("role").and_then(Value::as_str) == Some("hookMessage") {
                    message.insert("role".to_string(), Value::String("custom".to_string()));
                }
            }
        }
    }
}

fn migrate_to_current_version(entries: &mut [FileEntry]) -> bool {
    let header = entries.iter().find(|entry| is_session_header(entry));
    let version = header
        .and_then(|header| header.get("version"))
        .and_then(Value::as_i64)
        .unwrap_or(1);

    if version >= CURRENT_SESSION_VERSION {
        return false;
    }

    let mut owned: Vec<FileEntry> = entries.to_vec();
    if version < 2 {
        migrate_v1_to_v2(&mut owned);
    }
    if version < 3 {
        migrate_v2_to_v3(&mut owned);
    }
    for (index, entry) in owned.into_iter().enumerate() {
        entries[index] = entry;
    }
    true
}

pub fn migrate_session_entries(entries: &mut [FileEntry]) {
    migrate_to_current_version(entries);
}

pub fn parse_session_entries(content: &str) -> Vec<FileEntry> {
    let mut entries: Vec<FileEntry> = Vec::new();
    for line in content.trim().split('\n') {
        if line.trim().is_empty() {
            continue;
        }
        // Skip malformed lines.
        if let Ok(Value::Object(entry)) = serde_json::from_str::<Value>(line) {
            entries.push(rehydrate_session_file_entry(entry));
        }
    }
    apply_child_usage_attributions(&mut entries);
    entries
}

// ---------------------------------------------------------------------------
// Session tree shapes
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct SessionTreeFlatNode {
    pub entry: SessionEntry,
    pub label: Option<String>,
    pub label_timestamp: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SessionTreeNode {
    pub node: SessionTreeFlatNode,
    pub children: Vec<SessionTreeNode>,
}

/// `CustomMessageEntryContent` - `content: string | (TextContent | ImageContent)[]`.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub enum CustomMessageEntryContent {
    Text(String),
    Blocks(Vec<Value>),
}

impl CustomMessageEntryContent {
    fn to_value(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }
}

// ---------------------------------------------------------------------------
// SessionManager
// ---------------------------------------------------------------------------

pub struct SessionManager {
    session_id: String,
    session_file: Option<String>,
    session_dir: String,
    cwd: String,
    persist: bool,
    flushed: bool,
    file_entries: Vec<FileEntry>,
    by_id: indexmap::IndexMap<String, SessionEntry>,
    labels_by_id: indexmap::IndexMap<String, String>,
    label_timestamps_by_id: indexmap::IndexMap<String, String>,
    leaf_id: Option<String>,
    persist_listeners: Arc<Mutex<Vec<Option<SessionPersistListener>>>>,
    load_observation: Option<SessionLoadObservation>,
}

impl SessionManager {
    fn new_internal(
        cwd: String,
        session_dir: String,
        session_file: Option<String>,
        persist: bool,
        preloaded_entries: Option<Vec<FileEntry>>,
        preloaded_observation: Option<SessionLoadObservation>,
    ) -> Result<Self, String> {
        let mut manager = SessionManager {
            session_id: String::new(),
            session_file: None,
            session_dir,
            cwd,
            persist,
            flushed: false,
            file_entries: Vec::new(),
            by_id: indexmap::IndexMap::new(),
            labels_by_id: indexmap::IndexMap::new(),
            label_timestamps_by_id: indexmap::IndexMap::new(),
            leaf_id: None,
            persist_listeners: Arc::new(Mutex::new(Vec::new())),
            load_observation: None,
        };
        if persist && !manager.session_dir.is_empty() && !Path::new(&manager.session_dir).exists() {
            let _ = std::fs::create_dir_all(&manager.session_dir);
        }

        match session_file {
            Some(session_file) => {
                manager.set_session_file(&session_file, preloaded_entries, preloaded_observation)?
            }
            None => {
                manager.new_session(None)?;
            }
        }
        Ok(manager)
    }

    /// Switch to a different session file (used for resume and branching).
    /// preloadedEntries must be loadEntriesFromFile(sessionFile) for the same path; it
    /// lets the async daemon path skip the synchronous re-read.
    pub fn set_session_file(
        &mut self,
        session_file: &str,
        preloaded_entries: Option<Vec<FileEntry>>,
        preloaded_observation: Option<SessionLoadObservation>,
    ) -> Result<(), String> {
        let session_file = resolve_path(session_file);
        // Failure must leave the previous manager writable only at its old path.
        // Public preloaded callers cannot bypass the required repair gate.
        let repaired = self.persist && Path::new(&session_file).exists()
            && repair_jsonl_damage(&session_file)?;
        let preloaded_entries = if repaired { None } else { preloaded_entries };
        // A switch/reload must never report the prior transcript's bytes.
        self.load_observation = None;
        self.session_file = Some(session_file.clone());
        if Path::new(&session_file).exists() {
            match preloaded_entries {
                None => {
                    let loaded = load_entries_from_file_observed(&session_file);
                    self.file_entries = loaded.entries;
                    self.load_observation = loaded.observation;
                }
                Some(preloaded_entries) => {
                    self.load_observation = if preloaded_entries.is_empty() {
                        None
                    } else {
                        preloaded_observation
                    };
                    self.file_entries = preloaded_entries;
                }
            }

            // If file was empty or corrupted (no valid header), truncate and start fresh
            // to avoid appending messages without a session header (which breaks the session)
            if self.file_entries.is_empty() {
                let explicit_path = session_file;
                self.new_session(None)?;
                self.session_file = Some(explicit_path);
                self.rewrite_file()?;
                self.flushed = true;
                return Ok(());
            }

            let header = self
                .file_entries
                .iter()
                .find(|entry| is_session_header(entry))
                .cloned();
            self.session_id = header
                .as_ref()
                .and_then(|header| header.get("id"))
                .and_then(Value::as_str)
                .map(str::to_string)
                .unwrap_or_else(create_session_id);

            let mut should_rewrite = migrate_to_current_version(&mut self.file_entries);
            if let Some(header) = header.as_ref() {
                let has_parent = header_parent_session(header).is_some();
                if has_parent && header_rlm_depth(header).is_none() {
                    let depth = resolve_session_rlm_depth(header, &session_file);
                    if let Some(header_index) = self
                        .file_entries
                        .iter()
                        .position(|entry| is_session_header(entry))
                    {
                        self.file_entries[header_index]
                            .insert("rlmDepth".to_string(), Value::Number(depth.into()));
                    }
                    should_rewrite = true;
                }
            }
            if should_rewrite {
                self.rewrite_file()?;
            }

            self.build_index();
            self.flushed = true;
        } else {
            let explicit_path = session_file;
            self.new_session(None)?;
            // preserve explicit path from --resume selector
            self.session_file = Some(explicit_path);
        }
        Ok(())
    }

    pub fn new_session(
        &mut self,
        options: Option<&NewSessionOptions>,
    ) -> Result<Option<String>, String> {
        self.load_observation = None;
        let mut session_id = options
            .and_then(|options| options.id.clone())
            .unwrap_or_else(create_session_id);
        let mut session_file: Option<String> = None;
        let has_explicit_rlm_depth = options
            .map(NewSessionOptions::has_explicit_rlm_depth)
            .unwrap_or(false);
        let mut parent_header: Option<Map<String, Value>> = None;
        if let Some(parent_session) = options.and_then(|options| options.parent_session.clone()) {
            if !has_explicit_rlm_depth {
                // Unavailable parent metadata leaves the child depth unknown.
                parent_header = read_session_header(&parent_session);
            }
        }
        if self.persist {
            match options.and_then(|options| options.id.clone()) {
                Some(explicit_id) => {
                    session_file =
                        Some(get_session_file_path(&self.get_session_dir(), &session_id));
                    if let Some(file) = session_file.as_ref() {
                        if Path::new(file).exists() {
                            return Err(format!(
                                "Session file already exists for id \"{explicit_id}\": {file}"
                            ));
                        }
                    }
                }
                None => {
                    let target = create_unique_session_file_target(&self.get_session_dir())?;
                    session_id = target.0;
                    session_file = Some(target.1);
                }
            }
        }

        self.session_id = session_id;
        let timestamp = iso_now();
        let git = if self.persist {
            capture_git_context(&self.cwd)
        } else {
            None
        };
        let rlm_depth = if has_explicit_rlm_depth {
            options.and_then(|options| options.rlm_depth)
        } else if options
            .and_then(|options| options.parent_session.as_ref())
            .is_some()
        {
            derive_child_rlm_depth(parent_header.as_ref())
        } else {
            Some(root_rlm_depth_from_env()?)
        };
        let header = SessionHeader {
            type_: "session".to_string(),
            version: Some(CURRENT_SESSION_VERSION),
            id: self.session_id.clone(),
            timestamp,
            cwd: self.cwd.clone(),
            parent_session: options.and_then(|options| options.parent_session.clone()),
            rlm_depth,
            git,
        };
        self.file_entries = vec![serde_json::to_value(&header)
            .ok()
            .and_then(|value| value.as_object().cloned())
            .unwrap_or_default()];
        self.by_id.clear();
        self.labels_by_id.clear();
        self.label_timestamps_by_id.clear();
        self.leaf_id = None;
        self.flushed = false;

        if self.persist {
            self.session_file = session_file;
        }
        Ok(self.session_file.clone())
    }

    fn build_index(&mut self) {
        self.by_id.clear();
        self.labels_by_id.clear();
        self.label_timestamps_by_id.clear();
        self.leaf_id = None;
        for entry in &self.file_entries {
            if is_session_header(entry) {
                continue;
            }
            let id = entry_id(entry);
            self.by_id.insert(id.clone(), entry.clone());
            self.leaf_id = Some(id);
            if entry_type(entry) == "label" {
                let target_id = entry
                    .get("targetId")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let label = entry.get("label").and_then(Value::as_str);
                match label {
                    Some(label) => {
                        self.labels_by_id
                            .insert(target_id.clone(), label.to_string());
                        self.label_timestamps_by_id
                            .insert(target_id, entry_timestamp(entry));
                    }
                    None => {
                        self.labels_by_id.shift_remove(&target_id);
                        self.label_timestamps_by_id.shift_remove(&target_id);
                    }
                }
            }
        }
    }

    // TS _rewriteFile throws synchronously on any write failure and only
    // notifies persistence observers after a successful commit
    // (session-manager.ts:1900-1914). Propagate instead of swallowing.
    fn rewrite_file(&mut self) -> Result<(), String> {
        if !self.persist || self.session_file.is_none() {
            return Ok(());
        }
        let session_file = self.session_file.clone().unwrap_or_default();
        let content = format!(
            "{}\n",
            self.file_entries
                .iter()
                .map(serialize_session_file_entry)
                .collect::<Vec<_>>()
                .join("\n")
        );
        let target_path = realpath_if_present_sync(&session_file);
        let directory = dirname(&target_path);
        std::fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
        let metadata = stat_metadata_if_present(&target_path);
        let mode = metadata.as_ref().map(|metadata| metadata.mode);
        write_file_atomic_sync(
            &target_path,
            &content,
            WriteFileAtomicOptions { mode, fsync: false },
            None,
        )?;
        // Observers see committed writes only.
        self.notify_persist_listeners();
        Ok(())
    }

    fn notify_persist_listeners(&self) {
        let Some(session_file) = self.session_file.as_ref() else {
            return;
        };
        let listeners = self
            .persist_listeners
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        for listener in listeners.iter().flatten() {
            // Persistence observers must not break session writes.
            listener(session_file);
        }
    }

    /// `onPersist(listener): () => void` - the returned closure unsubscribes.
    pub fn on_persist(&self, listener: SessionPersistListener) -> Box<dyn Fn() + Send + Sync> {
        let listeners = Arc::clone(&self.persist_listeners);
        let index = {
            let mut guard = listeners.lock().unwrap_or_else(|error| error.into_inner());
            guard.push(Some(listener));
            guard.len() - 1
        };
        Box::new(move || {
            let mut guard = listeners.lock().unwrap_or_else(|error| error.into_inner());
            if index < guard.len() {
                guard[index] = None;
            }
        })
    }

    pub fn is_persisted(&self) -> bool {
        self.persist
    }

    pub fn get_cwd(&self) -> String {
        self.cwd.clone()
    }

    pub fn get_session_dir(&self) -> String {
        self.session_dir.clone()
    }

    pub fn get_session_id(&self) -> String {
        self.session_id.clone()
    }

    pub fn get_session_file(&self) -> Option<String> {
        self.session_file.clone()
    }

    /// Primary transcript-read bytes for the most recent successful file load, excluding header/repair probes.
    pub fn get_load_observation(&self) -> Option<SessionLoadObservation> {
        self.load_observation
    }

    pub fn materialize_session_file(
        &mut self,
        session_dir: Option<&str>,
    ) -> Result<String, String> {
        if let Some(session_file) = self.session_file.clone() {
            return Ok(session_file);
        }
        let dir = session_dir.map(str::to_string).unwrap_or_else(|| {
            if self.session_dir.is_empty() {
                get_default_session_dir(&self.cwd, None)
            } else {
                self.session_dir.clone()
            }
        });
        if !Path::new(&dir).exists() {
            let _ = std::fs::create_dir_all(&dir);
        }
        let previous_header = self.get_header();
        let target = create_unique_session_file_target(&dir)?;
        self.session_dir = dir;
        self.session_id = target.0;
        self.session_file = Some(target.1.clone());
        self.persist = true;
        let timestamp = iso_now();
        let git = capture_git_context(&self.cwd);
        let header = SessionHeader {
            type_: "session".to_string(),
            version: Some(CURRENT_SESSION_VERSION),
            id: self.session_id.clone(),
            timestamp,
            cwd: self.cwd.clone(),
            parent_session: previous_header
                .as_ref()
                .and_then(|header| header.get("parentSession"))
                .and_then(Value::as_str)
                .map(str::to_string),
            rlm_depth: Some(resolve_session_rlm_depth(
                previous_header.as_ref().unwrap_or(&Map::new()),
                &target.1,
            )),
            git,
        };
        let mut entries: Vec<FileEntry> = vec![serde_json::to_value(&header)
            .ok()
            .and_then(|value| value.as_object().cloned())
            .unwrap_or_default()];
        entries.extend(self.get_entries());
        self.file_entries = entries;
        self.rewrite_file()?;
        self.flushed = true;
        Ok(self.session_file.clone().unwrap_or_default())
    }

    pub fn get_session_artifact_dir(&self) -> Option<String> {
        if self.persist {
            Some(get_session_artifact_path(
                &self.session_dir,
                &self.session_id,
            ))
        } else {
            None
        }
    }

    /// Force-write all in-memory entries to the session file immediately.
    /// This bypasses the no-assistant guard in `persist` so that
    /// pre-model entries (session header, goal state, settings changes)
    /// are durable on disk before the first assistant response.
    /// No-op for in-memory (non-persisted) sessions.
    pub fn flush_now(&mut self) -> Result<(), String> {
        if !self.persist {
            return Ok(());
        }
        let Some(session_file) = self.session_file.clone() else {
            return Ok(());
        };
        if self.flushed && Path::new(&session_file).exists() {
            return Ok(());
        }
        self.rewrite_file()?;
        self.flushed = true;
        Ok(())
    }

    fn persist(&mut self, entry: &SessionEntry) -> Result<(), String> {
        if !self.persist {
            return Ok(());
        }
        let Some(session_file) = self.session_file.clone() else {
            return Ok(());
        };

        let has_assistant = self
            .file_entries
            .iter()
            .any(|entry| entry_type(entry) == "message" && message_role(entry) == "assistant");
        let entry_type_name = entry_type(entry);
        let should_persist_without_assistant =
            entry_type_name == "session_state" || entry_type_name == "session_info";
        if !self.flushed && !has_assistant && !should_persist_without_assistant {
            return Ok(());
        }

        if !self.flushed || !Path::new(&session_file).exists() {
            self.rewrite_file()?;
            self.flushed = true;
        } else {
            let _ = std::fs::create_dir_all(dirname(&session_file));
            let mut file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&session_file)
                .map_err(|error| error.to_string())?;
            use std::io::Write;
            writeln!(file, "{}", serialize_session_file_entry(entry))
                .map_err(|error| error.to_string())?;
            self.notify_persist_listeners();
        }
        Ok(())
    }

    fn append_entry(&mut self, entry: SessionEntry) -> Result<(), String> {
        let id = entry_id(&entry);
        self.file_entries.push(entry.clone());
        self.by_id.insert(id.clone(), entry.clone());
        self.leaf_id = Some(id);
        let result = self.persist(&entry);
        if result.is_err() {
            // Retain the unsaved entry and force recovery to rewrite the complete
            // transcript, including any line an unsuccessful append partly wrote.
            self.flushed = false;
        }
        result
    }

    fn append_entry_with_rollback(
        &mut self,
        append: impl FnOnce(&mut Self) -> Result<String, String>,
    ) -> Result<String, String> {
        let previous_leaf_id = self.leaf_id.clone();
        let result = append(self);
        match result {
            Ok(entry_id) => match self.flush_now_checked() {
                Ok(()) => Ok(entry_id),
                Err(error) => {
                    self.rollback_last_append(previous_leaf_id);
                    Err(error)
                }
            },
            Err(error) => {
                self.rollback_last_append(previous_leaf_id);
                Err(error)
            }
        }
    }

    /// The append indexes the entry before persisting it; undo exactly that.
    fn rollback_last_append(&mut self, previous_leaf_id: Option<String>) {
        if self.leaf_id.is_some() && self.leaf_id != previous_leaf_id {
            if let Some(leaf) = self.leaf_id.clone() {
                self.by_id.shift_remove(&leaf);
            }
            self.file_entries.pop();
            self.leaf_id = previous_leaf_id;
            // The failed append may have left a torn line on disk. Restore the file
            // from the rolled-back entries now; if that also fails (e.g. the disk is
            // still full), fall back to forcing the next persist to rewrite.
            self.flushed = false;
            if self.flush_now_checked().is_err() {
                self.flushed = false;
            }
        }
    }

    fn flush_now_checked(&mut self) -> Result<(), String> {
        if !self.persist {
            return Ok(());
        }
        let Some(session_file) = self.session_file.clone() else {
            return Ok(());
        };
        if self.flushed && Path::new(&session_file).exists() {
            return Ok(());
        }
        self.rewrite_file()?;
        self.flushed = true;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Append APIs
    // -----------------------------------------------------------------------

    fn next_entry_id(&self) -> String {
        let by_id = &self.by_id;
        generate_id(&|candidate| by_id.contains_key(candidate))
    }

    pub fn append_message(&mut self, message: AgentMessage) -> Result<String, String> {
        let entry: SessionEntry = serde_json::json!({
            "type": "message",
            "id": self.next_entry_id(),
            "parentId": match &self.leaf_id {
                Some(leaf) => Value::String(leaf.clone()),
                None => Value::Null,
            },
            "timestamp": iso_now(),
            "message": serde_json::to_value(&message).unwrap_or(Value::Null),
        })
        .as_object()
        .cloned()
        .unwrap_or_default();
        let id = entry_id(&entry);
        self.append_entry(entry)?;
        Ok(id)
    }

    pub fn append_thinking_level_change(&mut self, thinking_level: &str) -> Result<String, String> {
        let entry: SessionEntry = serde_json::json!({
            "type": "thinking_level_change",
            "id": self.next_entry_id(),
            "parentId": match &self.leaf_id {
                Some(leaf) => Value::String(leaf.clone()),
                None => Value::Null,
            },
            "timestamp": iso_now(),
            "thinkingLevel": thinking_level,
        })
        .as_object()
        .cloned()
        .unwrap_or_default();
        let id = entry_id(&entry);
        self.append_entry(entry)?;
        Ok(id)
    }

    pub fn append_service_tier_change(
        &mut self,
        service_tier: &ServiceTier,
    ) -> Result<String, String> {
        let entry: SessionEntry = serde_json::json!({
            "type": "service_tier_change",
            "id": self.next_entry_id(),
            "parentId": match &self.leaf_id {
                Some(leaf) => Value::String(leaf.clone()),
                None => Value::Null,
            },
            "timestamp": iso_now(),
            "serviceTier": serde_json::to_value(service_tier).unwrap_or(Value::Null),
        })
        .as_object()
        .cloned()
        .unwrap_or_default();
        let id = entry_id(&entry);
        self.append_entry(entry)?;
        Ok(id)
    }

    pub fn append_model_change(
        &mut self,
        provider: &str,
        model_id: &str,
    ) -> Result<String, String> {
        let entry: SessionEntry = serde_json::json!({
            "type": "model_change",
            "id": self.next_entry_id(),
            "parentId": match &self.leaf_id {
                Some(leaf) => Value::String(leaf.clone()),
                None => Value::Null,
            },
            "timestamp": iso_now(),
            "provider": provider,
            "modelId": model_id,
        })
        .as_object()
        .cloned()
        .unwrap_or_default();
        let id = entry_id(&entry);
        self.append_entry(entry)?;
        Ok(id)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn append_compaction(
        &mut self,
        summary: &str,
        first_kept_entry_id: &str,
        tokens_before: f64,
        details: Option<Value>,
        from_hook: Option<bool>,
        custom_instructions: Option<&str>,
        usage: Option<&Usage>,
        harness_digest: Option<&str>,
    ) -> Result<String, String> {
        let mut entry: SessionEntry = serde_json::json!({
            "type": "compaction",
            "id": self.next_entry_id(),
            "parentId": match &self.leaf_id {
                Some(leaf) => Value::String(leaf.clone()),
                None => Value::Null,
            },
            "timestamp": iso_now(),
            "summary": summary,
            "firstKeptEntryId": first_kept_entry_id,
            "tokensBefore": tokens_before,
        })
        .as_object()
        .cloned()
        .unwrap_or_default();
        // Optional keys are only written when the caller passed them, matching
        // how a JavaScript object literal with `undefined` values still
        // serializes the key as absent through JSON.stringify.
        if let Some(details) = details {
            entry.insert("details".to_string(), details);
        }
        if let Some(from_hook) = from_hook {
            entry.insert("fromHook".to_string(), Value::Bool(from_hook));
        }
        if let Some(custom_instructions) = custom_instructions {
            entry.insert(
                "customInstructions".to_string(),
                Value::String(custom_instructions.to_string()),
            );
        }
        if let Some(usage) = usage {
            entry.insert(
                "usage".to_string(),
                serde_json::to_value(usage).unwrap_or(Value::Null),
            );
        }
        if let Some(harness_digest) = harness_digest {
            entry.insert(
                "harnessDigest".to_string(),
                Value::String(harness_digest.to_string()),
            );
        }
        let id = entry_id(&entry);
        self.append_entry_with_rollback(move |manager| {
            manager.append_entry(entry)?;
            Ok(id)
        })
    }

    pub fn append_custom_entry(
        &mut self,
        custom_type: &str,
        data: Option<Value>,
    ) -> Result<String, String> {
        let mut entry: SessionEntry = serde_json::json!({
            "type": "custom",
            "customType": custom_type,
            "id": self.next_entry_id(),
            "parentId": match &self.leaf_id {
                Some(leaf) => Value::String(leaf.clone()),
                None => Value::Null,
            },
            "timestamp": iso_now(),
        })
        .as_object()
        .cloned()
        .unwrap_or_default();
        if let Some(data) = data {
            entry.insert("data".to_string(), data);
        }
        let id = entry_id(&entry);
        self.append_entry(entry)?;
        Ok(id)
    }

    pub fn append_custom_entry_with_rollback(
        &mut self,
        custom_type: &str,
        data: Option<Value>,
    ) -> Result<String, String> {
        self.append_entry_with_rollback(|manager| {
            manager.append_custom_entry(custom_type, data.clone())
        })
    }

    pub fn append_child_usage_attribution(
        &mut self,
        target_id: &str,
        child_usage: &Usage,
        aggregate_usage: &Usage,
        origin: Option<&str>,
    ) -> Result<String, String> {
        let target_is_assistant = self
            .by_id
            .get(target_id)
            .map(|target| entry_type(target) == "message" && message_role(target) == "assistant")
            .unwrap_or(false);
        if !target_is_assistant {
            return Err(format!("Assistant message entry {target_id} not found"));
        }

        if let Some(target) = self.by_id.get_mut(target_id) {
            if let Some(message) = target.get_mut("message").and_then(Value::as_object_mut) {
                message.insert(
                    "usage".to_string(),
                    serde_json::to_value(clone_usage(aggregate_usage)).unwrap_or(Value::Null),
                );
            }
        }
        let mut entry: SessionEntry = serde_json::json!({
            "type": "child_usage_attributed",
            "id": self.next_entry_id(),
            "parentId": match &self.leaf_id {
                Some(leaf) => Value::String(leaf.clone()),
                None => Value::Null,
            },
            "timestamp": iso_now(),
            "targetId": target_id,
            "childUsage": serde_json::to_value(clone_usage(child_usage)).unwrap_or(Value::Null),
            "aggregateUsage": serde_json::to_value(clone_usage(aggregate_usage)).unwrap_or(Value::Null),
        })
        .as_object()
        .cloned()
        .unwrap_or_default();
        if let Some(origin) = origin {
            entry.insert("origin".to_string(), Value::String(origin.to_string()));
        }
        let id = entry_id(&entry);
        self.append_entry(entry)?;
        Ok(id)
    }

    pub fn append_session_info(&mut self, name: &str) -> Result<String, String> {
        let entry: SessionEntry = serde_json::json!({
            "type": "session_info",
            "id": self.next_entry_id(),
            "parentId": match &self.leaf_id {
                Some(leaf) => Value::String(leaf.clone()),
                None => Value::Null,
            },
            "timestamp": iso_now(),
            "name": name.trim(),
        })
        .as_object()
        .cloned()
        .unwrap_or_default();
        let id = entry_id(&entry);
        self.append_entry(entry)?;
        Ok(id)
    }

    pub fn append_session_state(&mut self, state: &SessionState) -> Result<String, String> {
        let entry: SessionEntry = serde_json::json!({
            "type": "session_state",
            "id": self.next_entry_id(),
            "parentId": match &self.leaf_id {
                Some(leaf) => Value::String(leaf.clone()),
                None => Value::Null,
            },
            "timestamp": iso_now(),
            "state": { "status": state.status },
        })
        .as_object()
        .cloned()
        .unwrap_or_default();
        let id = entry_id(&entry);
        self.append_entry(entry)?;
        Ok(id)
    }

    pub fn get_session_name(&self) -> Option<String> {
        // Metadata is read on every streamed event; do not clone the transcript.
        for entry in self.file_entries.iter().rev() {
            if entry_type(entry) == "session_info" {
                return entry
                    .get("name")
                    .and_then(Value::as_str)
                    .map(|name| name.trim().to_string())
                    .filter(|name| !name.is_empty());
            }
        }
        None
    }

    pub fn get_session_state(&self) -> Option<SessionState> {
        for entry in self.file_entries.iter().rev() {
            if entry_type(entry) == "session_state" {
                let status = entry
                    .get("state")
                    .and_then(Value::as_object)
                    .and_then(|state| state.get("status"))
                    .and_then(normalize_session_state_status);
                if let Some(status) = status {
                    return Some(SessionState { status });
                }
            }
        }
        None
    }

    /// True when the session holds user-meaningful persisted content, as opposed to
    /// only daemon-written bookkeeping (session_state, agent_status, git_state) or
    /// the default model/thinking entries every new session is created with. Used by
    /// the daemon discard guard to decide whether a message-less draft is safe to
    /// delete (that guard always also requires zero messages).
    ///
    /// createAgentSession opens a new session with an optional leading `model_change`
    /// followed by `thinking_level_change` and `service_tier_change`. That creation
    /// prefix is skipped; anything beyond it is user content.
    pub fn has_user_content(&self) -> bool {
        let content_entries: Vec<SessionEntry> = self
            .get_entries()
            .into_iter()
            .filter(|entry| content_entry_types().contains(entry_type(entry)))
            .collect();
        let mut start = 0usize;
        if content_entries
            .get(start)
            .map(|entry| entry_type(entry) == "model_change")
            .unwrap_or(false)
        {
            start += 1;
        }
        if content_entries
            .get(start)
            .map(|entry| entry_type(entry) == "thinking_level_change")
            .unwrap_or(false)
        {
            start += 1;
        }
        if content_entries
            .get(start)
            .map(|entry| entry_type(entry) == "service_tier_change")
            .unwrap_or(false)
        {
            start += 1;
        }
        content_entries.len() > start
    }

    pub fn append_agent_status(&mut self, status: &AgentStatus) -> Result<String, String> {
        let entry: SessionEntry = serde_json::json!({
            "type": "agent_status",
            "id": self.next_entry_id(),
            "parentId": match &self.leaf_id {
                Some(leaf) => Value::String(leaf.clone()),
                None => Value::Null,
            },
            "timestamp": iso_now(),
            "status": {
                "summary": status.summary,
                "taskState": status.task_state,
                "basedOnMessageCount": status.based_on_message_count,
            },
        })
        .as_object()
        .cloned()
        .unwrap_or_default();
        let id = entry_id(&entry);
        self.append_entry(entry)?;
        Ok(id)
    }

    pub fn append_git_state(&mut self, git: &GitContext) -> Result<String, String> {
        let entry: SessionEntry = serde_json::json!({
            "type": "git_state",
            "id": self.next_entry_id(),
            "parentId": match &self.leaf_id {
                Some(leaf) => Value::String(leaf.clone()),
                None => Value::Null,
            },
            "timestamp": iso_now(),
            "git": serde_json::to_value(git).unwrap_or(Value::Null),
        })
        .as_object()
        .cloned()
        .unwrap_or_default();
        let id = entry_id(&entry);
        self.append_entry(entry)?;
        Ok(id)
    }

    pub fn record_git_state_if_changed(&mut self) -> Result<Option<String>, String> {
        if !self.persist {
            return Ok(None);
        }
        let git = match capture_git_context(&self.cwd) {
            Some(git) => git,
            None => return Ok(None),
        };
        let last = self.get_active_git_context();
        if let Some(last) = last {
            if git_contexts_equal(&last, &git) {
                return Ok(None);
            }
        }
        self.append_git_state(&git).map(Some)
    }

    fn get_active_git_context(&self) -> Option<GitContext> {
        let mut current = self
            .leaf_id
            .as_ref()
            .and_then(|leaf| self.by_id.get(leaf))
            .cloned();
        let mut visited = HashSet::new();
        while let Some(entry) = current {
            if !visited.insert(entry_id(&entry)) { break; }
            if entry_type(&entry) == "git_state" {
                return entry
                    .get("git")
                    .and_then(|git| serde_json::from_value::<GitContext>(git.clone()).ok());
            }
            current = entry_parent_id(&entry).and_then(|parent| self.by_id.get(&parent).cloned());
        }
        let header = self.file_entries.first();
        match header {
            Some(header) if is_session_header(header) => header
                .get("git")
                .and_then(|git| serde_json::from_value::<GitContext>(git.clone()).ok()),
            _ => None,
        }
    }

    pub fn get_latest_agent_status(&self) -> Option<AgentStatus> {
        // Walk the current leaf to root so we only read status on the active branch,
        // not a sibling branch's status that happens to sit later in the file.
        let mut current = self
            .leaf_id
            .as_ref()
            .and_then(|leaf| self.by_id.get(leaf))
            .cloned();
        let mut visited = HashSet::new();
        while let Some(entry) = current {
            if !visited.insert(entry_id(&entry)) { break; }
            if entry_type(&entry) == "agent_status" {
                return entry
                    .get("status")
                    .and_then(|status| serde_json::from_value::<AgentStatus>(status.clone()).ok());
            }
            current = entry_parent_id(&entry).and_then(|parent| self.by_id.get(&parent).cloned());
        }
        None
    }

    pub fn append_custom_message_entry(
        &mut self,
        custom_type: &str,
        content: &CustomMessageEntryContent,
        display: bool,
        details: Option<Value>,
    ) -> Result<String, String> {
        let mut entry: SessionEntry = serde_json::json!({
            "type": "custom_message",
            "customType": custom_type,
            "content": content.to_value(),
            "display": display,
            "id": self.next_entry_id(),
            "parentId": match &self.leaf_id {
                Some(leaf) => Value::String(leaf.clone()),
                None => Value::Null,
            },
            "timestamp": iso_now(),
        })
        .as_object()
        .cloned()
        .unwrap_or_default();
        if let Some(details) = details {
            entry.insert("details".to_string(), details);
        }
        let id = entry_id(&entry);
        self.append_entry(entry)?;
        Ok(id)
    }

    /// Append a custom message, undoing the append if persistence fails so a
    /// best-effort record never leaves an unsaved leaf for later entries.
    pub fn append_custom_message_entry_with_rollback(
        &mut self,
        custom_type: &str,
        content: &CustomMessageEntryContent,
        display: bool,
        details: Option<Value>,
    ) -> Result<String, String> {
        let content = content.clone();
        let custom_type = custom_type.to_string();
        self.append_entry_with_rollback(move |manager| {
            manager.append_custom_message_entry(&custom_type, &content, display, details.clone())
        })
    }

    pub fn get_leaf_id(&self) -> Option<String> {
        self.leaf_id.clone()
    }

    pub fn get_leaf_entry(&self) -> Option<SessionEntry> {
        self.leaf_id
            .as_ref()
            .and_then(|leaf| self.by_id.get(leaf))
            .cloned()
    }

    pub fn get_entry(&self, id: &str) -> Option<SessionEntry> {
        self.by_id.get(id).cloned()
    }

    pub fn get_children(&self, parent_id: &str) -> Vec<SessionEntry> {
        let mut children: Vec<SessionEntry> = Vec::new();
        for entry in self.by_id.values() {
            if entry_parent_id(entry).as_deref() == Some(parent_id) {
                children.push(entry.clone());
            }
        }
        children
    }

    pub fn get_label(&self, id: &str) -> Option<String> {
        self.labels_by_id.get(id).cloned()
    }

    pub fn append_label_change(
        &mut self,
        target_id: &str,
        label: Option<&str>,
    ) -> Result<String, String> {
        if !self.by_id.contains_key(target_id) {
            return Err(format!("Entry {target_id} not found"));
        }
        let entry: SessionEntry = serde_json::json!({
            "type": "label",
            "id": self.next_entry_id(),
            "parentId": match &self.leaf_id {
                Some(leaf) => Value::String(leaf.clone()),
                None => Value::Null,
            },
            "timestamp": iso_now(),
            "targetId": target_id,
            "label": match label {
                Some(label) => Value::String(label.to_string()),
                None => Value::Null,
            },
        })
        .as_object()
        .cloned()
        .unwrap_or_default();
        let timestamp = entry_timestamp(&entry);
        let id = entry_id(&entry);
        self.append_entry(entry)?;
        match label {
            Some(label) => {
                self.labels_by_id
                    .insert(target_id.to_string(), label.to_string());
                self.label_timestamps_by_id
                    .insert(target_id.to_string(), timestamp);
            }
            None => {
                self.labels_by_id.shift_remove(target_id);
                self.label_timestamps_by_id.shift_remove(target_id);
            }
        }
        Ok(id)
    }

    pub fn get_branch(&self, from_id: Option<&str>) -> Vec<SessionEntry> {
        let mut path = Vec::new();
        self.visit_branch(from_id, |entry| path.push(entry.clone()));
        path
    }

    /// Visit indexed branch entries in transcript order without cloning message
    /// bodies. The temporary path contains references only, including for very
    /// large restored Python state or tool output entries.
    pub fn visit_branch(&self, from_id: Option<&str>, mut visit: impl FnMut(&SessionEntry)) {
        let mut path = Vec::new();
        let mut current = from_id.or(self.leaf_id.as_deref()).and_then(|id| self.by_id.get(id));
        let mut visited = HashSet::new();
        while let Some(entry) = current {
            if !visited.insert(entry.get("id").and_then(Value::as_str).unwrap_or_default()) { break; }
            path.push(entry);
            current = entry.get("parentId").and_then(Value::as_str).and_then(|id| self.by_id.get(id));
        }
        for entry in path.into_iter().rev() { visit(entry); }
    }

    pub fn build_session_context(
        &self,
        target_model: Option<&pi_ai::types::Model>,
    ) -> SessionContext {
        // Pass fileEntries directly rather than getEntries(): the resolved context
        // is computed from the leaf-to-root walk over byId (which already excludes
        // the header), so the entries argument is only a fallback for an undefined
        // leaf - never hit here since leafId is always set or null. Avoids an O(n)
        // array copy on every call (attach, get_session_context, agent init, ...).
        let entries: Vec<SessionEntry> = self.file_entries.clone();
        build_session_context(
            &entries,
            Some(self.leaf_id.as_deref()),
            Some(&self.indexed_by_id()),
            target_model,
        )
    }

    pub fn build_session_context_with_entry_ids(
        &self,
        target_model: Option<&pi_ai::types::Model>,
    ) -> SessionContextWithEntryIds {
        let entries: Vec<SessionEntry> = self.file_entries.clone();
        build_session_context_with_entry_ids(
            &entries,
            Some(self.leaf_id.as_deref()),
            Some(&self.indexed_by_id()),
            target_model,
        )
    }

    pub fn build_session_history(
        &self,
        tip_entry_id: Option<&str>,
        target_model: Option<&pi_ai::types::Model>,
    ) -> Result<SessionHistorySnapshot, String> {
        if let Some(tip_entry_id) = tip_entry_id.or(self.leaf_id.as_deref()) {
            if !self.by_id.contains_key(tip_entry_id) {
                return Err(format!(
                    "Session history tip no longer exists: {tip_entry_id}"
                ));
            }
        }
        // TS default parameter: buildSessionHistory(tipEntryId = this.leafId)
        // (session-manager.ts:2427) — an omitted tip resolves to the current
        // leaf instead of yielding an empty snapshot. (An explicit JSON null in
        // the daemon body still maps to None here, which selects the leaf like
        // an omitted call.)
        let effective_tip = tip_entry_id.or(self.leaf_id.as_deref());
        let entries: Vec<SessionEntry> = self.file_entries.clone();
        let context = build_session_context_with_entry_ids(
            &entries,
            Some(effective_tip),
            Some(&self.indexed_by_id()),
            target_model,
        );
        let mut snapshot = order_session_context_for_transcript(&context);
        snapshot.tip_entry_id = effective_tip.map(str::to_string);
        Ok(snapshot)
    }

    /// The TypeScript passes `this.byId` (a `Map<string, SessionEntry>`).
    fn indexed_by_id(&self) -> BTreeMap<String, SessionEntry> {
        self.by_id
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect()
    }

    pub fn get_header(&self) -> Option<SessionEntry> {
        self.file_entries
            .iter()
            .find(|entry| is_session_header(entry))
            .cloned()
    }

    /// Count non-header entries without cloning transcript content.
    pub fn get_entry_count(&self) -> usize {
        self.file_entries.iter().filter(|entry| !is_session_header(entry)).count()
    }

    pub fn get_entries(&self) -> Vec<SessionEntry> {
        self.file_entries
            .iter()
            .filter(|entry| !is_session_header(entry))
            .cloned()
            .collect()
    }

    pub fn get_flat_tree(&self) -> Vec<SessionTreeFlatNode> {
        self.get_entries()
            .into_iter()
            .map(|entry| {
                let id = entry_id(&entry);
                SessionTreeFlatNode {
                    label: self.labels_by_id.get(&id).cloned(),
                    label_timestamp: self.label_timestamps_by_id.get(&id).cloned(),
                    entry,
                }
            })
            .collect()
    }

    pub fn get_tree(&self) -> Vec<SessionTreeNode> {
        let entries = self.get_flat_tree();
        let mut node_map: indexmap::IndexMap<String, SessionTreeNode> = indexmap::IndexMap::new();
        let mut roots: Vec<SessionTreeNode> = Vec::new();

        for flat_node in &entries {
            node_map.insert(
                entry_id(&flat_node.entry),
                SessionTreeNode {
                    node: flat_node.clone(),
                    children: Vec::new(),
                },
            );
        }

        for flat_node in &entries {
            let entry = &flat_node.entry;
            let id = entry_id(entry);
            if !node_map.contains_key(&id) {
                continue;
            }
            let parent_id = entry_parent_id(entry);
            let is_root = parent_id.is_none() || parent_id.as_deref() == Some(id.as_str());
            if is_root {
                if let Some(node) = node_map.get(&id) {
                    roots.push(node.clone());
                }
            } else {
                let parent_key = parent_id.unwrap_or_default();
                if node_map.contains_key(&parent_key) {
                    if let Some(node) = node_map.get(&id).cloned() {
                        if let Some(parent) = node_map.get_mut(&parent_key) {
                            parent.children.push(node);
                        }
                    }
                } else if let Some(node) = node_map.get(&id) {
                    roots.push(node.clone());
                }
            }
        }

        // Sort children by timestamp (oldest first, newest at bottom)
        // Use iterative approach to avoid stack overflow on deep trees
        let mut stack: Vec<SessionTreeNode> = roots.clone();
        let mut sorted: indexmap::IndexMap<String, Vec<SessionTreeNode>> =
            indexmap::IndexMap::new();
        while let Some(node) = stack.pop() {
            let mut children = node.children.clone();
            children.sort_by(|a, b| {
                iso_to_millis(&entry_timestamp(&a.node.entry))
                    .partial_cmp(&iso_to_millis(&entry_timestamp(&b.node.entry)))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for child in &children {
                stack.push(child.clone());
            }
            sorted.insert(entry_id(&node.node.entry), children);
        }
        fn apply(
            node: &mut SessionTreeNode,
            sorted: &indexmap::IndexMap<String, Vec<SessionTreeNode>>,
        ) {
            let id = entry_id(&node.node.entry);
            if let Some(children) = sorted.get(&id) {
                node.children = children.clone();
            }
            for child in node.children.iter_mut() {
                apply(child, sorted);
            }
        }
        for root in roots.iter_mut() {
            apply(root, &sorted);
        }

        roots
    }

    pub fn branch(&mut self, branch_from_id: &str) -> Result<(), String> {
        if !self.by_id.contains_key(branch_from_id) {
            return Err(format!("Entry {branch_from_id} not found"));
        }
        self.leaf_id = Some(branch_from_id.to_string());
        Ok(())
    }

    pub fn reset_leaf(&mut self) {
        self.leaf_id = None;
    }

    pub fn branch_with_summary(
        &mut self,
        branch_from_id: Option<&str>,
        summary: &str,
        details: Option<Value>,
        from_hook: Option<bool>,
        usage: Option<&Usage>,
    ) -> Result<String, String> {
        if let Some(branch_from_id) = branch_from_id {
            if !self.by_id.contains_key(branch_from_id) {
                return Err(format!("Entry {branch_from_id} not found"));
            }
        }
        self.leaf_id = branch_from_id.map(str::to_string);
        let mut entry: SessionEntry = serde_json::json!({
            "type": "branch_summary",
            "id": self.next_entry_id(),
            "parentId": match branch_from_id {
                Some(id) => Value::String(id.to_string()),
                None => Value::Null,
            },
            "timestamp": iso_now(),
            "fromId": branch_from_id.unwrap_or("root"),
            "summary": summary,
        })
        .as_object()
        .cloned()
        .unwrap_or_default();
        if let Some(details) = details {
            entry.insert("details".to_string(), details);
        }
        if let Some(from_hook) = from_hook {
            entry.insert("fromHook".to_string(), Value::Bool(from_hook));
        }
        if let Some(usage) = usage {
            entry.insert(
                "usage".to_string(),
                serde_json::to_value(usage).unwrap_or(Value::Null),
            );
        }
        let id = entry_id(&entry);
        self.append_entry(entry)?;
        Ok(id)
    }

    pub fn create_branched_session(&mut self, leaf_id: &str) -> Result<Option<String>, String> {
        let previous_session_file = self.session_file.clone();
        let path = self.get_branch(Some(leaf_id));
        if path.is_empty() {
            return Err(format!("Entry {leaf_id} not found"));
        }

        let path_without_labels: Vec<SessionEntry> = path
            .into_iter()
            .filter(|entry| entry_type(entry) != "label")
            .collect();

        let target = if self.persist {
            create_unique_session_file_target(&self.get_session_dir())?
        } else {
            (create_session_id(), String::new())
        };
        let new_session_id = target.0;
        let timestamp = iso_now();
        let new_session_file = if self.persist { Some(target.1) } else { None };

        let header = SessionHeader {
            type_: "session".to_string(),
            version: Some(CURRENT_SESSION_VERSION),
            id: new_session_id.clone(),
            timestamp,
            cwd: self.cwd.clone(),
            parent_session: if self.persist {
                previous_session_file.clone()
            } else {
                None
            },
            rlm_depth: Some(resolve_session_rlm_depth(
                self.get_header().as_ref().unwrap_or(&Map::new()),
                previous_session_file
                    .as_deref()
                    .or(new_session_file.as_deref())
                    .unwrap_or(""),
            )),
            git: if self.persist {
                capture_git_context(&self.cwd)
            } else {
                None
            },
        };

        let mut path_entry_ids: BTreeSet<String> =
            path_without_labels.iter().map(entry_id).collect();
        let mut labels_to_write: Vec<(String, String, String)> = Vec::new();
        for (target_id, label) in &self.labels_by_id {
            if path_entry_ids.contains(target_id) {
                labels_to_write.push((
                    target_id.clone(),
                    label.clone(),
                    self.label_timestamps_by_id
                        .get(target_id)
                        .cloned()
                        .unwrap_or_default(),
                ));
            }
        }

        if self.persist {
            let last_entry_id = path_without_labels.last().map(entry_id);
            let mut parent_id = last_entry_id;
            let mut label_entries: Vec<SessionEntry> = Vec::new();
            for (target_id, label, label_timestamp) in &labels_to_write {
                let mut known: BTreeSet<String> = path_entry_ids.clone();
                for entry in &label_entries {
                    known.insert(entry_id(entry));
                }
                let entry: SessionEntry = serde_json::json!({
                    "type": "label",
                    "id": generate_id(&|candidate| known.contains(candidate)),
                    "parentId": match &parent_id {
                        Some(id) => Value::String(id.clone()),
                        None => Value::Null,
                    },
                    "timestamp": label_timestamp,
                    "targetId": target_id,
                    "label": label,
                })
                .as_object()
                .cloned()
                .unwrap_or_default();
                path_entry_ids.insert(entry_id(&entry));
                parent_id = Some(entry_id(&entry));
                label_entries.push(entry);
            }

            let mut entries: Vec<FileEntry> = vec![serde_json::to_value(&header)
                .ok()
                .and_then(|value| value.as_object().cloned())
                .unwrap_or_default()];
            entries.extend(path_without_labels);
            entries.extend(label_entries);
            self.file_entries = entries;
            self.session_id = new_session_id;
            self.session_file = new_session_file.clone();
            self.build_index();

            // Only write the file now if it contains an assistant message.
            // Otherwise defer to _persist(), which creates the file on the
            // first assistant response, matching the newSession() contract.
            let has_assistant = self
                .file_entries
                .iter()
                .any(|entry| entry_type(entry) == "message" && message_role(entry) == "assistant");
            if has_assistant {
                self.rewrite_file()?;
                self.flushed = true;
            } else {
                self.flushed = false;
            }

            return Ok(new_session_file);
        }

        let mut label_entries: Vec<SessionEntry> = Vec::new();
        let mut parent_id = path_without_labels.last().map(entry_id);
        for (target_id, label, label_timestamp) in &labels_to_write {
            let mut known: BTreeSet<String> = path_entry_ids.clone();
            for entry in &label_entries {
                known.insert(entry_id(entry));
            }
            let entry: SessionEntry = serde_json::json!({
                "type": "label",
                "id": generate_id(&|candidate| known.contains(candidate)),
                "parentId": match &parent_id {
                    Some(id) => Value::String(id.clone()),
                    None => Value::Null,
                },
                "timestamp": label_timestamp,
                "targetId": target_id,
                "label": label,
            })
            .as_object()
            .cloned()
            .unwrap_or_default();
            parent_id = Some(entry_id(&entry));
            label_entries.push(entry);
        }
        let mut entries: Vec<FileEntry> = vec![serde_json::to_value(&header)
            .ok()
            .and_then(|value| value.as_object().cloned())
            .unwrap_or_default()];
        entries.extend(path_without_labels);
        entries.extend(label_entries);
        self.file_entries = entries;
        self.session_id = new_session_id;
        self.build_index();
        Ok(None)
    }
}

// ---------------------------------------------------------------------------
// Static constructors
// ---------------------------------------------------------------------------

impl SessionManager {
    pub fn create(cwd: &str, session_dir: Option<&str>) -> Result<Self, String> {
        let dir = session_dir
            .map(str::to_string)
            .unwrap_or_else(|| get_default_session_dir(cwd, None));
        SessionManager::new_internal(cwd.to_string(), dir, None, true, None, None)
    }

    pub fn open(
        path: &str,
        session_dir: Option<&str>,
        cwd_override: Option<&str>,
    ) -> Result<Self, String> {
        // Only the header's cwd is needed to construct the manager; the constructor
        // (setSessionFile) performs the full parse. Read just the first line here
        // instead of parsing the entire file a second time — that double parse is a
        // needless O(n) cost on open and is noticeable for long sessions.
        let mut cwd = cwd_override.map(str::to_string);
        if cwd.is_none() {
            let mut header = read_session_header(path);
            // readSessionHeader only inspects the first physical line. If that isn't a
            // valid session header (e.g. a leading blank/whitespace or malformed line),
            // fall back to the full loader, which trims and skips such lines exactly
            // like setSessionFile does — so this.cwd stays consistent with the header
            // the session is actually loaded with. This slow path is rare.
            let header_is_valid = header
                .as_ref()
                .map(|header| {
                    header.get("type").and_then(Value::as_str) == Some("session")
                        && header.get("id").and_then(Value::as_str).is_some()
                })
                .unwrap_or(false);
            if !header_is_valid {
                header = load_entries_from_file(path)
                    .into_iter()
                    .find(|entry| is_session_header(entry));
            }
            cwd = header
                .as_ref()
                .and_then(|header| header.get("cwd"))
                .and_then(Value::as_str)
                .map(str::to_string);
        }
        let dir = session_dir.map(str::to_string).unwrap_or_else(|| {
            Path::new(path)
                .parent()
                .map(|parent| parent.to_string_lossy().to_string())
                .filter(|parent| !parent.is_empty())
                .unwrap_or_else(|| resolve_path(".."))
        });
        let fallback_cwd = std::env::current_dir()
            .map(|cwd| cwd.to_string_lossy().to_string())
            .unwrap_or_default();
        SessionManager::new_internal(
            cwd.unwrap_or(fallback_cwd),
            dir,
            Some(path.to_string()),
            true,
            None,
            None,
        )
    }

    pub async fn open_async(
        path: &str,
        session_dir: Option<&str>,
        cwd_override: Option<&str>,
    ) -> Result<Self, String> {
        if !Path::new(path).exists() {
            return SessionManager::open(path, session_dir, cwd_override);
        }
        repair_jsonl_damage(path)?;
        let loaded = load_entries_from_file_async_observed(path, None).await;
        if loaded.entries.is_empty() {
            return SessionManager::open(path, session_dir, cwd_override);
        }
        let cwd = cwd_override.map(str::to_string).or_else(|| {
            loaded
                .entries
                .first()
                .and_then(|header| header.get("cwd"))
                .and_then(Value::as_str)
                .map(str::to_string)
        });
        let dir = session_dir.map(str::to_string).unwrap_or_else(|| {
            Path::new(path)
                .parent()
                .map(|parent| parent.to_string_lossy().to_string())
                .filter(|parent| !parent.is_empty())
                .unwrap_or_else(|| resolve_path(".."))
        });
        let fallback_cwd = std::env::current_dir()
            .map(|cwd| cwd.to_string_lossy().to_string())
            .unwrap_or_default();
        SessionManager::new_internal(
            cwd.unwrap_or(fallback_cwd),
            dir,
            Some(path.to_string()),
            true,
            Some(loaded.entries),
            loaded.observation,
        )
    }

    pub fn continue_recent(cwd: &str, session_dir: Option<&str>) -> Result<Self, String> {
        let dir = session_dir
            .map(str::to_string)
            .unwrap_or_else(|| get_default_session_dir(cwd, None));
        match find_most_recent_session_for_cwd(&dir, cwd) {
            Some(most_recent) => SessionManager::new_internal(
                cwd.to_string(),
                dir,
                Some(most_recent),
                true,
                None,
                None,
            ),
            None => SessionManager::new_internal(cwd.to_string(), dir, None, true, None, None),
        }
    }

    pub fn in_memory(cwd: Option<&str>, session_dir: Option<&str>) -> Result<Self, String> {
        let cwd = cwd.map(str::to_string).unwrap_or_else(|| {
            std::env::current_dir()
                .map(|cwd| cwd.to_string_lossy().to_string())
                .unwrap_or_default()
        });
        SessionManager::new_internal(
            cwd,
            session_dir.unwrap_or("").to_string(),
            None,
            false,
            None,
            None,
        )
    }

    pub fn fork_from(
        source_path: &str,
        target_cwd: &str,
        session_dir: Option<&str>,
    ) -> Result<Self, String> {
        let mut source_entries = load_entries_from_file(source_path);
        if source_entries.is_empty() {
            return Err(format!(
                "Cannot fork: source session file is empty or invalid: {source_path}"
            ));
        }

        let source_header = match source_entries
            .iter()
            .find(|entry| is_session_header(entry))
            .cloned()
        {
            Some(source_header) => source_header,
            None => {
                return Err(format!(
                    "Cannot fork: source session has no header: {source_path}"
                ))
            }
        };
        migrate_to_current_version(&mut source_entries);

        let dir = session_dir
            .map(str::to_string)
            .unwrap_or_else(|| get_default_session_dir(target_cwd, None));
        if !Path::new(&dir).exists() {
            let _ = std::fs::create_dir_all(&dir);
        }

        let target = create_unique_session_file_target(&dir)?;
        let new_session_id = target.0;
        let timestamp = iso_now();
        let new_session_file = target.1;

        let new_header = SessionHeader {
            type_: "session".to_string(),
            version: Some(CURRENT_SESSION_VERSION),
            id: new_session_id,
            timestamp,
            cwd: target_cwd.to_string(),
            parent_session: Some(source_path.to_string()),
            rlm_depth: Some(resolve_session_rlm_depth(&source_header, source_path)),
            git: capture_git_context(target_cwd),
        };
        {
            use std::io::Write;
            let mut file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&new_session_file)
                .map_err(|error| error.to_string())?;
            let _ = writeln!(
                file,
                "{}",
                serde_json::to_string(&new_header).unwrap_or_default()
            );
        }

        // Drop the source's git_state entries (re-linking children): they describe the source repo,
        // so the fork would otherwise report the source's git instead of its own target context.
        let mut dropped_parent: BTreeMap<String, Option<String>> = BTreeMap::new();
        for entry in &source_entries {
            if entry_type(entry) == "git_state" {
                dropped_parent.insert(entry_id(entry), entry_parent_id(entry));
            }
        }
        let live_parent = |parent_id: Option<String>| -> Option<String> {
            let mut pid = parent_id;
            let mut visited = HashSet::new();
            while let Some(current) = pid.clone() {
                if !visited.insert(current.clone()) { return None; }
                if !dropped_parent.contains_key(&current) {
                    return Some(current);
                }
                pid = dropped_parent.get(&current).cloned().flatten();
            }
            None
        };
        {
            use std::io::Write;
            let mut file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&new_session_file)
                .map_err(|error| error.to_string())?;
            for entry in &source_entries {
                if is_session_header(entry) || entry_type(entry) == "git_state" {
                    continue;
                }
                let parent_id = live_parent(entry_parent_id(entry));
                let out = if parent_id == entry_parent_id(entry) {
                    entry.clone()
                } else {
                    let mut relinked = entry.clone();
                    relinked.insert(
                        "parentId".to_string(),
                        match parent_id {
                            Some(parent) => Value::String(parent),
                            None => Value::Null,
                        },
                    );
                    relinked
                };
                let _ = writeln!(file, "{}", serialize_session_file_entry(&out));
            }
        }

        SessionManager::new_internal(
            target_cwd.to_string(),
            dir,
            Some(new_session_file),
            true,
            None,
            None,
        )
    }

    pub async fn list(
        cwd: &str,
        session_dir: Option<&str>,
        callbacks: Option<SessionListCallbacks>,
    ) -> Vec<SessionInfo> {
        let dir = session_dir
            .map(str::to_string)
            .unwrap_or_else(|| get_default_session_dir(cwd, None));
        let matches_cwd = |session: &SessionInfo| session_info_matches_cwd(session, cwd);
        // TS routes the cwd-scoped SUBSET through the item events (onSession) and
        // forwards progress unfiltered (session-manager.ts:2740-2751).
        let scoped_callbacks = callbacks.map(|callbacks| {
            let cwd = cwd.to_string();
            SessionListCallbacks {
                on_progress: callbacks.on_progress,
                on_session: callbacks.on_session.map(|on_session| {
                    let cwd = cwd.clone();
                    let wrapped: Box<SessionListItem> = Box::new(move |session: &SessionInfo| {
                        if session_info_matches_cwd(session, &cwd) {
                            on_session(session);
                        }
                    });
                    wrapped
                }),
            }
        });
        let mut sessions = list_sessions_from_dir(&dir, scoped_callbacks.as_ref(), 0, None)
            .await
            .into_iter()
            .filter(|session| matches_cwd(session))
            .collect::<Vec<_>>();
        sessions.sort_by(|a, b| {
            b.modified
                .partial_cmp(&a.modified)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sessions
    }

    pub async fn list_all(
        callbacks: Option<&SessionListCallbacks>,
        session_dir: Option<&str>,
    ) -> Vec<SessionInfo> {
        let sessions_dir = session_dir
            .map(str::to_string)
            .unwrap_or_else(|| get_sessions_dir(&get_default_agent_dir()));
        let mut sessions = list_sessions_from_dir(&sessions_dir, callbacks, 0, None).await;
        sessions.sort_by(|a, b| {
            b.modified
                .partial_cmp(&a.modified)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sessions
    }
}

#[cfg(test)]
#[path = "session_manager_safety_tests.rs"]
mod safety_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn borrowed_branch_visitor_preserves_order_tip_and_entry_identity() {
        let mut manager = SessionManager::in_memory(Some("/work"), Some("")).unwrap();
        let first = manager.append_message(user_message(&"large".repeat(100_000), 1)).unwrap();
        let abandoned = manager.append_custom_entry("other-branch", None).unwrap();
        manager.branch(&first).unwrap();
        let last = manager.append_custom_entry("current-branch", None).unwrap();
        let mut visited = Vec::new();
        manager.visit_branch(None, |entry| {
            let id = entry_id(entry);
            assert!(std::ptr::eq(entry, manager.by_id.get(&id).unwrap()), "borrow indexed entry, not a cloned body");
            visited.push(id);
        });
        assert_eq!(visited, [first.clone(), last]);
        visited.clear();
        manager.visit_branch(Some(&abandoned), |entry| visited.push(entry_id(entry)));
        assert_eq!(visited, [first, abandoned]);
        manager.visit_branch(Some("missing"), |_| panic!("invalid tip must not fall back to current branch"));
    }

    fn temp_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("pi-session-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn user_message(text: &str, timestamp: i64) -> AgentMessage {
        AgentMessage::Message(Message::User(pi_ai::types::UserMessage::new(
            pi_ai::types::UserContent::Text(text.to_string()),
            timestamp,
        )))
    }

    fn assistant_message(model: &str, timestamp: i64) -> AgentMessage {
        let mut message = pi_ai::types::AssistantMessage {
            role: "assistant".to_string(),
            content: vec![pi_ai::types::ContentBlock::Text(
                pi_ai::types::TextContent::new("hi"),
            )],
            api: "openai-completions".to_string(),
            provider: "openai".to_string(),
            model: model.to_string(),
            response_model: None,
            response_id: None,
            diagnostics: None,
            usage: Usage::default(),
            stop_reason: "stop".to_string(),
            stop_reason_raw: None,
            error_message: None,
            timestamp,
        };
        message.usage.input = 10.0;
        message.usage.output = 5.0;
        message.usage.total_tokens = 15.0;
        message.usage.cost.total = 0.25;
        AgentMessage::Message(Message::Assistant(message))
    }

    #[test]
    fn in_memory_sessions_have_no_file_and_start_empty() {
        let manager = SessionManager::in_memory(Some("/work"), Some("")).unwrap();
        assert!(!manager.is_persisted());
        assert!(manager.get_session_file().is_none());
        assert_eq!(manager.get_cwd(), "/work");
        assert!(manager.get_entries().is_empty());
        assert!(manager.get_leaf_id().is_none());
    }

    #[test]
    fn header_round_trips_through_json() {
        let dir = temp_dir();
        let mut manager = SessionManager::create("/work", Some(&dir.to_string_lossy())).unwrap();
        manager.new_session(None).unwrap();
        let header = manager.get_header().unwrap();
        assert_eq!(header.get("type").and_then(Value::as_str), Some("session"));
        assert_eq!(
            header.get("version").and_then(Value::as_i64),
            Some(CURRENT_SESSION_VERSION)
        );
        assert_eq!(header.get("cwd").and_then(Value::as_str), Some("/work"));
        assert!(header.get("id").and_then(Value::as_str).is_some());
        assert!(header.get("timestamp").and_then(Value::as_str).is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn explicit_rlm_depth_comes_from_the_env_when_absent() {
        std::env::remove_var("RLM_DEPTH");
        assert_eq!(root_rlm_depth_from_env().unwrap(), 0);
        std::env::set_var("RLM_DEPTH", "3");
        assert_eq!(root_rlm_depth_from_env().unwrap(), 3);
        std::env::set_var("RLM_DEPTH", "-1");
        assert!(root_rlm_depth_from_env().is_err());
        std::env::set_var("RLM_DEPTH", "abc");
        assert!(root_rlm_depth_from_env().is_err());
        std::env::remove_var("RLM_DEPTH");
    }

    #[test]
    fn child_depth_derives_from_the_parent_header() {
        let mut parent = Map::new();
        parent.insert("rlmDepth".to_string(), Value::Number(2.into()));
        assert_eq!(derive_child_rlm_depth(Some(&parent)), Some(3));
        assert_eq!(derive_child_rlm_depth(None), None);
        let mut invalid = Map::new();
        invalid.insert("rlmDepth".to_string(), Value::Number((-1).into()));
        assert_eq!(derive_child_rlm_depth(Some(&invalid)), None);
    }

    #[test]
    fn legacy_depth_falls_back_to_sub_session_segments() {
        assert_eq!(legacy_child_depth_from_path("/a/sub-0123abcd/b.jsonl"), 1);
        assert_eq!(
            legacy_child_depth_from_path("/a/sub-0123abcd/sub-4567abcd/b.jsonl"),
            2
        );
        assert_eq!(legacy_child_depth_from_path("/a/not-sub/b.jsonl"), 0);
        assert_eq!(legacy_child_depth_from_path("/a/sub-ZZZZ/b.jsonl"), 0);
    }

    #[test]
    fn appends_chain_parent_ids_and_persist() {
        let dir = temp_dir();
        let mut manager = SessionManager::create("/work", Some(&dir.to_string_lossy())).unwrap();
        let file = manager.new_session(None).unwrap().unwrap();
        let first = manager.append_message(user_message("hello", 1)).unwrap();
        let second = manager
            .append_message(assistant_message("gpt-5", 2))
            .unwrap();
        assert_eq!(manager.get_leaf_id().as_deref(), Some(second.as_str()));
        assert_eq!(
            manager
                .get_entry(&second)
                .unwrap()
                .get("parentId")
                .and_then(Value::as_str),
            Some(first.as_str())
        );

        let contents = std::fs::read_to_string(&file).unwrap();
        let lines: Vec<&str> = contents.trim_end().split('\n').collect();
        assert_eq!(lines.len(), 3);
        let header: Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(header["type"], "session");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn build_session_context_walks_the_leaf_to_root() {
        let mut manager = SessionManager::in_memory(Some("/work"), Some("")).unwrap();
        manager.append_message(user_message("first", 1)).unwrap();
        manager.append_thinking_level_change("high").unwrap();
        manager.append_model_change("openai", "gpt-5").unwrap();
        manager
            .append_message(assistant_message("gpt-5", 2))
            .unwrap();

        let context = manager.build_session_context(None);
        assert_eq!(context.messages.len(), 2);
        assert_eq!(context.thinking_level, "high");
        assert_eq!(
            context.model,
            Some(SessionContextModel {
                provider: "openai".to_string(),
                model_id: "gpt-5".to_string()
            })
        );
        let with_ids = manager.build_session_context_with_entry_ids(None);
        assert_eq!(with_ids.entry_ids.len(), with_ids.messages.len());
    }

    #[test]
    fn compaction_keeps_retained_messages_after_the_summary() {
        let mut manager = SessionManager::in_memory(Some("/work"), Some("")).unwrap();
        let first = manager.append_message(user_message("old", 1)).unwrap();
        let kept = manager.append_message(user_message("kept", 2)).unwrap();
        manager
            .append_message(assistant_message("gpt-5", 3))
            .unwrap();
        manager
            .append_compaction("summary text", &kept, 100.0, None, None, None, None, None)
            .unwrap();
        let tail = manager.append_message(user_message("new", 4)).unwrap();

        let context = manager.build_session_context(None);
        let roles: Vec<&str> = context
            .messages
            .iter()
            .map(|message| message.role())
            .collect();
        assert_eq!(
            roles,
            vec!["compactionSummary", "user", "assistant", "user"]
        );
        assert_ne!(first, kept);
        assert_eq!(manager.get_leaf_id().as_deref(), Some(tail.as_str()));

        // The omitted tip resolves to the current leaf (TS default parameter).
        let snapshot = manager.build_session_history(None, None).unwrap();
        assert_eq!(snapshot.tip_entry_id, manager.get_leaf_id());
    }

    #[test]
    fn history_snapshot_places_the_summary_at_the_retained_boundary() {
        let context = SessionContextWithEntryIds {
            messages: vec![
                AgentMessage::Custom(CustomAgentMessage::CompactionSummary {
                    summary: "s".to_string(),
                    provider_context: None,
                    tokens_before: 1.0,
                    retained_message_count: Some(1.0),
                    custom_instructions: None,
                    harness_digest: None,
                    timestamp: 10,
                }),
                user_message("a", 1),
                user_message("b", 2),
            ],
            entry_ids: vec!["c".to_string(), "a".to_string(), "b".to_string()],
            thinking_level: "off".to_string(),
            service_tier: None,
            model: None,
        };
        let snapshot = order_session_context_for_transcript(&context);
        assert_eq!(snapshot.entry_ids, vec!["a", "c", "b"]);
        assert_eq!(snapshot.messages.len(), 3);
    }

    #[test]
    fn labels_track_targets_and_replicate_into_a_branch() {
        let mut manager = SessionManager::in_memory(Some("/work"), Some("")).unwrap();
        let first = manager.append_message(user_message("one", 1)).unwrap();
        manager
            .append_label_change(&first, Some("important"))
            .unwrap();
        assert_eq!(manager.get_label(&first).as_deref(), Some("important"));
        manager.append_label_change(&first, None).unwrap();
        assert!(manager.get_label(&first).is_none());
        assert!(manager.append_label_change("missing", Some("x")).is_err());
    }

    #[test]
    fn branch_and_tree_shapes_follow_parent_ids() {
        let mut manager = SessionManager::in_memory(Some("/work"), Some("")).unwrap();
        let first = manager.append_message(user_message("one", 1)).unwrap();
        let second = manager.append_message(user_message("two", 2)).unwrap();
        assert_eq!(manager.get_children(&first).len(), 1);
        manager.branch(&first).unwrap();
        let third = manager.append_message(user_message("three", 3)).unwrap();
        assert_eq!(manager.get_branch(None).len(), 2);
        manager.branch(&second).unwrap();
        assert_eq!(manager.get_branch(None).len(), 2);
        assert!(manager.branch("missing").is_err());
        manager.reset_leaf();
        assert!(manager.get_leaf_id().is_none());
        let tree = manager.get_tree();
        assert!(!tree.is_empty());
        assert_ne!(third, second);
    }

    #[test]
    fn has_user_content_skips_the_creation_prefix() {
        let mut manager = SessionManager::in_memory(Some("/work"), Some("")).unwrap();
        assert!(!manager.has_user_content());
        manager.append_model_change("openai", "gpt-5").unwrap();
        manager.append_thinking_level_change("off").unwrap();
        manager
            .append_service_tier_change(&Some(Some("default".to_string())))
            .unwrap();
        assert!(!manager.has_user_content());
        manager.append_message(user_message("real", 1)).unwrap();
        assert!(manager.has_user_content());
    }

    #[test]
    fn session_name_state_and_agent_status_read_the_latest_entry() {
        let mut manager = SessionManager::in_memory(Some("/work"), Some("")).unwrap();
        manager.append_session_info("  named  ").unwrap();
        assert_eq!(manager.get_session_name().as_deref(), Some("named"));
        manager
            .append_session_state(&SessionState {
                status: SessionStateStatus::Archived,
            })
            .unwrap();
        assert_eq!(
            manager.get_session_state(),
            Some(SessionState {
                status: SessionStateStatus::Archived
            })
        );
        manager
            .append_agent_status(&AgentStatus {
                summary: "ok".to_string(),
                task_state: Some(AgentTaskState::Completed),
                based_on_message_count: 3,
            })
            .unwrap();
        let status = manager.get_latest_agent_status().unwrap();
        assert_eq!(status.summary, "ok");
        assert_eq!(status.task_state, Some(AgentTaskState::Completed));
    }

    #[test]
    fn child_usage_attribution_updates_the_target_message() {
        let mut manager = SessionManager::in_memory(Some("/work"), Some("")).unwrap();
        let assistant_id = manager
            .append_message(assistant_message("gpt-5", 2))
            .unwrap();
        let child = Usage {
            input: 3.0,
            output: 1.0,
            total_tokens: 4.0,
            ..Usage::default()
        };
        let aggregate = Usage {
            input: 13.0,
            output: 6.0,
            total_tokens: 19.0,
            ..Usage::default()
        };
        manager
            .append_child_usage_attribution(&assistant_id, &child, &aggregate, Some("spawn_task"))
            .unwrap();
        let entry = manager.get_entry(&assistant_id).unwrap();
        assert_eq!(entry["message"]["usage"]["input"].as_f64(), Some(13.0));
        assert!(manager
            .append_child_usage_attribution("missing", &child, &aggregate, None)
            .is_err());
    }

    #[test]
    fn inline_tool_text_references_shrink_and_rehydrate() {
        let long = "x".repeat(500);
        let mut entry = Map::new();
        entry.insert("type".to_string(), Value::String("message".to_string()));
        entry.insert("id".to_string(), Value::String("a".to_string()));
        entry.insert(
            "message".to_string(),
            serde_json::json!({
                "role": "toolResult",
                "toolCallId": "t",
                "toolName": "bash",
                "content": [{"type": "text", "text": long}],
                "details": {"stdout": long},
                "isError": false,
                "timestamp": 1,
            }),
        );
        let serialized = serialize_session_file_entry(&entry);
        assert!(serialized.contains(INLINE_TOOL_TEXT_ENCODING_KEY));
        let parsed: Map<String, Value> = serde_json::from_str(&serialized).unwrap();
        let rehydrated = rehydrate_session_file_entry(parsed);
        assert!(rehydrated.get(INLINE_TOOL_TEXT_ENCODING_KEY).is_none());
        assert_eq!(
            rehydrated["message"]["details"]["stdout"].as_str(),
            Some(long.as_str())
        );
    }

    #[test]
    fn rehydration_reports_an_unreconstructable_reference() {
        let mut entry = Map::new();
        entry.insert("type".to_string(), Value::String("message".to_string()));
        entry.insert("id".to_string(), Value::String("a".to_string()));
        entry.insert(
            INLINE_TOOL_TEXT_ENCODING_KEY.to_string(),
            serde_json::json!({
                "version": 1,
                "kind": INLINE_TOOL_TEXT_ENCODING_KIND,
                "fields": ["stdout"],
            }),
        );
        entry.insert(
            "message".to_string(),
            serde_json::json!({
                "role": "toolResult",
                "content": "short",
                "details": {"stdout": {"$primeToolText": {"version": 1, "source": "content", "contentIndex": 0, "start": 0, "length": 99, "sha256": "0".repeat(64)}}},
                "isError": false,
            }),
        );
        let rehydrated = rehydrate_session_file_entry(entry);
        assert_eq!(rehydrated["message"]["isError"], Value::Bool(true));
        assert!(rehydrated["message"]["details"]["stdout"]
            .as_str()
            .unwrap()
            .contains("session recovery error"));
    }

    #[test]
    fn migration_upgrades_v1_and_v2_sessions() {
        let mut entries: Vec<FileEntry> = vec![
            serde_json::json!({"type": "session", "id": "s", "timestamp": "t", "cwd": "/w"})
                .as_object()
                .cloned()
                .unwrap(),
            serde_json::json!({"type": "message", "message": {"role": "hookMessage"}})
                .as_object()
                .cloned()
                .unwrap(),
            serde_json::json!({"type": "compaction", "summary": "s", "firstKeptEntryIndex": 1})
                .as_object()
                .cloned()
                .unwrap(),
        ];
        assert!(migrate_to_current_version(&mut entries));
        assert_eq!(entries[0]["version"].as_i64(), Some(3));
        assert_eq!(entries[1]["message"]["role"], "custom");
        assert!(entries[2].get("firstKeptEntryIndex").is_none());
        assert_eq!(
            entries[2]["firstKeptEntryId"].as_str(),
            Some(entries[1]["id"].as_str().unwrap())
        );
        assert!(!migrate_to_current_version(&mut entries));
    }

    #[test]
    fn parse_session_entries_skips_malformed_lines_and_applies_attributions() {
        let usage = Usage {
            input: 1.0,
            ..Usage::default()
        };
        let content = format!(
            "{}\n{}\nnot json\n",
            serde_json::json!({"type": "session", "id": "s", "version": 3, "timestamp": "t", "cwd": "/w"}),
            serde_json::json!({"type": "message", "id": "a", "message": {"role": "assistant", "usage": {"input": 0.0, "output": 0.0, "cacheRead": 0.0, "cacheWrite": 0.0, "totalTokens": 0.0, "cost": {"input": 0.0, "output": 0.0, "cacheRead": 0.0, "cacheWrite": 0.0, "total": 0.0}}}}),
        );
        let entries = parse_session_entries(&content);
        assert_eq!(entries.len(), 2);
        let _ = usage;
    }

    #[test]
    fn load_entries_requires_a_valid_header() {
        let dir = temp_dir();
        let good = dir.join("good.jsonl");
        std::fs::write(
            &good,
            format!(
                "{}\n{}\n",
                serde_json::json!({"type": "session", "id": "s", "version": 3, "timestamp": "2026-01-01T00:00:00.000Z", "cwd": "/w"}),
                serde_json::json!({"type": "message", "id": "a", "parentId": null, "timestamp": "2026-01-01T00:00:01.000Z", "message": {"role": "user", "content": "hi", "timestamp": 1}})
            ),
        )
        .unwrap();
        let entries = load_entries_from_file(&good.to_string_lossy());
        assert_eq!(entries.len(), 2);

        let bad = dir.join("bad.jsonl");
        std::fs::write(&bad, "{\"type\":\"message\"}\n").unwrap();
        assert!(load_entries_from_file(&bad.to_string_lossy()).is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn session_info_scan_reads_name_state_and_usage() {
        let dir = temp_dir();
        let file = dir.join("scan.jsonl");
        std::fs::write(
            &file,
            format!(
                "{}\n{}\n{}\n{}\n",
                serde_json::json!({"type": "session", "id": "s1", "version": 3, "timestamp": "2026-01-01T00:00:00.000Z", "cwd": "/w"}),
                serde_json::json!({"type": "session_info", "id": "n", "name": "scan", "timestamp": "2026-01-01T00:00:01.000Z"}),
                serde_json::json!({"type": "session_state", "id": "st", "state": {"status": "hidden"}, "timestamp": "2026-01-01T00:00:02.000Z"}),
                serde_json::json!({"type": "message", "id": "m", "timestamp": "2026-01-01T00:00:03.000Z", "message": {"role": "user", "content": "hello there", "timestamp": 1000}})
            ),
        )
        .unwrap();
        let info = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(read_session_info(&file.to_string_lossy()))
            .unwrap();
        assert_eq!(info.id, "s1");
        assert_eq!(info.name.as_deref(), Some("scan"));
        assert_eq!(
            info.state,
            Some(SessionState {
                status: SessionStateStatus::Archived
            })
        );
        assert_eq!(info.message_count, 1);
        assert_eq!(info.first_message, "hello there");
        assert_eq!(info.rlm_depth, 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn most_recent_session_prefers_newer_mtime() {
        let dir = temp_dir();
        let older = dir.join("older.jsonl");
        let newer = dir.join("newer.jsonl");
        for file in [&older, &newer] {
            std::fs::write(
                file,
                format!(
                    "{}\n",
                    serde_json::json!({"type": "session", "id": "s", "version": 3, "timestamp": "2026-01-01T00:00:00.000Z", "cwd": "/w"})
                ),
            )
            .unwrap();
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
        std::fs::write(
            &newer,
            format!(
                "{}\n{}\n",
                serde_json::json!({"type": "session", "id": "s", "version": 3, "timestamp": "2026-01-01T00:00:00.000Z", "cwd": "/w"}),
                serde_json::json!({"type": "message", "id": "m", "timestamp": "2026-01-01T00:00:01.000Z", "message": {"role": "user", "content": "x", "timestamp": 1}})
            ),
        )
        .unwrap();
        let most_recent = find_most_recent_session(&dir.to_string_lossy()).unwrap();
        assert_eq!(Path::new(&most_recent).file_name().unwrap(), "newer.jsonl");
        let for_cwd = find_most_recent_session_for_cwd(&dir.to_string_lossy(), "/w").unwrap();
        assert_eq!(Path::new(&for_cwd).file_name().unwrap(), "newer.jsonl");
        assert!(find_most_recent_session_for_cwd(&dir.to_string_lossy(), "/other").is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn streamed_catalog_scan_preserves_torn_tail_and_append_metadata() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("streamed.jsonl");
        let path = file.to_string_lossy().into_owned();
        let header = serde_json::json!({"type":"session","id":"streamed","version":3,"timestamp":"2026-01-01T00:00:00Z","cwd":"/workspace"});
        let user = serde_json::json!({"type":"message","id":"user-1","message":{"role":"user","content":"catalog title","timestamp":1}});
        let tool = serde_json::json!({"type":"message","id":"tool-1","message":{"role":"toolResult","content":[{"type":"text","text":"x".repeat(2 * 1024 * 1024)}],"timestamp":2}});
        let tail = serde_json::json!({"type":"session_info","id":"name-1","name":"Renamed after output"}).to_string();
        std::fs::write(&file, format!("{header}\r\n{user}\r\n{tool}\n{tail}")).unwrap();
        let initial = read_session_info(&path).await.unwrap();
        assert_eq!(initial.message_count, 2);
        assert_eq!(initial.first_message, "catalog title");
        assert_eq!(initial.name.as_deref(), Some("Renamed after output"));
        let warm = read_session_info(&path).await.unwrap();
        assert_eq!(warm, initial);

        let next = serde_json::json!({"type":"message","id":"user-2","message":{"role":"user","content":"later searchable text","timestamp":3}});
        let mut writer = std::fs::OpenOptions::new().append(true).open(&file).unwrap();
        writeln!(writer, "\n{next}").unwrap();
        drop(writer);
        let appended = read_session_info(&path).await.unwrap();
        assert_eq!(appended.message_count, 3);
        assert_eq!(appended.name, initial.name);
        assert!(appended.all_messages_text.contains("later searchable text"));
        drop_session_scan_state(&path);
    }

    #[test]
    fn streamed_catalog_scan_obeys_captured_size_during_append() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("bounded.jsonl");
        let path = file.to_string_lossy().into_owned();
        let first = format!("{}\n", serde_json::json!({"type":"session","id":"bounded","cwd":"/workspace"}));
        std::fs::write(&file, format!("{first}{}\n", serde_json::json!({"type":"session_info","name":"later"}))).unwrap();
        let mut state = SessionScanState {
            file_size:0, mtime:None, dev:0, ino:0, offset:0, tail:Vec::new(),
            acc:create_session_scan_accumulator(), info:None, accounted_usage_entries:0,
        };
        assert!(scan_session_lines(&path, &mut state, first.len() as u64).unwrap().is_none());
        assert_eq!(state.offset, first.len() as u64);
        assert!(state.acc.name.is_none());
        assert!(scan_session_lines(&path, &mut state, std::fs::metadata(&file).unwrap().len()).unwrap().is_none());
        assert_eq!(state.acc.name.as_deref(), Some("later"));
    }

    #[test]
    fn artifact_paths_follow_the_session_file() {
        assert_eq!(
            get_session_artifacts_root("/a/sessions"),
            Path::new("/a").join("session-artifacts").to_string_lossy()
        );
        assert_eq!(
            get_session_artifact_path("/a/sessions", "id1"),
            Path::new("/a/session-artifacts/id1").to_string_lossy()
        );
        assert_eq!(
            get_session_artifact_path_for_file("/a/sessions/id1.jsonl", None),
            Path::new("/a/session-artifacts/id1").to_string_lossy()
        );
        assert_eq!(
            get_session_artifact_path_for_file("/a/sessions/id1.jsonl", Some("override")),
            Path::new("/a/session-artifacts/override").to_string_lossy()
        );
    }

    #[test]
    fn fork_from_relinks_children_around_git_state() {
        let dir = temp_dir();
        let source = dir.join("source.jsonl");
        let git_state_id = "g1";
        std::fs::write(
            &source,
            format!(
                "{}\n{}\n{}\n{}\n",
                serde_json::json!({"type": "session", "id": "src", "version": 3, "timestamp": "2026-01-01T00:00:00.000Z", "cwd": "/src"}),
                serde_json::json!({"type": "message", "id": "m1", "parentId": null, "timestamp": "2026-01-01T00:00:01.000Z", "message": {"role": "user", "content": "one", "timestamp": 1}}),
                serde_json::json!({"type": "git_state", "id": git_state_id, "parentId": "m1", "timestamp": "2026-01-01T00:00:02.000Z", "git": {"commit": "abc"}}),
                serde_json::json!({"type": "message", "id": "m2", "parentId": git_state_id, "timestamp": "2026-01-01T00:00:03.000Z", "message": {"role": "user", "content": "two", "timestamp": 2}})
            ),
        )
        .unwrap();
        let target_dir = temp_dir();
        let manager = SessionManager::fork_from(
            &source.to_string_lossy(),
            "/target",
            Some(&target_dir.to_string_lossy()),
        )
        .unwrap();
        let entries = manager.get_entries();
        assert_eq!(entries.len(), 2);
        assert_eq!(
            entries[1].get("parentId").and_then(Value::as_str),
            Some("m1")
        );
        let header = manager.get_header().unwrap();
        assert_eq!(header.get("cwd").and_then(Value::as_str), Some("/target"));
        assert_eq!(
            header.get("parentSession").and_then(Value::as_str),
            Some(source.to_string_lossy().as_ref())
        );
        assert!(SessionManager::fork_from(
            &dir.join("missing.jsonl").to_string_lossy(),
            "/target",
            None
        )
        .is_err());
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&target_dir);
    }

    #[test]
    fn create_branched_session_writes_labels_and_returns_a_new_file() {
        let dir = temp_dir();
        let mut manager = SessionManager::create("/work", Some(&dir.to_string_lossy())).unwrap();
        manager.new_session(None).unwrap();
        let first = manager.append_message(user_message("one", 1)).unwrap();
        let second = manager
            .append_message(assistant_message("gpt-5", 2))
            .unwrap();
        manager.append_label_change(&first, Some("keep")).unwrap();
        let new_file = manager.create_branched_session(&second).unwrap().unwrap();
        assert!(Path::new(&new_file).exists());
        assert_eq!(manager.get_label(&first).as_deref(), Some("keep"));
        assert!(manager.create_branched_session("missing").is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn record_git_state_if_changed_is_a_noop_outside_a_repo() {
        let dir = temp_dir();
        let mut manager =
            SessionManager::create("/definitely/not/a/repo", Some(&dir.to_string_lossy())).unwrap();
        assert_eq!(manager.record_git_state_if_changed().unwrap(), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn list_reads_sessions_from_the_directory_and_filters_by_cwd() {
        let dir = temp_dir();
        for (name, cwd) in [("a", "/work"), ("b", "/other")] {
            std::fs::write(
                dir.join(format!("{name}.jsonl")),
                format!(
                    "{}\n",
                    serde_json::json!({"type": "session", "id": name, "version": 3, "timestamp": "2026-01-01T00:00:00.000Z", "cwd": cwd})
                ),
            )
            .unwrap();
        }
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let sessions = runtime.block_on(SessionManager::list(
            "/work",
            Some(&dir.to_string_lossy()),
            None,
        ));
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].id, "a");
        let all = runtime.block_on(SessionManager::list_all(None, Some(&dir.to_string_lossy())));
        assert_eq!(all.len(), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn provider_checkpoints_are_detected_and_legacy_ones_are_read() {
        assert!(!has_provider_checkpoint(&Value::Null));
        assert!(has_provider_checkpoint(&serde_json::json!({
            "providerCheckpoint": {"version": 1, "provider": "openai", "api": "openai-responses", "model": "gpt-5", "baseUrl": "https://x", "items": [{"type": "message"}], "estimatedTokens": 1}
        })));
        let legacy = serde_json::json!({
            "strategy": "openai-responses-compaction-v2",
            "provider": "openai",
            "api": "openai-responses",
            "model": "gpt-5",
            "baseUrl": "https://x",
            "compactedWindow": [{"type": "compaction", "encrypted_content": "abc"}]
        });
        assert!(has_provider_checkpoint(&legacy));
        let checkpoint = get_provider_checkpoint(&legacy).unwrap();
        assert_eq!(checkpoint.version, 1);
        assert_eq!(checkpoint.model, "gpt-5");
        assert!(get_provider_checkpoint(&serde_json::json!({"strategy": "other"})).is_none());
    }

    #[test]
    fn oversized_message_summaries_are_extracted_without_full_parsing() {
        let long = "y".repeat(64);
        let line = format!(
            "{{\"type\":\"message\",\"timestamp\":\"2026-01-01T00:00:00.000Z\",\"message\":{{\"role\":\"user\",\"content\":\"{long}\"}}}}"
        );
        let summary = extract_oversized_message_summary(&line);
        assert_eq!(summary.role.as_deref(), Some("user"));
        assert!(summary.timestamp.is_some());
        assert_eq!(summary.text_preview.as_deref(), Some(long.as_str()));
        assert!(looks_like_message_entry(&line));
        assert!(!looks_like_message_entry("{\"type\":\"label\"}"));
    }

    #[test]
    fn search_text_is_capped() {
        let mut current = String::new();
        current = append_capped_search_text(&current, "one");
        current = append_capped_search_text(&current, "two");
        assert_eq!(current, "one two");
        let long = "z".repeat(SESSION_LIST_SEARCH_TEXT_MAX_CHARS + 10);
        let capped = append_capped_search_text(&String::new(), &long);
        assert_eq!(capped.len(), SESSION_LIST_SEARCH_TEXT_MAX_CHARS);
    }

    #[test]
    fn json_helpers_match_the_typescript_shapes() {
        assert_eq!(json_stringify(&serde_json::json!({"a": 1})), "{\"a\":1}");
        assert_eq!(is_safe_integer(&serde_json::json!(1.5)), None);
        assert_eq!(is_safe_integer(&serde_json::json!(2)), Some(2));
        assert_eq!(is_nonnegative_safe_integer(&serde_json::json!(-1)), None);
        assert!(iso_now().ends_with('Z'));
        assert_eq!(civil_from_days(0), (1970, 1, 1));
    }

    #[test]
    fn usage_helpers_clamp_at_zero() {
        let mut total = empty_usage();
        let usage = Usage {
            input: 5.0,
            output: 2.0,
            total_tokens: 7.0,
            ..Usage::default()
        };
        add_assistant_usage(&mut total, &usage);
        assert_eq!(total.input, 5.0);
        subtract_assistant_usage(&mut total, &usage);
        assert_eq!(total.input, 0.0);
        subtract_assistant_usage(&mut total, &usage);
        assert_eq!(total.input, 0.0);
        assert!(session_usage_summary_from(&empty_usage()).is_none());
        assert_eq!(
            session_usage_summary_from(&usage).unwrap().input_tokens,
            5.0
        );
    }

    #[test]
    fn accessors_report_the_configured_cwd_and_an_absent_file() {
        let manager = SessionManager::in_memory(Some("/work"), Some("")).unwrap();
        assert_eq!(manager.get_cwd(), "/work");
        assert!(manager.get_session_file().is_none());
        assert!(manager.get_load_observation().is_none());
    }
}
