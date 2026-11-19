//! Offline replay + benchmark harness for the LOCAL auto-compaction path.
//!
//! Subcommands (hand-rolled CLI; no new dependencies):
//!   gen-fixtures --out <dir>
//!       Deterministic fixture transcripts (mixed conversations, several content
//!       classes, sizes from 10 KiB to ~40 MiB plus a near-threshold case).
//!       Each case dir contains `transcript.json` and `manifest.json`.
//!   replay --fixture <dir|case> --out <json> [--provider-delay-ms N]
//!          [--summary-failure MSG] [--expect-outcome successful-compaction|summary-failure]
//!          --expect-summary-calls N  Exact provider-summary attempt budget for this fixture/options.
//!          [--context-window N] [--keep-recent N] [--work DIR]
//!          [--stamp SHA256] [--git DESCRIBE] [--timeout-secs N]
//!       Runs the REAL auto-compaction path end to end offline: private roots,
//!       in-process fixture provider with canned deterministic summaries, real
//!       SessionManager JSONL, CompactionMetrics phase rows via the recorder
//!       injection seam, provider call log, retained-state digests. Every case
//!       must pass expected-outcome admission and fresh-OS-process durable reopen parity.
//!   verify-reopen --expectation <json> --out <json>
//!       Read-only fresh-process reopen used by replay; report is outside the case root.
//!   self-test-r1 --work DIR --out <json>
//!       Deterministic positive and negative admission/reopen controls.
//!   verify-local --fixture <dir|case> --out <json> --arm <reviewed-r1-baseline|r2-candidate>
//!          --adapter-diff <complete-diff> [--expectation FROZEN_R1_BASELINE]
//!       Non-timed full-output local-kernel capture or exact semantic comparison.
//!   self-test-r2 --fixture <case> --work DIR --out <json>
//!       Deterministic R2 rejection controls, including invalid numeric CLI values.
//!   bench-local --fixture <dir|case> --iters N --out <json> [--warmup N]
//!          [--stamp SHA256] [--git DESCRIBE] [--chunk-context-window N]
//!       Microbenchmarks of the pure local kernels through their real entry
//!       points (estimate_tokens, find_cut_point, prepare_compaction,
//!       convert_to_llm, serialize_conversation, chunk slicing,
//!       SessionManager::build_session_context, jev prepare_context).
//!
//! Everything here is measurement tooling: no production file depends on it and
//! it must keep working unchanged BEFORE and AFTER optimization work.

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use pi_agent_core::performance_metrics::{
    AgentLoopPerformanceMetrics, PerformanceMetricEvent, PerformanceMetricIdScope,
    PerformanceMetricMeasurement as Measurement, PerformanceMetricOperation as Operation,
    PerformanceMetricOutcome as Outcome, PerformanceMetricRecorder,
};
use pi_agent_core::types::{AgentMessage, CustomAgentMessage};
use pi_ai::api_registry::{register_api_provider_simple, ApiProviderSimple, SimpleStreamFunction};
use pi_ai::types::StreamFunction;
use pi_ai::types::{
    AssistantMessage, AssistantMessageEvent, ContentBlock, Context, ImageOrTextContent,
    InputModality, Message, Model, SimpleStreamOptions, TextContent, ThinkingContent, ToolCall,
    ToolResultMessage, Usage, UserContent, UserMessage,
};
use pi_ai::utils::event_stream::{
    create_assistant_message_event_stream, AssistantMessageEventStream,
};
use pi_coding_agent::core::agent_session::AgentSessionEvent;
use pi_coding_agent::core::agent_session_services::{
    create_agent_session_from_services, create_agent_session_services,
    AgentSessionCreationOptions, CreateAgentSessionFromServicesOptions,
    CreateAgentSessionServicesOptions,
};
use pi_coding_agent::core::auth_storage::{AuthStorage, AuthStorageData, AuthStorageOptions};
use pi_ai::models::get_model_input_limit;
use pi_coding_agent::core::compaction::compaction::{
    build_summarization_prompt, default_compaction_settings, estimate_context_tokens,
    estimate_tokens, find_cut_point, prepare_compaction, should_compact_for_model,
    CompactionSessionEntry, COMPACTION_PROMPT_LEGACY, SUMMARY_UPDATE_POLICY_OFF,
    MAX_COMPACTION_CONTEXT_TOKENS,
};
use pi_coding_agent::core::compaction::utils::{
    serialize_conversation, SUMMARIZATION_SYSTEM_PROMPT,
};
use pi_coding_agent::core::jev_compaction::prepare_context;
use pi_coding_agent::core::messages::convert_to_llm;
use pi_coding_agent::core::model_registry::ModelRegistry;
use pi_coding_agent::core::model_tool_output_policy::ModelToolOutputPolicyOptions;
use pi_coding_agent::core::resource_loader::DefaultResourceLoaderOptions;
use pi_coding_agent::core::session_manager::{CustomMessageEntryContent, SessionManager};
use pi_coding_agent::core::settings_manager::SettingsManager;
use pi_jev::compaction::CompactionConfig;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

#[path = "compaction_replay/local_parity.rs"]
mod local_parity;
#[path = "compaction_replay/candidate_review.rs"]
mod candidate_review;

// ---------------------------------------------------------------------------
// Counting global allocator (std-only)
// ---------------------------------------------------------------------------

static ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);

struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
        System.realloc(ptr, layout, new_size)
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        System.alloc_zeroed(layout)
    }
}

#[global_allocator]
static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

fn alloc_snapshot() -> (u64, u64) {
    (
        ALLOC_COUNT.load(Ordering::Relaxed),
        ALLOC_BYTES.load(Ordering::Relaxed),
    )
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or(0)
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn semantic_sha256<T: serde::Serialize + ?Sized>(value: &T) -> Result<String, String> {
    let encoded = serde_json::to_string(value).map_err(|error| error.to_string())?;
    Ok(sha256_hex(encoded.as_bytes()))
}

/// Select the current branch from the exact durable JSONL values, without the
/// read-path rehydration used for model/UI callers. This is the byte-projection
/// counterpart to SessionManager::build_index + get_branch. Any malformed entry,
/// duplicate ID, broken parent, cycle, or missing header fails closed.
#[derive(Debug)]
struct RawDurableGraph {
    branch: Vec<Value>,
    entry_count: usize,
}

#[derive(Debug, Default, Clone, Copy, PartialEq)]
struct DurableMessageCounts {
    user: usize,
    assistant: usize,
    tool_result: usize,
    other: usize,
}

fn durable_message_counts(entries: &[Value]) -> Result<DurableMessageCounts, String> {
    let mut counts = DurableMessageCounts::default();
    for entry in entries {
        if entry.get("type").and_then(Value::as_str) != Some("message") {
            continue;
        }
        match entry.pointer("/message/role").and_then(Value::as_str) {
            Some("user") => counts.user += 1,
            Some("assistant") => counts.assistant += 1,
            Some("toolResult") => counts.tool_result += 1,
            Some("custom") | Some("branchSummary") | Some("compactionSummary") => {
                counts.other += 1
            }
            Some(other) => {
                return Err(stage_error(
                    "durable-message-shape",
                    format!("unsupported durable message role {other}"),
                ))
            }
            None => {
                return Err(stage_error(
                    "durable-message-shape",
                    "durable message has no role",
                ))
            }
        }
    }
    Ok(counts)
}

fn stage_error(stage: &str, reason: impl AsRef<str>) -> String {
    format!("[{stage}] {}", reason.as_ref())
}

fn ensure_stage(condition: bool, stage: &str, reason: impl AsRef<str>) -> Result<(), String> {
    if condition {
        Ok(())
    } else {
        Err(stage_error(stage, reason))
    }
}

/// Parse and validate the COMPLETE durable graph before selecting any branch.
/// SessionManager deliberately indexes current UI state and can hide malformed
/// off-branch entries; that behavior is not an admission check. This projection
/// retains exact JSON values while rejecting malformed/truncated lines, header
/// multiplicity, duplicate IDs, missing parents, disallowed roots and cycles.
fn raw_durable_graph(bytes: &[u8]) -> Result<RawDurableGraph, String> {
    let text = std::str::from_utf8(bytes)
        .map_err(|error| stage_error("durable-jsonl", format!("invalid UTF-8: {error}")))?;
    ensure_stage(
        !text.is_empty(),
        "durable-jsonl",
        "durable session file is empty",
    )?;
    ensure_stage(
        text.ends_with('\n'),
        "durable-jsonl",
        "last durable JSONL line is unterminated or truncated",
    )?;
    let mut by_id = HashMap::<String, Value>::new();
    let mut leaf_id = None::<String>;
    let mut header_count = 0usize;
    for (index, line) in text.strip_suffix('\n').unwrap_or(text).split('\n').enumerate() {
        ensure_stage(
            !line.is_empty() && !line.ends_with('\r'),
            "durable-jsonl",
            format!("blank or non-canonical durable JSONL line {}", index + 1),
        )?;
        let entry: Value = serde_json::from_str(line).map_err(|error| {
            stage_error(
                "durable-jsonl",
                format!("invalid durable JSONL line {}: {error}", index + 1),
            )
        })?;
        ensure_stage(
            entry.is_object(),
            "durable-jsonl",
            format!("non-object durable JSONL line {}", index + 1),
        )?;
        if entry.get("type").and_then(Value::as_str) == Some("session") {
            ensure_stage(
                entry.get("id").and_then(Value::as_str).is_some_and(|id| !id.is_empty()),
                "session-header",
                "durable session header has no nonempty ID",
            )?;
            header_count += 1;
            continue;
        }
        let id = entry
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
            .ok_or_else(|| {
                stage_error(
                    "full-graph-integrity",
                    format!("durable JSONL line {} has no nonempty ID", index + 1),
                )
            })?
            .to_string();
        match entry.get("parentId") {
            None | Some(Value::Null) => {}
            Some(Value::String(parent)) if parent.is_empty() => {
                return Err(stage_error(
                    "full-graph-integrity",
                    format!("durable entry {id} has an empty parent ID"),
                ))
            }
            Some(Value::String(_)) => {}
            Some(_) => {
                return Err(stage_error(
                    "full-graph-integrity",
                    format!("durable entry {id} has a non-string parent ID"),
                ))
            }
        }
        ensure_stage(
            by_id.insert(id.clone(), entry).is_none(),
            "full-graph-integrity",
            format!("duplicate durable entry ID {id}"),
        )?;
        leaf_id = Some(id);
    }
    ensure_stage(
        header_count == 1,
        "session-header",
        format!("expected exactly one session header, found {header_count}"),
    )?;

    // Validate every component, not just the branch reachable from the leaf.
    let mut complete = HashSet::<String>::new();
    for start_id in by_id.keys() {
        if complete.contains(start_id) {
            continue;
        }
        let mut path = Vec::<String>::new();
        let mut local = HashSet::<String>::new();
        let mut current_id = start_id.clone();
        loop {
            if complete.contains(&current_id) {
                break;
            }
            ensure_stage(
                local.insert(current_id.clone()),
                "full-graph-integrity",
                format!("cycle reaches durable entry {current_id}"),
            )?;
            let entry = by_id.get(&current_id).ok_or_else(|| {
                stage_error(
                    "full-graph-integrity",
                    format!("dangling parent ID {current_id}"),
                )
            })?;
            path.push(current_id.clone());
            current_id = match entry.get("parentId") {
                None | Some(Value::Null) => {
                    // SessionManager permits a transcript-rooted graph when no
                    // model/control entry was written first. A null parent is
                    // therefore a root for any otherwise well-formed entry.
                    break;
                }
                Some(Value::String(parent)) => parent.clone(),
                Some(_) => unreachable!("parent shape was validated above"),
            };
        }
        complete.extend(path);
    }

    let mut current_id = leaf_id.ok_or_else(|| {
        stage_error("full-graph-integrity", "durable JSONL has no branch entries")
    })?;
    let mut visited = HashSet::<String>::new();
    let mut branch = Vec::new();
    loop {
        ensure_stage(
            visited.insert(current_id.clone()),
            "full-graph-integrity",
            format!("selected durable branch cycle at {current_id}"),
        )?;
        let entry = by_id.get(&current_id).ok_or_else(|| {
            stage_error(
                "full-graph-integrity",
                format!("selected durable parent {current_id} is missing"),
            )
        })?;
        branch.push(entry.clone());
        current_id = match entry.get("parentId") {
            None | Some(Value::Null) => break,
            Some(Value::String(parent)) => parent.clone(),
            Some(_) => unreachable!("parent shape was validated above"),
        };
    }
    branch.reverse();
    Ok(RawDurableGraph {
        branch,
        entry_count: by_id.len(),
    })
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn timing_stats(mut samples: Vec<f64>) -> Value {
    if samples.is_empty() {
        return json!({});
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let count = samples.len();
    let mean = samples.iter().sum::<f64>() / count as f64;
    let variance =
        samples.iter().map(|s| (s - mean) * (s - mean)).sum::<f64>() / count as f64;
    let percentile = |fraction: f64| -> f64 {
        let index = ((fraction * (count - 1) as f64).round() as usize).min(count - 1);
        samples[index]
    };
    json!({
        "n": count,
        "mean_ms": mean,
        "p50_ms": percentile(0.5),
        "p95_ms": percentile(0.95),
        "stddev_ms": variance.sqrt(),
        "min_ms": samples[0],
        "max_ms": samples[count - 1],
    })
}

fn write_json(path: &Path, value: &Value) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|error| format!("mkdir: {error}"))?;
    }
    let body = serde_json::to_string_pretty(value).map_err(|error| error.to_string())?;
    std::fs::write(path, body).map_err(|error| format!("write {}: {error}", path.display()))
    .map(|_| ())
}

fn write_new_json(path: &Path, value: &Value) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|error| format!("mkdir: {error}"))?;
    }
    let body = serde_json::to_string_pretty(value).map_err(|error| error.to_string())?;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("create_new {}: {error}", path.display()))?;
    file.write_all(body.as_bytes())
        .map_err(|error| format!("write {}: {error}", path.display()))
}

fn exe_sha256() -> Result<String, String> {
    let exe = std::env::current_exe().map_err(|error| error.to_string())?;
    let bytes = std::fs::read(&exe).map_err(|error| format!("read exe: {error}"))?;
    Ok(sha256_hex(&bytes))
}

// ---------------------------------------------------------------------------
// Hand-rolled CLI
// ---------------------------------------------------------------------------

struct Cli {
    subcommand: String,
    options: HashMap<String, String>,
}

impl Cli {
    fn parse(args: &[String]) -> Result<Cli, String> {
        let mut subcommand = String::new();
        let mut options = HashMap::new();
        let mut index = 0usize;
        while index < args.len() {
            let arg = &args[index];
            index += 1;
            if subcommand.is_empty() {
                subcommand = arg.clone();
                continue;
            }
            if let Some(rest) = arg.strip_prefix("--") {
                let (key, value) = if let Some((key, value)) = rest.split_once('=') {
                    (key, value.to_string())
                } else if index < args.len() && !args[index].starts_with("--") {
                    // `--key value` form.
                    let value = args[index].clone();
                    index += 1;
                    (rest, value)
                } else {
                    (rest, String::new())
                };
                if options.insert(key.to_string(), value).is_some() {
                    return Err(format!("[cli-parse] duplicate option --{key}"));
                }
            } else {
                return Err(format!("unexpected positional argument: {arg}"));
            }
        }
        if subcommand.is_empty() {
            return Err(usage());
        }
        Ok(Cli { subcommand, options })
    }

    fn required(&self, key: &str) -> Result<String, String> {
        self.options
            .get(key)
            .cloned()
            .filter(|value| !value.is_empty())
            .ok_or_else(|| format!("missing required option --{key}"))
    }

    fn optional(&self, key: &str) -> Option<String> {
        self.options
            .get(key)
            .cloned()
            .filter(|value| !value.is_empty())
    }

    fn integer(
        &self,
        key: &str,
        default: usize,
        minimum: usize,
        maximum: usize,
    ) -> Result<usize, String> {
        let value = match self.options.get(key) {
            None => default,
            Some(raw) if raw.is_empty() => {
                return Err(stage_error(
                    &format!("cli-{key}"),
                    format!("--{key} is present but empty; a numeric value is required"),
                ))
            }
            Some(raw) => raw
                .parse::<usize>()
                .map_err(|error| stage_error(&format!("cli-{key}"), error.to_string()))?,
        };
        ensure_stage(
            (minimum..=maximum).contains(&value),
            &format!("cli-{key}"),
            format!("--{key} must be an integer in {minimum}..={maximum}, got {value}"),
        )?;
        Ok(value)
    }

    fn positive_number(&self, key: &str, default: f64, maximum: f64) -> Result<f64, String> {
        let value = match self.options.get(key) {
            None => default,
            Some(raw) if raw.is_empty() => {
                return Err(stage_error(
                    &format!("cli-{key}"),
                    format!("--{key} is present but empty; a numeric value is required"),
                ))
            }
            Some(raw) => raw
                .parse::<f64>()
                .map_err(|error| stage_error(&format!("cli-{key}"), error.to_string()))?,
        };
        ensure_stage(
            value.is_finite() && value > 0.0 && value <= maximum,
            &format!("cli-{key}"),
            format!("--{key} must be finite and in (0, {maximum}], got {value}"),
        )?;
        Ok(value)
    }
}

fn usage() -> String {
    "usage:\n\
     compaction_replay gen-fixtures --out <dir>\n\
     compaction_replay replay --fixture <dir|case> --out <json> [--provider-delay-ms N]\n\
     [--summary-failure MSG] [--context-window N] [--keep-recent N] [--work DIR]\n\
     [--stamp SHA256] [--git DESCRIBE] [--timeout-secs N]\n\
     compaction_replay bench-local --fixture <dir|case> --iters N --out <json> [--warmup N]\n\
     [--chunk-context-window N] [--stamp SHA256] [--git DESCRIBE]"
        .to_string()
}

// ---------------------------------------------------------------------------
// Fixture model (deterministic transcript records)
// ---------------------------------------------------------------------------

const FIXTURE_TIMESTAMP_BASE: i64 = 1_772_000_000_000;

#[derive(Clone)]
struct FixtureCall {
    id: String,
    name: String,
    arguments: Value,
}

#[derive(Clone)]
enum FixtureRecord {
    User {
        text: String,
    },
    Assistant {
        text: String,
        thinking: Option<String>,
        calls: Vec<FixtureCall>,
        usage: Option<(f64, f64)>,
    },
    ToolResult {
        call_id: String,
        name: String,
        content: String,
        is_error: bool,
    },
}

#[derive(Clone)]
struct LoadedFixture {
    name: String,
    records: Vec<FixtureRecord>,
    manifest: Value,
}

impl LoadedFixture {
    fn to_agent_messages(&self, model: &Model) -> Vec<AgentMessage> {
        let mut messages: Vec<AgentMessage> = Vec::with_capacity(self.records.len());
        for (index, record) in self.records.iter().enumerate() {
            let timestamp = FIXTURE_TIMESTAMP_BASE + index as i64 * 1000;
            match record {
                FixtureRecord::User { text } => {
                    messages.push(AgentMessage::Message(Message::User(UserMessage::new(
                        UserContent::Text(text.clone()),
                        timestamp,
                    ))));
                }
                FixtureRecord::Assistant {
                    text,
                    thinking,
                    calls,
                    usage,
                } => {
                    let mut content: Vec<ContentBlock> = Vec::new();
                    if let Some(thinking) = thinking {
                        content.push(ContentBlock::Thinking(ThinkingContent::new(
                            thinking.clone(),
                        )));
                    }
                    if !text.is_empty() {
                        content.push(ContentBlock::Text(TextContent::new(text.clone())));
                    }
                    for call in calls {
                        let arguments = match &call.arguments {
                            Value::Object(map) => map.clone(),
                            _ => Map::new(),
                        };
                        content.push(ContentBlock::ToolCall(ToolCall::new(
                            call.id.clone(),
                            call.name.clone(),
                            arguments,
                        )));
                    }
                    let (input, output) = usage.unwrap_or((1.0, 1.0));
                    messages.push(AgentMessage::Message(Message::Assistant(
                        AssistantMessage {
                            content,
                            api: model.api.clone(),
                            provider: model.provider.clone(),
                            model: model.id.clone(),
                            stop_reason: "stop".to_string(),
                            timestamp,
                            usage: Usage {
                                input,
                                output,
                                total_tokens: input + output,
                                ..Usage::zero()
                            },
                            ..Default::default()
                        },
                    )));
                }
                FixtureRecord::ToolResult {
                    call_id,
                    name,
                    content,
                    is_error,
                } => {
                    messages.push(AgentMessage::Message(Message::ToolResult(
                        ToolResultMessage::new(
                            call_id.clone(),
                            name.clone(),
                            vec![ImageOrTextContent::Text(TextContent::new(content.clone()))],
                            *is_error,
                            timestamp,
                        ),
                    )));
                }
            }
        }
        messages
    }

}

fn records_to_json(records: &[FixtureRecord]) -> Value {
    Value::Array(
        records
            .iter()
            .map(|record| match record {
                FixtureRecord::User { text } => json!({ "k": "user", "t": text }),
                FixtureRecord::Assistant {
                    text,
                    thinking,
                    calls,
                    usage,
                } => {
                    let mut entry = json!({ "k": "asst", "t": text });
                    if let Some(thinking) = thinking {
                        entry["th"] = json!(thinking);
                    }
                    if !calls.is_empty() {
                        entry["c"] = Value::Array(
                            calls
                                .iter()
                                .map(|call| {
                                    json!({ "id": call.id, "n": call.name, "a": call.arguments })
                                })
                                .collect(),
                        );
                    }
                    if let Some((input, output)) = usage {
                        entry["u"] = json!([input, output]);
                    }
                    entry
                }
                FixtureRecord::ToolResult {
                    call_id,
                    name,
                    content,
                    is_error,
                } => json!({
                    "k": "res",
                    "cid": call_id,
                    "n": name,
                    "t": content,
                    "e": is_error,
                }),
            })
            .collect(),
    )
}

fn records_from_json(value: &Value) -> Result<Vec<FixtureRecord>, String> {
    let array = value.as_array().ok_or("records must be an array")?;
    let mut records = Vec::with_capacity(array.len());
    for entry in array {
        let kind = entry.get("k").and_then(Value::as_str).unwrap_or_default();
        let text = entry
            .get("t")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        match kind {
            "user" => records.push(FixtureRecord::User { text }),
            "asst" => {
                let thinking = entry
                    .get("th")
                    .and_then(Value::as_str)
                    .map(str::to_string);
                let mut calls = Vec::new();
                if let Some(list) = entry.get("c").and_then(Value::as_array) {
                    for call in list {
                        calls.push(FixtureCall {
                            id: call
                                .get("id")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_string(),
                            name: call
                                .get("n")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_string(),
                            arguments: call.get("a").cloned().unwrap_or(Value::Null),
                        });
                    }
                }
                let usage = entry.get("u").and_then(Value::as_array).and_then(|pair| {
                    let input = pair.first().and_then(Value::as_f64)?;
                    let output = pair.get(1).and_then(Value::as_f64)?;
                    Some((input, output))
                });
                records.push(FixtureRecord::Assistant {
                    text,
                    thinking,
                    calls,
                    usage,
                });
            }
            "res" => records.push(FixtureRecord::ToolResult {
                call_id: entry
                    .get("cid")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                name: entry
                    .get("n")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                content: text,
                is_error: entry.get("e").and_then(Value::as_bool).unwrap_or(false),
            }),
            other => return Err(format!("unknown record kind: {other}")),
        }
    }
    Ok(records)
}

fn discover_fixtures(path: &Path) -> Result<Vec<PathBuf>, String> {
    if path.join("transcript.json").is_file() {
        return Ok(vec![path.to_path_buf()]);
    }
    let mut cases: Vec<PathBuf> = std::fs::read_dir(path)
        .map_err(|error| format!("read dir {}: {error}", path.display()))?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|child| child.join("transcript.json").is_file())
        .collect();
    cases.sort();
    if cases.is_empty() {
        return Err(format!(
            "no fixtures under {} (expected case dirs with transcript.json)",
            path.display()
        ));
    }
    Ok(cases)
}

fn load_fixture(case_dir: &Path) -> Result<LoadedFixture, String> {
    let manifest: Value = serde_json::from_str(
        &std::fs::read_to_string(case_dir.join("manifest.json"))
            .map_err(|error| format!("read manifest: {error}"))?,
    )
    .map_err(|error| format!("parse manifest: {error}"))?;
    let body: Value = serde_json::from_str(
        &std::fs::read_to_string(case_dir.join("transcript.json"))
            .map_err(|error| format!("read transcript: {error}"))?,
    )
    .map_err(|error| format!("parse transcript: {error}"))?;
    Ok(LoadedFixture {
        name: manifest
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        records: records_from_json(body.get("records").unwrap_or(&Value::Null))?,
        manifest,
    })
}

// ---------------------------------------------------------------------------
// gen-fixtures: deterministic content pools
// ---------------------------------------------------------------------------

const ASCII_SENTENCES: &[&str] = &[
    "The workspace build finished with three warnings and no errors.",
    "Renamed the parser module and updated the internal imports accordingly.",
    "Profiled the release binary before and after the change to keep evidence.",
    "The regression suite passes on the isolated profile with synthetic keys.",
    "Cache entries are invalidated whenever the underlying file changes.",
    "Traced the request through the gateway and the upstream provider.",
    "The kernel state snapshot was pruned before serialization completed.",
    "Reduced the allocation churn in the hot loop by reusing the buffer.",
    "Documented the deviation from the original plan in the report.",
    "The threshold check now consults the post-compaction usage only.",
    "Benchmark results were collected with warmup iterations disabled.",
    "The retention anchor keeps the last two thousand characters of text.",
    "Every cut point avoids separating a tool call from its results.",
    "The serialized conversation exceeded one million characters today.",
    "Kept the deterministic digest stable across repeated offline runs.",
];

const ASCII_WORDS: &[&str] = &[
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "gateway", "harbor",
    "index", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
];

const CJK_SENTENCES: &[&str] = &[
    "ビルドが完了しました。警告は三件です。",
    "パーサーモジュールの名前を変更し、インポートを更新しました。",
    "リリースバイナリのプロファイルを取得して、証拠を保存しました。",
    "隔離されたプロファイルで回帰テストがすべて通過しました。",
    "キャッシュのエントリはファイルが変わると無効化されます。",
    "カーネルの状態スナップショットを直列化前に整理しました。",
    "ホットループのアロケーションを減らして、バッファを再利用しました。",
    "保持アンカーは最後の二千文字を保持します。",
    "要約の検証は見出しごとに本文が空でないことを確認します。",
    "決定論的なダイジェストが繰り返し実行でも一致しました。",
];

const CJK_WORDS: &[&str] = &[
    "設定", "状態", "履歴", "要約", "検証", "保存", "復元", "測定", "実行", "結果",
    "閾値", "文脈", "保持", "削除", "追加", "更新",
];

const EMOJI_ITEMS: &[&str] = &[
    "\u{1F4E6}\u{2705}\u{1F41B}\u{1F525}\u{1F680}",
    "e\u{0301}\u{0302}le\u{0300}ve\u{0301} co\u{0302}te\u{0300}",
    "\u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467}\u{200D}\u{1F466}",
    "\u{1F1E6}\u{1F1FA} \u{1F1EF}\u{1F1F5} \u{1F1E9}\u{1F1EA}",
    "family: \u{1F3E1} with \u{1F9F8} and \u{1F3AE}",
    "cafe\u{0301} naive\u{0308} re\u{0301}sume\u{0301}",
    "\u{2764}\u{FE0F}\u{200D}\u{1F525} \u{1F44D}\u{1F3FC} \u{1F5A4}",
    "signs: \u{2192}\u{21D2}\u{221E} \u{2260} \u{2261}",
];

fn tool_names() -> Vec<(&'static str, Value)> {
    vec![
        (
            "bash",
            json!({
                "command": "cargo build --locked --release -p pi-coding-agent",
                "timeout_ms": 120000,
                "description": "build the workspace crate in release mode"
            }),
        ),
        (
            "bash",
            json!({
                "command": "rg -n \"find_cut_point\" crates --type rust",
                "timeout_ms": 30000,
                "description": "search the tree for the cut point function"
            }),
        ),
        (
            "read",
            json!({ "path": "crates/pi-coding-agent/src/core/compaction/compaction.rs" }),
        ),
        (
            "edit",
            json!({
                "path": "crates/pi-coding-agent/src/core/compaction/utils.rs",
                "old": "let chars: Vec<char> = text.chars().collect();",
                "new": "let chars: Vec<char> = text.chars().collect(); // measured by the harness"
            }),
        ),
        (
            "search",
            json!({ "query": "estimate_context_tokens", "scope": "repository", "limit": 50 }),
        ),
    ]
}

#[derive(Clone, Copy, PartialEq)]
enum ContentClass {
    Ascii,
    Cjk,
    Emoji,
    MixedCrlf,
    ToolHuge,
    ManySmall,
}

impl ContentClass {
    fn key(self) -> &'static str {
        match self {
            ContentClass::Ascii => "ascii",
            ContentClass::Cjk => "cjk",
            ContentClass::Emoji => "emoji",
            ContentClass::MixedCrlf => "mixed-crlf",
            ContentClass::ToolHuge => "tool-huge",
            ContentClass::ManySmall => "many-small",
        }
    }

}

struct ContentGenerator {
    rng: StdRng,
    class: ContentClass,
    call_counter: usize,
    produced_chars: usize,
    /// Round bulk factor: larger fixtures use bigger turns, not more of them.
    bulk: usize,
    /// Cap for a single huge tool-result line, scaled to the fixture size.
    huge_line_cap: usize,
}

impl ContentGenerator {
    fn new(seed: u64, class: ContentClass, target_bytes: usize) -> ContentGenerator {
        let bulk = (target_bytes / (256 * 1024)).max(1);
        ContentGenerator {
            rng: StdRng::seed_from_u64(seed),
            class,
            call_counter: 0,
            produced_chars: 0,
            bulk,
            huge_line_cap: (target_bytes / 8).clamp(4096, 1 << 20),
        }
    }

    fn sentence(&mut self) -> String {
        let mut parts: Vec<String> = Vec::new();
        match self.class {
            ContentClass::Ascii | ContentClass::ToolHuge | ContentClass::ManySmall => {
                let sentences = 1 + self.rng.gen_range(0..3);
                for _ in 0..sentences {
                    parts.push(
                        ASCII_SENTENCES
                            .choose(&mut self.rng)
                            .unwrap_or(&"")
                            .to_string(),
                    );
                }
            }
            ContentClass::Cjk => {
                let sentences = 1 + self.rng.gen_range(0..3);
                for _ in 0..sentences {
                    parts.push(
                        CJK_SENTENCES
                            .choose(&mut self.rng)
                            .unwrap_or(&"")
                            .to_string(),
                    );
                }
            }
            ContentClass::Emoji => {
                let items = 3 + self.rng.gen_range(0..5);
                for _ in 0..items {
                    parts.push(
                        EMOJI_ITEMS
                            .choose(&mut self.rng)
                            .unwrap_or(&"")
                            .to_string(),
                    );
                }
            }
            ContentClass::MixedCrlf => {
                let sentences = 2 + self.rng.gen_range(0..3);
                for index in 0..sentences {
                    if index % 2 == 0 {
                        parts.push(
                            ASCII_SENTENCES
                                .choose(&mut self.rng)
                                .unwrap_or(&"")
                                .to_string(),
                        );
                    } else {
                        parts.push(
                            CJK_SENTENCES
                                .choose(&mut self.rng)
                                .unwrap_or(&"")
                                .to_string(),
                        );
                    }
                }
            }
        }
        self.join(parts)
    }

    fn join(&mut self, parts: Vec<String>) -> String {
        let separator = match self.class {
            ContentClass::MixedCrlf => "\r\n",
            _ => "\n",
        };
        let joined = parts.join(separator);
        self.produced_chars += joined.chars().count();
        joined
    }

    fn user_text(&mut self) -> String {
        let paragraphs = (1 + self.rng.gen_range(0..3)) * self.bulk.max(1);
        let mut parts = Vec::new();
        for _ in 0..paragraphs {
            parts.push(self.sentence());
        }
        parts.join(match self.class {
            ContentClass::MixedCrlf => "\r\n\r\n",
            _ => "\n\n",
        })
    }

    fn assistant_text(&mut self) -> String {
        let sentences = (1 + self.rng.gen_range(0..2)) * self.bulk.max(1);
        let mut parts = Vec::new();
        for _ in 0..sentences {
            parts.push(self.sentence());
        }
        parts.join(match self.class {
            ContentClass::MixedCrlf => "\r\n",
            _ => "\n",
        })
    }

    fn thinking_text(&mut self) -> String {
        let mut parts = Vec::new();
        for _ in 0..(2 + self.rng.gen_range(0..3)) {
            let pool: &[&str] = match self.class {
                ContentClass::Cjk => CJK_WORDS,
                ContentClass::Emoji => EMOJI_ITEMS,
                _ => ASCII_WORDS,
            };
            parts.push(pool.choose(&mut self.rng).unwrap_or(&"").to_string());
        }
        self.join(parts)
    }

    fn huge_single_line(&mut self, target_chars: usize) -> String {
        // One single line, no newlines, JSON-safe punctuation only.
        let mut line = String::with_capacity(target_chars + 32);
        let mut written = 0usize;
        while written < target_chars {
            let chunk = format!(
                "{}:{:04x} ",
                ASCII_WORDS.choose(&mut self.rng).unwrap_or(&"alpha"),
                self.rng.gen_range(0..65536usize)
            );
            let take = (target_chars - written).min(chunk.chars().count());
            for character in chunk.chars().take(take) {
                line.push(character);
                written += 1;
            }
        }
        self.produced_chars += line.chars().count();
        line
    }

    fn next_call_id(&mut self) -> String {
        self.call_counter += 1;
        format!("call-{:06}", self.call_counter)
    }

    fn tool_round(&mut self) -> Vec<FixtureRecord> {
        let mut records = Vec::new();
        let call_count = 1 + self.rng.gen_range(0..2);
        let mut calls: Vec<FixtureCall> = Vec::new();
        for _ in 0..call_count {
            let tools = tool_names();
            let (name, arguments) = tools
                .choose(&mut self.rng)
                .unwrap_or(&("", Value::Null));
            calls.push(FixtureCall {
                id: self.next_call_id(),
                name: name.to_string(),
                arguments: arguments.clone(),
            });
        }
        let mut assistant_text = String::new();
        if self.rng.gen_bool(0.5) {
            assistant_text = self.assistant_text();
        }
        records.push(FixtureRecord::Assistant {
            text: assistant_text,
            thinking: None,
            calls,
            usage: None,
        });
        // Exactly one result per call, in call order, so tool pairing stays
        // complete exactly like a production transcript.
        let pending: Vec<FixtureCall> = match records.last() {
            Some(FixtureRecord::Assistant { calls, .. }) => calls.clone(),
            _ => Vec::new(),
        };
        for call in pending {
            let is_error = self.rng.gen_bool(0.08);
            let content = if self.class == ContentClass::ToolHuge && self.rng.gen_bool(0.85) {
                let target = (self.huge_line_cap / 2)
                    + self.rng.gen_range(0..self.huge_line_cap.saturating_add(1));
                self.huge_single_line(target)
            } else {
                self.assistant_text()
            };
            records.push(FixtureRecord::ToolResult {
                call_id: call.id,
                name: call.name,
                content,
                is_error,
            });
        }
        records
    }
}

struct FixturePlan {
    name: String,
    class: ContentClass,
    target_bytes: usize,
    seed: u64,
}

fn seed_for(name: &str) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in name.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash & 0x7fff_ffff_ffff
}

fn generate_fixture(plan: &FixturePlan, out_root: &Path) -> Result<Value, String> {
    let class_key = plan.class.key();
    let mut generator = ContentGenerator::new(plan.seed, plan.class, plan.target_bytes);
    let mut records: Vec<FixtureRecord> = Vec::new();
    let mut produced_bytes = 0usize;
    let small = plan.class == ContentClass::ManySmall;
    let near_threshold = plan.name.starts_with("near-threshold");
    loop {
        // User turn.
        let user_text = if small {
            let mut parts = Vec::new();
            for _ in 0..(1 + generator.rng.gen_range(0..2)) {
                parts.push(
                    ASCII_SENTENCES
                        .choose(&mut generator.rng)
                        .unwrap_or(&"")
                        .to_string(),
                );
            }
            generator.join(parts)
        } else {
            generator.user_text()
        };
        produced_bytes += user_text.len();
        records.push(FixtureRecord::User { text: user_text });
        // Assistant turn (sometimes with thinking).
        let thinking = if generator.rng.gen_bool(0.5) {
            Some(generator.thinking_text())
        } else {
            None
        };
        produced_bytes += thinking.as_ref().map(|t| t.len()).unwrap_or(0);
        let assistant_text = generator.assistant_text();
        produced_bytes += assistant_text.len();
        records.push(FixtureRecord::Assistant {
            text: assistant_text,
            thinking,
            calls: Vec::new(),
            usage: None,
        });
        // Tool round.
        if generator.rng.gen_bool(if small { 0.15 } else { 0.6 }) {
            let tool_records = generator.tool_round();
            produced_bytes += tool_records
                .iter()
                .map(|record| match record {
                    FixtureRecord::User { text } => text.len(),
                    FixtureRecord::Assistant { text, thinking, calls, .. } => {
                        text.len()
                            + thinking.as_ref().map(|t| t.len()).unwrap_or(0)
                            + calls
                                .iter()
                                .map(|call| call.arguments.to_string().len())
                                .sum::<usize>()
                    }
                    FixtureRecord::ToolResult { content, .. } => content.len(),
                })
                .sum::<usize>();
            records.extend(tool_records);
            let closing = generator.assistant_text();
            produced_bytes += closing.len();
            records.push(FixtureRecord::Assistant {
                text: closing,
                thinking: None,
                calls: Vec::new(),
                usage: None,
            });
        }
        if produced_bytes >= plan.target_bytes {
            break;
        }
    }
    // Deterministic assistant usage ramp: grows toward the default threshold so
    // stale-usage guards see realistic, ordered usage values.
    let threshold = MAX_COMPACTION_CONTEXT_TOKENS;
    let mut usage_index = 0usize;
    let assistants = records.len();
    for record in &mut records {
        if let FixtureRecord::Assistant { usage, .. } = record {
            usage_index += 1;
            let progress = usage_index as f64 / assistants.max(1) as f64;
            let total = if near_threshold {
                (threshold * 0.98 * progress).min(threshold - 512.0)
            } else {
                (1000.0 + 9000.0 * progress).min(10_000.0)
            };
            *usage = Some(((total * 0.95).max(1.0), (total * 0.05).max(1.0)));
        }
    }

    let case_dir = out_root.join(&plan.name);
    std::fs::create_dir_all(&case_dir).map_err(|error| format!("mkdir: {error}"))?;
    let transcript = json!({
        "schema": 1,
        "name": plan.name,
        "class": class_key,
        "seed": plan.seed,
        "records": records_to_json(&records),
    });
    let transcript_path = case_dir.join("transcript.json");
    std::fs::write(&transcript_path, serde_json::to_string(&transcript).unwrap())
        .map_err(|error| format!("write transcript: {error}"))?;

    // Manifest: the repo's own estimator over the exact messages the replay seeds.
    let model = fixture_model_for_estimates();
    let messages: Vec<AgentMessage> = records
        .iter()
        .enumerate()
        .map(|(index, record)| record_to_agent_message(record, index, &model))
        .collect();
    // The no-usage fallback the session performs when no assistant carries
    // usable usage (agent_session get_threshold_context_tokens): a plain sum of
    // the per-message estimator.
    let estimated_no_usage: f64 = messages.iter().map(estimate_tokens).sum();
    let estimated_with_usage = estimate_context_tokens(&messages).tokens;
    let per_message_tokens: Vec<f64> = messages.iter().map(estimate_tokens).collect();
    let char_count: usize = records
        .iter()
        .map(|record| match record {
            FixtureRecord::User { text } => text.chars().count(),
            FixtureRecord::Assistant { text, thinking, calls, .. } => {
                text.chars().count()
                    + thinking.as_ref().map(|t| t.chars().count()).unwrap_or(0)
                    + calls
                        .iter()
                        .map(|call| call.arguments.to_string().chars().count())
                        .sum::<usize>()
            }
            FixtureRecord::ToolResult { content, .. } => content.chars().count(),
        })
        .sum();
    let tool_calls = records
        .iter()
        .filter(|record| {
            matches!(
                record,
                FixtureRecord::Assistant { calls, .. } if !calls.is_empty()
            )
        })
        .count();
    let tool_results = records
        .iter()
        .filter(|record| matches!(record, FixtureRecord::ToolResult { .. }))
        .count();
    let file_bytes = std::fs::metadata(&transcript_path)
        .map(|meta| meta.len() as usize)
        .unwrap_or(0);
    let keep_recent = (estimated_no_usage * 0.10)
        .round()
        .clamp(64.0, 200_000.0);
    let manifest = json!({
        "schema": 1,
        "name": plan.name,
        "class": class_key,
        "seed": plan.seed,
        "file": "transcript.json",
        "bytes": file_bytes,
        "chars": char_count,
        "estimated_tokens_no_usage": estimated_no_usage,
        "estimated_tokens_with_usage": estimated_with_usage,
        "message_count_value": per_message_tokens.len(),
        "message_count": records.len(),
        "tool_call_messages": tool_calls,
        "tool_results": tool_results,
        "keep_recent_tokens_hint": keep_recent,
        "threshold_tokens_at_defaults": threshold,
        "content_class": class_key,
    });
    write_json(&case_dir.join("manifest.json"), &manifest)?;
    Ok(manifest)
}

fn record_to_agent_message(
    record: &FixtureRecord,
    index: usize,
    model: &Model,
) -> AgentMessage {
    let timestamp = FIXTURE_TIMESTAMP_BASE + index as i64 * 1000;
    match record {
        FixtureRecord::User { text } => AgentMessage::Message(Message::User(UserMessage::new(
            UserContent::Text(text.clone()),
            timestamp,
        ))),
        FixtureRecord::Assistant {
            text,
            thinking,
            calls,
            usage,
        } => {
            let mut content: Vec<ContentBlock> = Vec::new();
            if let Some(thinking) = thinking {
                content.push(ContentBlock::Thinking(ThinkingContent::new(thinking.clone())));
            }
            if !text.is_empty() {
                content.push(ContentBlock::Text(TextContent::new(text.clone())));
            }
            for call in calls {
                let arguments = match &call.arguments {
                    Value::Object(map) => map.clone(),
                    _ => Map::new(),
                };
                content.push(ContentBlock::ToolCall(ToolCall::new(
                    call.id.clone(),
                    call.name.clone(),
                    arguments,
                )));
            }
            let (input, output) = usage.unwrap_or((1.0, 1.0));
            AgentMessage::Message(Message::Assistant(AssistantMessage {
                content,
                api: model.api.clone(),
                provider: model.provider.clone(),
                model: model.id.clone(),
                stop_reason: "stop".to_string(),
                timestamp,
                usage: Usage {
                    input,
                    output,
                    total_tokens: input + output,
                    ..Usage::zero()
                },
                ..Default::default()
            }))
        }
        FixtureRecord::ToolResult {
            call_id,
            name,
            content,
            is_error,
        } => AgentMessage::Message(Message::ToolResult(ToolResultMessage::new(
            call_id.clone(),
            name.clone(),
            vec![ImageOrTextContent::Text(TextContent::new(content.clone()))],
            *is_error,
            timestamp,
        ))),
    }
}

fn fixture_model_for_estimates() -> Model {
    let mut model = Model::new(
        "compaction-replay-model",
        "compaction-replay-model",
        "harness-estimate",
        "harness-estimate",
        "https://harness.fixture.invalid/v1",
    );
    model.context_window = 1_000_000.0;
    model.max_tokens = 8192.0;
    model.input = vec![InputModality::Text];
    model
}

fn run_gen_fixtures(cli: &Cli) -> Result<i32, String> {
    let out_root = PathBuf::from(cli.required("out")?);
    std::fs::create_dir_all(&out_root).map_err(|error| format!("mkdir: {error}"))?;
    let classes: &[ContentClass] = &[
        ContentClass::Ascii,
        ContentClass::Cjk,
        ContentClass::Emoji,
        ContentClass::MixedCrlf,
        ContentClass::ToolHuge,
        ContentClass::ManySmall,
    ];
    let sizes: &[(&str, usize, &[ContentClass])] = &[
        ("small", 10 * 1024, classes),
        ("medium", 1024 * 1024, classes),
        ("large", 10 * 1024 * 1024, classes),
        (
            "xlarge",
            30 * 1024 * 1024,
            &[ContentClass::Ascii, ContentClass::ToolHuge],
        ),
    ];
    let mut plans: Vec<FixturePlan> = Vec::new();
    for (size, bytes, size_classes) in sizes {
        for class in *size_classes {
            let name = format!("{size}-{}", class.key());
            plans.push(FixturePlan {
                name: name.clone(),
                class: *class,
                target_bytes: *bytes,
                seed: seed_for(&name),
            });
        }
    }
    // Near-threshold: sized so the repo estimator lands just above the default
    // threshold (250k tokens = min(MAX_COMPACTION_CONTEXT_TOKENS, window - reserve)).
    plans.push(FixturePlan {
        name: "near-threshold-ascii".to_string(),
        class: ContentClass::Ascii,
        target_bytes: (MAX_COMPACTION_CONTEXT_TOKENS as usize) * 4 + 40_960,
        seed: seed_for("near-threshold-ascii"),
    });

    let total = plans.len();
    let mut manifests = Vec::new();
    for (index, plan) in plans.iter().enumerate() {
        let started = Instant::now();
        let manifest = generate_fixture(plan, &out_root)?;
        println!(
            "[{}/{}] {} -> {} bytes, {} messages, {:.0} est tokens ({:.1}s)",
            index + 1,
            total,
            plan.name,
            manifest["bytes"].as_u64().unwrap_or(0),
            manifest["message_count"].as_u64().unwrap_or(0),
            manifest["estimated_tokens_no_usage"].as_f64().unwrap_or(0.0),
            elapsed_ms(started) / 1000.0
        );
        manifests.push(manifest);
    }
    let index = json!({
        "schema": 1,
        "generated_by": "compaction_replay gen-fixtures",
        "case_count": manifests.len(),
        "cases": manifests,
    });
    write_json(&out_root.join("index.json"), &index)?;
    println!("wrote {} fixtures to {}", manifests.len(), out_root.display());
    Ok(0)
}

// ---------------------------------------------------------------------------
// Fixture provider (in-process, deterministic, no network)
// ---------------------------------------------------------------------------

/// A section-complete summary valid for BOTH the conversation and turn-prefix
/// formats (mirrors the parity-suite fixture reply).
const HARNESS_SUMMARY: &str = "## Goal\nHarness replay summary body.\n## Constraints & Preferences\nNone.\n## Progress\nSummarized offline.\n## Key Decisions\nPreserve evidence digests.\n## Next Steps\nContinue.\n## Critical Context\nFixture transcript.\n## Original Request\nFixture task.\n## Early Progress\nSummarized.\n## Context for Suffix\nContinue.";

struct ProviderState {
    api: String,
    /// Case root: every path under it is normalized to `<ROOT>` before hashing
    /// so request digests stay comparable across runs with different roots.
    root: String,
    /// Live session id: the system prompt embeds the conversation-log path
    /// `sessions/<uuid>.jsonl`, so the id is normalized to `<SESSION>`.
    session_id: Mutex<Option<String>>,
    calls: Mutex<Vec<Value>>,
    replies: Mutex<Vec<AssistantMessage>>,
    summary_failure: Mutex<Option<String>>,
    summary_calls: AtomicU64,
    turn_calls: AtomicU64,
    delay_ms: AtomicU64,
}

impl ProviderState {
    fn new(api: &str, root: &str) -> ProviderState {
        ProviderState {
            api: api.to_string(),
            root: root.to_string(),
            session_id: Mutex::new(None),
            calls: Mutex::new(Vec::new()),
            replies: Mutex::new(Vec::new()),
            summary_failure: Mutex::new(None),
            summary_calls: AtomicU64::new(0),
            turn_calls: AtomicU64::new(0),
            delay_ms: AtomicU64::new(0),
        }
    }

    /// Replace every occurrence of the case root (both slash forms) and the
    /// live session id with placeholders so per-request digests exclude the
    /// volatile root path and the random per-run session uuid.
    fn normalize(&self, text: &str) -> String {
        let mut normalized = text.replace(&self.root, "<ROOT>");
        if self.root.contains('\\') {
            let forward = self.root.replace('\\', "/");
            normalized = normalized.replace(&forward, "<ROOT>");
        }
        if let Some(session_id) = self.session_id.lock().unwrap().clone() {
            normalized = normalized.replace(&session_id, "<SESSION>");
        }
        normalized
    }
}

fn text_message(text: &str, stop_reason: &str) -> AssistantMessage {
    AssistantMessage {
        content: vec![ContentBlock::Text(TextContent::new(text))],
        stop_reason: stop_reason.to_string(),
        ..Default::default()
    }
}

fn events_for(message: AssistantMessage) -> Vec<AssistantMessageEvent> {
    let mut events: Vec<AssistantMessageEvent> = Vec::new();
    events.push(AssistantMessageEvent::Start {
        partial: message.clone(),
    });
    let mut partial = message.clone();
    partial.content = Vec::new();
    events.push(AssistantMessageEvent::TextStart {
        content_index: 0,
        partial: partial.clone(),
    });
    let text = message
        .content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text(text) => Some(text.text.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    partial.content = vec![ContentBlock::Text(TextContent::new(text.clone()))];
    events.push(AssistantMessageEvent::TextDelta {
        content_index: 0,
        delta: text.clone(),
        partial: partial.clone(),
    });
    events.push(AssistantMessageEvent::TextEnd {
        content_index: 0,
        content: text,
        partial: partial.clone(),
    });
    events.push(AssistantMessageEvent::Done {
        reason: message.stop_reason.clone(),
        message,
    });
    events
}

fn build_reply(state: &Arc<ProviderState>, model: &Model, is_summary_call: bool) -> AssistantMessage {
    let mut message = if is_summary_call {
        let failure = state.summary_failure.lock().unwrap().clone();
        match failure {
            Some(failure) => {
                let mut failed = text_message("", "error");
                failed.error_message = Some(failure);
                failed
            }
            None => text_message(HARNESS_SUMMARY, "stop"),
        }
    } else {
        state
            .replies
            .lock()
            .unwrap()
            .pop()
            .unwrap_or_else(|| text_message("harness replay turn reply", "stop"))
    };
    message.api = model.api.clone();
    message.provider = model.provider.clone();
    message.model = model.id.clone();
    if message.usage.total_tokens == 0.0 && message.usage.input == 0.0 {
        message.usage = Usage {
            input: 1.0,
            output: 1.0,
            total_tokens: 2.0,
            ..Usage::zero()
        };
    }
    if message.timestamp == 0 {
        message.timestamp = now_ms();
    }
    message
}

fn count_context_chars(context: &Context) -> (usize, usize) {
    let mut total = 0usize;
    let mut longest = 0usize;
    let mut observe = |text: &str| {
        let count = text.chars().count();
        total += count;
        longest = longest.max(count);
    };
    for message in &context.messages {
        match message {
            Message::User(user) => match &user.content {
                UserContent::Text(text) => observe(text),
                UserContent::Blocks(blocks) => {
                    for block in blocks {
                        if let ImageOrTextContent::Text(text) = block {
                            observe(&text.text);
                        }
                    }
                }
            },
            Message::Assistant(assistant) => {
                for block in &assistant.content {
                    match block {
                        ContentBlock::Text(text) => observe(&text.text),
                        ContentBlock::Thinking(thinking) => observe(&thinking.thinking),
                        ContentBlock::ToolCall(call) => {
                            let rendered = call
                                .arguments
                                .iter()
                                .map(|(key, value)| {
                                    format!("{key}={}", serde_json::to_string(value).unwrap_or_default())
                                })
                                .collect::<Vec<_>>()
                                .join(", ");
                            observe(&rendered);
                        }
                    }
                }
            }
            Message::ToolResult(result) => {
                for block in &result.content {
                    if let ImageOrTextContent::Text(text) = block {
                        observe(&text.text);
                    }
                }
            }
        }
    }
    (total, longest)
}

/// Digest of one provider request's payload: system prompt, message roles,
/// text/thinking contents, tool-call names + serialized arguments, message
/// count and total chars. Volatile fields are excluded: message timestamps,
/// tool-call ids, image blocks, and every path under the case root (normalized
/// to `<ROOT>`). This catches chunking/serialization regressions that keep the
/// chunk count and sizes unchanged.
fn request_payload_digest(
    state: &ProviderState,
    context: &Context,
) -> (String, usize, usize, Option<usize>) {
    let mut messages: Vec<Value> = Vec::with_capacity(context.messages.len());
    let mut total_chars = 0usize;
    for message in &context.messages {
        let role = message.role();
        let mut texts: Vec<String> = Vec::new();
        let mut tool_calls: Vec<Value> = Vec::new();
        match message {
            Message::User(user) => match &user.content {
                UserContent::Text(text) => texts.push(state.normalize(text)),
                UserContent::Blocks(blocks) => {
                    for block in blocks {
                        if let ImageOrTextContent::Text(text) = block {
                            texts.push(state.normalize(&text.text));
                        }
                    }
                }
            },
            Message::Assistant(assistant) => {
                for block in &assistant.content {
                    match block {
                        ContentBlock::Text(text) => texts.push(state.normalize(&text.text)),
                        ContentBlock::Thinking(thinking) => {
                            texts.push(state.normalize(&thinking.thinking))
                        }
                        ContentBlock::ToolCall(call) => {
                            let arguments = call
                                .arguments
                                .iter()
                                .map(|(key, value)| {
                                    format!(
                                        "{key}={}",
                                        serde_json::to_string(value).unwrap_or_default()
                                    )
                                })
                                .collect::<Vec<_>>()
                                .join(", ");
                            tool_calls.push(json!({
                                "name": call.name,
                                "arguments": state.normalize(&arguments),
                            }));
                        }
                    }
                }
            }
            Message::ToolResult(result) => {
                for block in &result.content {
                    if let ImageOrTextContent::Text(text) = block {
                        texts.push(state.normalize(&text.text));
                    }
                }
            }
        }
        total_chars += texts.iter().map(|text| text.chars().count()).sum::<usize>();
        messages.push(json!({ "role": role, "texts": texts, "tool_calls": tool_calls }));
    }
    let system_prompt = context
        .system_prompt
        .as_ref()
        .map(|prompt| state.normalize(prompt));
    let payload = json!({
        "system_prompt": system_prompt,
        "message_count": context.messages.len(),
        "total_chars": total_chars,
        "messages": messages,
    });
    let encoded = serde_json::to_string(&payload).unwrap_or_default();
    // Debug aid: dump the normalized payload when the env var is set, so a
    // volatile field can be found and excluded from the digest.
    if std::env::var("COMPACTION_REPLAY_DUMP_REQUESTS").is_ok() {
        static DUMP_COUNTER: AtomicU64 = AtomicU64::new(0);
        let seq = DUMP_COUNTER.fetch_add(1, Ordering::SeqCst);
        let _ = std::fs::write(
            Path::new(&state.root).join(format!("request-dump-{seq}.json")),
            &encoded,
        );
    }
    let digest = sha256_hex(encoded.as_bytes());
    let system_prompt_chars = context
        .system_prompt
        .as_ref()
        .map(|prompt| prompt.chars().count());
    (digest, total_chars, context.messages.len(), system_prompt_chars)
}

fn record_call(
    state: &Arc<ProviderState>,
    seq: u64,
    is_summary_call: bool,
    message_count: usize,
    total_chars: usize,
    longest_chars: usize,
    max_tokens: Option<f64>,
    service_ms: f64,
    payload_sha256: String,
    system_prompt_chars: Option<usize>,
    reply: &AssistantMessage,
) {
    let response_sha256 = serde_json::to_string(reply)
        .map(|encoded| sha256_hex(encoded.as_bytes()))
        .unwrap_or_default();
    state.calls.lock().unwrap().push(json!({
        "seq": seq,
        "is_summary_call": is_summary_call,
        "message_count": message_count,
        "total_chars": total_chars,
        "longest_text_chars": longest_chars,
        "max_tokens": max_tokens,
        "service_ms": service_ms,
        "payload_sha256": payload_sha256,
        "system_prompt_chars": system_prompt_chars,
        "stop_reason": reply.stop_reason,
        "error_message": reply.error_message,
        "response_sha256": response_sha256,
    }));
}

fn respond(
    state: &Arc<ProviderState>,
    model: &Model,
    context: &Context,
    options: Option<&SimpleStreamOptions>,
) -> AssistantMessageEventStream {
    let started = Instant::now();
    let is_summary_call = context.system_prompt.as_deref() == Some(SUMMARIZATION_SYSTEM_PROMPT);
    let (total_chars, longest_chars) = count_context_chars(context);
    let message_count = context.messages.len();
    let max_tokens = options.and_then(|options| options.stream.max_tokens);
    let (payload_sha256, _, _, system_prompt_chars) = request_payload_digest(state, context);
    let reply = build_reply(state, model, is_summary_call);
    let stream = create_assistant_message_event_stream();
    let delay_ms = state.delay_ms.load(Ordering::SeqCst);
    let seq = if is_summary_call {
        state.summary_calls.fetch_add(1, Ordering::SeqCst) + 1
    } else {
        state.turn_calls.fetch_add(1, Ordering::SeqCst) + 1
    };
    if delay_ms == 0 {
        for event in events_for(reply.clone()) {
            stream.push(event);
        }
        stream.end(Some(reply.clone()));
        record_call(
            state,
            seq,
            is_summary_call,
            message_count,
            total_chars,
            longest_chars,
            max_tokens,
            elapsed_ms(started),
            payload_sha256,
            system_prompt_chars,
            &reply,
        );
        return stream;
    }
    // Emulated network latency: the reply lands after the delay.
    let state_for_task = Arc::clone(state);
    let handle = tokio::runtime::Handle::try_current();
    match handle {
        Ok(handle) => {
            let task_stream = stream.clone();
            handle.spawn(async move {
                tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                for event in events_for(reply.clone()) {
                    task_stream.push(event);
                }
                task_stream.end(Some(reply.clone()));
                record_call(
                    &state_for_task,
                    seq,
                    is_summary_call,
                    message_count,
                    total_chars,
                    longest_chars,
                    max_tokens,
                    elapsed_ms(started),
                    payload_sha256,
                    system_prompt_chars,
                    &reply,
                );
            });
        }
        Err(_) => {
            for event in events_for(reply.clone()) {
                stream.push(event);
            }
            stream.end(Some(reply.clone()));
            record_call(
                state,
                seq,
                is_summary_call,
                message_count,
                total_chars,
                longest_chars,
                max_tokens,
                elapsed_ms(started),
                payload_sha256,
                system_prompt_chars,
                &reply,
            );
        }
    }
    stream
}

fn register_fixture_provider(state: &Arc<ProviderState>) {
    let simple_state = Arc::clone(state);
    let stream_simple: SimpleStreamFunction = Arc::new(
        move |model: &Model, context: &Context, options: Option<&SimpleStreamOptions>| {
            respond(&simple_state, model, context, options)
        },
    );
    let stream_state = Arc::clone(state);
    let stream: StreamFunction = Arc::new(
        move |model: &Model, context: &Context, _options: Option<&pi_ai::types::StreamOptions>| {
            let simple = SimpleStreamOptions::default();
            respond(&stream_state, model, context, Some(&simple))
        },
    );
    register_api_provider_simple(
        ApiProviderSimple {
            api: state.api.clone(),
            stream,
            stream_simple,
            compact: None,
            supports_compaction: None,
        },
        Some(format!("compaction-replay-{}", state.api)),
    );
}

// ---------------------------------------------------------------------------
// Metrics recorder (in-memory, phase rows for the JSON evidence)
// ---------------------------------------------------------------------------

struct HarnessRecorder {
    started: Instant,
    ids: AtomicU64,
    events: Mutex<Vec<PerformanceMetricEvent>>,
}

impl HarnessRecorder {
    fn new() -> HarnessRecorder {
        HarnessRecorder {
            started: Instant::now(),
            ids: AtomicU64::new(0),
            events: Mutex::new(Vec::new()),
        }
    }
}

impl PerformanceMetricRecorder for HarnessRecorder {
    fn session_id(&self) -> &str {
        "compaction-replay"
    }
    fn monotonic_now(&self) -> f64 {
        self.started.elapsed().as_secs_f64() * 1000.0
    }
    fn next_id(&self, scope: PerformanceMetricIdScope) -> String {
        format!("{scope:?}-{}", self.ids.fetch_add(1, Ordering::SeqCst))
    }
    fn record(&self, event: PerformanceMetricEvent) {
        self.events.lock().unwrap().push(event);
    }
    fn flush(&self) {}
    fn close(&self) {}
}

fn phase_rows(events: &[PerformanceMetricEvent]) -> Vec<Value> {
    events
        .iter()
        .filter(|event| {
            matches!(
                event.operation,
                Operation::Compaction
                    | Operation::CompactionPrepare
                    | Operation::CompactionHistory
                    | Operation::CompactionPrefix
                    | Operation::CompactionNative
                    | Operation::CompactionPersist
                    | Operation::CompactionRestore
            )
        })
        .filter(|event| {
            matches!(
                event.outcome,
                Some(Outcome::Success | Outcome::Failure | Outcome::Cancelled | Outcome::Unavailable)
            )
        })
        .map(|event| {
            json!({
                "operation": event.operation.as_str(),
                "outcome": event.outcome.map(|outcome| outcome.as_str()),
                "total_ms": event
                    .measurements
                    .as_ref()
                    .and_then(|map| map.get(&Measurement::TotalMs).cloned().flatten()),
                "serialized_bytes": event
                    .measurements
                    .as_ref()
                    .and_then(|map| map.get(&Measurement::SerializedBytes).cloned().flatten()),
            })
        })
        .collect()
}

fn provider_attempt_rows(events: &[PerformanceMetricEvent]) -> Vec<Value> {
    events
        .iter()
        .filter(|event| event.operation == Operation::ProviderAttempt)
        .filter(|event| {
            matches!(
                event.outcome,
                Some(Outcome::Success | Outcome::Failure | Outcome::Cancelled | Outcome::Unavailable)
            )
        })
        .map(|event| {
            json!({
                "outcome": event.outcome.map(|outcome| outcome.as_str()),
                "total_ms": event
                    .measurements
                    .as_ref()
                    .and_then(|map| map.get(&Measurement::TotalMs).cloned().flatten()),
                "dispatch_to_network_terminal_ms": event
                    .measurements
                    .as_ref()
                    .and_then(|map| map.get(&Measurement::DispatchToNetworkTerminalMs).cloned().flatten()),
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Normalized retained-state digest (deterministic across runs)
// ---------------------------------------------------------------------------

/// Message shape with volatile fields (timestamps, random UUIDs) removed, and
/// tool results keyed by their pairing ordinal instead of the random call id.
fn normalize_messages(messages: &[AgentMessage]) -> Value {
    let mut call_ordinals: HashMap<String, usize> = HashMap::new();

    let mut normalized: Vec<Value> = Vec::with_capacity(messages.len());
    let mut next_ordinal = 0usize;
    for message in messages {
        match message {
            AgentMessage::Message(Message::User(user)) => {
                let texts = match &user.content {
                    UserContent::Text(text) => vec![text.clone()],
                    UserContent::Blocks(blocks) => blocks
                        .iter()
                        .filter_map(|block| match block {
                            ImageOrTextContent::Text(text) => Some(text.text.clone()),
                            _ => None,
                        })
                        .collect(),
                };
                normalized.push(json!({ "role": "user", "texts": texts }));
            }
            AgentMessage::Message(Message::Assistant(assistant)) => {
                let mut texts: Vec<String> = Vec::new();
                let mut thinking: Vec<String> = Vec::new();
                let mut calls: Vec<Value> = Vec::new();
                for block in &assistant.content {
                    match block {
                        ContentBlock::Text(text) => texts.push(text.text.clone()),
                        ContentBlock::Thinking(block) => thinking.push(block.thinking.clone()),
                        ContentBlock::ToolCall(call) => {
                            let ordinal = next_ordinal;
                            next_ordinal += 1;
                            call_ordinals.insert(call.id.clone(), ordinal);
                            calls.push(json!({
                                "pair": ordinal,
                                "name": call.name,
                                "arguments": serde_json::to_string(&call.arguments)
                                    .unwrap_or_default(),
                            }));
                        }
                    }
                }
                normalized.push(json!({
                    "role": "assistant",
                    "texts": texts,
                    "thinking": thinking,
                    "calls": calls,
                }));
            }
            AgentMessage::Message(Message::ToolResult(result)) => {
                let content: String = result
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        ImageOrTextContent::Text(text) => Some(text.text.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                let pair = call_ordinals.get(&result.tool_call_id).copied();
                normalized.push(json!({
                    "role": "toolResult",
                    "pair": pair,
                    "name": result.tool_name,
                    "content": content,
                    "is_error": result.is_error,
                }));
            }
            AgentMessage::Custom(custom) => {
                normalized.push(normalize_custom(custom));
            }
        }
    }
    Value::Array(normalized)
}

fn normalize_custom(custom: &CustomAgentMessage) -> Value {
    match custom {
        CustomAgentMessage::BashExecution {
            command,
            output,
            exit_code,
            cancelled,
            truncated,
            ..
        } => json!({
            "role": "bashExecution",
            "command": command,
            "output": output,
            "exit_code": exit_code,
            "cancelled": cancelled,
            "truncated": truncated,
        }),
        CustomAgentMessage::Custom {
            custom_type,
            content,
            ..
        } => {
            let texts = match content {
                pi_agent_core::types::CustomMessageContent::Text(text) => {
                    vec![text.clone()]
                }
                pi_agent_core::types::CustomMessageContent::Blocks(blocks) => blocks
                    .iter()
                    .filter_map(|block| match block {
                        pi_agent_core::types::ContentBlock::Text(text) => {
                            Some(text.text.clone())
                        }
                        pi_agent_core::types::ContentBlock::Image(_) => None,
                    })
                    .collect(),
            };
            json!({ "role": "custom", "custom_type": custom_type, "texts": texts })
        }
        CustomAgentMessage::BranchSummary { summary, .. } => {
            json!({ "role": "branchSummary", "summary": summary })
        }
        CustomAgentMessage::CompactionSummary {
            summary,
            tokens_before,
            retained_message_count,
            custom_instructions,
            harness_digest,
            ..
        } => json!({
            "role": "compactionSummary",
            "summary": summary,
            "tokens_before": tokens_before,
            "retained_message_count": retained_message_count,
            "custom_instructions": custom_instructions,
            "harness_digest": harness_digest,
        }),
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct PairingStats {
    tool_calls: usize,
    tool_results: usize,
    matched_results: usize,
    unmatched_results: usize,
    calls_without_result: usize,
    duplicate_tool_call_ids: usize,
    duplicate_tool_result_ids: usize,
}

fn record_tool_call(stats: &mut PairingStats, calls: &mut HashMap<String, bool>, id: &str) {
    stats.tool_calls += 1;
    if calls.insert(id.to_string(), false).is_some() {
        stats.duplicate_tool_call_ids += 1;
    }
}

fn record_tool_result(stats: &mut PairingStats, calls: &mut HashMap<String, bool>, id: &str) {
    stats.tool_results += 1;
    match calls.get_mut(id) {
        Some(seen) if *seen => stats.duplicate_tool_result_ids += 1,
        Some(seen) => {
            stats.matched_results += 1;
            *seen = true;
        }
        None => stats.unmatched_results += 1,
    }
}

fn pairing_stats(messages: &[AgentMessage]) -> PairingStats {
    let mut calls = HashMap::<String, bool>::new();
    let mut stats = PairingStats::default();
    for message in messages {
        match message {
            AgentMessage::Message(Message::Assistant(assistant)) => {
                for block in &assistant.content {
                    if let ContentBlock::ToolCall(call) = block {
                        record_tool_call(&mut stats, &mut calls, &call.id);
                    }
                }
            }
            AgentMessage::Message(Message::ToolResult(result)) => {
                record_tool_result(&mut stats, &mut calls, &result.tool_call_id);
            }
            _ => {}
        }
    }
    stats.calls_without_result = calls.values().filter(|seen| !**seen).count();
    stats
}

/// Pair durable tool records directly, before compaction can trim a duplicate
/// or malformed record out of the live projection. ID multiplicity is counted
/// independently of one-to-one pairing so duplicate IDs cannot balance counts.
fn raw_durable_pairing(entries: &[Value]) -> Result<PairingStats, String> {
    let mut calls = HashMap::<String, bool>::new();
    let mut stats = PairingStats::default();
    for entry in entries {
        if entry.get("type").and_then(Value::as_str) != Some("message") {
            continue;
        }
        let message = entry.get("message").ok_or_else(|| {
            stage_error("tool-record", "durable message entry has no message object")
        })?;
        match message.get("role").and_then(Value::as_str) {
            Some("assistant") => {
                let content = message.get("content").and_then(Value::as_array).ok_or_else(|| {
                    stage_error("tool-record", "assistant message content is not an array")
                })?;
                for block in content {
                    if block.get("type").and_then(Value::as_str) == Some("toolCall") {
                        let id = block.get("id").and_then(Value::as_str).filter(|id| !id.is_empty()).ok_or_else(|| {
                            stage_error("tool-record", "tool call has no nonempty ID")
                        })?;
                        record_tool_call(&mut stats, &mut calls, id);
                    }
                }
            }
            Some("toolResult") => {
                let id = message.get("toolCallId").and_then(Value::as_str).filter(|id| !id.is_empty()).ok_or_else(|| {
                    stage_error("tool-record", "tool result has no nonempty toolCallId")
                })?;
                record_tool_result(&mut stats, &mut calls, id);
            }
            _ => {}
        }
    }
    stats.calls_without_result = calls.values().filter(|seen| !**seen).count();
    Ok(stats)
}

fn pairing_value(pairing: &PairingStats) -> Value {
    json!({
        "tool_calls": pairing.tool_calls,
        "tool_results": pairing.tool_results,
        "matched_results": pairing.matched_results,
        "unmatched_results": pairing.unmatched_results,
        "calls_without_result": pairing.calls_without_result,
        "duplicate_tool_call_ids": pairing.duplicate_tool_call_ids,
        "duplicate_tool_result_ids": pairing.duplicate_tool_result_ids,
    })
}

fn protected_state_projection(entries: &[Value]) -> Value {
    let custom_messages: Vec<Value> = entries
        .iter()
        .filter(|entry| entry.get("type").and_then(Value::as_str) == Some("custom_message"))
        .cloned()
        .collect();
    let checkpoints: Vec<Value> = entries
        .iter()
        .filter(|entry| entry.get("type").and_then(Value::as_str) == Some("compaction"))
        .cloned()
        .collect();
    let session_controls: Vec<Value> = entries
        .iter()
        .filter(|entry| {
            matches!(
                entry.get("type").and_then(Value::as_str),
                Some("model_change" | "thinking_level_change" | "service_tier_change" | "custom" | "custom_message" | "branch_summary" | "session_info" | "label")
            )
        })
        .cloned()
        .collect();
    json!({
        "custom_messages": custom_messages,
        "checkpoints": checkpoints,
        "session_controls": session_controls,
    })
}

fn validate_checkpoint_links(entries: &[Value]) -> Result<(), String> {
    let ids: HashSet<&str> = entries.iter().filter_map(|entry| entry.get("id").and_then(Value::as_str)).collect();
    for entry in entries {
        if entry.get("type").and_then(Value::as_str) != Some("compaction") {
            continue;
        }
        let kept = entry.get("firstKeptEntryId").and_then(Value::as_str).filter(|id| !id.is_empty()).ok_or_else(|| {
            stage_error("checkpoint", "compaction has no nonempty firstKeptEntryId")
        })?;
        ensure_stage(ids.contains(kept), "checkpoint", format!("checkpoint firstKeptEntryId {kept} is missing from the durable graph"))?;
        ensure_stage(entry.get("summary").and_then(Value::as_str).is_some_and(|summary| !summary.is_empty()), "checkpoint", "compaction has an empty durable summary")?;
    }
    Ok(())
}

fn validate_durable_turn_delta(
    seeded: &[Value],
    final_branch: &[Value],
    total_entry_count: usize,
    expected: ReplayOutcomeExpectation,
) -> Result<Value, String> {
    // A replay writes a linear local branch. Anything else is durable state an
    // ordinary reopen can silently ignore and therefore must not be admitted.
    ensure_stage(
        total_entry_count == final_branch.len(),
        "unexpected-durable-entry",
        format!("durable graph has {} entries outside the selected branch", total_entry_count.saturating_sub(final_branch.len())),
    )?;
    let final_by_id: HashMap<&str, &Value> = final_branch
        .iter()
        .filter_map(|entry| entry.get("id").and_then(Value::as_str).map(|id| (id, entry)))
        .collect();
    for entry in seeded {
        let id = entry.get("id").and_then(Value::as_str).ok_or_else(|| stage_error("durable-seed", "seed entry has no ID"))?;
        ensure_stage(
            final_by_id.get(id).is_some_and(|final_entry| *final_entry == entry),
            "durable-seed",
            format!("seeded entry {id} was changed or removed by the single replay turn"),
        )?;
    }
    let seeded_ids: HashSet<&str> = seeded.iter().filter_map(|entry| entry.get("id").and_then(Value::as_str)).collect();
    let added: Vec<&Value> = final_branch.iter().filter(|entry| {
        entry.get("id").and_then(Value::as_str).is_none_or(|id| !seeded_ids.contains(id))
    }).collect();
    let mut added_messages = DurableMessageCounts::default();
    let mut added_compactions = 0usize;
    let mut added_custom_types = Vec::<String>::new();
    for entry in &added {
        match entry.get("type").and_then(Value::as_str) {
            Some("message") => match entry.pointer("/message/role").and_then(Value::as_str) {
                Some("user") => added_messages.user += 1,
                Some("assistant") => added_messages.assistant += 1,
                Some("toolResult") => added_messages.tool_result += 1,
                Some(_) => added_messages.other += 1,
                None => return Err(stage_error("durable-message-shape", "new durable message lacks a role")),
            },
            Some("compaction") => added_compactions += 1,
            Some("custom_message") => {
                let custom_type = entry.get("customType").and_then(Value::as_str).unwrap_or_default();
                added_custom_types.push(custom_type.to_string());
            }
            Some(other) => {
                return Err(stage_error(
                    "unexpected-durable-turns",
                    format!("single replay turn added unexpected durable entry type {other}"),
                ))
            }
            None => return Err(stage_error("durable-entry-shape", "new durable entry has no type")),
        }
    }
    ensure_stage(
        added_messages.user == 1 && added_messages.assistant == 1
            && added_messages.tool_result == 0 && added_messages.other == 0,
        "unexpected-durable-turns",
        format!("expected exactly one added user and assistant message and no others; got {added_messages:?}"),
    )?;
    let expected_compactions = if expected == ReplayOutcomeExpectation::SuccessfulCompaction { 1 } else { 0 };
    ensure_stage(
        added_compactions == expected_compactions,
        "checkpoint-count",
        format!("expected {expected_compactions} added compaction entries, got {added_compactions}"),
    )?;
    added_custom_types.sort();
    let mut expected_custom_types = vec!["harness_digest".to_string()];
    if expected == ReplayOutcomeExpectation::SummaryFailure {
        expected_custom_types.push("compaction_outcome".to_string());
        expected_custom_types.sort();
    }
    ensure_stage(
        added_custom_types == expected_custom_types,
        "protected-custom-state",
        format!("expected added custom types {expected_custom_types:?}, got {added_custom_types:?}"),
    )?;
    let expected_total = if expected == ReplayOutcomeExpectation::SuccessfulCompaction { 4 } else { 4 };
    ensure_stage(
        added.len() == expected_total,
        "unexpected-durable-turns",
        format!("expected exactly {expected_total} added durable entries, got {}", added.len()),
    )?;
    Ok(json!({
        "added_entry_count": added.len(),
        "added_message_counts": {
            "user": added_messages.user,
            "assistant": added_messages.assistant,
            "tool_results": added_messages.tool_result,
            "other": added_messages.other,
        },
        "added_compaction_entries": added_compactions,
        "added_custom_types": added_custom_types,
        "seed_unchanged": true,
        "graph_entry_count": total_entry_count,
        "branch_entry_count": final_branch.len(),
    }))
}

// ---------------------------------------------------------------------------
// replay subcommand
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReplayOutcomeExpectation {
    SuccessfulCompaction,
    SummaryFailure,
}

impl ReplayOutcomeExpectation {
    fn parse(raw: &str) -> Result<Self, String> {
        match raw {
            "successful-compaction" => Ok(Self::SuccessfulCompaction),
            "summary-failure" => Ok(Self::SummaryFailure),
            other => Err(format!(
                "invalid --expect-outcome {other:?}: expected successful-compaction or summary-failure"
            )),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::SuccessfulCompaction => "successful-compaction",
            Self::SummaryFailure => "summary-failure",
        }
    }
}

struct ReplayOptions {
    provider_delay_ms: u64,
    summary_failure: Option<String>,
    expected_outcome: ReplayOutcomeExpectation,
    expected_summary_calls: usize,
    context_window: f64,
    keep_recent: Option<f64>,
    work: PathBuf,
    timeout_secs: u64,
}

#[derive(Default)]
struct ObservedCompaction {
    starts: Mutex<Vec<String>>,
    ends: Mutex<Vec<Value>>,
}

fn private_auth_options() -> Option<AuthStorageOptions> {
    Some(AuthStorageOptions {
        prime_cli_config_path: None,
        use_prime_cli_config: false,
    })
}

const HARNESS_API_KEY: &str = "harness-synthetic-key";

fn export_case_env(root: &Path) {
    let agent_dir = root.join("agent");
    let session_dir = root.join("sessions");
    let temp_dir = root.join("temp");
    std::env::set_var(
        "PRIME_AGENT_CODING_AGENT_DIR",
        agent_dir.to_string_lossy().to_string(),
    );
    std::env::set_var(
        "PRIME_AGENT_SESSION_DIR",
        session_dir.to_string_lossy().to_string(),
    );
    std::env::set_var(
        "PRIME_AGENT_CODING_AGENT_SESSION_DIR",
        session_dir.to_string_lossy().to_string(),
    );
    std::env::set_var("TMPDIR", temp_dir.to_string_lossy().to_string());
    std::env::set_var("TEMP", temp_dir.to_string_lossy().to_string());
    std::env::set_var("TMP", temp_dir.to_string_lossy().to_string());
}

fn json_type_name(value: Option<&Value>) -> &'static str {
    match value {
        None => "missing",
        Some(Value::Null) => "null",
        Some(Value::Bool(_)) => "boolean",
        Some(Value::Number(_)) => "number",
        Some(Value::String(_)) => "string",
        Some(Value::Array(_)) => "array",
        Some(Value::Object(_)) => "object",
    }
}

fn json_mismatch_detail(expected: Option<&Value>, actual: Option<&Value>, pointer: &str) -> Value {
    let expected_sha = expected
        .map(|value| sha256_hex(value.to_string().as_bytes()))
        .unwrap_or_else(|| "missing".to_string());
    let actual_sha = actual
        .map(|value| sha256_hex(value.to_string().as_bytes()))
        .unwrap_or_else(|| "missing".to_string());
    json!({
        "pointer": pointer,
        "expected_type": json_type_name(expected),
        "actual_type": json_type_name(actual),
        "expected_sha256": expected_sha,
        "actual_sha256": actual_sha,
    })
}

fn first_json_difference(expected: Option<&Value>, actual: Option<&Value>, pointer: &str) -> Option<Value> {
    if expected == actual {
        return None;
    }
    match (expected, actual) {
        (Some(Value::Array(expected_items)), Some(Value::Array(actual_items))) => {
            if expected_items.len() != actual_items.len() {
                let mut difference = json_mismatch_detail(expected, actual, pointer);
                difference["expected_length"] = json!(expected_items.len());
                difference["actual_length"] = json!(actual_items.len());
                return Some(difference);
            }
            for (index, (expected_item, actual_item)) in
                expected_items.iter().zip(actual_items).enumerate()
            {
                if let Some(difference) = first_json_difference(
                    Some(expected_item),
                    Some(actual_item),
                    &format!("{pointer}/{index}"),
                ) {
                    return Some(difference);
                }
            }
        }
        (Some(Value::Object(expected_items)), Some(Value::Object(actual_items))) => {
            let mut keys: Vec<&str> = expected_items
                .keys()
                .chain(actual_items.keys())
                .map(String::as_str)
                .collect();
            keys.sort_unstable();
            keys.dedup();
            for key in keys {
                if let Some(difference) = first_json_difference(
                    expected_items.get(key),
                    actual_items.get(key),
                    &format!("{pointer}/{key}"),
                ) {
                    return Some(difference);
                }
            }
        }
        _ => {}
    }
    Some(json_mismatch_detail(expected, actual, pointer))
}

async fn run_replay_case(
    fixture: &LoadedFixture,
    options: &ReplayOptions,
    case_counter: &AtomicU64,
) -> Result<Value, String> {
    let case_root = options
        .work
        .join(format!(
            "replay-{}-{}-{}",
            fixture.name,
            std::process::id(),
            case_counter.fetch_add(1, Ordering::SeqCst)
        ));
    let cwd = case_root.join("workspace");
    let agent_dir = case_root.join("agent");
    let session_dir = case_root.join("sessions");
    let temp_dir = case_root.join("temp");
    for dir in [&case_root, &cwd, &agent_dir, &session_dir, &temp_dir] {
        std::fs::create_dir_all(dir).map_err(|error| format!("mkdir: {error}"))?;
    }
    export_case_env(&case_root);

    let api = format!("harness-api-{}", fixture.name);
    let provider = format!("harness-provider-{}", fixture.name);
    let state = Arc::new(ProviderState::new(&api, &case_root.to_string_lossy()));
    state
        .delay_ms
        .store(options.provider_delay_ms, Ordering::SeqCst);
    *state.summary_failure.lock().unwrap() = options.summary_failure.clone();
    register_fixture_provider(&state);

    let mut model = Model::new(
        "compaction-replay-model",
        "compaction-replay-model",
        api.clone(),
        provider.clone(),
        format!("https://{api}.fixture.invalid/v1"),
    );
    model.context_window = options.context_window;
    model.max_tokens = 8192.0;
    model.input = vec![InputModality::Text];

    let keep_recent = options
        .keep_recent
        .unwrap_or_else(|| fixture.manifest.get("keep_recent_tokens_hint").and_then(Value::as_f64).unwrap_or(20000.0));
    let mut compaction_settings = default_compaction_settings();
    compaction_settings.enabled = true;
    compaction_settings.reserve_tokens = 16384.0;
    compaction_settings.keep_recent_tokens = keep_recent;
    compaction_settings.summary_update_policy = None;
    let threshold = f64::min(
        MAX_COMPACTION_CONTEXT_TOKENS,
        model.context_window - compaction_settings.reserve_tokens,
    );
    let trigger_usage_total = threshold + 64.0;
    let expected_fire = should_compact_for_model(trigger_usage_total, &model, &compaction_settings);

    let settings_json = json!({
        "autoRefine": {"enabled": false},
        "retry": {"enabled": false},
        "compaction": {
            "enabled": true,
            "reserveTokens": compaction_settings.reserve_tokens,
            "keepRecentTokens": keep_recent,
        },
        "telemetry": {"enabled": false},
        "agentTraces": {"enabled": false},
        "quietStartup": true,
    });
    let settings = Arc::new(Mutex::new(SettingsManager::in_memory(
        settings_json
            .as_object()
            .cloned()
            .ok_or("settings object")?,
    )));
    let auth_storage = Arc::new(tokio::sync::Mutex::new(AuthStorage::in_memory(
        AuthStorageData::new(),
        private_auth_options(),
    )));
    auth_storage
        .lock()
        .await
        .set_runtime_api_key(&model.provider, HARNESS_API_KEY);
    let model_registry = Arc::new(Mutex::new(ModelRegistry::in_memory(AuthStorage::in_memory(
        AuthStorageData::new(),
        private_auth_options(),
    ))));
    model_registry
        .lock()
        .unwrap()
        .set_runtime_api_key(&model.provider, HARNESS_API_KEY);

    let session_manager = Arc::new(Mutex::new(SessionManager::create(
        &cwd.to_string_lossy(),
        Some(&session_dir.to_string_lossy()),
    )?));

    let loader_options = DefaultResourceLoaderOptions {
        cwd: cwd.to_string_lossy().to_string(),
        agent_dir: agent_dir.to_string_lossy().to_string(),
        no_extensions: true,
        no_skills: true,
        no_prompt_templates: true,
        no_themes: true,
        no_context_files: true,
        bundled_skills_dir: Some(None),
        settings_manager: Some(Arc::clone(&settings)),
        ..Default::default()
    };
    let services = create_agent_session_services(CreateAgentSessionServicesOptions {
        cwd: cwd.to_string_lossy().to_string(),
        agent_dir: Some(agent_dir.to_string_lossy().to_string()),
        auth_storage: Some(Arc::clone(&auth_storage)),
        settings_manager: Some(Arc::clone(&settings)),
        model_registry: Some(Arc::clone(&model_registry)),
        extension_flag_values: None,
        no_builtin_herdr_reporter: Some(true),
        telemetry_disabled: Some(true),
        resource_loader_options: Some(loader_options),
    })
    .await?;
    let services = Arc::new(services);

    let created = create_agent_session_from_services(CreateAgentSessionFromServicesOptions {
        services: Arc::clone(&services),
        session_manager: Arc::clone(&session_manager),
        session_start_event: None,
        creation: AgentSessionCreationOptions {
            model: Some(model.clone()),
            no_tools: Some("all".to_string()),
            prewarm_ipython_kernel: Some(false),
            telemetry_disabled: Some(true),
            include_goals: Some(false),
            include_compact_skill: Some(false),
            ..Default::default()
        },
    })
    .await?;
    let session = created.session;
    // The system prompt embeds the conversation-log path with this random id.
    *state.session_id.lock().unwrap() = Some(session.session_id());

    // Recorder injection seam: phase rows for Prepare/Native/History/Prefix/
    // Persist/Restore/Total + provider attempts.
    let recorder = Arc::new(HarnessRecorder::new());
    session.agent.set_performance_metrics(Some(AgentLoopPerformanceMetrics::new(
        recorder.clone(),
    )));

    let observed = Arc::new(ObservedCompaction::default());
    {
        let observed = Arc::clone(&observed);
        session.subscribe(Arc::new(move |event: AgentSessionEvent| match event {
            AgentSessionEvent::CompactionStart { reason, .. } => {
                observed.starts.lock().unwrap().push(reason);
            }
            AgentSessionEvent::CompactionEnd {
                reason,
                result,
                aborted,
                will_retry,
                error_message,
                ..
            } => {
                observed.ends.lock().unwrap().push(json!({
                    "reason": reason,
                    "aborted": aborted,
                    "will_retry": will_retry,
                    "error_message": error_message,
                    "tokens_before": result.as_ref().map(|result| result.tokens_before),
                    "first_kept_entry_id": result.as_ref().map(|result| result.first_kept_entry_id.clone()),
                }));
            }
            _ => {}
        }));
    }

    // Roots privacy gate (mirrors the parity suite's V00 posture): a case whose
    // effective roots escape the private root fails instead of merely recording.
    let artifact_dir = session_manager
        .lock()
        .unwrap()
        .get_session_artifact_dir()
        .unwrap_or_default();
    let roots_private = Path::new(&artifact_dir).starts_with(&case_root);
    if !roots_private {
        session.dispose_async(Some(false)).await;
        return Err(format!(
            "effective roots are not private: artifact dir {} escapes case root {}",
            artifact_dir,
            case_root.display()
        ));
    }

    // Seed the durable transcript and the live state without provider calls.
    let messages = fixture.to_agent_messages(&model);
    let seed_started = Instant::now();
    {
        let mut manager = session.session_manager.lock().unwrap();
        for message in &messages {
            manager
                .append_message(message.clone())
                .map_err(|error| format!("append_message: {error}"))?;
        }
    }
    {
        let mut agent_state = session.agent.state();
        agent_state.messages = session.build_session_context().messages;
        session.agent.set_state(agent_state);
    }
    // Snapshot the exact durable seed bytes, not get_branch()'s rehydrated
    // public values. Disk serialization may replace repeated tool-detail text
    // with an inline reference; get_branch reconstructs it for model/UI use.
    let seeded_session_file = {
        let mut manager = session.session_manager.lock().unwrap();
        manager.flush_now().map_err(|error| stage_error("durable-seed", format!("flush seeded session: {error}")))?;
        manager.get_session_file().ok_or_else(|| stage_error("durable-seed", "seeded durable session has no file path"))?
    };
    let seeded_bytes = std::fs::read(&seeded_session_file)
        .map_err(|error| stage_error("durable-seed", format!("read exact seeded JSONL: {error}")))?;
    let seeded_graph = raw_durable_graph(&seeded_bytes)?;
    ensure_stage(
        seeded_graph.entry_count == seeded_graph.branch.len(),
        "durable-seed",
        "seed contains entries outside its selected durable branch",
    )?;
    let seeded_branch = seeded_graph.branch;
    let seeded_message_counts = durable_message_counts(&seeded_branch)?;
    ensure_stage(
        seeded_message_counts.user + seeded_message_counts.assistant
            + seeded_message_counts.tool_result + seeded_message_counts.other == messages.len(),
        "durable-seed",
        "durable seed count differs from the loaded fixture messages",
    )?;
    let seeded_evidence = json!({
        "fixture_message_count": messages.len(),
        "branch_entry_count": seeded_branch.len(),
        "message_counts": {
            "user": seeded_message_counts.user,
            "assistant": seeded_message_counts.assistant,
            "tool_results": seeded_message_counts.tool_result,
            "other": seeded_message_counts.other,
        },
        "branch_sha256": semantic_sha256(&seeded_branch)?,
    });
    let seed_ms = elapsed_ms(seed_started);

    // Drive the session so the threshold check fires at turn end. The reply's
    // usage is set to just cross the real threshold formula.
    state.replies.lock().unwrap().push(AssistantMessage {
        content: vec![ContentBlock::Text(TextContent::new(
            "harness replay trigger turn",
        ))],
        api: api.clone(),
        provider: provider.clone(),
        model: model.id.clone(),
        stop_reason: "stop".to_string(),
        timestamp: now_ms(),
        usage: Usage {
            input: (trigger_usage_total - 16.0).max(1.0),
            output: 16.0,
            total_tokens: trigger_usage_total,
            ..Usage::zero()
        },
        ..Default::default()
    });
    let turn_started = Instant::now();
    let prompt_result = session
        .prompt("harness replay trigger turn", None)
        .await
        .map_err(|error| format!("prompt: {error}"))?;
    let _ = prompt_result;
    session
        .wait_for_headless_idle()
        .await
        .map_err(|error| format!("idle: {error}"))?;
    let turn_ms = elapsed_ms(turn_started);

    // Outcome + digests.
    let branch = session.session_manager.lock().unwrap().get_branch(None);
    let compaction_entries: Vec<&Map<String, Value>> = branch
        .iter()
        .filter(|entry| entry.get("type").and_then(Value::as_str) == Some("compaction"))
        .collect();
    let last_compaction = compaction_entries.last().cloned();
    let summary_text = last_compaction
        .and_then(|entry| entry.get("summary"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let first_kept_entry_id = last_compaction
        .and_then(|entry| entry.get("firstKeptEntryId"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let tokens_before = last_compaction
        .and_then(|entry| entry.get("tokensBefore"))
        .and_then(Value::as_f64);
    let live = session.messages();
    let normalized = normalize_messages(&live);
    let retained_state_sha256 = sha256_hex(
        serde_json::to_string(&normalized)
            .map_err(|error| error.to_string())?
            .as_bytes(),
    );
    // Fresh reopen parity uses the durable semantic projection. In particular,
    // an intentional summary failure has a live-only auto_compaction_failure_notice;
    // requiring that transient UI notice to survive reopen would reject valid
    // persistence. The live digest above and notice count below remain intact.
    let writer_restored_messages = session
        .session_manager
        .lock()
        .unwrap()
        .build_session_context(None)
        .messages;
    let normalized_for_reopen = normalize_messages(&writer_restored_messages);
    let reopen_context_sha256 = semantic_sha256(&normalized_for_reopen)?;
    let transient_failure_notice_count = normalized
        .as_array()
        .map(|messages| {
            messages
                .iter()
                .filter(|message| {
                    message.get("role").and_then(Value::as_str) == Some("custom")
                        && message.get("custom_type").and_then(Value::as_str)
                            == Some("auto_compaction_failure_notice")
                })
                .count()
        })
        .unwrap_or(0);
    let summary_sha256 = sha256_hex(summary_text.as_bytes());
    let pairing = pairing_stats(&live);
    let session_file = session
        .session_file()
        .ok_or_else(|| "durable replay has no session JSONL path".to_string())?;

    let events = recorder.events.lock().unwrap().clone();
    let starts = observed.starts.lock().unwrap().clone();
    let ends = observed.ends.lock().unwrap().clone();
    let provider_calls = state.calls.lock().unwrap().clone();
    let summary_calls = state.summary_calls.load(Ordering::SeqCst);
    let turn_calls = state.turn_calls.load(Ordering::SeqCst);
    let fired = !starts.is_empty();
    // Per-call request payload digests, sorted, plus an aggregate: a
    // chunking/serialization regression that preserves chunk counts and sizes
    // still changes these. The list is SORTED because the split-turn history
    // and turn-prefix summary calls run concurrently (tokio::try_join!), so
    // their completion order — and with it the record order — is not
    // deterministic; the ordered per-call log stays in `provider_calls`.
    let mut provider_request_digests: Vec<String> = provider_calls
        .iter()
        .map(|call| {
            call.get("payload_sha256")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string()
        })
        .collect();
    provider_request_digests.sort();
    let provider_requests_sha256 = sha256_hex(provider_request_digests.join("\n").as_bytes());

    session.dispose_async(Some(false)).await;

    let session_file_bytes = std::fs::read(&session_file)
        .map_err(|error| format!("read durable session JSONL after dispose: {error}"))?;
    let session_file_text = String::from_utf8(session_file_bytes.clone())
        .map_err(|error| format!("durable session JSONL is not UTF-8: {error}"))?;
    let raw_durable_graph = raw_durable_graph(&session_file_bytes)?;
    let raw_durable_state = raw_durable_graph.branch;
    let turn_delta = validate_durable_turn_delta(
        &seeded_branch,
        &raw_durable_state,
        raw_durable_graph.entry_count,
        options.expected_outcome,
    )?;
    validate_checkpoint_links(&raw_durable_state)?;
    let raw_durable_pairing = raw_durable_pairing(&raw_durable_state)?;
    let restored_pairing = pairing_stats(&writer_restored_messages);
    let protected_state = protected_state_projection(&raw_durable_state);
    let expected_admission = json!({
        "summary_calls": options.expected_summary_calls,
        "turn_calls": 1,
        "injected_summary_failure": options.summary_failure,
    });
    let reopen_evidence = json!({
        "expected_admission": expected_admission,
        "seeded_state": seeded_evidence,
        "turn_delta": turn_delta,
        "raw_durable_pairing": pairing_value(&raw_durable_pairing),
        "restored_pairing": pairing_value(&restored_pairing),
        "protected_state": protected_state,
    });
    let reopen_started = Instant::now();
    let fresh_process_reopen = fresh_process_reopen(
        &case_root,
        case_counter,
        &session_file,
        &raw_durable_state,
        &normalized_for_reopen,
        &reopen_evidence,
        options.timeout_secs,
    )
    .await?;
    let fresh_reopen_ms = elapsed_ms(reopen_started);
    let session_file_stats = json!({
        "path": session_file,
        "bytes": session_file_bytes.len(),
        "lines": session_file_text.lines().count(),
    });

    let pairing_json = pairing_value(&pairing);
    let determinism_key = json!({
        "fixture": fixture.name,
        "retained_state_sha256": retained_state_sha256,
        "summary_sha256": summary_sha256,
        "pairing": pairing_json,
        "compaction_entries": compaction_entries.len(),
        "tokens_before": tokens_before,
        "trigger_usage_total": trigger_usage_total,
        "keep_recent_tokens": keep_recent,
        "expected_fire": expected_fire,
        "fired": fired,
        "summary_calls": summary_calls,
        "turn_calls": turn_calls,
        "live_message_count": live.len(),
        "starts": starts.len(),
        "ends": ends.len(),
        "provider_request_digests": provider_request_digests,
        "provider_requests_sha256": provider_requests_sha256,
    });
    let case_result = json!({
        "case": fixture.name,
        "fixture_manifest": fixture.manifest,
        "roots_private": roots_private,
        "settings": {
            "reserve_tokens": compaction_settings.reserve_tokens,
            "keep_recent_tokens": keep_recent,
            "context_window": model.context_window,
            "model_max_tokens": model.max_tokens,
        },
        "trigger": {
            "threshold_tokens": threshold,
            "trigger_usage_total": trigger_usage_total,
            "expected_fire": expected_fire,
            "fired": fired,
            "start_reasons": starts,
        },
        "stage_totals_ms": {
            "seed": seed_ms,
            "turn_and_compaction": turn_ms,
            "fresh_process_reopen": fresh_reopen_ms,
        },
        "phases": phase_rows(&events),
        "provider_attempts": provider_attempt_rows(&events),
        "expected_admission": expected_admission,
        "seeded_state": seeded_evidence,
        "durable_turn_delta": turn_delta,
        "raw_durable_pairing": pairing_value(&raw_durable_pairing),
        "restored_pairing": pairing_value(&restored_pairing),
        "protected_state": protected_state,
        "provider_calls": provider_calls,
        "provider_call_counts": {
            "summary": summary_calls,
            "turn": turn_calls,
        },
        "compaction_end_events": ends,
        "outcome": {
            "compaction_entries": compaction_entries.len(),
            "first_kept_entry_id": first_kept_entry_id,
            "tokens_before": tokens_before,
            "summary_chars": summary_text.chars().count(),
            "live_message_count": live.len(),
            "writer_restored_message_count": writer_restored_messages.len(),
            "live_minus_restored_message_count": live.len() as i64 - writer_restored_messages.len() as i64,
            "transient_failure_notice_count": transient_failure_notice_count,
            "branch_entry_count": branch.len(),
        },
        "digests": {
            "retained_state_sha256": retained_state_sha256,
            "reopen_context_sha256": reopen_context_sha256,
            "summary_sha256": summary_sha256,
            "pairing": pairing_json,
            "provider_requests_sha256": provider_requests_sha256,
        },
        "session_file": session_file_stats,
        "fresh_process_reopen": fresh_process_reopen,
        "determinism_key": determinism_key,
    });
    Ok(case_result)
}

fn env_stamp(cli: &Cli) -> Value {
    let computed = exe_sha256();
    let stamp = cli
        .optional("stamp")
        .or_else(|| computed.clone().ok());
    json!({
        "exe_sha256": stamp,
        "exe_sha256_source": if cli.optional("stamp").is_some() { "--stamp" } else if computed.is_ok() { "computed-from-running-exe" } else { "unavailable" },
        "git": cli.optional("git"),
        "pid": std::process::id(),
        "argv": std::env::args().collect::<Vec<String>>(),
    })
}

fn runtime_stamp() -> Value {
    let available = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1);
    json!({
        "charcount": {
            "requested_mode": std::env::var("PRIME_AGENT_COMPACT_RUST_COUNT").unwrap_or_else(|_| "auto".to_string()),
            "effective_backend": pi_coding_agent::core::compaction::charcount::backend().name(),
            "dispatch_diagnostic": pi_coding_agent::core::compaction::charcount::diagnostic(),
        },
        "parallel": {
            "requested_mode": std::env::var("PRIME_AGENT_COMPACTION_PARALLEL").unwrap_or_else(|_| "auto".to_string()),
            "available_parallelism": available,
            "effective_workers": available.min(6),
            "fallback_count": pi_coding_agent::core::compaction::parallel::parallel_fallback_count(),
        }
    })
}



fn reopen_expectation_value<'a>(expected: &'a Value, pointer: &str) -> Result<&'a str, String> {
    required_replay_value(expected, pointer)?
        .as_str()
        .ok_or_else(|| format!("reopen expectation {pointer} is not text"))
}

fn reopen_expectation_json<'a>(expected: &'a Value, pointer: &str) -> Result<&'a Value, String> {
    expected
        .pointer(pointer)
        .ok_or_else(|| format!("reopen expectation lacks {pointer}"))
}

fn reopen_projection_difference(
    expected: &Value,
    actual: &Value,
    pointer: &str,
    stage: &str,
) -> Result<(), String> {
    if let Some(difference) = first_json_difference(Some(expected), Some(actual), pointer) {
        return Err(stage_error(stage, format!("projection mismatch: {difference}")));
    }
    Ok(())
}

fn verify_reopen_projection(expected: &Value) -> Result<Value, String> {
    let session_file = reopen_expectation_value(expected, "/session_file")?;
    let input = PathBuf::from(session_file);
    ensure_stage(input.is_file(), "session-open", format!("reopen input is not a readable file: {session_file}"))?;
    let session_root = input
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| stage_error("report-confinement", "reopen input is not under sessions/<file>"))?;
    let expected_writer = required_replay_u64(expected, "/writer_pid")?;
    ensure_stage(
        expected_writer != std::process::id() as u64,
        "fresh-process",
        "verify-reopen must run in a fresh OS process",
    )?;
    let before = std::fs::read(&input)
        .map_err(|error| stage_error("session-open", format!("read reopen input {session_file}: {error}")))?;
    ensure_stage(!before.is_empty(), "session-open", "reopen input is empty")?;
    let before_sha = sha256_hex(&before);
    ensure_stage(
        before_sha == reopen_expectation_value(expected, "/input_sha256")?,
        "fixture-input-hash",
        "reopen input differs from the writer's closed-session bytes",
    )?;

    // This is deliberately before SessionManager::open or any selected-branch
    // operation. Off-branch corruption must not be hidden by the production UI index.
    let graph = raw_durable_graph(&before)?;
    let raw_branch = graph.branch;
    let raw_graph_sha = semantic_sha256(&raw_branch)?;
    let expected_durable = reopen_expectation_json(expected, "/durable_branch")?;
    let actual_durable = serde_json::to_value(&raw_branch).map_err(|error| error.to_string())?;
    reopen_projection_difference(expected_durable, &actual_durable, "/durable_branch", "raw-durable-parity")?;
    ensure_stage(
        raw_graph_sha == reopen_expectation_value(expected, "/durable_sha256")?,
        "raw-durable-hash",
        "reopened durable branch hash differs from the writer",
    )?;
    ensure_stage(
        graph.entry_count == required_replay_u64(expected, "/graph_entry_count")? as usize,
        "full-graph-integrity",
        "full graph size differs from the writer",
    )?;
    validate_checkpoint_links(&raw_branch)?;

    let expected_protected = reopen_expectation_json(expected, "/protected_state")?;
    let actual_protected = protected_state_projection(&raw_branch);
    reopen_projection_difference(
        expected_protected.pointer("/checkpoints").ok_or_else(|| stage_error("checkpoint", "missing checkpoint expectation"))?,
        actual_protected.pointer("/checkpoints").unwrap_or(&Value::Null),
        "/protected_state/checkpoints",
        "checkpoint",
    )?;
    reopen_projection_difference(
        expected_protected.pointer("/custom_messages").ok_or_else(|| stage_error("protected-custom-state", "missing custom-state expectation"))?,
        actual_protected.pointer("/custom_messages").unwrap_or(&Value::Null),
        "/protected_state/custom_messages",
        "protected-custom-state",
    )?;
    reopen_projection_difference(
        expected_protected.pointer("/session_controls").ok_or_else(|| stage_error("protected-custom-state", "missing control-state expectation"))?,
        actual_protected.pointer("/session_controls").unwrap_or(&Value::Null),
        "/protected_state/session_controls",
        "protected-custom-state",
    )?;
    let raw_pairing = raw_durable_pairing(&raw_branch)?;
    let expected_raw_pairing = reopen_expectation_json(expected, "/raw_durable_pairing")?;
    reopen_projection_difference(expected_raw_pairing, &pairing_value(&raw_pairing), "/raw_durable_pairing", "durable-tool-pairing")?;
    validate_pairing_value(expected_raw_pairing, "durable-tool-pairing")?;

    let session_dir = input
        .parent()
        .ok_or_else(|| stage_error("session-open", "reopen input has no session directory"))?;
    let manager = SessionManager::open(session_file, Some(&session_dir.to_string_lossy()), None)
        .map_err(|error| stage_error("session-open", error))?;
    let rehydrated_branch = manager.get_branch(None);
    let rehydrated_sha = semantic_sha256(&rehydrated_branch)?;
    let restored_messages = manager.build_session_context(None).messages;
    let normalized_restored_messages = normalize_messages(&restored_messages);
    let retained_sha = semantic_sha256(&normalized_restored_messages)?;
    let restored_message_count = restored_messages.len();
    let branch_entry_count = raw_branch.len();
    drop(manager);

    let expected_normalized = reopen_expectation_json(expected, "/normalized_messages")?;
    reopen_projection_difference(
        expected_normalized,
        &normalized_restored_messages,
        "/normalized_messages",
        "normalized-message-parity",
    )?;
    ensure_stage(
        retained_sha == reopen_expectation_value(expected, "/retained_state_sha256")?,
        "retained-state-hash",
        "reopened restored message hash differs from the writer",
    )?;
    ensure_stage(
        restored_message_count == required_replay_u64(expected, "/restored_message_count")? as usize,
        "normalized-message-count",
        "restored message count differs from the writer",
    )?;
    let restored_pairing = pairing_stats(&restored_messages);
    let expected_restored_pairing = reopen_expectation_json(expected, "/restored_pairing")?;
    reopen_projection_difference(expected_restored_pairing, &pairing_value(&restored_pairing), "/restored_pairing", "restored-tool-pairing")?;
    validate_pairing_value(expected_restored_pairing, "restored-tool-pairing")?;

    // Test-only fault injection for self-test-r1's disposable session. A normal
    // expectation never sets this marker. The guard below must still reject the
    // mutation even though the writer's input SHA matched on entry.
    if expected.get("selftest_mutate_input_after_open").and_then(Value::as_bool) == Some(true) {
        let mut mutated = before.clone();
        mutated.extend_from_slice(b"selftest reader mutation\n");
        std::fs::write(&input, &mutated)
            .map_err(|error| stage_error("reader-mutation", format!("write selftest reader mutation: {error}")))?;
    }
    let after = std::fs::read(&input)
        .map_err(|error| stage_error("raw-byte-integrity", format!("read reopen input after open/drop {session_file}: {error}")))?;
    ensure_stage(after == before, "raw-byte-integrity", "read-only reopen changed the session JSONL bytes")?;
    // Revalidate after close too: the opener must not hide or normalize an
    // initially malformed graph while keeping the selected branch superficially valid.
    raw_durable_graph(&after)?;
    Ok(json!({
        "schema": 1,
        "operation": "verify-reopen",
        "reader": {
            "pid": std::process::id(),
            "exe_sha256": exe_sha256()?,
        },
        "session_root": session_root,
        "session_file": session_file,
        "input_bytes_before_after": before.len(),
        "input_sha256_before_after": before_sha,
        "durable_sha256": raw_graph_sha,
        "graph_entry_count": graph.entry_count,
        "rehydrated_branch_sha256": rehydrated_sha,
        "retained_state_sha256": retained_sha,
        "branch_entry_count": branch_entry_count,
        "restored_message_count": restored_message_count,
        "raw_durable_pairing": pairing_value(&raw_pairing),
        "restored_pairing": pairing_value(&restored_pairing),
        "additional_user_or_provider_turns": 0,
        "read_only_reopen": true,
        "fresh_process": true,
        "parity": "full graph + exact raw durable branch + normalized durable build_session_context + protected custom/checkpoint state",
    }))
}
fn run_verify_reopen(cli: &Cli) -> Result<i32, String> {
    let expectation_path = PathBuf::from(cli.required("expectation")?);
    let out = PathBuf::from(cli.required("out")?);
    let expected: Value = serde_json::from_str(
        &std::fs::read_to_string(&expectation_path)
            .map_err(|error| format!("read expectation: {error}"))?,
    )
    .map_err(|error| format!("parse expectation: {error}"))?;
    let input = PathBuf::from(reopen_expectation_value(&expected, "/session_file")?);
    let session_root = input
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| stage_error("report-confinement", "reopen expectation has no case root"))?;
    if out.starts_with(session_root) {
        return Err(stage_error("report-confinement", "reopen report must be outside the measured case root"));
    }
    let report = verify_reopen_projection(&expected)?;
    write_new_json(&out, &report)?;
    Ok(0)
}

async fn execute_verify_reopen(
    expectation_path: &Path,
    report_path: &Path,
    timeout_secs: u64,
) -> Result<Value, String> {
    let exe = std::env::current_exe().map_err(|error| error.to_string())?;
    let mut command = tokio::process::Command::new(exe);
    command
        .arg("verify-reopen")
        .arg("--expectation")
        .arg(expectation_path)
        .arg("--out")
        .arg(report_path)
        .kill_on_drop(true);
    let output = tokio::time::timeout(Duration::from_secs(timeout_secs), command.output())
        .await
        .map_err(|_| format!("fresh-process reopen timed out after {timeout_secs}s"))?
        .map_err(|error| format!("spawn fresh-process reopen: {error}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(format!(
            "fresh-process reopen failed ({}): {} {}",
            output.status,
            stderr.chars().take(4000).collect::<String>(),
            stdout.chars().take(1000).collect::<String>()
        ));
    }
    let report: Value = serde_json::from_str(
        &std::fs::read_to_string(report_path)
            .map_err(|error| format!("read reopen report: {error}"))?,
    )
    .map_err(|error| format!("parse reopen report: {error}"))?;
    if required_replay_u64(&report, "/reader/pid")? == std::process::id() as u64 {
        return Err("reopen evidence was not produced by a fresh process".to_string());
    }
    let reader_exe = reopen_expectation_value(&report, "/reader/exe_sha256")?;
    if reader_exe != exe_sha256()? {
        return Err("reader executable identity differs from replay writer".to_string());
    }
    if !required_replay_bool(&report, "/read_only_reopen")?
        || !required_replay_bool(&report, "/fresh_process")?
        || required_replay_u64(&report, "/additional_user_or_provider_turns")? != 0
    {
        return Err("reopen evidence contract is incomplete".to_string());
    }
    Ok(report)
}

async fn fresh_process_reopen(
    case_root: &Path,
    case_counter: &AtomicU64,
    session_file: &str,
    durable_branch: &[Value],
    normalized_messages: &Value,
    reopen_evidence: &Value,
    timeout_secs: u64,
) -> Result<Value, String> {
    let verification_root = case_root
        .parent()
        .ok_or_else(|| "case root has no parent for external verification".to_string())?
        .join("r1-verification");
    let sequence = case_counter.fetch_add(1, Ordering::SeqCst);
    let stem = format!("reopen-{}-{sequence}", std::process::id());
    let expectation_path = verification_root.join(format!("{stem}-expectation.json"));
    let report_path = verification_root.join(format!("{stem}-report.json"));
    let mut expected = json!({
        "schema": 1,
        "writer_pid": std::process::id(),
        "session_file": session_file,
        "input_sha256": sha256_hex(&std::fs::read(session_file).map_err(|error| format!("read closed session: {error}"))?),
        "durable_sha256": semantic_sha256(durable_branch)?,
        "retained_state_sha256": semantic_sha256(normalized_messages)?,
        "durable_branch": durable_branch,
        "normalized_messages": normalized_messages,
        "restored_message_count": normalized_messages.as_array().map_or(0, Vec::len),
        "graph_entry_count": durable_branch.len(),
    });
    let evidence = reopen_evidence.as_object().ok_or_else(|| stage_error("admission-evidence", "reopen evidence is not an object"))?;
    expected.as_object_mut().ok_or_else(|| stage_error("admission-evidence", "reopen expectation is not an object"))?
        .extend(evidence.clone());
    write_new_json(&expectation_path, &expected)?;

    let report = execute_verify_reopen(&expectation_path, &report_path, timeout_secs).await?;
    Ok(json!({
        "expectation": expectation_path,
        "report_path": report_path,
        "report": report,
    }))
}

fn required_replay_value<'a>(case: &'a Value, pointer: &str) -> Result<&'a Value, String> {
    case.pointer(pointer)
        .ok_or_else(|| stage_error("admission-evidence", format!("missing replay evidence field {pointer}")))
}

fn required_replay_bool(case: &Value, pointer: &str) -> Result<bool, String> {
    required_replay_value(case, pointer)?
        .as_bool()
        .ok_or_else(|| stage_error("admission-evidence", format!("replay evidence {pointer} is not boolean")))
}

fn required_replay_u64(case: &Value, pointer: &str) -> Result<u64, String> {
    required_replay_value(case, pointer)?
        .as_u64()
        .ok_or_else(|| stage_error("admission-evidence", format!("replay evidence {pointer} is not a nonnegative integer")))
}

fn required_replay_nonempty_string(case: &Value, pointer: &str) -> Result<(), String> {
    let value = required_replay_value(case, pointer)?
        .as_str()
        .ok_or_else(|| stage_error("admission-evidence", format!("replay evidence {pointer} is not text")))?;
    if value.is_empty() {
        return Err(stage_error("admission-evidence", format!("replay evidence {pointer} is empty")));
    }
    Ok(())
}

fn validate_pairing_value(pairing: &Value, stage: &str) -> Result<(), String> {
    let tool_calls = required_replay_u64(pairing, "/tool_calls")?;
    let tool_results = required_replay_u64(pairing, "/tool_results")?;
    let matched = required_replay_u64(pairing, "/matched_results")?;
    let unmatched = required_replay_u64(pairing, "/unmatched_results")?;
    let calls_without_result = required_replay_u64(pairing, "/calls_without_result")?;
    let duplicate_calls = required_replay_u64(pairing, "/duplicate_tool_call_ids")?;
    let duplicate_results = required_replay_u64(pairing, "/duplicate_tool_result_ids")?;
    ensure_stage(
        tool_calls == tool_results && matched == tool_results && unmatched == 0
            && calls_without_result == 0 && duplicate_calls == 0 && duplicate_results == 0,
        stage,
        format!(
            "tool pairing is not one-to-one: calls={tool_calls} results={tool_results} matched={matched} unmatched={unmatched} missing={calls_without_result} duplicate_call_ids={duplicate_calls} duplicate_result_ids={duplicate_results}"
        ),
    )
}

fn validate_provider_calls(case: &Value, expected: ReplayOutcomeExpectation) -> Result<(), String> {
    let expected_summaries = required_replay_u64(case, "/expected_admission/summary_calls")?;
    let expected_turns = required_replay_u64(case, "/expected_admission/turn_calls")?;
    let summaries = required_replay_u64(case, "/provider_call_counts/summary")?;
    let turns = required_replay_u64(case, "/provider_call_counts/turn")?;
    ensure_stage(
        summaries == expected_summaries,
        "summary-call-count",
        format!("expected exactly {expected_summaries} summary calls, observed {summaries}"),
    )?;
    ensure_stage(
        turns == expected_turns && expected_turns == 1,
        "provider-turn-count",
        format!("expected exactly one ordinary provider turn, observed {turns}"),
    )?;
    let calls = required_replay_value(case, "/provider_calls")?
        .as_array()
        .ok_or_else(|| stage_error("provider-call-record", "provider_calls is not an array"))?;
    ensure_stage(
        calls.len() as u64 == summaries + turns,
        "provider-call-record",
        format!("provider row count {} differs from typed counters {}", calls.len(), summaries + turns),
    )?;
    let injected_failure = match expected {
        ReplayOutcomeExpectation::SummaryFailure => Some(
            required_replay_value(case, "/expected_admission/injected_summary_failure")?
                .as_str()
                .filter(|error| !error.is_empty())
                .ok_or_else(|| stage_error("summary-error-admission", "summary failure requires a nonempty injected provider error"))?,
        ),
        ReplayOutcomeExpectation::SuccessfulCompaction => None,
    };
    let mut summary_rows = 0u64;
    let mut turn_rows = 0u64;
    let mut seen_sequences = HashSet::<(bool, u64)>::new();
    for (index, call) in calls.iter().enumerate() {
        let is_summary = call.get("is_summary_call").and_then(Value::as_bool).ok_or_else(|| {
            stage_error("provider-call-record", format!("provider row {index} has no Boolean is_summary_call"))
        })?;
        if is_summary {
            summary_rows += 1;
        } else {
            turn_rows += 1;
        }
        let seq = call.get("seq").and_then(Value::as_u64).filter(|seq| *seq > 0).ok_or_else(|| {
            stage_error("provider-call-record", format!("provider row {index} has no positive sequence"))
        })?;
        ensure_stage(
            seen_sequences.insert((is_summary, seq)),
            "provider-call-record",
            format!("duplicate provider sequence for summary={is_summary}, seq={seq}"),
        )?;
        required_replay_nonempty_string(call, "/response_sha256")?;
        required_replay_nonempty_string(call, "/payload_sha256")?;
        let stop_reason = call.get("stop_reason").and_then(Value::as_str).ok_or_else(|| {
            stage_error("provider-outcome", format!("provider row {index} has no typed stop_reason"))
        })?;
        let provider_error = match call.get("error_message") {
            None | Some(Value::Null) => None,
            Some(Value::String(error)) if error.is_empty() => None,
            Some(Value::String(error)) => Some(error.as_str()),
            Some(_) => return Err(stage_error("provider-outcome", format!("provider row {index} has a non-string error_message"))),
        };
        if is_summary {
            match injected_failure {
                Some(injected) => {
                    ensure_stage(
                        stop_reason == "error" && provider_error == Some(injected),
                        "summary-provider-outcome",
                        format!("summary provider row {index} did not return the exact injected error"),
                    )?;
                }
                None => {
                    ensure_stage(
                        stop_reason == "stop" && provider_error.is_none(),
                        "summary-provider-outcome",
                        format!("successful summary provider row {index} is not a clean stop"),
                    )?;
                }
            }
        } else {
            ensure_stage(
                stop_reason == "stop" && provider_error.is_none(),
                "ordinary-provider-outcome",
                format!("ordinary provider row {index} failed or stopped unexpectedly"),
            )?;
        }
    }
    ensure_stage(
        summary_rows == expected_summaries && turn_rows == expected_turns,
        "provider-call-record",
        format!("typed provider rows summary={summary_rows} turn={turn_rows} do not match the exact expectation"),
    )
}

fn validate_replay_case(
    case: &Value,
    expected: ReplayOutcomeExpectation,
) -> Result<(), String> {
    ensure_stage(required_replay_bool(case, "/roots_private")?, "private-roots", "session roots escaped the private case root")?;
    ensure_stage(required_replay_bool(case, "/trigger/expected_fire")?, "expected-fire", "fixture threshold unexpectedly did not expect compaction")?;
    ensure_stage(required_replay_bool(case, "/trigger/fired")?, "expected-fire", "expected compaction did not fire")?;
    ensure_stage(
        required_replay_u64(case, "/determinism_key/starts")? == 1
            && required_replay_u64(case, "/determinism_key/ends")? == 1,
        "compaction-event-pair",
        "expected exactly one compaction start/end pair",
    )?;
    // Keep the original ordinary-turn gate and add the independently supplied exact budget.
    ensure_stage(
        required_replay_u64(case, "/provider_call_counts/turn")? == 1,
        "provider-turn-count",
        "expected exactly one ordinary provider turn and no synthetic extra turn",
    )?;
    ensure_stage(
        required_replay_u64(case, "/provider_call_counts/summary")? > 0,
        "summary-call-count",
        "compaction fired without a summary request",
    )?;
    ensure_stage(
        required_replay_u64(case, "/session_file/bytes")? > 0
            && required_replay_u64(case, "/session_file/lines")? > 0,
        "durable-session",
        "durable session JSONL is empty or unreadable",
    )?;
    required_replay_nonempty_string(case, "/session_file/path")?;
    for pointer in [
        "/digests/retained_state_sha256",
        "/digests/reopen_context_sha256",
        "/digests/summary_sha256",
        "/digests/provider_requests_sha256",
        "/fresh_process_reopen/report/durable_sha256",
        "/fresh_process_reopen/report/retained_state_sha256",
    ] {
        required_replay_nonempty_string(case, pointer)?;
    }
    validate_pairing_value(required_replay_value(case, "/digests/pairing")?, "live-tool-pairing")?;
    validate_pairing_value(required_replay_value(case, "/raw_durable_pairing")?, "durable-tool-pairing")?;
    validate_pairing_value(required_replay_value(case, "/restored_pairing")?, "restored-tool-pairing")?;
    validate_provider_calls(case, expected)?;

    let delta = required_replay_value(case, "/durable_turn_delta")?;
    ensure_stage(required_replay_bool(delta, "/seed_unchanged")?, "durable-seed", "the single replay turn changed its seeded durable history")?;
    ensure_stage(required_replay_u64(delta, "/added_entry_count")? == 4, "unexpected-durable-turns", "single replay turn must add exactly four durable entries")?;
    ensure_stage(
        required_replay_u64(delta, "/added_message_counts/user")? == 1
            && required_replay_u64(delta, "/added_message_counts/assistant")? == 1
            && required_replay_u64(delta, "/added_message_counts/tool_results")? == 0
            && required_replay_u64(delta, "/added_message_counts/other")? == 0,
        "unexpected-durable-turns",
        "durable turn delta is not exactly one user plus one assistant message",
    )?;
    ensure_stage(
        required_replay_u64(delta, "/graph_entry_count")? == required_replay_u64(delta, "/branch_entry_count")?
            && required_replay_u64(delta, "/branch_entry_count")? == required_replay_u64(case, "/outcome/branch_entry_count")?,
        "unexpected-durable-entry",
        "full graph, selected branch and outcome counts disagree",
    )?;
    ensure_stage(
        required_replay_u64(case, "/fresh_process_reopen/report/restored_message_count")?
            == required_replay_u64(case, "/outcome/writer_restored_message_count")?,
        "normalized-message-count",
        "fresh-process restored message count differs from the writer",
    )?;

    match expected {
        ReplayOutcomeExpectation::SuccessfulCompaction => {
            ensure_stage(required_replay_u64(case, "/outcome/compaction_entries")? == 1, "successful-compaction", "successful-compaction requires exactly one durable compaction entry")?;
            ensure_stage(required_replay_u64(delta, "/added_compaction_entries")? == 1, "checkpoint-count", "successful replay must add exactly one checkpoint")?;
            ensure_stage(required_replay_u64(case, "/outcome/summary_chars")? > 0, "successful-compaction", "successful-compaction returned an empty summary")?;
            required_replay_nonempty_string(case, "/outcome/first_kept_entry_id")?;
            ensure_stage(required_replay_u64(case, "/outcome/live_message_count")? > 0, "successful-compaction", "successful-compaction has an empty restored context")?;
            let end = required_replay_value(case, "/compaction_end_events/0")?;
            ensure_stage(end.get("aborted").and_then(Value::as_bool) == Some(false), "successful-compaction", "successful-compaction ended aborted")?;
            ensure_stage(end.get("will_retry").and_then(Value::as_bool) == Some(false), "successful-compaction", "successful-compaction unexpectedly requires retry")?;
            ensure_stage(
                end.get("error_message").is_none_or(|value| value.is_null() || value.as_str().is_none_or(str::is_empty)),
                "successful-compaction",
                "successful-compaction ended with an error",
            )?;
            ensure_stage(end.get("tokens_before").and_then(Value::as_f64).is_some(), "successful-compaction", "successful-compaction has no tokens_before result")?;
            ensure_stage(required_replay_u64(case, "/outcome/transient_failure_notice_count")? == 0, "summary-error-persistence", "successful-compaction has a transient failure notice")?;
            ensure_stage(
                required_replay_value(case, "/outcome/live_minus_restored_message_count")?.as_i64() == Some(0),
                "normalized-message-count",
                "successful live and durable restored projections have unexpected extra turns",
            )?;
        }
        ReplayOutcomeExpectation::SummaryFailure => {
            ensure_stage(required_replay_u64(case, "/outcome/compaction_entries")? == 0, "summary-failure", "summary-failure unexpectedly committed compaction")?;
            ensure_stage(required_replay_u64(delta, "/added_compaction_entries")? == 0, "checkpoint-count", "summary failure must add no checkpoint")?;
            let injected = required_replay_value(case, "/expected_admission/injected_summary_failure")?.as_str().ok_or_else(|| stage_error("summary-error-admission", "injected failure is not text"))?;
            let error = required_replay_value(case, "/compaction_end_events/0/error_message")?.as_str().ok_or_else(|| stage_error("end-summary-error", "summary-failure ended without an error message"))?;
            let intended = format!("Auto-compaction failed: Summarization failed: {injected}");
            ensure_stage(error == intended, "end-summary-error", format!("summary-failure end error {error:?} does not match the exact injected summarization outcome {intended:?}"))?;
            ensure_stage(required_replay_u64(case, "/outcome/transient_failure_notice_count")? == 1, "summary-error-persistence", "summary-failure does not have exactly one live failure notice")?;
            ensure_stage(
                required_replay_value(case, "/outcome/live_minus_restored_message_count")?.as_i64() == Some(1),
                "summary-error-persistence",
                "summary-failure durable context must contain zero transient notices while live state contains exactly one",
            )?;
        }
    }
    let report = required_replay_value(case, "/fresh_process_reopen/report")?;
    ensure_stage(required_replay_bool(report, "/fresh_process")? && required_replay_bool(report, "/read_only_reopen")?, "fresh-process", "fresh-process and read-only evidence is required")?;
    ensure_stage(required_replay_u64(report, "/additional_user_or_provider_turns")? == 0, "unexpected-durable-turns", "reopen reported an additional user/provider turn")?;
    Ok(())
}
async fn run_replay(cli: &Cli) -> Result<i32, String> {
    let fixture_path = PathBuf::from(cli.required("fixture")?);
    let out = PathBuf::from(cli.required("out")?);
    let cases = discover_fixtures(&fixture_path)?;
    let summary_failure = cli.optional("summary-failure");
    let expected_outcome = match cli.optional("expect-outcome") {
        Some(raw) => ReplayOutcomeExpectation::parse(&raw)?,
        None if summary_failure.is_some() => ReplayOutcomeExpectation::SummaryFailure,
        None => ReplayOutcomeExpectation::SuccessfulCompaction,
    };
    if expected_outcome == ReplayOutcomeExpectation::SuccessfulCompaction && summary_failure.is_some() {
        return Err(
            "--expect-outcome successful-compaction conflicts with --summary-failure".to_string(),
        );
    }
    if expected_outcome == ReplayOutcomeExpectation::SummaryFailure && summary_failure.is_none() {
        return Err("--expect-outcome summary-failure requires --summary-failure MSG".to_string());
    }
    ensure_stage(
        cli.options.contains_key("expect-summary-calls"),
        "cli-expect-summary-calls",
        "--expect-summary-calls N is required and must come from the fixture/options baseline, never the observed call count",
    )?;
    let expected_summary_calls = cli.integer("expect-summary-calls", 1, 1, 1_000_000)?;
    let options = ReplayOptions {
        provider_delay_ms: cli.integer("provider-delay-ms", 0, 0, 3_600_000)? as u64,
        summary_failure,
        expected_outcome,
        expected_summary_calls,
        context_window: cli.positive_number("context-window", 1_000_000.0, 1_000_000_000.0)?,
        keep_recent: match cli.options.contains_key("keep-recent") {
            true => Some(cli.positive_number("keep-recent", 1.0, 1_000_000_000.0)?),
            false => None,
        },
        work: cli
            .optional("work")
            .map(PathBuf::from)
            .unwrap_or_else(|| std::env::temp_dir().join("compaction-replay-work")),
        timeout_secs: cli.integer("timeout-secs", 900, 1, 86_400)? as u64,
    };
    std::fs::create_dir_all(&options.work).map_err(|error| format!("mkdir work: {error}"))?;
    let counter = AtomicU64::new(0);
    let mut case_results: Vec<Value> = Vec::new();
    let mut failures: Vec<String> = Vec::new();
    for case_dir in &cases {
        let fixture = load_fixture(case_dir)?;
        println!("replaying {} ({} records)", fixture.name, fixture.records.len());
        let case = tokio::time::timeout(
            Duration::from_secs(options.timeout_secs),
            run_replay_case(&fixture, &options, &counter),
        )
        .await;
        match case {
            Ok(Ok(value)) => match validate_replay_case(&value, options.expected_outcome) {
                Ok(()) => case_results.push(value),
                Err(error) => {
                    let failure = format!("{}: admission failed: {error}", fixture.name);
                    eprintln!("replay failed: {failure}");
                    failures.push(failure);
                }
            },
            Ok(Err(error)) => {
                let failure = format!("{}: {error}", fixture.name);
                eprintln!("replay failed: {failure}");
                failures.push(failure);
            }
            Err(_) => {
                let failure = format!("{}: timeout after {}s", fixture.name, options.timeout_secs);
                eprintln!("replay failed: {failure}");
                failures.push(failure);
            }
        }
    }
    let output = json!({
        "harness": "compaction_replay",
        "subcommand": "replay",
        "schema": 1,
        "env": env_stamp(cli),
        "runtime": runtime_stamp(),
        "config": {
            "provider_delay_ms": options.provider_delay_ms,
            "summary_failure": options.summary_failure,
            "expect_outcome": options.expected_outcome.as_str(),
            "expect_summary_calls": options.expected_summary_calls,
            "context_window": options.context_window,
            "keep_recent": options.keep_recent,
            "timeout_secs": options.timeout_secs,
        },
        "case_count": case_results.len(),
        "cases": case_results,
        "failures": failures,
    });
    write_new_json(&out, &output)?;
    println!("wrote {} ({} cases, {} failures)", out.display(), output["case_count"], failures.len());
    if failures.is_empty() {
        Ok(0)
    } else {
        Ok(2)
    }
}



// ---------------------------------------------------------------------------
// bench-local subcommand
// ---------------------------------------------------------------------------

fn bench_kernel(
    name: &str,
    detail: Value,
    warmup: usize,
    iters: usize,
    mut kernel: impl FnMut() -> f64,
) -> Value {
    let mut checksum = 0.0f64;
    for _ in 0..warmup {
        checksum += kernel();
    }
    let mut samples: Vec<f64> = Vec::with_capacity(iters);
    let mut allocs: Option<(u64, u64)> = None;
    let before_first = alloc_snapshot();
    for index in 0..iters {
        let started = Instant::now();
        checksum += kernel();
        samples.push(elapsed_ms(started));
        if index == 0 {
            let after_first = alloc_snapshot();
            allocs = Some((
                after_first.0.saturating_sub(before_first.0),
                after_first.1.saturating_sub(before_first.1),
            ));
        }
    }
    let (alloc_count, alloc_bytes) = allocs.unwrap_or((0, 0));
    let stats = timing_stats(samples);
    assert_eq!(stats["n"].as_u64(), Some(iters as u64), "timed sample count must equal requested iterations");
    json!({
        "name": name,
        "detail": detail,
        "warmup": warmup,
        "stats": stats,
        "allocs_first_iter": {
            "count": alloc_count,
            "bytes": alloc_bytes,
        },
        "checksum": checksum,
    })
}

fn bench_kernel_fallible(
    name: &str,
    detail: Value,
    warmup: usize,
    iters: usize,
    mut kernel: impl FnMut() -> Result<f64, String>,
) -> Result<Value, String> {
    ensure_stage(warmup > 0 && iters > 0, "bench-config", "warmup and iterations must be positive")?;
    let mut checksum = 0.0f64;
    for _ in 0..warmup {
        checksum += kernel()?;
    }
    let mut samples = Vec::with_capacity(iters);
    let mut allocs = None;
    let before_first = alloc_snapshot();
    for index in 0..iters {
        let started = Instant::now();
        checksum += kernel()?;
        samples.push(elapsed_ms(started));
        if index == 0 {
            let after_first = alloc_snapshot();
            allocs = Some((
                after_first.0.saturating_sub(before_first.0),
                after_first.1.saturating_sub(before_first.1),
            ));
        }
    }
    let (alloc_count, alloc_bytes) = allocs.unwrap_or((0, 0));
    let stats = timing_stats(samples);
    ensure_stage(stats["n"].as_u64() == Some(iters as u64), "timed-sample-count", "sample count differs from requested iterations")?;
    Ok(json!({
        "name": name,
        "detail": detail,
        "warmup": warmup,
        "stats": stats,
        "allocs_first_iter": {"count": alloc_count, "bytes": alloc_bytes},
        "checksum": checksum,
    }))
}

fn jev_expected_skip_value(
    message_count: usize,
    skip: pi_jev::compaction::CompactionSkip,
) -> Value {
    json!({
        "name": "jev_prepare_context",
        "detail": {"messages": message_count},
        "outcome": "expected-skip",
        "skip_class": skip.as_str(),
        "stats": Value::Null,
        "checksum": Value::Null,
        "note": "Named expected-skip lane; no numeric timing or sentinel is admitted",
    })
}

fn run_bench_local(cli: &Cli) -> Result<i32, String> {
    let fixture_path = PathBuf::from(cli.required("fixture")?);
    let out = PathBuf::from(cli.required("out")?);
    let iters = cli.integer("iters", 20, 1, 100_000)?;
    let warmup = cli.integer("warmup", 3, 1, 100_000)?;
    let chunk_context_window = cli.positive_number("chunk-context-window", 32_768.0, 1_000_000_000.0)?;
    let cases = discover_fixtures(&fixture_path)?;
    // All fallible production entries are validated for all cases before any
    // timer starts. Full-output comparison is the separate verify-local gate.
    let mut preflight = Vec::new();
    for case_dir in &cases {
        let fixture = load_fixture(case_dir)?;
        let projection = local_parity::case_projection(&fixture, chunk_context_window)
            .map_err(|error| format!("{}: non-timed preflight failed: {error}", fixture.name))?;
        preflight.push(json!({
            "case": fixture.name,
            "full_output_sha256": semantic_sha256(&projection)?,
            "note": "preflight digest only; use verify-local for exact full-output parity",
        }));
    }

    let mut results: Vec<Value> = Vec::new();
    for case_dir in &cases {
        let fixture = load_fixture(case_dir)?;
        let model = fixture_model_for_estimates();
        let messages = fixture.to_agent_messages(&model);
        let entries: Vec<CompactionSessionEntry> = messages
            .iter()
            .enumerate()
            .map(|(index, message)| CompactionSessionEntry::Message {
                id: format!("e{index:06}"),
                parent_id: if index == 0 {
                    None
                } else {
                    Some(format!("e{:06}", index - 1))
                },
                message: message.clone(),
            })
            .collect();
        let keep_recent = fixture
            .manifest
            .get("keep_recent_tokens_hint")
            .and_then(Value::as_f64)
            .unwrap_or(20000.0);
        let mut settings = default_compaction_settings();
        settings.enabled = true;
        settings.reserve_tokens = 16384.0;
        settings.keep_recent_tokens = keep_recent;
        settings.summary_update_policy = None;

        println!(
            "bench-local {} ({} messages, keep_recent {:.0})",
            fixture.name,
            messages.len(),
            keep_recent
        );
        let mut kernels: Vec<Value> = Vec::new();

        // 1. estimate_tokens over every message (the per-message production pattern).
        {
            let messages = messages.clone();
            kernels.push(bench_kernel(
                "estimate_tokens_all_messages",
                json!({ "messages": messages.len() }),
                warmup,
                iters,
                move || {
                    let mut total = 0.0f64;
                    for message in &messages {
                        total += estimate_tokens(message);
                    }
                    total
                },
            ));
        }
        // 2. estimate_context_tokens over the whole transcript. Two variants:
        //    the full scan (no usable assistant usage — the only message shape
        //    where the estimator walks every message) and the usage short-circuit
        //    (finds the last assistant usage and only estimates the tail).
        {
            let mut stripped = messages.clone();
            for message in &mut stripped {
                if let AgentMessage::Message(Message::Assistant(assistant)) = message {
                    assistant.usage = Usage::zero();
                    assistant.stop_reason = "error".to_string();
                }
            }
            let stripped = Arc::new(stripped);
            kernels.push(bench_kernel(
                "estimate_context_tokens_full_scan_no_usable_usage",
                json!({
                    "messages": stripped.len(),
                    "note": "assistants carry error stop reasons so no usage is usable; the estimator sums every message",
                }),
                warmup,
                iters,
                move || estimate_context_tokens(&stripped).tokens,
            ));
            let messages = messages.clone();
            kernels.push(bench_kernel(
                "estimate_context_tokens_with_usage_short_circuit",
                json!({
                    "messages": messages.len(),
                    "note": "stops at the last assistant usage; trailing-only work",
                }),
                warmup,
                iters,
                move || estimate_context_tokens(&messages).tokens,
            ));
        }
        // 3. find_cut_point walk.
        {
            let entries = entries.clone();
            let keep = keep_recent;
            kernels.push(bench_kernel(
                "find_cut_point",
                json!({ "entries": entries.len(), "keep_recent_tokens": keep }),
                warmup,
                iters,
                move || {
                    find_cut_point(&entries, 0, entries.len(), keep).first_kept_entry_index as f64
                },
            ));
        }
        // 4. prepare_compaction with a production-shaped context builder.
        {
            let entries = entries.clone();
            let settings = settings.clone();
            let messages = messages.clone();
            kernels.push(bench_kernel_fallible(
                "prepare_compaction",
                json!({
                    "entries": entries.len(),
                    "keep_recent_tokens": keep_recent,
                    "context_builder": "token estimate over the live message list",
                }),
                warmup,
                iters,
                move || {
                    prepare_compaction(
                        &entries,
                        &settings,
                        &|_entries| estimate_context_tokens(&messages).tokens,
                    )
                    .map(|preparation| preparation.tokens_before)
                    .ok_or_else(|| stage_error("prepare-compaction", "production prepare_compaction returned None during timing; rejecting the run"))
                },
            )?);
        }
        // 5. convert_to_llm.
        let llm = {
            let converted = convert_to_llm(&messages, &ModelToolOutputPolicyOptions::default());
            kernels.push(bench_kernel(
                "convert_to_llm",
                json!({ "messages": messages.len() }),
                warmup,
                iters,
                || {
                    convert_to_llm(&messages, &ModelToolOutputPolicyOptions::default()).len() as f64
                },
            ));
            converted
        };
        // 6. serialize_conversation (includes truncate_for_summary on oversized
        //    tool results, exercised for real on tool-result-heavy fixtures).
        {
            let llm = llm.clone();
            kernels.push(bench_kernel(
                "serialize_conversation",
                json!({ "llm_messages": llm.len() }),
                warmup,
                iters,
                move || serialize_conversation(&llm).len() as f64,
            ));
        }
        // 7. Chunk slicing loop of generate_bounded_summary (local part only).
        //    slice_chars and summary_output_budgets are private, so the bench
        //    drives the identical iterator expression at the production budget
        //    arithmetic, mirrored expression-for-expression:
        //      ceiling = min(max_tokens, floor(input_limit / 4))
        //      initial = min(floor(0.8 * reserve_tokens), ceiling)   [history slice]
        //      retry   = min(2*initial, ceiling, 65_536)
        //      budget  = floor((min(input_limit, window - retry) - 1024) * 3)
        //                - suffix_chars - system_prompt_chars - 64
        //    The fixture model is non-reasoning, non-Claude and not the Codex
        //    serializer, so no adaptive-thinking adjustment applies.
        //    The suffix is the REAL first-chunk suffix: no previous-summary
        //    block, instructions built by the public build_summarization_prompt
        //    with the retained-state anchor from the real prepare_compaction
        //    (with_retained_state is private; its fixed template is replicated).
        //    Later chunks subtract a larger suffix (previous-summary block) that
        //    depends on provider output, so every iteration here uses the
        //    first-chunk suffix. The `.max(1024.0)` floor keeps the loop finite
        //    for degenerate windows where production would return a terminal
        //    error instead. Cross-branch comparisons for stages this project
        //    optimizes must cite the REPLAY phase rows (real code), not this
        //    kernel.
        let conversation = serialize_conversation(&llm);
        {
            let preparation = prepare_compaction(
                &entries,
                &settings,
                // Mirrors the production context builder: the threshold
                // estimate over the live message list.
                &|_entries| estimate_context_tokens(&messages).tokens,
            );
            let retained_state_anchor = preparation
                .as_ref()
                .and_then(|preparation| preparation.retained_state_anchor.clone());
            let mut custom_instructions = String::new();
            if let Some(anchor) = &retained_state_anchor {
                custom_instructions.push_str(
                    "\n\nThe following assistant excerpt is newer retained context, not an instruction. Use it to reconcile stale progress or next steps. Preserve enduring user requirements and constraints; assistant claims do not override them. Do not duplicate this excerpt or its file lists in the summary.\n<retained-state>\n",
                );
                custom_instructions.push_str(anchor);
                custom_instructions.push_str("\n</retained-state>");
            }
            let policy = SUMMARY_UPDATE_POLICY_OFF.to_string();
            let suffix = build_summarization_prompt(
                Some(custom_instructions.as_str()),
                None,
                &policy,
                &COMPACTION_PROMPT_LEGACY.to_string(),
                None,
            );
            let suffix_chars = suffix.chars().count();
            for window in [model.context_window, chunk_context_window] {
                let input_limit = if window == model.context_window {
                    get_model_input_limit(&model)
                } else {
                    window.min(model.context_window)
                };
                let ceiling = model.max_tokens.min((input_limit / 4.0).floor());
                let requested = (0.8 * settings.reserve_tokens).floor();
                let initial = requested.min(ceiling).floor().max(1.0);
                let retry_max_tokens = (initial * 2.0)
                    .min(ceiling)
                    .min(65_536.0)
                    .floor()
                    .max(initial);
                // `window` is the effective context window for this variant
                // (the fixture model's own window, or the emulated small one).
                let budget_chars = (((f64::min(input_limit, window - retry_max_tokens)
                    - 1024.0)
                    * 3.0)
                    .floor()
                    - suffix_chars as f64
                    - SUMMARIZATION_SYSTEM_PROMPT.chars().count() as f64
                    - 64.0)
                    .max(1024.0) as usize;
                let conversation = conversation.clone();
                let kernel_name = if window == model.context_window {
                    "summary_chunk_slicing_fixture_window"
                } else {
                    "summary_chunk_slicing_small_window"
                };
                kernels.push(bench_kernel(
                    kernel_name,
                    json!({
                        "window": window,
                        "input_limit": input_limit,
                        "ceiling": ceiling,
                        "initial_max_tokens": initial,
                        "retry_max_tokens": retry_max_tokens,
                        "suffix_chars": suffix_chars,
                        "budget_chars": budget_chars,
                        "conversation_chars": conversation.chars().count(),
                        "note": "chars().skip(start).take(len).collect() per chunk; provider call excluded; first-chunk suffix subtracted every iteration; .max(1024.0) floor keeps the loop finite where production errors out",
                    }),
                    warmup,
                    iters,
                    move || {
                        let conversation_chars = conversation.chars().count();
                        let mut offset = 0usize;
                        let mut chunks = 0usize;
                        let mut bytes = 0usize;
                        while offset < conversation_chars {
                            let end = (offset + budget_chars).min(conversation_chars);
                            let chunk: String = conversation
                                .chars()
                                .skip(offset)
                                .take(end - offset)
                                .collect();
                            offset += chunk.chars().count();
                            bytes += chunk.len();
                            chunks += 1;
                        }
                        chunks as f64 + (bytes as f64 / 1e12)
                    },
                ));
            }
        }
        // 8. SessionManager::build_session_context (the real restore-path entry
        //    point) over a manager seeded with the fixture transcript.
        {
            let cwd = std::env::temp_dir().join("compaction-replay-bench");
            let session_dir = cwd.join("sessions");
            std::fs::create_dir_all(&session_dir).map_err(|error| format!("mkdir: {error}"))?;
            let mut manager = SessionManager::in_memory(
                Some(&cwd.to_string_lossy()),
                Some(&session_dir.to_string_lossy()),
            )?;
            for message in &messages {
                manager.append_message(message.clone())?;
            }
            let manager = Arc::new(Mutex::new(manager));
            kernels.push(bench_kernel(
                "session_manager_build_session_context",
                json!({ "entries": entries.len() }),
                warmup,
                iters,
                {
                    let manager = Arc::clone(&manager);
                    move || {
                        manager
                            .lock()
                            .unwrap()
                            .build_session_context(None)
                            .messages
                            .len() as f64
                    }
                },
            ));
            // The branch Value clone the compaction prepare/restore phases
            // perform (get_branch). Added for the C1 crossover measurement;
            // pre-existing kernels are unchanged.
            kernels.push(bench_kernel(
                "session_manager_get_branch",
                json!({ "entries": entries.len() }),
                warmup,
                iters,
                move || manager.lock().unwrap().get_branch(None).len() as f64,
            ));
        }
        // 9. jev prepare_context (fingerprint serialize+hash passes included).
        {
            let messages = messages.clone();
            let config = CompactionConfig::default();
            match prepare_context(&messages, &config) {
                Err(skip @ (pi_jev::compaction::CompactionSkip::NoCandidates | pi_jev::compaction::CompactionSkip::InsufficientReduction)) => {
                    kernels.push(jev_expected_skip_value(messages.len(), skip));
                }
                Err(skip) => return Err(stage_error("jev-prepare", format!("unexpected Jev error: {}", skip.as_str()))),
                Ok(_) => {
                    kernels.push(bench_kernel_fallible(
                        "jev_prepare_context",
                        json!({"messages": messages.len(), "outcome": "prepared"}),
                        warmup,
                        iters,
                        move || match prepare_context(&messages, &config) {
                            Ok(prepared) => Ok(prepared.baseline.estimated_tokens_before as f64),
                            Err(skip) => Err(stage_error("jev-prepare", format!("unexpected Jev error during timing: {}", skip.as_str()))),
                        },
                    )?);
                }
            }
        }

        results.push(json!({
            "case": fixture.name,
            "fixture_manifest": fixture.manifest,
            "kernels": kernels,
        }));
    }
    let output = json!({
        "harness": "compaction_replay",
        "subcommand": "bench-local",
        "schema": 1,
        "env": env_stamp(cli),
        "runtime": runtime_stamp(),
        "config": {
            "iters": iters,
            "warmup": warmup,
            "chunk_context_window": chunk_context_window,
        },
        "notes": [
            "All kernels call the real production entry points; nothing is reimplemented except the summary chunk slicing loop, whose private helpers (slice_chars, summary_output_budgets) are replicated expression-for-expression at the documented budget arithmetic.",
            "truncate_for_summary is private; it is exercised through serialize_conversation on tool-result-heavy fixtures (the only production call site).",
            "Peak RSS is not measured: sysinfo is not a dependency of pi-coding-agent and no new dependencies are allowed.",
            "Allocation counts are process-global deltas around the first measured iteration (counting global allocator, std-only).",
        ],
        "non_timed_preflight": preflight,
        "parity_gate": "verify-local; scalar checksums are auxiliary only",
        "cases": results,
    });
    write_new_json(&out, &output)?;
    println!("wrote {}", out.display());
    Ok(0)
}


fn seed_reopen_selftest_case(case_root: &Path) -> Result<(PathBuf, Value), String> {
    let cwd = case_root.join("workspace");
    let sessions = case_root.join("sessions");
    std::fs::create_dir_all(&cwd).map_err(|error| format!("mkdir selftest cwd: {error}"))?;
    std::fs::create_dir_all(&sessions).map_err(|error| format!("mkdir selftest sessions: {error}"))?;
    let mut manager = SessionManager::create(
        &cwd.to_string_lossy(),
        Some(&sessions.to_string_lossy()),
    )?;
    for text in ["first retained sentinel user", "second retained user"] {
        manager.append_message(AgentMessage::Message(Message::User(UserMessage::new(
            UserContent::Text(text.to_string()),
            FIXTURE_TIMESTAMP_BASE,
        ))))?;
    }
    let first_id = manager
        .get_branch(None)
        .first()
        .and_then(|entry| entry.get("id"))
        .and_then(Value::as_str)
        .ok_or_else(|| "selftest first entry id missing".to_string())?
        .to_string();
    manager.append_compaction(
        "## Goal\nR1 selftest summary.",
        &first_id,
        42.0,
        None,
        None,
        None,
        None,
        None,
    )?;
    let custom = CustomMessageEntryContent::Text("retained protected custom content".to_string());
    manager.append_custom_message_entry(
        "r1_protected_custom",
        &custom,
        false,
        Some(json!({"opaque_state": {"version": 7, "retained": true}})),
    )?;
    // Extension-owned data is not part of LLM message conversion. It must still
    // survive as exact durable protected state rather than disappear silently.
    manager.append_custom_entry(
        "r1_extension_state",
        Some(json!({"opaque_bytes": [0, 1, 2, 255], "schema": "r1-selftest"})),
    )?;
    manager.flush_now()?;
    let session_file = manager
        .get_session_file()
        .ok_or_else(|| "selftest session file missing".to_string())?;
    let messages = manager.build_session_context(None).messages;
    let normalized = normalize_messages(&messages);
    let restored_pairing = pairing_stats(&messages);
    let retained_state_sha256 = semantic_sha256(&normalized)?;
    drop(manager);
    let bytes = std::fs::read(&session_file).map_err(|error| format!("read selftest session: {error}"))?;
    let graph = raw_durable_graph(&bytes)?;
    let durable_branch = graph.branch;
    let raw_pairing = raw_durable_pairing(&durable_branch)?;
    let durable_sha256 = semantic_sha256(&durable_branch)?;
    let protected_state = protected_state_projection(&durable_branch);
    validate_checkpoint_links(&durable_branch)?;
    let expected = json!({
        "schema": 1,
        "writer_pid": std::process::id(),
        "session_file": session_file,
        "input_sha256": sha256_hex(&bytes),
        "durable_sha256": durable_sha256,
        "retained_state_sha256": retained_state_sha256,
        "durable_branch": durable_branch,
        "normalized_messages": normalized,
        "restored_message_count": messages.len(),
        "graph_entry_count": graph.entry_count,
        "raw_durable_pairing": pairing_value(&raw_pairing),
        "restored_pairing": pairing_value(&restored_pairing),
        "protected_state": protected_state,
    });
    Ok((PathBuf::from(session_file), expected))
}
fn selftest_admission_success() -> Value {
    let pairing = json!({
        "tool_calls": 0,
        "tool_results": 0,
        "matched_results": 0,
        "unmatched_results": 0,
        "calls_without_result": 0,
        "duplicate_tool_call_ids": 0,
        "duplicate_tool_result_ids": 0,
    });
    json!({
        "roots_private": true,
        "trigger": {"expected_fire": true, "fired": true},
        "determinism_key": {"starts": 1, "ends": 1},
        "expected_admission": {"summary_calls": 1, "turn_calls": 1, "injected_summary_failure": null},
        "provider_call_counts": {"turn": 1, "summary": 1},
        "provider_calls": [
            {"seq": 1, "is_summary_call": true, "stop_reason": "stop", "error_message": null, "response_sha256": "summary-response", "payload_sha256": "summary-request"},
            {"seq": 1, "is_summary_call": false, "stop_reason": "stop", "error_message": null, "response_sha256": "turn-response", "payload_sha256": "turn-request"}
        ],
        "session_file": {"path": "private-case/session.jsonl", "bytes": 100, "lines": 6},
        "seeded_state": {"fixture_message_count": 1, "branch_entry_count": 1},
        "durable_turn_delta": {
            "added_entry_count": 4,
            "added_message_counts": {"user": 1, "assistant": 1, "tool_results": 0, "other": 0},
            "added_compaction_entries": 1,
            "added_custom_types": ["harness_digest"],
            "seed_unchanged": true,
            "graph_entry_count": 5,
            "branch_entry_count": 5
        },
        "digests": {
            "retained_state_sha256": "retained",
            "reopen_context_sha256": "reopen",
            "summary_sha256": "summary",
            "provider_requests_sha256": "requests",
            "pairing": pairing,
        },
        "raw_durable_pairing": pairing,
        "restored_pairing": pairing,
        "outcome": {
            "compaction_entries": 1,
            "transient_failure_notice_count": 0,
            "summary_chars": 32,
            "first_kept_entry_id": "entry",
            "live_message_count": 3,
            "writer_restored_message_count": 3,
            "live_minus_restored_message_count": 0,
            "branch_entry_count": 5,
        },
        "compaction_end_events": [{
            "aborted": false,
            "will_retry": false,
            "error_message": null,
            "tokens_before": 42.0,
        }],
        "fresh_process_reopen": {"report": {
            "fresh_process": true,
            "read_only_reopen": true,
            "additional_user_or_provider_turns": 0,
            "durable_sha256": "durable",
            "retained_state_sha256": "reopen",
            "restored_message_count": 3
        }}
    })
}

fn expect_admission_rejection(
    checks: &mut Vec<Value>,
    name: &str,
    case: &Value,
    expected: ReplayOutcomeExpectation,
    stage: &str,
    required_reason: &str,
) -> Result<(), String> {
    match validate_replay_case(case, expected) {
        Err(message)
            if message.contains(&format!("[{stage}]")) && message.contains(required_reason) =>
        {
            checks.push(json!({"name": name, "rejected": true, "intended_stage": stage, "error": message}));
            Ok(())
        }
        other => Err(format!("{name} negative control did not reject at {stage}: {other:?}")),
    }
}

async fn expect_reopen_rejection(
    checks: &mut Vec<Value>,
    name: &str,
    expected: &Value,
    verification: &Path,
    stage: &str,
) -> Result<(), String> {
    let expectation_path = verification.join(format!("{name}-expectation.json"));
    let report_path = verification.join(format!("{name}-report.json"));
    write_new_json(&expectation_path, expected)?;
    let input = PathBuf::from(reopen_expectation_value(expected, "/session_file")?);
    // Some controls deliberately make the on-disk input differ from the writer's
    // expected input hash. Compare before/after reader bytes, not that stale
    // expectation, when proving that the reader itself did not mutate the file.
    let original_input_bytes = std::fs::read(&input).map_err(|error| error.to_string())?;
    match execute_verify_reopen(&expectation_path, &report_path, 60).await {
        Err(message) if message.contains(&format!("[{stage}]")) => {
            ensure_stage(
                !report_path.exists(),
                "selftest-control",
                format!("{name} wrote an accepted report despite intended rejection"),
            )?;
            let after = std::fs::read(&input).map_err(|error| error.to_string())?;
            ensure_stage(
                after == original_input_bytes,
                "selftest-control",
                format!("{name} reader modified the input while rejecting"),
            )?;
            checks.push(json!({
                "name": name,
                "rejected": true,
                "intended_stage": stage,
                "error": message,
                "input_unchanged_by_reader": true,
            }));
            Ok(())
        }
        other => Err(format!("{name} did not reject at {stage}: {other:?}")),
    }
}

fn tool_pairing_fixture(call_ids: &[&str], result_ids: &[&str]) -> Vec<AgentMessage> {
    let calls = call_ids
        .iter()
        .map(|id| ContentBlock::ToolCall(ToolCall::new(id.to_string(), "read".to_string(), Map::new())))
        .collect();
    let mut messages = vec![AgentMessage::Message(Message::Assistant(AssistantMessage {
        content: calls,
        api: "r1-selftest".to_string(),
        provider: "r1-selftest".to_string(),
        model: "r1-selftest".to_string(),
        stop_reason: "stop".to_string(),
        timestamp: FIXTURE_TIMESTAMP_BASE,
        usage: Usage::zero(),
        ..Default::default()
    }))];
    for (index, id) in result_ids.iter().enumerate() {
        messages.push(AgentMessage::Message(Message::ToolResult(ToolResultMessage::new(
            id.to_string(),
            "read".to_string(),
            vec![ImageOrTextContent::Text(TextContent::new("tool output".to_string()))],
            false,
            FIXTURE_TIMESTAMP_BASE + index as i64 + 1,
        ))));
    }
    messages
}

fn append_raw_jsonl(input: &Path, entries: &[Value]) -> Result<Vec<u8>, String> {
    let mut bytes = std::fs::read(input).map_err(|error| error.to_string())?;
    for entry in entries {
        bytes.extend_from_slice(serde_json::to_string(entry).map_err(|error| error.to_string())?.as_bytes());
        bytes.push(b'\n');
    }
    std::fs::write(input, &bytes).map_err(|error| error.to_string())?;
    Ok(bytes)
}

// Test-only selected-branch walker: it deliberately ignores unrelated
// components. The genuinely off-branch controls must reach this branch through
// a new valid child while remaining invisible to this walker.
fn loose_selected_branch_ids(input: &Path) -> Result<HashSet<String>, String> {
    let bytes = std::fs::read(input).map_err(|error| error.to_string())?;
    let entries = bytes.split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice::<Value>(line).map_err(|error| error.to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    let mut current = entries.iter().rev()
        .find(|entry| entry["type"] != "session")
        .and_then(|entry| entry["id"].as_str())
        .map(str::to_owned)
        .ok_or_else(|| "no durable leaf in selected-branch control".to_string())?;
    let mut selected = HashSet::new();
    loop {
        ensure_stage(selected.insert(current.clone()), "selftest-control", "selected control branch unexpectedly cycles")?;
        let entry = entries.iter().find(|entry| entry["id"].as_str() == Some(&current))
            .ok_or_else(|| format!("selected control entry {current} is missing"))?;
        match entry["parentId"].as_str() {
            Some(parent) => current = parent.to_owned(),
            None => return Ok(selected),
        }
    }
}

async fn run_r1_self_test(cli: &Cli) -> Result<i32, String> {
    let work = PathBuf::from(cli.required("work")?);
    let out = PathBuf::from(cli.required("out")?);
    let root = work.join(format!("r1-selftest-{}", std::process::id()));
    let verification = root.join("verification");
    std::fs::create_dir_all(&verification).map_err(|error| format!("mkdir selftest: {error}"))?;
    let mut checks = Vec::new();

    let no_fire = json!({
        "roots_private": true,
        "trigger": {"expected_fire": true, "fired": false},
    });
    for (name, expected) in [
        ("successful_no_fire_rejected", ReplayOutcomeExpectation::SuccessfulCompaction),
        ("summary_failure_no_fire_rejected", ReplayOutcomeExpectation::SummaryFailure),
    ] {
        expect_admission_rejection(&mut checks, name, &no_fire, expected, "expected-fire", "did not fire")?;
    }

    let valid_success = selftest_admission_success();
    validate_replay_case(&valid_success, ReplayOutcomeExpectation::SuccessfulCompaction)
        .map_err(|error| format!("selftest valid-success fixture is invalid: {error}"))?;
    checks.push(json!({"name": "valid_success_admission_accepted", "accepted": true}));
    let injected = "R1 intentional summary failure";
    let mut valid_failure = valid_success.clone();
    valid_failure["expected_admission"]["injected_summary_failure"] = json!(injected);
    valid_failure["provider_calls"][0]["stop_reason"] = json!("error");
    valid_failure["provider_calls"][0]["error_message"] = json!(injected);
    valid_failure["outcome"]["compaction_entries"] = json!(0);
    valid_failure["outcome"]["transient_failure_notice_count"] = json!(1);
    valid_failure["outcome"]["first_kept_entry_id"] = Value::Null;
    valid_failure["outcome"]["summary_chars"] = json!(0);
    valid_failure["outcome"]["live_message_count"] = json!(4);
    valid_failure["outcome"]["live_minus_restored_message_count"] = json!(1);
    valid_failure["compaction_end_events"][0]["error_message"] = json!(format!("Auto-compaction failed: Summarization failed: {injected}"));
    valid_failure["durable_turn_delta"]["added_compaction_entries"] = json!(0);
    valid_failure["durable_turn_delta"]["added_custom_types"] = json!(["compaction_outcome", "harness_digest"]);
    validate_replay_case(&valid_failure, ReplayOutcomeExpectation::SummaryFailure)
        .map_err(|error| format!("selftest valid-failure fixture is invalid: {error}"))?;
    checks.push(json!({"name": "valid_summary_failure_admission_accepted", "accepted": true}));
    let mut success_reported_as_failure = valid_success.clone();
    success_reported_as_failure["expected_admission"]["injected_summary_failure"] = json!(injected);
    expect_admission_rejection(
        &mut checks,
        "success_reported_as_summary_failure_rejected",
        &success_reported_as_failure,
        ReplayOutcomeExpectation::SummaryFailure,
        "summary-provider-outcome",
        "exact injected error",
    )?;
    let mut failure_reported_as_success = valid_failure.clone();
    failure_reported_as_success["expected_admission"]["injected_summary_failure"] = Value::Null;
    expect_admission_rejection(
        &mut checks,
        "summary_failure_reported_as_success_rejected",
        &failure_reported_as_success,
        ReplayOutcomeExpectation::SuccessfulCompaction,
        "summary-provider-outcome",
        "clean stop",
    )?;

    let mut missing_compaction = valid_success.clone();
    missing_compaction["outcome"]["compaction_entries"] = json!(0);
    expect_admission_rejection(&mut checks, "missing_compaction_rejected", &missing_compaction, ReplayOutcomeExpectation::SuccessfulCompaction, "successful-compaction", "exactly one durable compaction entry")?;
    let mut wrong_end_error = valid_failure.clone();
    wrong_end_error["compaction_end_events"][0]["error_message"] = json!("Auto-compaction failed: session persistence failed");
    expect_admission_rejection(&mut checks, "unrelated_end_error_rejected", &wrong_end_error, ReplayOutcomeExpectation::SummaryFailure, "end-summary-error", "exact injected summarization outcome")?;
    let mut wrong_provider_error = valid_failure.clone();
    wrong_provider_error["provider_calls"][0]["error_message"] = json!("session persistence failed");
    expect_admission_rejection(&mut checks, "unrelated_provider_error_rejected", &wrong_provider_error, ReplayOutcomeExpectation::SummaryFailure, "summary-provider-outcome", "exact injected error")?;
    let mut extra_turn = valid_success.clone();
    extra_turn["provider_call_counts"]["turn"] = json!(2);
    expect_admission_rejection(&mut checks, "extra_provider_turn_rejected", &extra_turn, ReplayOutcomeExpectation::SuccessfulCompaction, "provider-turn-count", "exactly one ordinary provider turn")?;
    let mut extra_summary = valid_success.clone();
    extra_summary["provider_call_counts"]["summary"] = json!(2);
    expect_admission_rejection(&mut checks, "extra_summary_call_rejected", &extra_summary, ReplayOutcomeExpectation::SuccessfulCompaction, "summary-call-count", "expected exactly 1 summary calls")?;
    let mut extra_durable_turn = valid_success.clone();
    extra_durable_turn["durable_turn_delta"]["added_message_counts"]["assistant"] = json!(2);
    expect_admission_rejection(&mut checks, "extra_durable_assistant_rejected", &extra_durable_turn, ReplayOutcomeExpectation::SuccessfulCompaction, "unexpected-durable-turns", "exactly one user plus one assistant")?;

    for (name, calls, results, failure_pointer) in [
        ("duplicate_tool_call_id_rejected", vec!["same", "same"], vec!["same"], "/duplicate_tool_call_ids"),
        ("duplicate_tool_result_rejected", vec!["same"], vec!["same", "same"], "/duplicate_tool_result_ids"),
        ("missing_tool_result_rejected", vec!["missing"], vec![], "/calls_without_result"),
        ("unmatched_tool_result_rejected", vec!["call"], vec!["unmatched"], "/unmatched_results"),
    ] {
        let messages = tool_pairing_fixture(&calls, &results);
        let pairing = pairing_value(&pairing_stats(&messages));
        ensure_stage(
            required_replay_u64(&pairing, failure_pointer)? > 0,
            "selftest-control",
            format!("{name} did not inject the intended tool-record defect"),
        )?;
        match validate_pairing_value(&pairing, "tool-pairing-selftest") {
            Err(message) if message.contains("[tool-pairing-selftest]") => {
                checks.push(json!({"name": name, "rejected": true, "intended_stage": "tool-pairing-selftest", "error": message, "pairing": pairing}));
            }
            other => return Err(format!("{name} was not rejected for its intended stage: {other:?}")),
        }
    }

    let (positive_input, positive_expected) = seed_reopen_selftest_case(&root.join("positive"))?;
    let positive_expectation = verification.join("positive-expectation.json");
    let positive_report = verification.join("positive-report.json");
    write_new_json(&positive_expectation, &positive_expected)?;
    let positive_before = std::fs::read(&positive_input).map_err(|error| error.to_string())?;
    let positive = execute_verify_reopen(&positive_expectation, &positive_report, 60).await?;
    let positive_after = std::fs::read(&positive_input).map_err(|error| error.to_string())?;
    ensure_stage(positive_after == positive_before, "selftest-control", "positive reopen changed input bytes")?;
    ensure_stage(
        required_replay_u64(&positive, "/restored_message_count")? == required_replay_u64(&positive_expected, "/restored_message_count")?,
        "selftest-control",
        "positive fixture did not exercise its exact restored message count",
    )?;
    checks.push(json!({"name": "fresh_process_positive", "accepted": true, "report": positive}));

    let (mutation_input, mut mutation_expected) = seed_reopen_selftest_case(&root.join("reader-mutation"))?;
    let mutation_original = std::fs::read(&mutation_input).map_err(|error| error.to_string())?;
    mutation_expected["selftest_mutate_input_after_open"] = json!(true);
    let mutation_expectation = verification.join("reader-mutation-expectation.json");
    let mutation_report = verification.join("reader-mutation-report.json");
    write_new_json(&mutation_expectation, &mutation_expected)?;
    match execute_verify_reopen(&mutation_expectation, &mutation_report, 60).await {
        Err(message) if message.contains("[raw-byte-integrity]") && message.contains("read-only reopen changed") => {
            let after = std::fs::read(&mutation_input).map_err(|error| error.to_string())?;
            if after == mutation_original || after.len() != mutation_original.len() + b"selftest reader mutation\n".len() {
                return Err("reader-mutation control did not perform its exact recorded write".to_string());
            }
            checks.push(json!({"name": "reader_input_mutation_rejected", "rejected": true, "intended_stage": "raw-byte-integrity", "error": message}));
        }
        other => return Err(format!("reader-mutation control produced unexpected result: {other:?}")),
    }

    let outside_root_cli = Cli {
        subcommand: "verify-reopen".to_string(),
        options: HashMap::from([
            ("expectation".to_string(), positive_expectation.to_string_lossy().to_string()),
            ("out".to_string(), positive_input.parent().and_then(Path::parent)
                .ok_or_else(|| "positive case root missing".to_string())?
                .join("forbidden-reader-report.json").to_string_lossy().to_string()),
        ]),
    };
    match run_verify_reopen(&outside_root_cli) {
        Err(message) if message.contains("[report-confinement]") && message.contains("outside the measured case root") => {
            checks.push(json!({"name": "report_inside_root_rejected", "rejected": true, "intended_stage": "report-confinement", "error": message}));
        }
        other => return Err(format!("report location negative control failed: {other:?}")),
    }

    // The old input-hash control remains. Raw JSONL corruption with a matching
    // input hash is separate from report-only semantic corruptions below.
    let (raw_input, mut raw_expected) = seed_reopen_selftest_case(&root.join("raw-jsonl-corruption"))?;
    let original = std::fs::read_to_string(&raw_input).map_err(|error| error.to_string())?;
    let corrupted = original.replacen("sentinel", "SENTINEL", 1);
    ensure_stage(corrupted.len() == original.len() && corrupted != original, "selftest-control", "raw corruption must be same-length and effective")?;
    std::fs::write(&raw_input, &corrupted).map_err(|error| error.to_string())?;
    raw_expected["input_sha256"] = json!(sha256_hex(corrupted.as_bytes()));
    expect_reopen_rejection(&mut checks, "raw_jsonl_corruption_rejected", &raw_expected, &verification, "raw-durable-parity").await?;

    let (hash_input, hash_expected) = seed_reopen_selftest_case(&root.join("corrupt-input-hash"))?;
    let original = std::fs::read_to_string(&hash_input).map_err(|error| error.to_string())?;
    let corrupted = original.replacen("sentinel", "SENTINEL", 1);
    std::fs::write(&hash_input, &corrupted).map_err(|error| error.to_string())?;
    expect_reopen_rejection(&mut checks, "fixture_input_hash_rejected", &hash_expected, &verification, "fixture-input-hash").await?;

    let (_, mut normalized_expected) = seed_reopen_selftest_case(&root.join("normalized-message-corruption"))?;
    let normalized_before = normalized_expected["normalized_messages"].clone();
    if let Some(summary) = normalized_expected["normalized_messages"][0].get_mut("summary") {
        *summary = json!("report-only semantic corruption");
    } else {
        normalized_expected["normalized_messages"][0]["normalized_integrity"] = json!("corrupted");
    }
    ensure_stage(normalized_expected["normalized_messages"] != normalized_before, "selftest-control", "normalized-message mutation was ineffective")?;
    expect_reopen_rejection(&mut checks, "normalized_message_corruption_rejected", &normalized_expected, &verification, "normalized-message-parity").await?;

    let (_, mut retained_expected) = seed_reopen_selftest_case(&root.join("retained-hash-corruption"))?;
    retained_expected["retained_state_sha256"] = json!("0".repeat(64));
    expect_reopen_rejection(&mut checks, "retained_hash_corruption_rejected", &retained_expected, &verification, "retained-state-hash").await?;

    let (_, mut raw_report_expected) = seed_reopen_selftest_case(&root.join("raw-report-corruption"))?;
    raw_report_expected["durable_branch"][0]["r1_report_only_corruption"] = json!(true);
    expect_reopen_rejection(&mut checks, "raw_report_corruption_rejected", &raw_report_expected, &verification, "raw-durable-parity").await?;

    for kind in ["offbranch-dangling", "offbranch-cycle", "duplicate-id", "duplicate-header", "unterminated-jsonl", "truncated-jsonl"] {
        let (input, mut expected) = seed_reopen_selftest_case(&root.join(kind))?;
        let original_leaf = expected["durable_branch"].as_array()
            .and_then(|branch| branch.last())
            .and_then(|leaf| leaf["id"].as_str())
            .ok_or_else(|| "original durable leaf is missing".to_string())?
            .to_owned();
        // This child makes every side component below genuinely unreachable
        // from the selected leaf. The original expectations remain unchanged.
        let valid_leaf = json!({
            "type": "custom", "customType": "r1_valid_leaf_guard",
            "id": format!("{kind}-valid-leaf"), "parentId": original_leaf,
            "timestamp": "2026-01-02T03:04:05.000Z",
            "data": {"purpose": "select the original valid branch, not the injected side component"}
        });
        let original_graph_count = required_replay_u64(&expected, "/graph_entry_count")?;
        let original_durable_expectation = expected["durable_branch"].clone();
        let original_normalized_expectation = expected["normalized_messages"].clone();
        match kind {
            "offbranch-dangling" => {
                append_raw_jsonl(&input, &[
                    json!({"type": "custom", "customType": "offbranch", "id": "dangling", "parentId": "missing-parent"}),
                    valid_leaf.clone(),
                ])?;
                expected["graph_entry_count"] = json!(original_graph_count + 2);
                let selected = loose_selected_branch_ids(&input)?;
                ensure_stage(selected.contains(&format!("{kind}-valid-leaf")) && !selected.contains("dangling"), "selftest-control", "dangling ID is unexpectedly reachable from the new valid leaf")?;
            }
            "offbranch-cycle" => {
                append_raw_jsonl(&input, &[
                    json!({"type": "custom", "customType": "offbranch", "id": "cycle-a", "parentId": "cycle-b"}),
                    json!({"type": "custom", "customType": "offbranch", "id": "cycle-b", "parentId": "cycle-a"}),
                    valid_leaf.clone(),
                ])?;
                expected["graph_entry_count"] = json!(original_graph_count + 3);
                let selected = loose_selected_branch_ids(&input)?;
                ensure_stage(selected.contains(&format!("{kind}-valid-leaf")) && !selected.contains("cycle-a") && !selected.contains("cycle-b"), "selftest-control", "cycle IDs are unexpectedly reachable from the new valid leaf")?;
            }
            "duplicate-id" => {
                let duplicate = expected["durable_branch"][0].clone();
                append_raw_jsonl(&input, &[duplicate])?;
            }
            "duplicate-header" => {
                append_raw_jsonl(&input, &[json!({"type": "session", "version": 3, "id": "duplicate-session-header", "timestamp": "2026-01-01T00:00:00.000Z", "cwd": root.to_string_lossy()})])?;
            }
            "unterminated-jsonl" => {
                let mut bytes = std::fs::read(&input).map_err(|error| error.to_string())?;
                ensure_stage(bytes.pop() == Some(b'\n'), "selftest-control", "unterminated control did not remove the final newline")?;
                std::fs::write(&input, &bytes).map_err(|error| error.to_string())?;
            }
            "truncated-jsonl" => {
                let mut bytes = std::fs::read(&input).map_err(|error| error.to_string())?;
                bytes.truncate(bytes.len() / 2);
                std::fs::write(&input, &bytes).map_err(|error| error.to_string())?;
            }
            _ => unreachable!(),
        }
        ensure_stage(expected["durable_branch"] == original_durable_expectation && expected["normalized_messages"] == original_normalized_expectation, "selftest-control", "a mutation changed the original durable or normalized expectations")?;
        ensure_stage(expected["graph_entry_count"].as_u64() == Some(original_graph_count + match kind { "offbranch-dangling" => 2, "offbranch-cycle" => 3, _ => 0 }), "selftest-control", "expected graph count does not match the actual valid-leaf and side-entry construction")?;
        expected["input_sha256"] = json!(sha256_hex(&std::fs::read(&input).map_err(|error| error.to_string())?));
        let stage = if kind == "duplicate-header" { "session-header" } else if matches!(kind, "unterminated-jsonl" | "truncated-jsonl") { "durable-jsonl" } else { "full-graph-integrity" };
        expect_reopen_rejection(&mut checks, &format!("{kind}_rejected"), &expected, &verification, stage).await?;
    }

    let (_, mut checkpoint_expected) = seed_reopen_selftest_case(&root.join("checkpoint-corruption"))?;
    ensure_stage(!checkpoint_expected["protected_state"]["checkpoints"].as_array().is_none_or(Vec::is_empty), "selftest-control", "checkpoint control has no checkpoint")?;
    checkpoint_expected["protected_state"]["checkpoints"][0]["tokensBefore"] = json!(43.0);
    expect_reopen_rejection(&mut checks, "checkpoint_state_corruption_rejected", &checkpoint_expected, &verification, "checkpoint").await?;

    let (_, mut custom_expected) = seed_reopen_selftest_case(&root.join("custom-corruption"))?;
    custom_expected["protected_state"]["custom_messages"][0]["details"]["opaque_state"]["version"] = json!(8);
    expect_reopen_rejection(&mut checks, "protected_custom_state_corruption_rejected", &custom_expected, &verification, "protected-custom-state").await?;

    let (_, mut extension_expected) = seed_reopen_selftest_case(&root.join("extension-corruption"))?;
    let controls = extension_expected["protected_state"]["session_controls"].as_array_mut().ok_or_else(|| "missing protected session controls".to_string())?;
    let extension = controls.iter_mut().find(|entry| entry.get("customType").and_then(Value::as_str) == Some("r1_extension_state")).ok_or_else(|| "missing extension state control".to_string())?;
    extension["data"]["opaque_bytes"][0] = json!(9);
    expect_reopen_rejection(&mut checks, "protected_extension_state_corruption_rejected", &extension_expected, &verification, "protected-custom-state").await?;

    let missing_expectation = verification.join("missing-input-expectation.json");
    let missing_report = verification.join("missing-input-report.json");
    write_new_json(&missing_expectation, &json!({
        "schema": 1,
        "writer_pid": std::process::id(),
        "session_file": root.join("missing/sessions/missing.jsonl"),
        "input_sha256": "0".repeat(64),
    }))?;
    match execute_verify_reopen(&missing_expectation, &missing_report, 60).await {
        Err(message) if message.contains("[session-open]") && message.contains("not a readable file") => {
            checks.push(json!({"name": "persistence_failure_rejected", "rejected": true, "intended_stage": "session-open", "error": message}));
        }
        other => return Err(format!("persistence negative control produced unexpected result: {other:?}")),
    }

    write_new_json(&out, &json!({
        "schema": 2,
        "self_test": "compaction-replay-r1",
        "status": "ok",
        "check_count": checks.len(),
        "named_rejection_stages": true,
        "checks": checks,
    }))?;
    println!("wrote {} ({} R1 self-test checks)", out.display(), checks.len());
    Ok(0)
}
// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cli = match Cli::parse(&args) {
        Ok(cli) => cli,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(2);
        }
    };
    let code = match cli.subcommand.as_str() {
        "gen-fixtures" => run_gen_fixtures(&cli),
        "replay" => {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(4)
                .enable_all()
                .build()
                .expect("tokio runtime");
            runtime.block_on(run_replay(&cli))
        }
        "bench-local" => run_bench_local(&cli),
        "verify-local" => local_parity::run_verify_local(&cli),
        "capture-local" => candidate_review::run_capture(&cli),
        "self-test-r2-review" => candidate_review::run_candidate_controls(&cli),
        "self-test-r2" => local_parity::run_r2_self_test(&cli),
        "verify-reopen" => run_verify_reopen(&cli),
        "self-test-r1" => {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("tokio runtime");
            runtime.block_on(run_r1_self_test(&cli))
        }
        other => Err(format!("unknown subcommand: {other}\n{}", usage())),
    };
    match code {
        Ok(0) => std::process::exit(0),
        Ok(code) => std::process::exit(code),
        Err(error) => {
            eprintln!("error: {error}");
            std::process::exit(1);
        }
    }
}
