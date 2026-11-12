//! Offline replay + benchmark harness for the LOCAL auto-compaction path.
//!
//! Subcommands (hand-rolled CLI; no new dependencies):
//!   gen-fixtures --out <dir>
//!       Deterministic fixture transcripts (mixed conversations, several content
//!       classes, sizes from 10 KiB to ~40 MiB plus a near-threshold case).
//!       Each case dir contains `transcript.json` and `manifest.json`.
//!   replay --fixture <dir|case> --out <json> [--provider-delay-ms N]
//!          [--summary-failure MSG] [--context-window N] [--keep-recent N]
//!          [--work DIR] [--stamp SHA256] [--git DESCRIBE] [--timeout-secs N]
//!       Runs the REAL auto-compaction path end to end offline: private roots,
//!       in-process fixture provider with canned deterministic summaries, real
//!       SessionManager JSONL, CompactionMetrics phase rows via the recorder
//!       injection seam, provider call log, retained-state digests.
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
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Component, Path, PathBuf};
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
    ServiceTier, ToolResultMessage, Usage, UserContent, UserMessage,
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
use pi_coding_agent::core::session_manager::{
    get_provider_checkpoint, has_provider_checkpoint, SessionContextModel,
    SessionContextWithEntryIds, SessionEntry, SessionManager, CURRENT_SESSION_VERSION,
};
use pi_coding_agent::core::settings_manager::SettingsManager;
use pi_jev::compaction::CompactionConfig;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde::de::{self, Deserialize, MapAccess, SeqAccess, Visitor};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

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
                if let Some((key, value)) = rest.split_once('=') {
                    options.insert(key.to_string(), value.to_string());
                } else if index < args.len() && !args[index].starts_with("--") {
                    // `--key value` form.
                    options.insert(rest.to_string(), args[index].clone());
                    index += 1;
                } else {
                    options.insert(rest.to_string(), String::new());
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

    fn number(&self, key: &str, default: f64) -> Result<f64, String> {
        match self.optional(key) {
            None => Ok(default),
            Some(raw) => raw
                .parse::<f64>()
                .map_err(|error| format!("--{key}: {error}")),
        }
    }

    /// New boolean flags are present only without an attached/following value.
    /// Legacy modes deliberately keep their old permissive option handling.
    fn flag(&self, key: &str) -> Result<bool, String> {
        match self.options.get(key) {
            None => Ok(false),
            Some(value) if value.is_empty() => Ok(true),
            Some(value) => Err(format!("--{key} must be valueless, got {value:?}")),
        }
    }

    fn reject_unknown_options(&self, allowed: &[&str]) -> Result<(), String> {
        let allowed: HashSet<&str> = allowed.iter().copied().collect();
        let mut unknown: Vec<&str> = self
            .options
            .keys()
            .map(String::as_str)
            .filter(|key| !allowed.contains(key))
            .collect();
        unknown.sort_unstable();
        if unknown.is_empty() {
            Ok(())
        } else {
            Err(format!("unknown option(s): {}", unknown.join(", ")))
        }
    }
}

fn usage() -> String {
    "usage:\n\
     compaction_replay gen-fixtures --out <dir>\n\
     compaction_replay replay --fixture <dir|case> --out <json> [--provider-delay-ms N]\n\
     [--summary-failure MSG] [--context-window N] [--keep-recent N] [--work DIR]\n\
     [--stamp SHA256] [--git DESCRIBE] [--timeout-secs N]\n\
     [--export-session-projection]\n\
     compaction_replay bench-local --fixture <dir|case> --iters N --out <json> [--warmup N]\n\
     [--chunk-context-window N] [--stamp SHA256] [--git DESCRIBE] [--raw-samples]\n\
     compaction_replay verify-reopen --session-file <existing.jsonl> --work-root <private-root>\n\
     --out <new.json> [--git DESCRIBE]\n\
     New boolean options must be valueless: --raw-samples and\n\
     --export-session-projection. Each opt-in enforces known-option checking;\n\
     verify-reopen is always strict."
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
) {
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
        stream.end(Some(reply));
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
                task_stream.end(Some(reply));
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
                );
            });
        }
        Err(_) => {
            for event in events_for(reply.clone()) {
                stream.push(event);
            }
            stream.end(Some(reply));
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

struct PairingStats {
    tool_calls: usize,
    tool_results: usize,
    matched_results: usize,
    unmatched_results: usize,
    calls_without_result: usize,
}

fn pairing_stats(messages: &[AgentMessage]) -> PairingStats {
    let mut call_ordinals: HashMap<String, bool> = HashMap::new();
    let mut stats = PairingStats {
        tool_calls: 0,
        tool_results: 0,
        matched_results: 0,
        unmatched_results: 0,
        calls_without_result: 0,
    };
    for message in messages {
        match message {
            AgentMessage::Message(Message::Assistant(assistant)) => {
                for block in &assistant.content {
                    if let ContentBlock::ToolCall(call) = block {
                        stats.tool_calls += 1;
                        call_ordinals.insert(call.id.clone(), false);
                    }
                }
            }
            AgentMessage::Message(Message::ToolResult(result)) => {
                stats.tool_results += 1;
                match call_ordinals.get_mut(&result.tool_call_id) {
                    Some(seen) => {
                        stats.matched_results += 1;
                        *seen = true;
                    }
                    None => stats.unmatched_results += 1,
                }
            }
            _ => {}
        }
    }
    stats.calls_without_result = call_ordinals.values().filter(|seen| !**seen).count();
    stats
}


// ---------------------------------------------------------------------------
// PR3 normalized session projection (post-timing evidence only)
// ---------------------------------------------------------------------------

const NORMALIZER_VERSION: &str = "pr3-session-v1";
const COMMON_READER_SCHEMA: &str = "strict-pr3-evidence-v2";

fn value_contains_key(value: &Value, wanted: &str) -> bool {
    match value {
        Value::Object(map) => {
            map.contains_key(wanted) || map.values().any(|value| value_contains_key(value, wanted))
        }
        Value::Array(values) => values.iter().any(|value| value_contains_key(value, wanted)),
        _ => false,
    }
}

fn value_contains_legacy_checkpoint_strategy(value: &Value) -> bool {
    match value {
        Value::Object(map) => {
            map.get("strategy").and_then(Value::as_str)
                == Some("openai-responses-compaction-v2")
                || map
                    .values()
                    .any(value_contains_legacy_checkpoint_strategy)
        }
        Value::Array(values) => values
            .iter()
            .any(value_contains_legacy_checkpoint_strategy),
        _ => false,
    }
}

/// Provider/native restoration is deliberately outside this bounded offline
/// projection. Rejection occurs before normalized accepted hashes can exist.
fn reject_provider_dependent_value(value: &Value) -> Result<(), String> {
    const FORBIDDEN: &[&str] = &[
        "providerCheckpoint",
        "providerContext",
        "provider_context",
        "nativeAuxiliary",
        "native_auxiliary",
        "providerAuxiliary",
        "provider_auxiliary",
        "nativeCompaction",
        "compactedWindow",
        "encrypted_content",
        "encryptedContent",
    ];
    if FORBIDDEN
        .iter()
        .any(|key| value_contains_key(value, key))
        || value_contains_legacy_checkpoint_strategy(value)
    {
        return Err("PROVIDER_DEPENDENT_AUXILIARY_UNSUPPORTED".into());
    }
    Ok(())
}

fn reject_provider_dependent_entries(entries: &[SessionEntry]) -> Result<(), String> {
    for entry in entries {
        reject_provider_dependent_value(&Value::Object(entry.clone()))?;
        if entry.get("type").and_then(Value::as_str) == Some("compaction") {
            let details = entry.get("details").cloned().unwrap_or(Value::Null);
            if has_provider_checkpoint(&details) || get_provider_checkpoint(&details).is_some() {
                return Err("PROVIDER_DEPENDENT_AUXILIARY_UNSUPPORTED".into());
            }
        }
    }
    Ok(())
}

fn reject_provider_dependent_messages(messages: &[AgentMessage]) -> Result<(), String> {
    for message in messages {
        if matches!(
            message,
            AgentMessage::Custom(CustomAgentMessage::CompactionSummary {
                provider_context: Some(_),
                ..
            })
        ) {
            return Err("PROVIDER_DEPENDENT_AUXILIARY_UNSUPPORTED".into());
        }
        let serialized = serde_json::to_value(message).map_err(|error| error.to_string())?;
        reject_provider_dependent_value(&serialized)?;
    }
    Ok(())
}

/// Explicit canonical JSON: recursively sorted object keys, ordered arrays,
/// original scalar types, compact serde_json UTF-8 encoding and no map feature
/// assumptions. This is separate from semantic/byte equality claims.
fn encode_canonical_json(value: &Value, out: &mut Vec<u8>) -> Result<(), String> {
    match value {
        Value::Object(map) => {
            out.push(b'{');
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            for (index, key) in keys.into_iter().enumerate() {
                if index != 0 {
                    out.push(b',');
                }
                serde_json::to_writer(&mut *out, key).map_err(|error| error.to_string())?;
                out.push(b':');
                encode_canonical_json(&map[key], out)?;
            }
            out.push(b'}');
        }
        Value::Array(values) => {
            out.push(b'[');
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    out.push(b',');
                }
                encode_canonical_json(value, out)?;
            }
            out.push(b']');
        }
        scalar => {
            serde_json::to_writer(&mut *out, scalar).map_err(|error| error.to_string())?;
        }
    }
    Ok(())
}

fn canonical_json_bytes(value: &Value) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    encode_canonical_json(value, &mut out)?;
    Ok(out)
}

fn canonical_sha256(value: &Value) -> Result<String, String> {
    Ok(sha256_hex(&canonical_json_bytes(value)?))
}

fn with_projection_hashes(mut projection: Value) -> Result<Value, String> {
    let (durable_domain, context_domain) = {
        let object = projection
            .as_object_mut()
            .ok_or("session projection root must be an object")?;
        object.remove("hashes");
        let durable_domain = json!({
            "header": object.get("header").cloned().ok_or("projection header missing")?,
            "durable": object.get("durable").cloned().ok_or("projection durable missing")?,
        });
        let context_domain = object
            .get("rebuilt")
            .cloned()
            .ok_or("projection rebuilt missing")?;
        (durable_domain, context_domain)
    };
    let durable_sha256 = canonical_sha256(&durable_domain)?;
    let context_sha256 = canonical_sha256(&context_domain)?;
    // Only the entire root hashes member is excluded. observed_live is never
    // inserted into this root and therefore is never folded into this hash.
    let projection_sha256 = canonical_sha256(&projection)?;
    projection
        .as_object_mut()
        .ok_or("session projection root must be an object")?
        .insert(
            "hashes".into(),
            json!({
                "canonicalization": "recursive-key-sorted compact serde_json UTF-8; ordered arrays; scalar types preserved",
                "durable_sha256": durable_sha256,
                "context_sha256": context_sha256,
                "projection_sha256": projection_sha256,
            }),
        );
    Ok(projection)
}

#[derive(Clone)]
struct SessionNormalizer {
    entry_ids: HashMap<String, String>,
    tool_ids: HashMap<String, String>,
    private_roots: Vec<String>,
}

impl SessionNormalizer {
    fn new(header: &SessionEntry, entries: &[SessionEntry]) -> Result<Self, String> {
        let _ = header
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
            .ok_or("session header ID missing")?;
        let cwd = header
            .get("cwd")
            .and_then(Value::as_str)
            .ok_or("session header cwd missing")?;
        let case_root = Path::new(cwd)
            .parent()
            .ok_or("replay workspace has no case root")?;
        let mut roots = vec![case_root.to_string_lossy().to_string()];
        if let Ok(canonical) = std::fs::canonicalize(case_root) {
            roots.push(canonical.to_string_lossy().to_string());
        }
        let mut private_roots = Vec::new();
        for root in roots {
            if root.is_empty() || root == "/" || root == "\\" {
                return Err("unsafe private-root normalization spelling".into());
            }
            private_roots.push(root.replace('\\', "/"));
            private_roots.push(root.replace('/', "\\"));
            private_roots.push(root);
        }
        private_roots.sort_by_key(|root| std::cmp::Reverse(root.len()));
        private_roots.dedup();

        let mut entry_ids = HashMap::new();
        for (index, entry) in entries.iter().enumerate() {
            let id = entry
                .get("id")
                .and_then(Value::as_str)
                .filter(|id| !id.is_empty())
                .ok_or_else(|| format!("durable entry {index} has no required ID"))?
                .to_string();
            if entry_ids.contains_key(&id) {
                return Err(format!("duplicate durable entry ID: {id}"));
            }
            entry_ids.insert(id, format!("entry:{index:06}"));
        }

        let mut tool_ids = HashMap::new();
        fn collect(value: &Value, ids: &mut HashMap<String, String>) -> Result<(), String> {
            match value {
                Value::Object(map) => {
                    let is_call = matches!(
                        map.get("type").and_then(Value::as_str),
                        Some("toolCall" | "tool_call")
                    );
                    if is_call {
                        let id = map
                            .get("id")
                            .and_then(Value::as_str)
                            .filter(|id| !id.is_empty())
                            .ok_or("tool call has no required ID")?
                            .to_string();
                        if !ids.contains_key(&id) {
                            let next = ids.len();
                            ids.insert(id, format!("tool:{next:06}"));
                        }
                    }
                    if let Some(raw) = map.get("toolCallId") {
                        let id = raw
                            .as_str()
                            .filter(|id| !id.is_empty())
                            .ok_or("tool result has an invalid toolCallId")?
                            .to_string();
                        if !ids.contains_key(&id) {
                            let next = ids.len();
                            ids.insert(id, format!("tool:{next:06}"));
                        }
                    }
                    for value in map.values() {
                        collect(value, ids)?;
                    }
                }
                Value::Array(values) => {
                    for value in values {
                        collect(value, ids)?;
                    }
                }
                _ => {}
            }
            Ok(())
        }
        for entry in entries {
            if let Some(message) = entry.get("message") {
                collect(message, &mut tool_ids)?;
            }
        }
        Ok(Self {
            entry_ids,
            tool_ids,
            private_roots,
        })
    }

    fn private_root_text(&self, text: &str) -> String {
        let mut text = text.to_string();
        for root in &self.private_roots {
            text = text.replace(root, "<PRIVATE_ROOT>");
        }
        text
    }

    /// Non-message/custom-detail data is not scrubbed for timestamps or IDs.
    /// Only exact private-root path variants are replaced in these strings.
    fn normalize_other(&self, value: &Value) -> Value {
        match value {
            Value::Object(map) => Value::Object(
                map.iter()
                    .map(|(key, value)| (key.clone(), self.normalize_other(value)))
                    .collect(),
            ),
            Value::Array(values) => {
                Value::Array(values.iter().map(|value| self.normalize_other(value)).collect())
            }
            Value::String(text) => Value::String(self.private_root_text(text)),
            scalar => scalar.clone(),
        }
    }

    fn normalize_timestamp(value: &Value) -> Result<Value, String> {
        let source_type = match value {
            Value::Number(_) => "number",
            Value::String(_) => "string",
            _ => return Err("legitimate timestamp has an unexpected JSON type".into()),
        };
        Ok(json!({"volatile": "timestamp", "source_type": source_type}))
    }

    fn normalize_link(
        value: &Value,
        ids: &HashMap<String, String>,
        field: &str,
    ) -> Result<Value, String> {
        match value {
            Value::Null => Ok(Value::Null),
            Value::String(id) if !id.is_empty() => {
                // Unrecognized links intentionally remain residual and visible.
                Ok(Value::String(ids.get(id).cloned().unwrap_or_else(|| id.clone())))
            }
            _ => Err(format!("{field} must be a nonempty string or null")),
        }
    }

    fn normalize_message_at_root(&self, value: &Value, at_root: bool) -> Result<Value, String> {
        match value {
            Value::Object(map) => {
                let is_tool_call = matches!(
                    map.get("type").and_then(Value::as_str),
                    Some("toolCall" | "tool_call")
                );
                if is_tool_call {
                    map.get("id")
                        .and_then(Value::as_str)
                        .filter(|id| !id.is_empty())
                        .ok_or("tool call has no required ID")?;
                }
                let mut out = Map::new();
                for (key, value) in map {
                    let normalized = if at_root && key == "timestamp" {
                        Self::normalize_timestamp(value)?
                    } else if at_root && key == "fromId" {
                        Self::normalize_link(value, &self.entry_ids, key)?
                    } else if key == "toolCallId" {
                        Self::normalize_link(value, &self.tool_ids, key)?
                    } else if is_tool_call && key == "id" {
                        Self::normalize_link(value, &self.tool_ids, key)?
                    } else if key == "details" {
                        self.normalize_other(value)
                    } else {
                        self.normalize_message_at_root(value, false)?
                    };
                    out.insert(key.clone(), normalized);
                }
                Ok(Value::Object(out))
            }
            Value::Array(values) => values
                .iter()
                .map(|value| self.normalize_message_at_root(value, false))
                .collect::<Result<Vec<_>, _>>()
                .map(Value::Array),
            Value::String(text) => Ok(Value::String(self.private_root_text(text))),
            scalar => Ok(scalar.clone()),
        }
    }

    fn normalize_entry(&self, entry: &SessionEntry) -> Result<Value, String> {
        let mut out = Map::new();
        for (key, value) in entry {
            let normalized = match key.as_str() {
                "id" => Self::normalize_link(value, &self.entry_ids, key)?,
                "parentId" | "firstKeptEntryId" | "fromId" => {
                    Self::normalize_link(value, &self.entry_ids, key)?
                }
                "timestamp" => Self::normalize_timestamp(value)?,
                "message" => self.normalize_message_at_root(value, true)?,
                _ => self.normalize_other(value),
            };
            out.insert(key.clone(), normalized);
        }
        Ok(Value::Object(out))
    }

    fn normalize_header(&self, header: &SessionEntry) -> Result<Value, String> {
        let mut out = Map::new();
        for (key, value) in header {
            let normalized = match key.as_str() {
                "id" => Value::String("session".into()),
                "timestamp" => Self::normalize_timestamp(value)?,
                _ => self.normalize_other(value),
            };
            out.insert(key.clone(), normalized);
        }
        Ok(Value::Object(out))
    }

    fn normalize_agent_message(&self, message: &AgentMessage) -> Result<Value, String> {
        self.normalize_message_at_root(
            &serde_json::to_value(message).map_err(|error| error.to_string())?,
            true,
        )
    }

    fn normalize_entry_id(&self, id: &str) -> String {
        self.entry_ids.get(id).cloned().unwrap_or_else(|| id.to_string())
    }

    fn normalize_tool_id(&self, id: &str) -> String {
        self.tool_ids.get(id).cloned().unwrap_or_else(|| id.to_string())
    }
}

struct ObservedWriterContext {
    live_messages: Vec<AgentMessage>,
    context_messages: Vec<AgentMessage>,
    thinking_level: String,
    service_tier: ServiceTier,
    model: Option<SessionContextModel>,
}

struct SessionProjectionRecord {
    projection: Value,
    observed_live: Option<Value>,
}

fn session_model_json(model: &Option<SessionContextModel>) -> Value {
    match model {
        Some(model) => json!({"provider": model.provider, "model_id": model.model_id}),
        None => Value::Null,
    }
}

fn availability_for_keys(entries: &[SessionEntry], keys: &[&str]) -> &'static str {
    let found = entries.iter().any(|entry| {
        let value = Value::Object(entry.clone());
        keys.iter().any(|key| value_contains_key(&value, key))
    });
    if found { "emitted" } else { "not_emitted" }
}


fn pending_protected_item(
    test_name: &str,
    coverage_aspect: &str,
) -> Value {
    json!({
        "applicability": "not_exercised_by_fixture",
        "reason_code": "PROTECTED_ITEM_ABSENT_FROM_FIXTURE",
        "verdict": "not_ready_actual_per_arm_named_receipts_required",
        "value": null,
        "supporting_test_refs": [],
        "required_named_test": {
            "target": "compaction_recovery",
            "test_name": test_name,
            "coverage_aspect": coverage_aspect,
            "baseline_and_candidate_result_records_and_raw_log_sha256": "not_ready",
            "verified_nonzero_test_pass_per_arm": "not_ready",
        },
    })
}

/// These are intentionally incomplete source-export holders, never a completed
/// pair's operation_evidence. The external validator must replace not_ready
/// with independently checked actual values and nonempty per-arm receipts.
fn replay_operation_evidence_pending() -> Value {
    json!({
        "operation_profile": "replay",
        "record_status": "not_ready_not_a_complete_v2_pair_record",
        "fresh_process_session_reopen": {
            "applicability": "applicable",
            "reason_code": "APPLICABLE",
            "verdict": "not_ready_real_distinct_pid_reader_required",
            "value": null,
            "supporting_test_refs": [],
        },
        "protected_goals": pending_protected_item(
            "near_limit_auto_compaction_retains_goals_constraints_subagent_and_evidence",
            "goals retained through compaction",
        ),
        "protected_constraints": pending_protected_item(
            "near_limit_auto_compaction_retains_goals_constraints_subagent_and_evidence",
            "constraints retained through compaction",
        ),
        "protected_evidence": pending_protected_item(
            "near_limit_auto_compaction_retains_goals_constraints_subagent_and_evidence",
            "evidence retained through compaction",
        ),
        "pending_subagent": pending_protected_item(
            "compaction_with_active_subagent_preserves_the_pending_result",
            "active subagent and pending result retained",
        ),
        "tool_result_identity": pending_protected_item(
            "tool_output_compaction_preserves_result_identity_and_count",
            "tool result identity and count retained",
        ),
        "post_restart_provider_continuation": {
            "applicability": "not_exercised_by_fixture",
            "reason_code": "REOPEN_HAS_NO_ADDITIONAL_TURNS",
            "verdict": "not_ready_actual_per_arm_named_receipts_required",
            "value": null,
            "supporting_test_refs": [],
            "required_named_tests": [
                {
                    "target": "compaction_turn_flow",
                    "test_name": "d2_interactive_turn_crossing_threshold_compacts_and_continues",
                    "coverage_aspect": "interactive zero-turn reopen continuation",
                },
                {
                    "target": "compaction_turn_flow",
                    "test_name": "d2_queued_user_prompt_completes_after_threshold_compaction",
                    "coverage_aspect": "queued zero-turn reopen continuation",
                },
            ],
            "additional_user_or_provider_turns_in_reader": 0,
            "reader_fact_is_not_a_continuation_pass": true,
        },
    })
}

fn projection_tool_identities(
    rebuilt: &SessionContextWithEntryIds,
    normalizer: &SessionNormalizer,
) -> Result<Value, String> {
    struct CallRecord {
        raw_id: String,
        normalized_id: String,
        message_index: usize,
        content_index: usize,
        entry_id: String,
        result_message_indices: Vec<usize>,
    }
    let mut calls = Vec::new();
    let mut results = Vec::new();
    for (message_index, message) in rebuilt.messages.iter().enumerate() {
        let entry_id = rebuilt
            .entry_ids
            .get(message_index)
            .map(|id| normalizer.normalize_entry_id(id))
            .ok_or("rebuilt message/entry-ID alignment mismatch")?;
        match message {
            AgentMessage::Message(Message::Assistant(assistant)) => {
                for (content_index, block) in assistant.content.iter().enumerate() {
                    if let ContentBlock::ToolCall(call) = block {
                        if call.id.is_empty() {
                            return Err("rebuilt tool call ID missing".into());
                        }
                        calls.push(CallRecord {
                            normalized_id: normalizer.normalize_tool_id(&call.id),
                            raw_id: call.id.clone(),
                            message_index,
                            content_index,
                            entry_id: entry_id.clone(),
                            result_message_indices: Vec::new(),
                        });
                    }
                }
            }
            AgentMessage::Message(Message::ToolResult(result)) => {
                if result.tool_call_id.is_empty() {
                    return Err("rebuilt tool-result ID missing".into());
                }
                results.push((message_index, entry_id, result));
            }
            _ => {}
        }
    }
    let mut result_values = Vec::new();
    for (message_index, entry_id, result) in &results {
        let matching_calls: Vec<usize> = calls
            .iter()
            .filter(|call| call.raw_id == result.tool_call_id)
            .map(|call| call.message_index)
            .collect();
        for call in calls.iter_mut().filter(|call| call.raw_id == result.tool_call_id) {
            call.result_message_indices.push(*message_index);
        }
        result_values.push(json!({
            "tool_call_id": normalizer.normalize_tool_id(&result.tool_call_id),
            "message_index": message_index,
            "entry_id": entry_id,
            "matched_call_message_indices": matching_calls,
            "orphan_in_rebuilt": matching_calls.is_empty(),
        }));
    }
    let call_values: Vec<Value> = calls
        .iter()
        .map(|call| {
            json!({
                "id": call.normalized_id,
                "message_index": call.message_index,
                "content_index": call.content_index,
                "entry_id": call.entry_id,
                "result_message_indices": call.result_message_indices,
                "orphan_in_rebuilt": call.result_message_indices.is_empty(),
            })
        })
        .collect();
    let pairing = pairing_stats(&rebuilt.messages);
    let calls_with_result = calls
        .iter()
        .filter(|call| !call.result_message_indices.is_empty())
        .count();
    Ok(json!({
        "tool_calls": call_values,
        "tool_results": result_values,
        "pairing": {
            "tool_calls": pairing.tool_calls,
            "tool_results": pairing.tool_results,
            "matched_results": pairing.matched_results,
            "unmatched_results": pairing.unmatched_results,
            "calls_without_result": pairing.calls_without_result,
            "calls_with_result": calls_with_result,
        },
    }))
}

/// Shared by the writer and strict fresh-process reader. `observed` exists only
/// for the real writer and is deliberately returned outside the projection
/// hash domains.
fn session_projection(
    manager: &SessionManager,
    observed: Option<ObservedWriterContext>,
) -> Result<SessionProjectionRecord, String> {
    let header = manager.get_header().ok_or("session header missing")?;
    let entries = manager.get_entries();
    let branch = manager.get_branch(None);
    let leaf = manager.get_leaf_id();
    let rebuilt = manager.build_session_context_with_entry_ids(None);
    reject_provider_dependent_entries(&entries)?;
    reject_provider_dependent_entries(&branch)?;
    reject_provider_dependent_messages(&rebuilt.messages)?;
    if rebuilt.messages.len() != rebuilt.entry_ids.len() {
        return Err("rebuilt message/entry-ID alignment mismatch".into());
    }

    let normalizer = SessionNormalizer::new(&header, &entries)?;
    let normalized_header = normalizer.normalize_header(&header)?;
    let normalized_entries = entries
        .iter()
        .map(|entry| normalizer.normalize_entry(entry))
        .collect::<Result<Vec<_>, _>>()?;
    let normalized_branch: Vec<Value> = branch
        .iter()
        .map(|entry| {
            let id = entry
                .get("id")
                .and_then(Value::as_str)
                .ok_or("branch entry ID missing")?;
            Ok(Value::String(normalizer.normalize_entry_id(id)))
        })
        .collect::<Result<_, String>>()?;
    let normalized_leaf = match leaf.as_deref() {
        Some(id) => Value::String(normalizer.normalize_entry_id(id)),
        None => Value::Null,
    };

    let mut items = Vec::with_capacity(rebuilt.messages.len());
    let mut normalized_rebuilt_messages = Vec::with_capacity(rebuilt.messages.len());
    for (message, entry_id) in rebuilt.messages.iter().zip(&rebuilt.entry_ids) {
        let normalized_message = normalizer.normalize_agent_message(message)?;
        normalized_rebuilt_messages.push(normalized_message.clone());
        items.push(json!({
            "entry_id": normalizer.normalize_entry_id(entry_id),
            "normalized_message": normalized_message,
        }));
    }
    let tool_identities = projection_tool_identities(&rebuilt, &normalizer)?;
    let epilogue_start_index = usize::from(matches!(
        rebuilt.messages.first(),
        Some(AgentMessage::Custom(CustomAgentMessage::CompactionSummary { .. }))
    ));
    let rebuilt_value = json!({
        "items": items,
        "thinking_level": rebuilt.thinking_level,
        "service_tier": rebuilt.service_tier,
        "model": session_model_json(&rebuilt.model),
        "epilogue_start_index": epilogue_start_index,
        "tool_calls": tool_identities["tool_calls"],
        "tool_results": tool_identities["tool_results"],
        "pairing": tool_identities["pairing"],
    });
    let availability = json!({
        "provider_context": "not_emitted",
        "pending_subagent": availability_for_keys(&entries, &[
            "pendingSubagent", "pending_subagent", "pending_subagent_result",
            "activeSubagent", "active_subagent", "pendingResult", "pending_result",
        ]),
        "extensions": if entries.iter().any(|entry| {
            entry.get("type").and_then(Value::as_str) == Some("custom_message")
                || value_contains_key(&Value::Object(entry.clone()), "extension")
        }) { "emitted" } else { "not_emitted" },
        "input_release": availability_for_keys(&entries, &[
            "inputRelease", "input_release", "releaseInput",
        ]),
        "semantic_comparison_status": {
            "pending_subagent": "not_identifiable",
            "extensions": "not_identifiable",
            "input_release": "not_identifiable",
        },
        "semantics": "emission facts only; not_identifiable is not a pass; missing protected items require their own typed coverage evidence",
    });

    let projection = with_projection_hashes(json!({
        "normalization": {
            "version": NORMALIZER_VERSION,
            "entry_id_scheme": "durable file encounter order: entry:%06",
            "tool_id_scheme": "durable tool encounter order: tool:%06",
            "entry_count": entries.len(),
            "tool_id_count": normalizer.tool_ids.len(),
            "common_measurement_source_sha256": sha256_hex(include_bytes!("compaction_replay.rs")),
            "timestamps": "known entry/message timestamps tagged by original JSON type",
            "private_roots": "<PRIVATE_ROOT> from the replay header workspace parent and exact path variants",
        },
        "header": normalized_header,
        "durable": {
            "entries": normalized_entries,
            "branch_entry_ids": normalized_branch,
            "leaf_entry_id": normalized_leaf,
        },
        "rebuilt": rebuilt_value,
        "availability": availability,
    }))?;

    let observed_live = match observed {
        None => None,
        Some(observed) => {
            let normalized_live = observed
                .live_messages
                .iter()
                .map(|message| normalizer.normalize_agent_message(message))
                .collect::<Result<Vec<_>, _>>()?;
            let normalized_real_context = observed
                .context_messages
                .iter()
                .map(|message| normalizer.normalize_agent_message(message))
                .collect::<Result<Vec<_>, _>>()?;
            let writer_thinking = json!(observed.thinking_level);
            let writer_service = serde_json::to_value(observed.service_tier)
                .map_err(|error| error.to_string())?;
            let writer_model = session_model_json(&observed.model);
            let rebuilt_thinking = rebuilt_value["thinking_level"].clone();
            let rebuilt_service = rebuilt_value["service_tier"].clone();
            let rebuilt_model = rebuilt_value["model"].clone();
            let messages_match = normalized_live == normalized_rebuilt_messages;
            let real_context_messages_match = normalized_real_context == normalized_rebuilt_messages;
            let thinking_matches = writer_thinking == rebuilt_thinking;
            let service_matches = writer_service == rebuilt_service;
            let model_matches = writer_model == rebuilt_model;
            if !messages_match
                || !real_context_messages_match
                || !thinking_matches
                || !service_matches
                || !model_matches
            {
                return Err("real writer live/context does not equal normalized rebuilt context".into());
            }
            Some(json!({
                "messages": normalized_live,
                "real_context_messages": normalized_real_context,
                "thinking_level": writer_thinking,
                "service_tier": writer_service,
                "model": writer_model,
                "matches_rebuilt": messages_match,
                "real_context_messages_match_rebuilt": real_context_messages_match,
                "thinking_level_matches_rebuilt": thinking_matches,
                "service_tier_matches_rebuilt": service_matches,
                "model_matches_rebuilt": model_matches,
            }))
        }
    };
    Ok(SessionProjectionRecord {
        projection,
        observed_live,
    })
}

// ---------------------------------------------------------------------------
// replay subcommand
// ---------------------------------------------------------------------------

struct ReplayOptions {
    provider_delay_ms: u64,
    summary_failure: Option<String>,
    context_window: f64,
    keep_recent: Option<f64>,
    work: PathBuf,
    timeout_secs: u64,
    export_session_projection: bool,
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
    let summary_sha256 = sha256_hex(summary_text.as_bytes());
    let pairing = pairing_stats(&live);
    let session_file = session.session_file();
    let session_file_stats = session_file
        .as_deref()
        .map(|path| {
            let bytes = std::fs::metadata(path).map(|meta| meta.len()).unwrap_or(0);
            let lines = std::fs::read_to_string(path)
                .map(|body| body.lines().count())
                .unwrap_or(0);
            json!({ "path": path, "bytes": bytes, "lines": lines })
        })
        .unwrap_or(Value::Null);

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

    // All real writer/live comparisons and restored normalized bodies are
    // additive evidence after turn_and_compaction has ended at turn_ms.
    let projection_result: Option<Result<SessionProjectionRecord, String>> =
        if options.export_session_projection {
            Some((|| -> Result<SessionProjectionRecord, String> {
                let real_writer_context = session.build_session_context();
                let observed = ObservedWriterContext {
                    live_messages: live.clone(),
                    context_messages: real_writer_context.messages,
                    thinking_level: real_writer_context.thinking_level,
                    service_tier: real_writer_context.service_tier,
                    model: real_writer_context.model,
                };
                let manager = session.session_manager.lock().unwrap();
                session_projection(&manager, Some(observed))
            })())
        } else {
            None
        };

    session.dispose_async(Some(false)).await;
    let projection_record = projection_result.transpose()?;
    let closed_session_evidence = if options.export_session_projection {
        let path = session_file
            .as_deref()
            .ok_or("projection replay did not produce a closed session file")?;
        let closed_bytes = std::fs::read(path)
            .map_err(|error| format!("read disposed session file: {error}"))?;
        Some(json!({
            "path": path,
            "byte_count": closed_bytes.len(),
            "sha256": sha256_hex(&closed_bytes),
            "captured_after_dispose": true,
        }))
    } else {
        None
    };

    let pairing_json = json!({
        "tool_calls": pairing.tool_calls,
        "tool_results": pairing.tool_results,
        "matched_results": pairing.matched_results,
        "unmatched_results": pairing.unmatched_results,
        "calls_without_result": pairing.calls_without_result,
    });
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
    let mut case_result = json!({
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
        },
        "phases": phase_rows(&events),
        "provider_attempts": provider_attempt_rows(&events),
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
            "branch_entry_count": branch.len(),
        },
        "digests": {
            "retained_state_sha256": retained_state_sha256,
            "summary_sha256": summary_sha256,
            "pairing": pairing_json,
            "provider_requests_sha256": provider_requests_sha256,
        },
        "session_file": session_file_stats,
        "determinism_key": determinism_key,
    });
    if let Some(record) = projection_record {
        case_result["session_projection"] = record.projection;
        case_result["observed_live"] = record
            .observed_live
            .ok_or("real writer projection requires observed_live evidence")?;
        case_result["closed_session_file"] = closed_session_evidence
            .ok_or("real writer projection requires disposed-file byte evidence")?;
        case_result["operation_evidence_pending"] = replay_operation_evidence_pending();
    }
    Ok(case_result)
}


// ---------------------------------------------------------------------------
// Strict synchronous fresh-process reader
// ---------------------------------------------------------------------------

fn path_metadata_is_link(metadata: &std::fs::Metadata) -> bool {
    if metadata.file_type().is_symlink() {
        return true;
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x400;
        if metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
            return true;
        }
    }
    false
}

fn reject_link_components(path: &Path) -> Result<(), String> {
    for ancestor in path.ancestors() {
        match std::fs::symlink_metadata(ancestor) {
            Ok(metadata) if path_metadata_is_link(&metadata) => {
                return Err(format!("symlink/reparse path component rejected: {}", ancestor.display()));
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => break,
            Err(error) => return Err(format!("inspect path component {}: {error}", ancestor.display())),
        }
    }
    Ok(())
}

fn reject_parent_components(path: &Path) -> Result<(), String> {
    if path
        .components()
        .any(|component| matches!(component, Component::ParentDir))
    {
        Err(format!("parent-directory path component is forbidden: {}", path.display()))
    } else {
        Ok(())
    }
}

fn rooted_absolute(root: &Path, value: &Path) -> Result<PathBuf, String> {
    reject_parent_components(value)?;
    if value.is_absolute() {
        Ok(value.to_path_buf())
    } else {
        Ok(root.join(value))
    }
}

fn modified_unix_nanos(metadata: &std::fs::Metadata) -> Option<u64> {
    metadata
        .modified()
        .ok()?
        .duration_since(UNIX_EPOCH)
        .ok()
        .map(|duration| duration.as_nanos().min(u64::MAX as u128) as u64)
}

#[cfg(windows)]
fn windows_by_handle_identity(
    path: &Path,
    metadata: &std::fs::Metadata,
) -> Result<(Option<u32>, Option<u32>, Option<u64>), String> {
    use std::os::windows::fs::{MetadataExt, OpenOptionsExt};
    use std::os::windows::io::AsRawHandle;

    const FILE_SHARE_READ: u32 = 0x00000001;
    const FILE_SHARE_WRITE: u32 = 0x00000002;
    const FILE_SHARE_DELETE: u32 = 0x00000004;
    const FILE_FLAG_BACKUP_SEMANTICS: u32 = 0x02000000;
    const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x00200000;
    const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x00000400;

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct ByHandleFileInformation {
        dw_file_attributes: u32,
        ft_creation_time_low: u32,
        ft_creation_time_high: u32,
        ft_last_access_time_low: u32,
        ft_last_access_time_high: u32,
        ft_last_write_time_low: u32,
        ft_last_write_time_high: u32,
        dw_volume_serial_number: u32,
        n_file_size_high: u32,
        n_file_size_low: u32,
        n_number_of_links: u32,
        n_file_index_high: u32,
        n_file_index_low: u32,
    }

    const _: [(); 52] = [(); std::mem::size_of::<ByHandleFileInformation>()];
    const _: [(); 4] = [(); std::mem::align_of::<ByHandleFileInformation>()];

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetFileInformationByHandle(
            file: *mut std::ffi::c_void,
            information: *mut ByHandleFileInformation,
        ) -> i32;
    }

    // Query-only access avoids asking for file-data read/write permission.
    // BACKUP_SEMANTICS permits a directory handle. OPEN_REPARSE_POINT avoids
    // following a reparse target to replace its identity with the target's.
    let opened_file = std::fs::OpenOptions::new()
        .access_mode(0)
        .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE)
        .custom_flags(FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT)
        .open(path)
        .map_err(|error| format!("open metadata handle {}: {error}", path.display()))?;
    let opened_metadata = opened_file
        .metadata()
        .map_err(|error| format!("read opened metadata {}: {error}", path.display()))?;
    if path_metadata_is_link(&opened_metadata)
        || opened_metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0
    {
        return Err(format!(
            "symlink/reparse metadata handle rejected: {}",
            path.display()
        ));
    }
    // A separate handle is not an atomic snapshot of the passed Metadata.
    // Reject an ordinary path whose type, size, attributes, or stable times
    // changed between those queries; whole before/after stamps still compare.
    if opened_metadata.is_file() != metadata.is_file()
        || opened_metadata.is_dir() != metadata.is_dir()
        || opened_metadata.len() != metadata.len()
        || opened_metadata.file_attributes() != metadata.file_attributes()
        || opened_metadata.creation_time() != metadata.creation_time()
        || opened_metadata.last_write_time() != metadata.last_write_time()
    {
        return Err(format!(
            "path changed while reading metadata: {}",
            path.display()
        ));
    }

    // The C output is thirteen u32 slots: one attribute slot, six FILETIME
    // slots, and six volume/size/link/index slots. Unused time/size fields
    // time/size fields remain present so all following field offsets are exact.
    // Default supplies valid all-integer zero initialization, never a fallback
    // identity. The successful API call supplies every value copied below.
    let mut information = ByHandleFileInformation::default();
    // SAFETY: opened_file owns the borrowed raw handle throughout this call.
    // No ownership is transferred and CloseHandle is never called directly.
    // `information` is a live, writable, exactly 52-byte #[repr(C)] object of
    // alignment four. Its pointer remains valid for the call's duration. No
    // raw handle is cached or returned; File drops only after scalar copying.
    let success = unsafe {
        GetFileInformationByHandle(opened_file.as_raw_handle(), &mut information)
    };
    if success == 0 {
        return Err(format!(
            "GetFileInformationByHandle failed for {}: {}",
            path.display(),
            std::io::Error::last_os_error()
        ));
    }
    if information.dw_file_attributes & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
        return Err(format!(
            "symlink/reparse identity rejected: {}",
            path.display()
        ));
    }
    // These are actual path-metadata identities: Some(real u32/u32/u64), in
    // the original field types. High/low index halves stay in Win32 order.
    // open/query failure returned Err above, never null-success or fabricated
    // timestamp/hash identities. File remains alive through these copies.
    Ok((
        Some(information.n_number_of_links),
        Some(information.dw_volume_serial_number),
        Some(
            ((information.n_file_index_high as u64) << 32)
                | information.n_file_index_low as u64,
        ),
    ))
}

fn stable_file_stamp(path: &Path, metadata: &std::fs::Metadata) -> Result<Value, String> {
    let kind = if metadata.is_file() {
        "file"
    } else if metadata.is_dir() {
        "directory"
    } else {
        return Err(format!("non-regular session-root entry rejected: {}", path.display()));
    };
    let mut stamp = json!({
        "kind": kind,
        "length": metadata.len(),
        "modified_unix_nanos": modified_unix_nanos(metadata),
        "read_only": metadata.permissions().readonly(),
    });
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        stamp["unix_dev"] = json!(metadata.dev());
        stamp["unix_ino"] = json!(metadata.ino());
        stamp["unix_mode"] = json!(metadata.mode());
        stamp["unix_nlink"] = json!(metadata.nlink());
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        stamp["windows_file_attributes"] = json!(metadata.file_attributes());
        stamp["windows_creation_time"] = json!(metadata.creation_time());
        stamp["windows_last_write_time"] = json!(metadata.last_write_time());
        let (number_of_links, volume_serial_number, file_index) =
            windows_by_handle_identity(path, metadata)?;
        stamp["windows_number_of_links"] = json!(number_of_links);
        stamp["windows_volume_serial_number"] = json!(volume_serial_number);
        stamp["windows_file_index"] = json!(file_index);
    }
    Ok(stamp)
}

fn relative_slash_path(root: &Path, path: &Path) -> Result<String, String> {
    let relative = path
        .strip_prefix(root)
        .map_err(|_| format!("{} escapes root {}", path.display(), root.display()))?;
    Ok(relative
        .components()
        .map(|component| component.as_os_str().to_string_lossy().to_string())
        .collect::<Vec<_>>()
        .join("/"))
}

fn snapshot_root_inventory(root: &Path) -> Result<BTreeMap<String, Value>, String> {
    fn visit(root: &Path, path: &Path, out: &mut BTreeMap<String, Value>) -> Result<(), String> {
        let metadata = std::fs::symlink_metadata(path)
            .map_err(|error| format!("inventory metadata {}: {error}", path.display()))?;
        if path_metadata_is_link(&metadata) {
            return Err(format!("symlink/reparse inventory entry rejected: {}", path.display()));
        }
        let relative = relative_slash_path(root, path)?;
        out.insert(relative, stable_file_stamp(path, &metadata)?);
        if metadata.is_dir() {
            let mut children = std::fs::read_dir(path)
                .map_err(|error| format!("inventory read directory {}: {error}", path.display()))?
                .map(|entry| entry.map(|entry| entry.path()).map_err(|error| error.to_string()))
                .collect::<Result<Vec<_>, _>>()?;
            children.sort();
            for child in children {
                visit(root, &child, out)?;
            }
        }
        Ok(())
    }
    let mut inventory = BTreeMap::new();
    visit(root, root, &mut inventory)?;
    Ok(inventory)
}

fn without_directory_mtime(stamp: &Value) -> Value {
    let mut stamp = stamp.clone();
    if let Some(object) = stamp.as_object_mut() {
        object.remove("modified_unix_nanos");
        object.remove("windows_last_write_time");
    }
    stamp
}

fn assert_inventory_unchanged(
    before: &BTreeMap<String, Value>,
    after: &BTreeMap<String, Value>,
    output_relative: &str,
    output_parent_directories: &[String],
    output_is_expected: bool,
) -> Result<(), String> {
    let mut paths = before.keys().chain(after.keys()).collect::<Vec<_>>();
    paths.sort();
    paths.dedup();
    let mut unexpected = Vec::new();
    for path in paths {
        let old = before.get(path);
        let new = after.get(path);
        let unchanged = match (old, new) {
            (Some(before_stamp), Some(after_stamp))
                if path.is_empty()
                    && before_stamp["kind"] == "directory"
                    && after_stamp["kind"] == "directory" =>
            {
                without_directory_mtime(before_stamp) == without_directory_mtime(after_stamp)
            }
            _ => old == new,
        };
        if unchanged {
            continue;
        }
        // The sole new entry is the create_new report. Its containing directory
        // mtime and those of existing ancestors may changed only because of it.
        if output_is_expected && path == output_relative && old.is_none() && new.is_some() {
            continue;
        }
        if output_is_expected
            && output_parent_directories.contains(path)
            && old.is_some_and(|stamp| stamp["kind"] == "directory")
            && new.is_some_and(|stamp| stamp["kind"] == "directory")
            && without_directory_mtime(old.expect("checked old")) == without_directory_mtime(new.expect("checked new"))
        {
            continue;
        }
        unexpected.push(format!("{path}: before={old:?}; after={new:?}"));
    }
    if unexpected.is_empty() {
        Ok(())
    } else {
        unexpected.sort();
        Err(format!("unexpected new/changed session-root entries: {}", unexpected.join(", ")))
    }
}

struct ParsedReplayShape {
    case_root: PathBuf,
    session_id: String,
}

fn safe_single_component_session_id(id: &str) -> bool {
    !id.is_empty()
        && id != "."
        && id != ".."
        && !id.chars().any(|character| {
            matches!(character, '/' | '\\' | ':') || character.is_control()
        })
}

/// Validation-only recursive JSON value. It checks decoded object keys without
/// constructing an ordinary Value/Map that would silently discard duplicates.
#[derive(Clone, Copy)]
struct StrictJsonValue;

impl<'de> Deserialize<'de> for StrictJsonValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        deserializer.deserialize_any(StrictJsonValue)
    }
}

impl<'de> Visitor<'de> for StrictJsonValue {
    type Value = StrictJsonValue;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("any valid JSON value with no duplicate key in each object")
    }

    fn visit_bool<E>(self, _value: bool) -> Result<Self::Value, E> {
        Ok(StrictJsonValue)
    }

    fn visit_i64<E>(self, _value: i64) -> Result<Self::Value, E> {
        Ok(StrictJsonValue)
    }

    fn visit_u64<E>(self, _value: u64) -> Result<Self::Value, E> {
        Ok(StrictJsonValue)
    }

    fn visit_f64<E>(self, _value: f64) -> Result<Self::Value, E> {
        Ok(StrictJsonValue)
    }

    fn visit_str<E>(self, _value: &str) -> Result<Self::Value, E> {
        Ok(StrictJsonValue)
    }

    fn visit_borrowed_str<E>(self, _value: &'de str) -> Result<Self::Value, E> {
        Ok(StrictJsonValue)
    }

    fn visit_string<E>(self, _value: String) -> Result<Self::Value, E> {
        Ok(StrictJsonValue)
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(StrictJsonValue)
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(StrictJsonValue)
    }

    fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        StrictJsonValue::deserialize(deserializer)
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        while seq.next_element::<StrictJsonValue>()?.is_some() {}
        Ok(StrictJsonValue)
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        // Exactly one set per JSON object, not a global/descendant set. Serde
        // decodes escapes in String keys, so escaped-equivalent repeats collide.
        let mut keys = HashSet::new();
        while let Some(key) = map.next_key::<String>()? {
            if !keys.insert(key.clone()) {
                return Err(de::Error::custom(format!(
                    "DUPLICATE_OBJECT_KEY: {key:?}"
                )));
            }
            map.next_value::<StrictJsonValue>()?;
        }
        Ok(StrictJsonValue)
    }
}

fn reject_duplicate_json_object_keys(line: &str, row: usize) -> Result<(), String> {
    let mut parser = serde_json::Deserializer::from_str(line);
    if let Err(error) = StrictJsonValue::deserialize(&mut parser) {
        let message = error.to_string();
        if error.classify() == serde_json::error::Category::Data
            && message.starts_with("DUPLICATE_OBJECT_KEY: ")
        {
            return Err(format!(
                "DUPLICATE_OBJECT_KEY in session JSON row {row}: {message}"
            ));
        }
        return Err(format!("invalid session JSON at row {row}: {error}"));
    }
    parser
        .end()
        .map_err(|error| format!("invalid session JSON at row {row}: {error}"))
}

fn parse_strict_replay_shape(
    bytes: &[u8],
    input: &Path,
    work_root: &Path,
) -> Result<ParsedReplayShape, String> {
    if bytes.is_empty() {
        return Err("EMPTY_SESSION_FILE".into());
    }
    if bytes.contains(&0) {
        return Err("NUL_SESSION_TAIL_REQUIRES_REPAIR_REJECTED".into());
    }
    if !bytes.ends_with(b"\n") {
        return Err("UNTERMINATED_SESSION_TAIL_REQUIRES_REPAIR_REJECTED".into());
    }
    let body = std::str::from_utf8(bytes).map_err(|_| "session file is not UTF-8".to_string())?;
    let mut entries = Vec::new();
    for (index, line) in body.split_terminator('\n').enumerate() {
        if line.trim().is_empty() {
            return Err(format!("blank session JSONL row {} rejected", index + 1));
        }
        // Reject duplicate raw object keys before Value/Map can collapse them.
        // Ordinary conversion and every existing shape/provider check stay intact.
        reject_duplicate_json_object_keys(line, index + 1)?;
        let value: Value = serde_json::from_str(line)
            .map_err(|error| format!("invalid session JSON at row {}: {error}", index + 1))?;
        let object = value
            .as_object()
            .cloned()
            .ok_or_else(|| format!("session row {} is not an object", index + 1))?;
        if index == 0 && object.get("type").and_then(Value::as_str) != Some("session") {
            return Err("HEADERLESS_SESSION_REJECTED".into());
        }
        if index != 0 && object.get("type").and_then(Value::as_str) == Some("session") {
            return Err("duplicate session header rejected".into());
        }
        entries.push(object);
    }
    let header = entries.first().cloned().ok_or("HEADERLESS_SESSION_REJECTED")?;
    if header.get("version").and_then(Value::as_i64) != Some(CURRENT_SESSION_VERSION) {
        return Err("SESSION_VERSION_MIGRATION_REQUIRED_REJECTED".into());
    }
    let session_id = header
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| safe_single_component_session_id(id))
        .ok_or("unsafe session header ID")?
        .to_string();
    if header
        .get("parentSessionId")
        .is_some_and(|value| !value.is_null())
        || header.get("parentSession").is_some_and(|value| !value.is_null())
    {
        return Err("PARENT_SESSION_DEPTH_MIGRATION_UNSUPPORTED".into());
    }
    let mut required_ids = HashSet::new();
    for entry in entries.iter().skip(1) {
        let id = entry
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
            .ok_or("MISSING_REQUIRED_ENTRY_ID")?
            .to_string();
        if !required_ids.insert(id.clone()) {
            return Err(format!("DUPLICATE_REQUIRED_ENTRY_ID: {id}"));
        }
    }
    for entry in entries.iter().skip(1) {
        reject_provider_dependent_value(&Value::Object(entry.clone()))?;
        if entry.get("type").and_then(Value::as_str) == Some("compaction") {
            let details = entry.get("details").cloned().unwrap_or(Value::Null);
            if has_provider_checkpoint(&details) || get_provider_checkpoint(&details).is_some() {
                return Err("PROVIDER_DEPENDENT_AUXILIARY_UNSUPPORTED".into());
            }
        }
    }

    // This experiment's replay tree is a single chronological chain. Reject a
    // damaged/ambiguous branch rather than allowing open/build-index recovery.
    let entries_without_header = &entries[1..];
    if entries_without_header.is_empty() {
        return Err("non-replay session has no entries".into());
    }
    let mut previous_id: Option<&str> = None;
    for entry in entries_without_header {
        let id = entry
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
            .ok_or("MISSING_REQUIRED_ENTRY_ID")?;
        let parent = match entry.get("parentId") {
            None | Some(Value::Null) => None,
            Some(Value::String(id)) if !id.is_empty() => Some(id.as_str()),
            _ => return Err("invalid durable parentId type".into()),
        };
        if parent != previous_id {
            return Err("nonlinear/repaired replay lineage rejected".into());
        }
        previous_id = Some(id);
    }
    let compactions: Vec<&SessionEntry> = entries_without_header
        .iter()
        .filter(|entry| entry.get("type").and_then(Value::as_str) == Some("compaction"))
        .collect();
    if compactions.len() != 1 {
        return Err("exactly one intended branch compaction is required".into());
    }
    let compaction = compactions[0];
    if compaction.get("summary").and_then(Value::as_str).is_none()
        || !compaction
            .get("tokensBefore")
            .and_then(Value::as_f64)
            .is_some_and(f64::is_finite)
    {
        return Err("unsafe replay compaction shape".into());
    }
    let first_kept = compaction
        .get("firstKeptEntryId")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .ok_or("compaction firstKeptEntryId missing")?;
    if !required_ids.contains(first_kept) {
        return Err("compaction firstKeptEntryId does not resolve".into());
    }
    let synthetic_model = entries_without_header.iter().any(|entry| {
        let message = entry.get("message");
        message.and_then(|message| message.get("role")).and_then(Value::as_str) == Some("assistant")
            && message.and_then(|message| message.get("model")).and_then(Value::as_str)
                == Some("compaction-replay-model")
            && message
                .and_then(|message| message.get("api"))
                .and_then(Value::as_str)
                .is_some_and(|api| api.starts_with("harness-api-"))
            && message
                .and_then(|message| message.get("provider"))
                .and_then(Value::as_str)
                .is_some_and(|provider| provider.starts_with("harness-provider-"))
    });
    if !synthetic_model {
        return Err("NON_REPLAY_OR_PRODUCTION_MODEL_SHAPE_REJECTED".into());
    }

    let cwd_raw = header
        .get("cwd")
        .and_then(Value::as_str)
        .ok_or("session header cwd missing")?;
    let cwd_input = PathBuf::from(cwd_raw);
    if !cwd_input.is_absolute() {
        return Err("replay header cwd must be absolute".into());
    }
    reject_parent_components(&cwd_input)?;
    reject_link_components(&cwd_input)?;
    let cwd = std::fs::canonicalize(&cwd_input).map_err(|error| format!("canonicalize replay cwd: {error}"))?;
    if cwd.file_name().and_then(|name| name.to_str()) != Some("workspace")
        || !cwd.starts_with(work_root)
    {
        return Err("NON_REPLAY_WORKSPACE_OR_ESCAPE_REJECTED".into());
    }
    let case_root = cwd
        .parent()
        .ok_or("replay case root missing")?
        .to_path_buf();
    if !case_root.starts_with(work_root) {
        return Err("replay case root escapes private work root".into());
    }
    let sessions_root = std::fs::canonicalize(case_root.join("sessions"))
        .map_err(|error| format!("canonicalize replay sessions root: {error}"))?;
    if !input.starts_with(&sessions_root) || !sessions_root.starts_with(work_root) {
        return Err("NON_REPLAY_SESSION_DIRECTORY_OR_ESCAPE_REJECTED".into());
    }
    Ok(ParsedReplayShape {
        case_root,
        session_id,
    })
}

struct ReopenPreflight {
    work_root: PathBuf,
    input: PathBuf,
    output: PathBuf,
    output_relative: String,
    output_parent_directories: Vec<String>,
    before_bytes: Vec<u8>,
    before_sha256: String,
    before_input_stamp: Value,
    before_inventory: BTreeMap<String, Value>,
    replay: ParsedReplayShape,
}

fn reopen_preflight(cli: &Cli) -> Result<ReopenPreflight, String> {
    cli.reject_unknown_options(&["session-file", "work-root", "out", "git"])?;
    let work_root_raw = PathBuf::from(cli.required("work-root")?);
    reject_parent_components(&work_root_raw)?;
    reject_link_components(&work_root_raw)?;
    let work_root = std::fs::canonicalize(&work_root_raw)
        .map_err(|error| format!("canonicalize private work root: {error}"))?;
    let root_metadata = std::fs::symlink_metadata(&work_root)
        .map_err(|error| format!("work-root metadata: {error}"))?;
    if !root_metadata.is_dir() || path_metadata_is_link(&root_metadata) {
        return Err("work-root must be a real non-symlink directory".into());
    }

    let input_argument = rooted_absolute(&work_root, &PathBuf::from(cli.required("session-file")?))?;
    reject_link_components(&input_argument)?;
    let input = std::fs::canonicalize(&input_argument)
        .map_err(|error| format!("canonicalize existing session file: {error}"))?;
    if input == work_root || !input.starts_with(&work_root) {
        return Err("session file escapes private work root".into());
    }
    let input_metadata = std::fs::symlink_metadata(&input)
        .map_err(|error| format!("session-file metadata: {error}"))?;
    if !input_metadata.is_file() || path_metadata_is_link(&input_metadata) {
        return Err("session file must be a regular non-symlink file".into());
    }
    reject_link_components(&input)?;
    let before_input_stamp = stable_file_stamp(&input, &input_metadata)?;

    let output_argument = rooted_absolute(&work_root, &PathBuf::from(cli.required("out")?))?;
    reject_link_components(&output_argument)?;
    let output_parent = output_argument
        .parent()
        .ok_or("reader output has no parent directory")?;
    let output_name = output_argument
        .file_name()
        .filter(|name| !name.is_empty())
        .ok_or("reader output must name a new file")?;
    let canonical_output_parent = std::fs::canonicalize(output_parent)
        .map_err(|error| format!("canonicalize reader output parent: {error}"))?;
    if !canonical_output_parent.starts_with(&work_root) {
        return Err("reader output escapes private work root".into());
    }
    let output = canonical_output_parent.join(output_name);
    if output == input {
        return Err("reader output must be disjoint from input".into());
    }
    if output.extension().and_then(|name| name.to_str()) != Some("json") {
        return Err("reader output must be a .json report".into());
    }
    if output.parent() == input.parent() {
        return Err("reader output cannot be a sibling of the session input".into());
    }
    match std::fs::symlink_metadata(&output) {
        Ok(_) => return Err("reader output already exists; create_new refuses to replace it".into()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(format!("inspect reader output: {error}")),
    }
    let output_relative = relative_slash_path(&work_root, &output)?;
    if output_relative.is_empty() {
        return Err("reader output must be beneath the root, not its directory".into());
    }
    let parent_relative = relative_slash_path(&work_root, &canonical_output_parent)?;
    let mut output_parent_directories = vec![String::new()];
    if !parent_relative.is_empty() {
        let mut current = PathBuf::new();
        for component in Path::new(&parent_relative).components() {
            current.push(component);
            output_parent_directories.push(current.to_string_lossy().replace('\\', "/"));
        }
    }

    let before_bytes = std::fs::read(&input).map_err(|error| format!("read original session: {error}"))?;
    let before_sha256 = sha256_hex(&before_bytes);
    let before_inventory = snapshot_root_inventory(&work_root)?;
    let replay = parse_strict_replay_shape(&before_bytes, &input, &work_root)?;
    // Strict reads themselves must not race the pre-open byte/metadata source.
    let current_bytes = std::fs::read(&input).map_err(|error| format!("reread original session: {error}"))?;
    let current_metadata = std::fs::symlink_metadata(&input)
        .map_err(|error| format!("original session metadata after preflight: {error}"))?;
    if current_bytes != before_bytes
        || stable_file_stamp(&input, &current_metadata)? != before_input_stamp
    {
        return Err("session changed during preflight; production open was not called".into());
    }
    let after_preflight_inventory = snapshot_root_inventory(&work_root)?;
    assert_inventory_unchanged(
        &before_inventory,
        &after_preflight_inventory,
        &output_relative,
        &output_parent_directories,
        false,
    )?;
    Ok(ReopenPreflight {
        work_root,
        input,
        output,
        output_relative,
        output_parent_directories,
        before_bytes,
        before_sha256,
        before_input_stamp,
        before_inventory,
        replay,
    })
}

fn reader_stage_bytes(preflight: &ReopenPreflight, stage: &str) -> Result<Value, String> {
    let bytes = std::fs::read(&preflight.input)
        .map_err(|error| format!("{stage}: read session input: {error}"))?;
    let digest = sha256_hex(&bytes);
    if bytes != preflight.before_bytes || digest != preflight.before_sha256 {
        return Err(format!("{stage}: session bytes or SHA-256 changed"));
    }
    let metadata = std::fs::symlink_metadata(&preflight.input)
        .map_err(|error| format!("{stage}: session metadata: {error}"))?;
    let stamp = stable_file_stamp(&preflight.input, &metadata)?;
    if stamp != preflight.before_input_stamp {
        return Err(format!("{stage}: session stable metadata changed"));
    }
    Ok(json!({
        "stage": stage,
        "byte_count": bytes.len(),
        "sha256": digest,
        "matches_before_open": true,
        "stable_metadata_unchanged": true,
    }))
}

fn reader_stage_inventory(
    preflight: &ReopenPreflight,
    output_is_expected: bool,
) -> Result<BTreeMap<String, Value>, String> {
    let current = snapshot_root_inventory(&preflight.work_root)?;
    assert_inventory_unchanged(
        &preflight.before_inventory,
        &current,
        &preflight.output_relative,
        &preflight.output_parent_directories,
        output_is_expected,
    )?;
    Ok(current)
}

/// The opener is injected only to let negative in-file tests prove preflight
/// returns before it can run. Production binds the one real public API below.
fn verify_reopen_using(
    cli: &Cli,
    open_manager: impl FnOnce(&str) -> Result<SessionManager, String>,
) -> Result<i32, String> {
    let preflight = reopen_preflight(cli)?;
    let input_string = preflight.input.to_string_lossy().to_string();
    let manager = match open_manager(&input_string) {
        Ok(manager) => manager,
        Err(open_error) => {
            // An error does not prove that production open left the input and
            // siblings untouched. Check both independently and never restore
            // over altered bytes or remove quarantine/unexpected evidence.
            let byte_check = reader_stage_bytes(&preflight, "immediately_after_open_error");
            let inventory_check = reader_stage_inventory(&preflight, false);
            return match (byte_check, inventory_check) {
                (Ok(_), Ok(_)) => Err(format!(
                    "SessionManager::open failed; post-error byte/inventory checks unchanged: {open_error}"
                )),
                (bytes, inventory) => Err(format!(
                    "SessionManager::open failed: {open_error}; post-error evidence check: bytes={}; root_inventory={}",
                    bytes.err().map_or_else(|| "unchanged".to_string(), |error| error),
                    inventory.err().map_or_else(|| "unchanged".to_string(), |error| error),
                )),
            };
        }
    };
    let after_open_result = (|| -> Result<(Value, BTreeMap<String, Value>), String> {
        Ok((
            reader_stage_bytes(&preflight, "immediately_after_open")?,
            reader_stage_inventory(&preflight, false)?,
        ))
    })();
    let (after_open, after_open_inventory) = match after_open_result {
        Ok(result) => result,
        Err(error) => {
            drop(manager);
            return Err(error);
        }
    };
    let projection_result = session_projection(&manager, None);
    drop(manager);
    let after_drop = reader_stage_bytes(&preflight, "after_drop")?;
    let after_drop_inventory = reader_stage_inventory(&preflight, false)?;
    let projection_record = projection_result?;

    // This is the only new filesystem entry permitted by the reader. The path
    // was strictly checked before open; create_new enforces disjointness again.
    let mut report_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&preflight.output)
        .map_err(|error| format!("create_new reader report {}: {error}", preflight.output.display()))?;
    let final_bytes = reader_stage_bytes(&preflight, "final")?;
    let final_inventory = reader_stage_inventory(&preflight, true)?;
    let current_exe = std::env::current_exe().map_err(|error| error.to_string())?;
    let reader_exe_sha256 = exe_sha256()?;
    let before = json!({
        "stage": "before_open",
        "byte_count": preflight.before_bytes.len(),
        "sha256": preflight.before_sha256,
        "matches_before_open": true,
        "stable_metadata_unchanged": true,
    });
    let report = json!({
        "schema": COMMON_READER_SCHEMA,
        "operation_profile": "verify-reopen",
        "raw_private_identities": {
            "work_root": preflight.work_root,
            "case_root": preflight.replay.case_root,
            "session_file": preflight.input,
            "session_id": preflight.replay.session_id,
            "output_report": preflight.output,
            "canonical_containment": true,
            "output_disjoint_create_new": true,
        },
        "reader": {
            "pid": std::process::id(),
            "argv": std::env::args().collect::<Vec<_>>(),
            "current_exe": current_exe,
            "exe_sha256": reader_exe_sha256,
            "exe_sha256_source": "computed-from-running-exe",
            "common_measurement_source_sha256": sha256_hex(include_bytes!("compaction_replay.rs")),
            "reader_schema": COMMON_READER_SCHEMA,
            "external_source_manifest_required": true,
            "git": cli.optional("git"),
            "external_identity_proof_required": [
                "parent-created pid/start/end/exit/timeout records",
                "reader PID differs from writer PID",
                "same-arm frozen executable/source hashes match",
                "final input and root inventory verification",
            ],
            "environment_epoch_records": "not_supplied_by_reader_required_from_external_runner",
        },
        "open_rebuild": {
            "sequence": [
                "strict preflight before production open",
                "SessionManager::open(path, None, None)",
                "session_projection/get_header/get_entries/get_branch/get_leaf_id",
                "build_session_context_with_entry_ids(None)",
                "drop(manager)",
            ],
            "open_rebuild_drop_only": true,
            "additional_user_or_provider_turns": 0,
            "in_process_restore_substitute": false,
            "provider_registry_or_endpoint_constructed": false,
            "repair_migration_quarantine_or_rewrite_performed": false,
            "manager_drop_completed": true,
        },
        "input_bytes": {
            "before_open": before,
            "immediately_after_open": after_open,
            "after_drop": after_drop,
            "final": final_bytes,
            "unchanged": true,
        },
        "root_inventory": {
            "scope": "root-relative stable identity/type/length/modification metadata; access time excluded",
            "before_open": preflight.before_inventory,
            "immediately_after_open": after_open_inventory,
            "after_drop": after_drop_inventory,
            "final": final_inventory,
            "unchanged_no_unexpected_sidecars": true,
            "sole_allowed_new_entry": preflight.output_relative,
        },
        "session_projection": projection_record.projection,
        "external_final_verification_required": true,
    });
    let encoded = serde_json::to_vec_pretty(&report).map_err(|error| error.to_string())?;
    report_file
        .write_all(&encoded)
        .and_then(|_| report_file.write_all(b"\n"))
        .map_err(|error| format!("write reader report: {error}"))?;
    report_file
        .sync_all()
        .map_err(|error| format!("sync reader report: {error}"))?;
    drop(report_file);
    Ok(0)
}

fn run_verify_reopen(cli: &Cli) -> Result<i32, String> {
    verify_reopen_using(cli, |path| SessionManager::open(path, None, None))
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

async fn run_replay(cli: &Cli) -> Result<i32, String> {
    let export_projection = cli.flag("export-session-projection")?;
    if export_projection {
        cli.reject_unknown_options(&[
            "fixture",
            "out",
            "provider-delay-ms",
            "summary-failure",
            "context-window",
            "keep-recent",
            "work",
            "stamp",
            "git",
            "timeout-secs",
            "export-session-projection",
        ])?;
    }
    let fixture_path = PathBuf::from(cli.required("fixture")?);
    let out = PathBuf::from(cli.required("out")?);
    // Strict projection mode validates all effective numeric settings before
    // replay. Legacy parsing below retains its original discovery-first order.
    let strict_numbers = if export_projection {
        Some((
            strict_finite_integral(cli, "provider-delay-ms", 0.0, true)? as u64,
            strict_finite_integral(cli, "context-window", 1_000_000.0, false)?,
            if cli.optional("keep-recent").is_some() {
                Some(strict_finite_integral(cli, "keep-recent", 0.0, true)?)
            } else {
                None
            },
            strict_finite_integral(cli, "timeout-secs", 900.0, false)? as u64,
        ))
    } else {
        None
    };
    let cases = discover_fixtures(&fixture_path)?;
    let (provider_delay_ms, context_window, keep_recent, timeout_secs) = match strict_numbers {
        Some(values) => values,
        None => (
            cli.number("provider-delay-ms", 0.0)? as u64,
            cli.number("context-window", 1_000_000.0)?,
            match cli.number("keep-recent", f64::NAN) {
                Ok(value) if value.is_finite() => Some(value),
                Ok(_) => None,
                Err(error) => return Err(error),
            },
            cli.number("timeout-secs", 900.0)? as u64,
        ),
    };
    let options = ReplayOptions {
        provider_delay_ms,
        summary_failure: cli.optional("summary-failure"),
        context_window,
        keep_recent,
        work: cli
            .optional("work")
            .map(PathBuf::from)
            .unwrap_or_else(|| std::env::temp_dir().join("compaction-replay-work")),
        timeout_secs,
        export_session_projection: export_projection,
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
            Ok(Ok(value)) => case_results.push(value),
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
    let mut output = json!({
        "harness": "compaction_replay",
        "subcommand": "replay",
        "schema": 1,
        "env": env_stamp(cli),
        "config": {
            "provider_delay_ms": options.provider_delay_ms,
            "summary_failure": options.summary_failure,
            "context_window": options.context_window,
            "keep_recent": options.keep_recent,
            "timeout_secs": options.timeout_secs,
        },
        "case_count": case_results.len(),
        "cases": case_results,
        "failures": failures,
    });
    if export_projection {
        let object = output.as_object_mut().ok_or("replay output object")?;
        object.insert("schema".into(), json!(2));
        object.insert("reader_schema".into(), json!(COMMON_READER_SCHEMA));
        object["config"]
            .as_object_mut()
            .ok_or("replay config object")?
            .insert("export_session_projection".into(), json!(true));
    }
    write_json(&out, &output)?;
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


fn required_raw_allocations(allocs: Option<(u64, u64)>) -> Result<(u64, u64), String> {
    allocs.ok_or_else(|| {
        "raw serializer requires real first-measured allocation diagnostics".to_string()
    })
}

fn raw_duration_array(value: &Value, field: &str, expected: usize) -> Result<Vec<f64>, String> {
    let values = value
        .get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("raw_samples.{field} must be an array of numbers"))?;
    if values.len() != expected {
        return Err(format!("raw_samples.{field} must contain exactly {expected} values"));
    }
    values
        .iter()
        .map(|value| {
            let duration = value
                .as_f64()
                .filter(|duration| duration.is_finite() && *duration >= 0.0)
                .ok_or_else(|| {
                    format!("raw_samples.{field} contains a nonnumeric/nonfinite/negative duration")
                })?;
            Ok(duration)
        })
        .collect()
}

fn validate_raw_kernel_result(result: &Value) -> Result<(Vec<f64>, Vec<f64>), String> {
    let raw_value = result.get("raw_samples").ok_or("raw_samples must be an object")?;
    let raw = raw_value
        .as_object()
        .ok_or("raw_samples must be an object")?;
    if raw.get("schema").and_then(Value::as_u64) != Some(1)
        || raw.get("phase_order") != Some(&json!(["warmup", "measured"]))
        || raw.get("warmup_count").and_then(Value::as_u64) != Some(5)
        || raw.get("measured_iteration_count").and_then(Value::as_u64) != Some(30)
    {
        return Err("raw_samples schema, phase order, or integral count drift".into());
    }
    let warmup_ms = raw_duration_array(raw_value, "warmup_ms", 5)?;
    let measured_ms = raw_duration_array(raw_value, "measured_ms", 30)?;
    if result["allocation_status"] != "measured"
        || result["detail"]["llm_messages"].as_u64()
            != result["raw_samples"]["input"]["llm_messages"].as_u64()
    {
        return Err("raw allocation status or converted-input count mismatch".into());
    }
    let count = result["allocs_first_iter"]["count"]
        .as_u64()
        .ok_or("first measured allocation count must be an actual integer")?;
    let bytes = result["allocs_first_iter"]["bytes"]
        .as_u64()
        .ok_or("first measured allocation bytes must be an actual integer")?;
    let (count, bytes) = required_raw_allocations(Some((count, bytes)))?;
    let _ = (count, bytes);
    let expected_algorithm = "sum_serialize_conversation_string_len_as_f64__5_warmup_then_30_measured";
    if result["output_checksum"]["algorithm"] != expected_algorithm
        || result["output_checksum"]["encoding"]
            != "serde_json JSON number; parse as IEEE-754 binary64; not hex, not a hash"
    {
        return Err("output_checksum algorithm/encoding drift".into());
    }
    let checksum = result["checksum"]
        .as_f64()
        .filter(|value| value.is_finite())
        .ok_or("actual numeric checksum missing")?;
    let output_checksum = result["output_checksum"]["value"]
        .as_f64()
        .filter(|value| value.is_finite())
        .ok_or("actual numeric output_checksum.value missing")?;
    if checksum.to_bits() != output_checksum.to_bits() {
        return Err("checksum and output_checksum.value differ".into());
    }
    for field in [
        "manifest_sha256",
        "transcript_sha256",
        "converted_input_sha256",
        "common_measurement_source_sha256",
    ] {
        if result["raw_samples"]["input"][field]
            .as_str()
            .map_or(true, |hash| hash.len() != 64 || !hash.bytes().all(|byte| byte.is_ascii_hexdigit()))
        {
            return Err(format!("raw input {field} must identify actual input bytes"));
        }
    }
    if result["raw_samples"]["input"]["common_measurement_source_sha256"]
        != sha256_hex(include_bytes!("compaction_replay.rs"))
    {
        return Err("raw common-source identity does not match this exact example source".into());
    }
    Ok((warmup_ms, measured_ms))
}

fn require_positive_raw_timing(warmup: &[f64], measured: &[f64]) -> Result<(), String> {
    if warmup.iter().chain(measured).any(|duration| *duration <= 0.0) {
        return Err(
            "raw timer resolution is insufficient for positive-duration evidence; measurement unsupported"
                .into(),
        );
    }
    Ok(())
}

fn bench_kernel(
    name: &str,
    detail: Value,
    warmup: usize,
    iters: usize,
    kernel: impl FnMut() -> f64,
) -> Value {
    bench_kernel_impl(name, detail, warmup, iters, kernel, false, None)
        .expect("legacy benchmark kernel is infallible")
}

/// One implementation for legacy and raw serializer measurement. Only raw mode
/// adds a timer to the five warmups and retains ordered vectors after timing.
#[allow(clippy::too_many_arguments)]
fn bench_kernel_impl(
    name: &str,
    detail: Value,
    warmup: usize,
    iters: usize,
    mut kernel: impl FnMut() -> f64,
    capture_samples: bool,
    raw_input: Option<Value>,
) -> Result<Value, String> {
    if capture_samples {
        if warmup != 5 || iters != 30 {
            return Err(format!(
                "raw serializer mode requires exactly 5 warmups and 30 measured iterations, got {warmup}/{iters}"
            ));
        }
        if raw_input.is_none() {
            return Err("raw serializer mode requires its actual fixture identity".into());
        }
    } else if raw_input.is_some() {
        return Err("raw fixture identity is valid only in capture-samples mode".into());
    }

    let mut checksum = 0.0f64;
    let mut warmup_samples = capture_samples.then(|| Vec::with_capacity(warmup));
    for _ in 0..warmup {
        if let Some(samples) = warmup_samples.as_mut() {
            let started = Instant::now();
            checksum += kernel();
            samples.push(elapsed_ms(started));
        } else {
            checksum += kernel();
        }
    }

    let mut samples: Vec<f64> = Vec::with_capacity(iters);
    let mut allocs: Option<(u64, u64)> = None;
    let before_first = alloc_snapshot();
    for index in 0..iters {
        if capture_samples {
            let started = Instant::now();
            checksum += kernel();
            samples.push(elapsed_ms(started));
        } else {
            let started = Instant::now();
            checksum += kernel();
            samples.push(elapsed_ms(started));
        }
        if index == 0 {
            let after_first = alloc_snapshot();
            allocs = Some((
                after_first.0.saturating_sub(before_first.0),
                after_first.1.saturating_sub(before_first.1),
            ));
        }
    }

    if capture_samples {
        let observed_warmups = warmup_samples.as_ref().map_or(0, Vec::len);
        if observed_warmups != warmup || samples.len() != iters {
            return Err(format!(
                "raw call count mismatch: {observed_warmups}/{warmup} warmups and {}/{} measured calls",
                samples.len(), iters
            ));
        }
        if warmup_samples
            .as_ref()
            .into_iter()
            .flatten()
            .chain(samples.iter())
            .any(|duration| !duration.is_finite())
        {
            return Err("raw serializer produced a non-finite duration".into());
        }
    }
    let stats_input = samples.clone(); // Post-timing only; raw mode retains ordered samples below.
    let stats = timing_stats(stats_input);
    let (alloc_count, alloc_bytes) = if capture_samples {
        required_raw_allocations(allocs)?
    } else {
        allocs.unwrap_or((0, 0))
    };
    if capture_samples && !checksum.is_finite() {
        return Err("raw serializer output-length checksum is non-finite".into());
    }

    let mut result = json!({
        "name": name,
        "detail": detail,
        "warmup": warmup,
        "stats": stats,
        "allocs_first_iter": {
            "count": alloc_count,
            "bytes": alloc_bytes,
        },
        "checksum": checksum,
    });
    if capture_samples {
        let warmup_samples = warmup_samples.ok_or("raw warmup storage missing")?;
        let object = result
            .as_object_mut()
            .ok_or("benchmark kernel result must be an object")?;
        object.insert("allocation_status".into(), json!("measured"));
        object.insert(
            "output_checksum".into(),
            json!({
                "algorithm": "sum_serialize_conversation_string_len_as_f64__5_warmup_then_30_measured",
                "encoding": "serde_json JSON number; parse as IEEE-754 binary64; not hex, not a hash",
                "value": checksum,
            }),
        );
        object.insert(
            "raw_samples".into(),
            json!({
                "schema": 1,
                "input": raw_input,
                "phase_order": ["warmup", "measured"],
                "warmup_count": warmup,
                "warmup_ms": warmup_samples,
                "measured_iteration_count": iters,
                "measured_ms": samples,
            }),
        );
    }
    Ok(result)
}


fn pending_applicable_local_evidence(value: Value) -> Value {
    json!({
        "applicability": "applicable",
        "reason_code": "APPLICABLE",
        "verdict": "not_ready_pending_independent_cross_arm_comparison",
        "value": value,
        "supporting_test_refs": [],
    })
}

/// These are only operation-limited local inapplicability facts. The external
/// pair validator must still compose a complete v2 operation_evidence object;
/// no process-local output here is a complete pair or a semantic-pass claim.
fn local_operation_evidence_pending(serializer: &Value) -> Result<Value, String> {
    let actual_input = serializer
        .get("raw_samples")
        .and_then(|raw| raw.get("input"))
        .cloned()
        .ok_or("local evidence requires actual raw input identity")?;
    let actual_checksum = serializer
        .get("output_checksum")
        .cloned()
        .ok_or("local evidence requires actual output checksum")?;
    let allocations = serializer.get("allocs_first_iter").cloned().ok_or(
        "local evidence requires actual first-measured allocation diagnostics",
    )?;
    let mut pending = Map::new();
    pending.insert(
        "input_identity".into(),
        pending_applicable_local_evidence(json!({
            "this_arm_actual_input": actual_input,
            "comparison_required": "immutable fixture/config/converted input plus byte-identical common measurement source; source manifests and each arm's separately frozen executable must be independently checked",
        })),
    );
    pending.insert(
        "output_checksum_parity".into(),
        pending_applicable_local_evidence(json!({
            "this_arm_actual_checksum": actual_checksum,
            "comparison_required": "independent baseline/candidate JSON numbers parsed as IEEE-754 binary64 and compared exactly; 35-call String::len() sum only",
        })),
    );
    let direct_fields = [
        ("provider_request_payloads", "DIRECT_SERIALIZER_NO_PROVIDER"),
        ("compaction_summary", "DIRECT_SERIALIZER_NO_SUMMARY"),
        ("session_semantics", "DIRECT_SERIALIZER_NO_SESSION_FILE"),
        ("protected_state", "DIRECT_SERIALIZER_NO_SESSION_STATE"),
        ("trigger_cut_call", "DIRECT_SERIALIZER_NO_COMPACTION_SESSION"),
        ("persistence_restart", "DIRECT_SERIALIZER_NO_PERSISTENCE"),
        ("fresh_process_session_reopen", "DIRECT_SERIALIZER_NO_SESSION_REOPEN"),
        ("post_restart_provider_continuation", "DIRECT_SERIALIZER_NO_CONTINUATION"),
    ];
    let mut inapplicable = Map::new();
    for (field, reason) in direct_fields {
        inapplicable.insert(
            field.into(),
            json!({
                "applicability": "inapplicable",
                "reason_code": reason,
                "verdict": "not_evaluable",
                "value": null,
                "supporting_test_refs": [],
            }),
        );
    }
    Ok(json!({
        "protocol": "pr3-paired-v2",
        "operation_profile": "bench-local",
        "record_status": "not_ready_not_a_complete_v2_pair_record",
        "operation_evidence_pending": Value::Object(pending),
        "operation_evidence_inapplicable": Value::Object(inapplicable),
        "allocation_evidence": {
            "allocation_status": "measured",
            "allocs_first_iter": allocations,
        },
        "composition_rule": "replace only the two pending applicable objects with actual cross-arm comparison findings; combine them with these exact eight operation-limited inapplicable envelopes; never turn these local facts into replay/provider/session waivers",
    }))
}

fn strict_finite_integral(
    cli: &Cli,
    key: &str,
    default: f64,
    allow_zero: bool,
) -> Result<f64, String> {
    let value = cli.number(key, default)?;
    if !value.is_finite()
        || value.fract() != 0.0
        || value < 0.0
        || (!allow_zero && value <= 0.0)
        || value >= u64::MAX as f64
    {
        return Err(format!(
            "--{key} must be a finite {}integer representable as u64",
            if allow_zero { "nonnegative " } else { "positive " }
        ));
    }
    Ok(value)
}

fn strict_raw_count(cli: &Cli, key: &str, expected: usize) -> Result<usize, String> {
    let raw = cli
        .optional(key)
        .ok_or_else(|| format!("raw serializer mode requires an explicit --{key}"))?;
    let value = raw
        .parse::<f64>()
        .map_err(|error| format!("--{key}: {error}"))?;
    if !value.is_finite() || value.fract() != 0.0 || value <= 0.0 || value > usize::MAX as f64 {
        return Err(format!("--{key} must be a finite positive integer"));
    }
    let count = value as usize;
    if count != expected {
        return Err(format!("raw serializer mode requires --{key} {expected}, got {count}"));
    }
    Ok(count)
}

fn apply_bench_output_mode(output: &mut Value, raw_mode: bool) -> Result<(), String> {
    if raw_mode {
        let object = output.as_object_mut().ok_or("bench output object")?;
        object.insert("schema".into(), json!(2));
        object["config"]
            .as_object_mut()
            .ok_or("bench config object")?
            .insert("raw_samples".into(), json!(true));
    }
    Ok(())
}

fn run_bench_local(cli: &Cli) -> Result<i32, String> {
    let raw_mode = cli.flag("raw-samples")?;
    if raw_mode {
        cli.reject_unknown_options(&[
            "fixture",
            "iters",
            "out",
            "warmup",
            "stamp",
            "git",
            "chunk-context-window",
            "raw-samples",
        ])?;
    }
    let fixture_path = PathBuf::from(cli.required("fixture")?);
    let out = PathBuf::from(cli.required("out")?);
    let iters = if raw_mode {
        strict_raw_count(cli, "iters", 30)?
    } else {
        cli.number("iters", 20.0)? as usize
    };
    let warmup = if raw_mode {
        strict_raw_count(cli, "warmup", 5)?
    } else {
        cli.number("warmup", 3.0)? as usize
    };
    let chunk_context_window = if raw_mode {
        strict_finite_integral(cli, "chunk-context-window", 32_768.0, true)?
    } else {
        cli.number("chunk-context-window", 32_768.0)?
    };
    let cases = discover_fixtures(&fixture_path)?;

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
            kernels.push(bench_kernel(
                "prepare_compaction",
                json!({
                    "entries": entries.len(),
                    "keep_recent_tokens": keep_recent,
                    "context_builder": "token estimate over the live message list",
                }),
                warmup,
                iters,
                move || {
                    // Mirrors the production context builder: the threshold
                    // estimate over the live message list.
                    prepare_compaction(
                        &entries,
                        &settings,
                        &|_entries| estimate_context_tokens(&messages).tokens,
                    )
                    .map(|preparation| preparation.tokens_before)
                    .unwrap_or(-1.0)
                },
            ));
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
            let converted_value = raw_mode
                .then(|| serde_json::to_value(&llm).map_err(|error| error.to_string()))
                .transpose()?;
            let converted_input_sha256 = converted_value
                .as_ref()
                .map(canonical_sha256)
                .transpose()?;
            let raw_input = if raw_mode {
                let manifest_bytes = std::fs::read(case_dir.join("manifest.json"))
                    .map_err(|error| format!("read raw fixture manifest: {error}"))?;
                let transcript_bytes = std::fs::read(case_dir.join("transcript.json"))
                    .map_err(|error| format!("read raw fixture transcript: {error}"))?;
                Some(json!({
                    "case": fixture.name,
                    "llm_messages": llm.len(),
                    "converted_input_sha256": converted_input_sha256
                        .ok_or("raw converted input identity missing")?,
                    "manifest_sha256": sha256_hex(&manifest_bytes),
                    "transcript_sha256": sha256_hex(&transcript_bytes),
                    "common_measurement_source_sha256": sha256_hex(include_bytes!("compaction_replay.rs")),
                    "conversion": "convert_to_llm with ModelToolOutputPolicyOptions::default()",
                    "config": {
                        "warmup": 5,
                        "iters": 30,
                        "chunk_context_window": chunk_context_window,
                    },
                }))
            } else {
                None
            };
            // The production call, its temporary String lifetime, and the start/end
            // of the timer remain inside the single shared implementation above.
            let mut serializer_result = bench_kernel_impl(
                "serialize_conversation",
                json!({ "llm_messages": llm.len() }),
                warmup,
                iters,
                move || serialize_conversation(&llm).len() as f64,
                raw_mode,
                raw_input,
            )?;
            if raw_mode {
                // Post-timing validation never replaces a missing/invalid value.
                let (warmup_samples, measured_samples) =
                    validate_raw_kernel_result(&serializer_result)?;
                require_positive_raw_timing(&warmup_samples, &measured_samples)?;
                serializer_result["local_evidence_pending"] =
                    local_operation_evidence_pending(&serializer_result)?;
            }
            kernels.push(serializer_result);
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
            let mut skip_note: Option<String> = None;
            if let Err(skip) = prepare_context(&messages, &config) {
                skip_note = Some(skip.to_string());
            }
            let note = skip_note;
            kernels.push(bench_kernel(
                "jev_prepare_context",
                json!({
                    "messages": messages.len(),
                    "skip_on_first_call": note,
                }),
                warmup,
                iters,
                move || match prepare_context(&messages, &config) {
                    Ok(prepared) => prepared.baseline.estimated_tokens_before as f64,
                    Err(_) => -1.0,
                },
            ));
        }

        results.push(json!({
            "case": fixture.name,
            "fixture_manifest": fixture.manifest,
            "kernels": kernels,
        }));
    }
    let mut output = json!({
        "harness": "compaction_replay",
        "subcommand": "bench-local",
        "schema": 1,
        "env": env_stamp(cli),
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
        "cases": results,
    });
    apply_bench_output_mode(&mut output, raw_mode)?;
    write_json(&out, &output)?;
    println!("wrote {}", out.display());
    Ok(0)
}


// ---------------------------------------------------------------------------
// Source tests only. Real writer/reader PID separation is an external
// smoke/decision requirement; these in-process tests never claim that proof.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod measurement_evidence_tests {
    use super::*;
    use std::cell::Cell;
    use std::rc::Rc;
    use tempfile::TempDir;

    struct ValidCase {
        _temp: TempDir,
        work_root: PathBuf,
        session_file: PathBuf,
        header: SessionEntry,
        entries: Vec<SessionEntry>,
        branch: Vec<SessionEntry>,
        projection: Value,
        observed_live: Value,
        user_before_cut: String,
        retained_call_entry: String,
        compaction_entry: String,
        post_compaction_entry: String,
    }

    fn harness_assistant(
        text: &str,
        tool_call: Option<(&str, &str)>,
        timestamp: i64,
    ) -> AgentMessage {
        let mut content = vec![ContentBlock::Text(TextContent::new(text.to_string()))];
        if let Some((id, argument)) = tool_call {
            content.push(ContentBlock::ToolCall(ToolCall::new(
                id.to_string(),
                "fixture_tool",
                json!({"argument": argument})
                    .as_object()
                    .cloned()
                    .expect("tool arguments object"),
            )));
        }
        AgentMessage::Message(Message::Assistant(AssistantMessage {
            content,
            api: "harness-api-test".into(),
            provider: "harness-provider-test".into(),
            model: "compaction-replay-model".into(),
            stop_reason: "stop".into(),
            timestamp,
            usage: Usage {
                input: 11.0,
                output: 3.0,
                total_tokens: 14.0,
                ..Usage::zero()
            },
            ..Default::default()
        }))
    }

    fn seed_valid_case(label: &str) -> ValidCase {
        let temp = tempfile::tempdir().expect("private measurement test root");
        let work_root = temp.path().join("work");
        let case_root = work_root.join(format!("replay-{label}"));
        let workspace = case_root.join("workspace");
        let sessions = case_root.join("sessions");
        let artifact_dir = work_root.join("session-artifacts/existing");
        for directory in [&workspace, &sessions, &artifact_dir] {
            std::fs::create_dir_all(directory).expect("private fixture directory");
        }
        // A nonempty root with existing siblings is the starting condition.
        std::fs::write(
            artifact_dir.join("semantic-edges.jsonl"),
            b"{\"existing\":true}\n",
        )
        .expect("existing semantic edges fixture");
        let mut manager = SessionManager::create(
            &workspace.to_string_lossy(),
            Some(&sessions.to_string_lossy()),
        )
        .expect("valid replay manager");
        manager.append_thinking_level_change("high").expect("thinking context");
        manager
            .append_service_tier_change(&Some(Some("default".into())))
            .expect("service context");
        let timestamp = 1_772_000_000_000 + label.len() as i64;
        let user_before_cut = manager
            .append_message(AgentMessage::Message(Message::User(UserMessage::new(
                UserContent::Text("summarized earlier turn".into()),
                timestamp,
            ))))
            .expect("earlier user");
        let call_id = format!("tool-call-{label}");
        let retained_call_entry = manager
            .append_message(harness_assistant(
                "retained tool call",
                Some((&call_id, "kept")),
                timestamp + 1,
            ))
            .expect("retained tool call");
        manager
            .append_message(AgentMessage::Message(Message::ToolResult(
                ToolResultMessage::new(
                    call_id,
                    "fixture_tool",
                    vec![ImageOrTextContent::Text(TextContent::new("tool output"))],
                    false,
                    timestamp + 2,
                ),
            )))
            .expect("paired tool result");
        // Preserve an orphan result rather than hiding it through normalization.
        manager
            .append_message(AgentMessage::Message(Message::ToolResult(
                ToolResultMessage::new(
                    format!("orphan-call-{label}"),
                    "fixture_tool",
                    vec![ImageOrTextContent::Text(TextContent::new("orphan output"))],
                    false,
                    timestamp + 3,
                ),
            )))
            .expect("orphan tool result");
        let details = json!({
            "customProtected": {
                "goals": ["preserve goal"],
                "constraints": ["preserve constraint"],
                "evidence": ["preserve evidence"],
                "pending_subagent": {"pendingResult": "opaque but non-provider"},
                "unknownTimestamp": 1_772_000_000_000i64,
            },
            "unknownFutureField": {"scalarTypes": [1, "1", true, null]},
        });
        let compaction_entry = manager
            .append_compaction(
                "retention summary",
                &retained_call_entry,
                19.0,
                Some(details),
                Some(false),
                Some("test custom instructions"),
                Some(&Usage {
                    input: 19.0,
                    output: 2.0,
                    total_tokens: 21.0,
                    ..Usage::zero()
                }),
                Some("test-harness-digest"),
            )
            .expect("intended compaction");
        let post_compaction_entry = manager
            .append_message(AgentMessage::Message(Message::User(UserMessage::new(
                UserContent::Text(format!(
                    "post-compaction private path {}/artifact.txt",
                    case_root.display()
                )),
                timestamp + 4,
            ))))
            .expect("post-compaction user");
        manager
            .append_message(harness_assistant("post-compaction assistant", None, timestamp + 5))
            .expect("post-compaction assistant");
        let branch = manager.get_branch(None);
        let entries = manager.get_entries();
        let header = manager.get_header().expect("valid header");
        let writer_context = manager.build_session_context(None);
        let observed = ObservedWriterContext {
            live_messages: writer_context.messages.clone(),
            context_messages: writer_context.messages,
            thinking_level: writer_context.thinking_level,
            service_tier: writer_context.service_tier,
            model: writer_context.model,
        };
        let projection = session_projection(&manager, Some(observed))
            .expect("valid real-manager projection");
        let session_file = PathBuf::from(
            manager.get_session_file().expect("durable replay session file"),
        );
        // Fixture-writer flush only; the strict reader has no flush call.
        manager.flush_now().expect("closed fixture bytes");
        drop(manager);
        ValidCase {
            _temp: temp,
            work_root,
            session_file,
            header,
            entries,
            branch,
            projection: projection.projection,
            observed_live: projection.observed_live.expect("writer observations"),
            user_before_cut,
            retained_call_entry,
            compaction_entry,
            post_compaction_entry,
        }
    }

    fn reopen_cli(case: &ValidCase) -> Cli {
        Cli {
            subcommand: "verify-reopen".into(),
            options: HashMap::from([
                (
                    "session-file".into(),
                    case.session_file.to_string_lossy().to_string(),
                ),
                (
                    "work-root".into(),
                    case.work_root.to_string_lossy().to_string(),
                ),
                (
                    "out".into(),
                    case.work_root
                        .join("verify-reopen.json")
                        .to_string_lossy()
                        .to_string(),
                ),
                ("git".into(), "source-test-not-a-source-identity-proof".into()),
            ]),
        }
    }

    fn assert_rejected_before_open(case: &ValidCase, cli: Cli) -> String {
        let before = std::fs::read(&case.session_file).expect("original negative-input bytes");
        let called = Cell::new(false);
        let output = cli
            .options
            .get("out")
            .map(PathBuf::from)
            .expect("negative reader output");
        let result = verify_reopen_using(&cli, |_| {
            called.set(true);
            panic!("negative preflight must return before any production open")
        });
        assert!(result.is_err(), "negative input must fail closed");
        assert!(!called.get(), "negative preflight called production open");
        assert_eq!(
            std::fs::read(&case.session_file).expect("negative input after rejection"),
            before,
            "negative preflight mutated or restored over original bytes"
        );
        assert!(!output.exists(), "negative preflight created an accepted report");
        result.expect_err("negative preflight error classification")
    }

    fn mutate_jsonl(case: &ValidCase, mutate: impl FnOnce(&mut Vec<Value>)) {
        let body = std::fs::read_to_string(&case.session_file).expect("negative fixture source");
        let mut rows = body
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).expect("fixture JSON"))
            .collect::<Vec<_>>();
        mutate(&mut rows);
        let mut bytes = rows
            .iter()
            .map(|row| serde_json::to_string(row).expect("negative JSON encoding"))
            .collect::<Vec<_>>()
            .join("\n");
        bytes.push('\n');
        std::fs::write(&case.session_file, bytes).expect("private negative fixture");
    }


    /// Prove ordinary Value interpretation is valid/identical, then inject only
    /// raw duplicate keys. Strict row validation must fire before production open.
    fn inject_raw_duplicate_object_key(case: &ValidCase, kind: &str) -> usize {
        let original = std::fs::read(&case.session_file).expect("original duplicate-base bytes");
        let ordinary_rows = |bytes: &[u8]| {
            bytes
                .split(|byte| *byte == b'\n')
                .filter(|line| !line.is_empty())
                .map(|line| serde_json::from_slice::<Value>(line).expect("ordinary raw JSON row"))
                .collect::<Vec<_>>()
        };
        let before = ordinary_rows(&original);
        assert!(!before.is_empty());
        let input = std::fs::canonicalize(&case.session_file).expect("valid duplicate base path");
        let work_root = std::fs::canonicalize(&case.work_root).expect("valid duplicate work root");
        parse_strict_replay_shape(&original, &input, &work_root)
            .expect("unmodified valid replay shape before raw duplicate mutation");
        let mut raw_rows: Vec<Vec<u8>> = original
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
            .map(<[u8]>::to_vec)
            .collect();
        assert_eq!(raw_rows.len(), before.len());

        let duplicate_row = match kind {
            "duplicate-header-current-version" => {
                assert_eq!(raw_rows[0].last(), Some(&b'}'));
                // In this Rust raw string, JSON vers\u0069on is a literal ASCII
                // escape. JSON decoding makes it the existing "version" key.
                let duplicate = format!(r#", "vers\u0069on":{}"#, CURRENT_SESSION_VERSION);
                raw_rows[0].pop();
                raw_rows[0].extend_from_slice(duplicate.as_bytes());
                raw_rows[0].push(b'}');
                1usize
            }
            "duplicate-nested-protected-key" => {
                let index = before
                    .iter()
                    .position(|row| row["type"] == "compaction")
                    .expect("actual protected compaction/detail payload");
                let original_line = String::from_utf8(raw_rows[index].clone())
                    .expect("protected compaction row UTF-8");
                // Actual path: details.customProtected.pending_subagent.pendingResult.
                // The second JSON key is the escaped-equivalent pendingResult.
                let occurrence = "\"pendingResult\":\"opaque but non-provider\"";
                let duplicate = "\"pendingResult\":\"opaque but non-provider\",\"pending\\u0052esult\":\"opaque but non-provider\"";
                assert_eq!(original_line.matches(occurrence).count(), 1);
                raw_rows[index] = original_line
                    .replacen(occurrence, duplicate, 1)
                    .into_bytes();
                index + 1 // Physical JSONL rows are one-based, including the header.
            }
            _ => panic!("unknown raw duplicate fixture: {kind}"),
        };
        let mut mutated = Vec::new();
        for (row_index, row) in raw_rows.iter().enumerate() {
            if row_index > 0 {
                mutated.push(b'\n');
            }
            mutated.extend_from_slice(row);
        }
        mutated.push(b'\n');
        let after = ordinary_rows(&mutated);
        assert_eq!(
            after, before,
            "only raw duplicate presence differs: ordinary Value must retain the last/current valid value"
        );
        assert_eq!(
            after[0]["version"].as_i64(),
            Some(CURRENT_SESSION_VERSION),
            "header final/current version survives ordinary last-key-wins parsing"
        );
        if kind == "duplicate-nested-protected-key" {
            let compaction = after
                .iter()
                .find(|row| row["type"] == "compaction")
                .expect("ordinary protected compaction");
            assert_eq!(
                compaction.pointer("/details/customProtected/pending_subagent/pendingResult"),
                Some(&json!("opaque but non-provider"))
            );
        }
        std::fs::write(&case.session_file, mutated).expect("raw duplicate negative input");
        duplicate_row
    }

    fn raw_input() -> Value {
        let converted = json!([{"role": "user", "content": "actual raw test converted input"}]);
        json!({
            "case": "source-test",
            "llm_messages": converted.as_array().expect("converted test array").len(),
            "converted_input_sha256": canonical_sha256(&converted).expect("test converted input hash"),
            "manifest_sha256": sha256_hex(b"raw source test manifest bytes"),
            "transcript_sha256": sha256_hex(b"raw source test transcript bytes"),
            "common_measurement_source_sha256": sha256_hex(include_bytes!("compaction_replay.rs")),
            "conversion": "convert_to_llm with ModelToolOutputPolicyOptions::default()",
            "config": {"warmup": 5, "iters": 30, "chunk_context_window": 32_768.0},
        })
    }

    #[test]
    fn raw_kernel_orders_35_real_calls_and_numeric_samples() {
        let calls = Rc::new(Cell::new(0usize));
        let observed = Rc::clone(&calls);
        let result = bench_kernel_impl(
            "serialize_conversation",
            json!({"llm_messages": 1}),
            5,
            30,
            move || {
                let next = observed.get() + 1;
                observed.set(next);
                next as f64
            },
            true,
            Some(raw_input()),
        )
        .expect("raw measurement kernel");
        assert_eq!(calls.get(), 35);
        assert_eq!(result["checksum"], json!(630.0));
        assert_eq!(result["allocation_status"], "measured");
        assert!(result["allocs_first_iter"]["count"].is_u64());
        assert!(result["allocs_first_iter"]["bytes"].is_u64());
        assert_eq!(result["raw_samples"]["warmup_count"], 5);
        assert_eq!(result["raw_samples"]["measured_iteration_count"], 30);
        assert_eq!(
            result["raw_samples"]["phase_order"],
            json!(["warmup", "measured"])
        );
        for (field, count) in [("warmup_ms", 5), ("measured_ms", 30)] {
            let values = result["raw_samples"][field]
                .as_array()
                .expect("numeric raw array");
            assert_eq!(values.len(), count);
            assert!(values
                .iter()
                .all(|value| value.as_f64().is_some_and(f64::is_finite)));
        }
        // Stats sorting may not mutate either original ordered sample vector.
        let stats_input = result["raw_samples"]["measured_ms"].clone();
        let expected_stats_input: Value = serde_json::from_str(
            &serde_json::to_string(&stats_input).expect("expected raw JSON normalization"),
        )
        .expect("expected raw JSON parse");
        let round_trip: Value = serde_json::from_str(
            &serde_json::to_string(&result).expect("raw JSON round trip"),
        )
        .expect("raw JSON parse");
        assert_eq!(round_trip["raw_samples"]["measured_ms"], expected_stats_input);
        assert!(round_trip["raw_samples"]["measured_iteration_count"].is_u64());
        let (checked_warmup, checked_measured) =
            validate_raw_kernel_result(&round_trip).expect("ordered raw validator");
        assert_eq!(checked_warmup.len(), 5);
        assert_eq!(checked_measured.len(), 30);
        for (field, bad) in [
            ("warmup_ms", json!("not-a-number")),
            ("measured_ms", json!(null)),
        ] {
            let mut invalid = result.clone();
            invalid["raw_samples"][field] = bad;
            assert!(validate_raw_kernel_result(&invalid).is_err());
        }
        let mut invalid_phase = result.clone();
        invalid_phase["raw_samples"]["phase_order"] = json!(["measured", "warmup"]);
        assert!(validate_raw_kernel_result(&invalid_phase).is_err());
        let mut negative_duration = result.clone();
        negative_duration["raw_samples"]["warmup_ms"][0] = json!(-1.0);
        assert!(validate_raw_kernel_result(&negative_duration).is_err());
        assert!(required_raw_allocations(None).is_err());
    }

    #[test]
    fn raw_local_evidence_envelopes_are_literal_and_do_not_fabricate_pair_pass() {
        let result = bench_kernel_impl(
            "serialize_conversation",
            json!({"llm_messages": 1}),
            5,
            30,
            || 7.0,
            true,
            Some(raw_input()),
        )
        .expect("raw local envelope test data");
        validate_raw_kernel_result(&result).expect("actual numeric raw inputs");
        let mut quantized = result.clone();
        quantized["raw_samples"]["warmup_ms"][0] = json!(0.0);
        let (quantized_warmup, quantized_measured) =
            validate_raw_kernel_result(&quantized).expect("literal zero remains raw data");
        assert!(require_positive_raw_timing(&quantized_warmup, &quantized_measured)
            .expect_err("quantized zero is not eligible timing evidence")
            .contains("unsupported"));
        let pending = local_operation_evidence_pending(&result).expect("pending local facts");
        assert_eq!(pending["record_status"], "not_ready_not_a_complete_v2_pair_record");
        let applicable = pending["operation_evidence_pending"]
            .as_object()
            .expect("exactly two pending applicable fields");
        assert_eq!(applicable.len(), 2);
        assert_eq!(
            applicable.keys().map(String::as_str).collect::<HashSet<_>>(),
            HashSet::from(["input_identity", "output_checksum_parity"])
        );
        for field in ["input_identity", "output_checksum_parity"] {
            assert_eq!(applicable[field]["reason_code"], "APPLICABLE");
            assert_eq!(applicable[field]["supporting_test_refs"], json!([]));
            assert!(applicable[field]["verdict"]
                .as_str()
                .expect("pending local verdict")
                .starts_with("not_ready"));
        }
        assert_eq!(
            pending["operation_evidence_pending"]["output_checksum_parity"]["value"]["this_arm_actual_checksum"],
            result["output_checksum"]
        );
        let inapplicable = pending["operation_evidence_inapplicable"]
            .as_object()
            .expect("exactly eight local envelopes");
        assert_eq!(inapplicable.len(), 8);
        for (field, reason) in [
            ("provider_request_payloads", "DIRECT_SERIALIZER_NO_PROVIDER"),
            ("compaction_summary", "DIRECT_SERIALIZER_NO_SUMMARY"),
            ("session_semantics", "DIRECT_SERIALIZER_NO_SESSION_FILE"),
            ("protected_state", "DIRECT_SERIALIZER_NO_SESSION_STATE"),
            ("trigger_cut_call", "DIRECT_SERIALIZER_NO_COMPACTION_SESSION"),
            ("persistence_restart", "DIRECT_SERIALIZER_NO_PERSISTENCE"),
            ("fresh_process_session_reopen", "DIRECT_SERIALIZER_NO_SESSION_REOPEN"),
            ("post_restart_provider_continuation", "DIRECT_SERIALIZER_NO_CONTINUATION"),
        ] {
            assert_eq!(inapplicable[field]["reason_code"], reason);
            assert_eq!(inapplicable[field]["applicability"], "inapplicable");
            assert_eq!(inapplicable[field]["verdict"], "not_evaluable");
            assert!(inapplicable[field]["value"].is_null());
            assert_eq!(inapplicable[field]["supporting_test_refs"], json!([]));
        }
    }

    #[test]
    fn raw_kernel_uses_real_serializer_len_sum_not_content_hash() {
        let llm = vec![Message::User(UserMessage::new(
            UserContent::Text("actual serializer checksum input".into()),
            7,
        ))];
        let expected_one = serialize_conversation(&llm).len() as f64;
        assert!(expected_one > 0.0);
        let calls = Rc::new(Cell::new(0usize));
        let observed = Rc::clone(&calls);
        let result = bench_kernel_impl(
            "serialize_conversation",
            json!({"llm_messages": llm.len()}),
            5,
            30,
            move || {
                observed.set(observed.get() + 1);
                serialize_conversation(&llm).len() as f64
            },
            true,
            Some(raw_input()),
        )
        .expect("actual serializer raw samples");
        assert_eq!(calls.get(), 35);
        assert_eq!(result["checksum"], json!(expected_one * 35.0));
        assert_eq!(
            result["output_checksum"]["algorithm"],
            "sum_serialize_conversation_string_len_as_f64__5_warmup_then_30_measured"
        );
        assert_eq!(result["output_checksum"]["value"], result["checksum"]);
        assert!(result["output_checksum"]["value"].is_number());
        assert_ne!(result["output_checksum"]["encoding"], "hex");
        let source = include_str!("compaction_replay.rs");
        assert_eq!(
            source
                .lines()
                .filter(|line| line.trim().trim_end_matches(',') == "move || serialize_conversation(&llm).len() as f64")
                .count(),
            1,
            "the production timed closure must have one exact source form"
        );
    }

    #[test]
    fn legacy_kernel_keys_and_raw_rejections_are_fail_closed() {
        let legacy = bench_kernel("legacy", json!({"x": 1}), 3, 2, || 2.0);
        let mut keys = legacy
            .as_object()
            .expect("legacy object")
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        keys.sort();
        assert_eq!(
            keys,
            ["allocs_first_iter", "checksum", "detail", "name", "stats", "warmup"]
        );
        assert_eq!(legacy["checksum"], 10.0);
        assert!(legacy.get("raw_samples").is_none());
        assert!(legacy.get("allocation_status").is_none());
        for (warmup, iters) in [(0, 0), (4, 30), (5, 29), (5, 31), (5, 0)] {
            let calls = Cell::new(0usize);
            let result = bench_kernel_impl(
                "serialize_conversation",
                json!({}),
                warmup,
                iters,
                || {
                    calls.set(calls.get() + 1);
                    1.0
                },
                true,
                Some(raw_input()),
            );
            assert!(result.is_err());
            assert_eq!(calls.get(), 0, "invalid raw counts must precede first call");
        }
    }

    #[test]
    fn cli_and_schema_boundaries_preserve_legacy_and_reject_new_drift() {
        let cli = Cli::parse(&[
            "bench-local".into(),
            "--fixture".into(),
            "f".into(),
            "--raw-samples=false".into(),
        ])
        .expect("parser retains raw option text");
        assert!(cli.flag("raw-samples").is_err());
        let valued = Cli::parse(&[
            "replay".into(),
            "--export-session-projection".into(),
            "false".into(),
        ])
        .expect("parser");
        assert!(valued.flag("export-session-projection").is_err());
        let unknown = Cli::parse(&[
            "verify-reopen".into(),
            "--unexpected-option".into(),
        ])
        .expect("parser");
        assert!(unknown
            .reject_unknown_options(&["session-file", "work-root", "out", "git"])
            .unwrap_err()
            .contains("unexpected-option"));
        let legacy = Cli::parse(&[
            "replay".into(),
            "--legacy-unknown-option".into(),
        ])
        .expect("legacy parser shape remains accepted");
        assert!(!legacy.flag("export-session-projection").expect("flag absent"));
        // No strict option validation is run globally for that legacy shape.
        for (raw, expected, error) in [
            ("29", None, true),
            ("30", Some(30), false),
            ("30.5", None, true),
            ("0", None, true),
            ("NaN", None, true),
            ("inf", None, true),
            ("-1", None, true),
        ] {
            let cli = Cli::parse(&[
                "bench-local".into(),
                "--iters".into(),
                raw.into(),
            ])
            .expect("count parser");
            let value = strict_raw_count(&cli, "iters", 30);
            if error {
                assert!(value.is_err(), "invalid raw count {raw}");
            } else {
                assert_eq!(value.expect("valid count"), expected.expect("count"));
            }
        }
        let missing = Cli::parse(&["bench-local".into()]).expect("parser");
        for key in ["iters", "warmup"] {
            assert!(strict_raw_count(&missing, key, 30).is_err());
        }
        let mut output = json!({"schema": 1, "config": {"iters": 20}, "cases": []});
        let original = output.clone();
        apply_bench_output_mode(&mut output, false).expect("legacy output unchanged");
        assert_eq!(output, original);
        apply_bench_output_mode(&mut output, true).expect("raw schema output");
        assert_eq!(output["schema"], 2);
        assert_eq!(output["config"]["raw_samples"], true);
        assert_eq!(output["config"]["iters"], 20);
    }

    #[test]
    fn strict_numeric_options_reject_nonfinite_nonintegral_and_zero_integer_casts() {
        for value in ["NaN", "inf", "1.5", "-1"] {
            let cli = Cli::parse(&[
                "replay".into(),
                "--work".into(),
                "unused-private-test-root".into(),
                "--provider-delay-ms".into(),
                value.into(),
            ])
            .expect("parser text");
            assert!(strict_finite_integral(&cli, "provider-delay-ms", 0.0, true).is_err());
        }
        let zero = Cli::parse(&[
            "replay".into(),
            "--context-window".into(),
            "0".into(),
        ])
        .expect("zero-value parser text");
        assert!(strict_finite_integral(&zero, "context-window", 1_000_000.0, false).is_err());
        let fractional = Cli::parse(&[
            "bench-local".into(),
            "--raw-samples".into(),
            "--chunk-context-window".into(),
            "32768.5".into(),
        ])
        .expect("raw parser text");
        assert!(strict_finite_integral(&fractional, "chunk-context-window", 32_768.0, true).is_err());
        let accepted = Cli::parse(&["replay".into()]).expect("strict replay defaults");
        assert_eq!(
            strict_finite_integral(&accepted, "provider-delay-ms", 0.0, true).unwrap(),
            0.0
        );
    }

    #[test]
    fn type7_reference_vectors_are_independent_of_sorted_export() {
        let values = (1..=10).map(|value| value as f64).collect::<Vec<_>>();
        fn type7(values: &[f64], fraction: f64) -> f64 {
            let mut sorted = values.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
            let index = fraction * (sorted.len() - 1) as f64;
            let lower = index.floor() as usize;
            let upper = index.ceil() as usize;
            sorted[lower] + (sorted[upper] - sorted[lower]) * index.fract()
        }
        assert_eq!(type7(&values, 0.5), 5.5);
        assert!((type7(&values, 0.95) - 9.55).abs() < 1e-12);
        assert_eq!(values, (1..=10).map(|v| v as f64).collect::<Vec<_>>());
    }

    #[test]
    fn canonical_json_sorts_keys_and_preserves_arrays_scalars_and_punctuation() {
        let value = json!({
            "z": [1, "1", true, null, {"b": 2, "a": 1}],
            "a": {"z": "}[,:]", "a": -0.0},
        });
        let canonical = String::from_utf8(canonical_json_bytes(&value).expect("canonical JSON"))
            .expect("UTF-8");
        assert_eq!(
            canonical,
            r#"{"a":{"a":-0.0,"z":"}[,:]"},"z":[1,"1",true,null,{"a":1,"b":2}]}"#
        );
        let reordered = json!({
            "a": {"a": -0.0, "z": "}[,:]"},
            "z": [1, "1", true, null, {"a": 1, "b": 2}],
        });
        assert_eq!(
            canonical_json_bytes(&value).expect("canonical"),
            canonical_json_bytes(&reordered).expect("canonical")
        );
        let mut changed_array = value.clone();
        changed_array["z"].as_array_mut().expect("array").swap(0, 1);
        assert_ne!(
            canonical_json_bytes(&value).expect("canonical"),
            canonical_json_bytes(&changed_array).expect("canonical")
        );
    }

    #[test]
    fn valid_projection_retains_full_order_entry_links_tools_and_epilogue() {
        let case = seed_valid_case("order");
        let entries = case.projection["durable"]["entries"]
            .as_array()
            .expect("full ordered durable history");
        assert_eq!(entries.len(), case.branch.len());
        assert_eq!(
            case.projection["durable"]["branch_entry_ids"]
                .as_array()
                .expect("branch IDs")
                .len(),
            case.branch.len()
        );
        let compact_index = case
            .branch
            .iter()
            .position(|entry| entry["id"] == case.compaction_entry)
            .expect("compaction index");
        let compaction_id = format!("entry:{compact_index:06}");
        let retained_index = case
            .branch
            .iter()
            .position(|entry| entry["id"] == case.retained_call_entry)
            .expect("retained call index");
        let retained_id = format!("entry:{retained_index:06}");
        let post_index = case
            .branch
            .iter()
            .position(|entry| entry["id"] == case.post_compaction_entry)
            .expect("post index");
        let items = case.projection["rebuilt"]["items"]
            .as_array()
            .expect("summary-first rebuilt context");
        assert_eq!(items[0]["entry_id"], compaction_id);
        assert_eq!(items[0]["normalized_message"]["role"], "compactionSummary");
        assert_eq!(case.projection["rebuilt"]["epilogue_start_index"], 1);
        assert!(items.iter().any(|item| item["entry_id"] == retained_id));
        let before_cut = case
            .branch
            .iter()
            .position(|entry| entry["id"] == case.user_before_cut)
            .expect("before-cut index");
        assert!(!items
            .iter()
            .any(|item| item["entry_id"] == format!("entry:{before_cut:06}")));
        assert!(items
            .iter()
            .any(|item| item["entry_id"] == format!("entry:{post_index:06}")));
        assert_eq!(
            items.last().expect("last rebuilt item")["normalized_message"]["role"],
            "assistant"
        );
        assert_eq!(
            entries[compact_index]["firstKeptEntryId"],
            format!("entry:{retained_index:06}")
        );
        let durable_compaction = &entries[compact_index];
        assert_eq!(durable_compaction["fromHook"], json!(false));
        assert_eq!(
            durable_compaction["customInstructions"],
            "test custom instructions"
        );
        assert_eq!(durable_compaction["harnessDigest"], "test-harness-digest");
        assert_eq!(durable_compaction["tokensBefore"], json!(19.0));
        assert_eq!(durable_compaction["usage"]["input"], json!(19.0));
        assert_eq!(durable_compaction["usage"]["output"], json!(2.0));
        assert_eq!(durable_compaction["usage"]["totalTokens"], json!(21.0));
        let protected = &durable_compaction["details"]["customProtected"];
        assert_eq!(protected["goals"], json!(["preserve goal"]));
        assert_eq!(protected["constraints"], json!(["preserve constraint"]));
        assert_eq!(protected["evidence"], json!(["preserve evidence"]));
        assert_eq!(
            protected["pending_subagent"]["pendingResult"],
            "opaque but non-provider"
        );
        assert_eq!(protected["unknownTimestamp"], json!(1_772_000_000_000i64));
        assert_eq!(
            durable_compaction["details"]["unknownFutureField"]["scalarTypes"],
            json!([1, "1", true, null])
        );
        let rebuilt_summary = &items[0]["normalized_message"];
        assert_eq!(rebuilt_summary["tokensBefore"], json!(19.0));
        assert_eq!(rebuilt_summary["customInstructions"], "test custom instructions");
        assert_eq!(rebuilt_summary["harnessDigest"], "test-harness-digest");
        assert_eq!(case.projection["availability"]["provider_context"], "not_emitted");
        assert_eq!(case.projection["availability"]["pending_subagent"], "emitted");
        assert_eq!(case.projection["availability"]["input_release"], "not_emitted");
        for feature in ["pending_subagent", "extensions", "input_release"] {
            assert_eq!(
                case.projection["availability"]["semantic_comparison_status"][feature],
                "not_identifiable"
            );
        }
        let tool_calls = case.projection["rebuilt"]["tool_calls"]
            .as_array()
            .expect("tool identity records");
        assert_eq!(tool_calls[0]["id"], "tool:000000");
        assert_eq!(tool_calls[0]["orphan_in_rebuilt"], false);
        let tool_results = case.projection["rebuilt"]["tool_results"]
            .as_array()
            .expect("tool result records");
        assert!(tool_results
            .iter()
            .any(|result| result["orphan_in_rebuilt"] == true));
        assert_eq!(case.projection["rebuilt"]["pairing"]["matched_results"], 1);
        assert_eq!(case.projection["rebuilt"]["pairing"]["unmatched_results"], 1);
        assert_eq!(case.projection["rebuilt"]["pairing"]["calls_with_result"], 1);
        assert!(case.observed_live["matches_rebuilt"].as_bool().expect("bool"));
        assert!(case.observed_live["real_context_messages_match_rebuilt"]
            .as_bool()
            .expect("bool"));
        assert!(case.observed_live["thinking_level_matches_rebuilt"]
            .as_bool()
            .expect("bool"));
        assert!(case.observed_live["service_tier_matches_rebuilt"]
            .as_bool()
            .expect("bool"));
        assert!(case.observed_live["model_matches_rebuilt"]
            .as_bool()
            .expect("bool"));
    }

    #[test]
    fn random_ids_roots_and_timestamps_normalize_but_nonvolatile_drift_survives() {
        let first = seed_valid_case("first-random");
        let second = seed_valid_case("second-random");
        assert_ne!(first.header["id"], second.header["id"]);
        assert_ne!(first.session_file, second.session_file);
        assert_eq!(first.projection, second.projection);
        assert_eq!(first.observed_live, second.observed_live);
        let compact_index = first
            .entries
            .iter()
            .position(|entry| entry["id"] == first.compaction_entry)
            .expect("compaction entry");
        let normalizer = SessionNormalizer::new(&first.header, &first.entries)
            .expect("session normalizer");
        let original = normalizer
            .normalize_entry(&first.entries[compact_index])
            .expect("normalized compaction");
        for (key, value) in [
            ("tokensBefore", json!("19")),
            ("fromHook", json!(1)),
            ("summary", json!("changed summary")),
        ] {
            let mut changed = first.entries[compact_index].clone();
            changed.insert(key.into(), value);
            assert_ne!(
                normalizer.normalize_entry(&changed).expect("changed normalization"),
                original,
                "{key} must not be normalized away"
            );
        }
        let mut changed_usage = first.entries[compact_index].clone();
        changed_usage["usage"]["input"] = json!(20.0);
        assert_ne!(
            normalizer
                .normalize_entry(&changed_usage)
                .expect("changed usage"),
            original
        );
        let mut unknown = first.entries[compact_index].clone();
        unknown["details"]["customProtected"]["newUnknown"] = json!(true);
        assert_ne!(
            normalizer.normalize_entry(&unknown).expect("unknown field"),
            original
        );
        let mut detail_timestamp = first.entries[compact_index].clone();
        detail_timestamp["details"]["customProtected"]["unknownTimestamp"] = json!(999);
        assert_ne!(
            normalizer
                .normalize_entry(&detail_timestamp)
                .expect("custom timestamp remains nonvolatile"),
            original,
            "unknown custom timestamps must not be broadly scrubbed"
        );
    }

    fn assert_projection_change(case: &ValidCase, change: impl FnOnce(&mut Value)) {
        let mut changed = case.projection.clone();
        change(&mut changed);
        let changed = with_projection_hashes(changed).expect("recompute mutated hashes");
        assert_ne!(changed, case.projection);
        assert_ne!(
            changed["hashes"]["projection_sha256"],
            case.projection["hashes"]["projection_sha256"]
        );
    }

    #[test]
    fn canonical_domains_and_mutations_are_independently_recomputable() {
        let case = seed_valid_case("hash-domains");
        let mut without_hashes = case.projection.clone();
        let old_hashes = without_hashes
            .as_object_mut()
            .expect("projection object")
            .remove("hashes")
            .expect("real hashes");
        let durable = json!({
            "header": without_hashes["header"],
            "durable": without_hashes["durable"],
        });
        assert_eq!(
            old_hashes["durable_sha256"],
            canonical_sha256(&durable).expect("durable domain")
        );
        assert_eq!(
            old_hashes["context_sha256"],
            canonical_sha256(&without_hashes["rebuilt"]).expect("context domain")
        );
        assert_eq!(
            old_hashes["projection_sha256"],
            canonical_sha256(&without_hashes).expect("projection domain")
        );
        assert_eq!(with_projection_hashes(without_hashes).expect("hash"), case.projection);
        for hash in ["durable_sha256", "context_sha256", "projection_sha256"] {
            let value = old_hashes[hash].as_str().expect("actual SHA-256");
            assert_eq!(value.len(), 64);
            assert!(value.bytes().all(|byte| byte.is_ascii_hexdigit()));
            assert_ne!(value, "0".repeat(64));
        }
        assert!(case.projection.get("observed_live").is_none());
        let compact_index = case
            .entries
            .iter()
            .position(|entry| entry["id"] == case.compaction_entry)
            .expect("compaction index");
        assert_projection_change(&case, |value| value["availability"]["input_release"] = json!("emitted"));
        assert_projection_change(&case, |value| {
            value["durable"]["entries"][compact_index]["unknown"] = json!("new");
        });
        assert_projection_change(&case, |value| {
            value["durable"]["entries"][compact_index]["usage"]["cost"]["total"] = json!(1.0);
        });
        assert_projection_change(&case, |value| {
            value["durable"]["entries"][compact_index]["details"]["customProtected"]
                ["pending_subagent"]["pendingResult"] = json!("different pending detail");
        });
        assert_projection_change(&case, |value| {
            value["durable"]["entries"][compact_index]["details"]["future"] = json!("custom");
        });
        assert_projection_change(&case, |value| value["rebuilt"]["epilogue_start_index"] = json!(2));
        assert_projection_change(&case, |value| {
            value["rebuilt"]["items"][1]["entry_id"] = json!("entry:changed");
        });
        assert_projection_change(&case, |value| {
            value["rebuilt"]["tool_results"][0]["matched_call_message_indices"] = json!([]);
        });
        assert_projection_change(&case, |value| {
            value["durable"]["entries"].as_array_mut().expect("entries").swap(0, 1);
        });
    }

    #[test]
    fn provider_shapes_fail_even_if_observed_and_rebuilt_messages_match() {
        for kind in [
            "providerCheckpoint",
            "openai-responses-compaction-v2",
            "providerContext",
            "compactedWindow",
            "nativeAuxiliary",
            "provider_auxiliary",
        ] {
            let case = seed_valid_case(&format!("provider-{kind}"));
            mutate_jsonl(&case, |rows| {
                let compaction = rows
                    .iter_mut()
                    .find(|row| row["type"] == "compaction")
                    .expect("compaction");
                match kind {
                    "providerCheckpoint" => {
                        compaction["details"]["providerCheckpoint"] = json!({"opaque": "checkpoint"});
                    }
                    "openai-responses-compaction-v2" => {
                        compaction["details"]["strategy"] = json!(kind);
                    }
                    "providerContext" => compaction["details"]["providerContext"] = json!({}),
                    "compactedWindow" => compaction["details"]["compactedWindow"] = json!([]),
                    "nativeAuxiliary" => compaction["details"]["nativeAuxiliary"] = json!({}),
                    "provider_auxiliary" => {
                        compaction["details"]["provider_auxiliary"] = json!({})
                    }
                    _ => unreachable!(),
                }
            });
            let error = verify_reopen_using(&reopen_cli(&case), |_| {
                panic!("provider preflight must reject before production open")
            })
            .expect_err("provider/native auxiliary must fail");
            assert!(error.contains("PROVIDER_DEPENDENT_AUXILIARY_UNSUPPORTED"),"{kind}: {error}");
        }
        let checkpoint = pi_ai::compaction::ProviderCompactionCheckpoint {
            version: 1,
            provider: "harness-provider-test".into(),
            api: "harness-api-test".into(),
            model: "compaction-replay-model".into(),
            base_url: "https://fixture.invalid/v1".into(),
            endpoint: None,
            items: vec![json!({"type": "user", "content": "opaque"})
                .as_object()
                .expect("checkpoint item")
                .clone()],
            estimated_tokens: 1.0,
        };
        let message = AgentMessage::Custom(CustomAgentMessage::CompactionSummary {
            summary: "same summary".into(),
            provider_context: Some(checkpoint),
            tokens_before: 1.0,
            retained_message_count: Some(0.0),
            custom_instructions: None,
            harness_digest: None,
            timestamp: 123,
        });
        let observed = vec![message.clone()];
        let rebuilt = vec![message];
        assert_eq!(observed, rebuilt, "message equality is intentionally true");
        assert!(reject_provider_dependent_messages(&observed).is_err());
        assert!(reject_provider_dependent_messages(&rebuilt).is_err());
    }

    #[test]
    fn observed_live_and_context_mismatches_fail_despite_matching_other_fields() {
        let case = seed_valid_case("observed-mismatch");
        let manager = SessionManager::open(
            &case.session_file.to_string_lossy(),
            None,
            None,
        )
        .expect("real-valid manager for observed comparison test");
        let context = manager.build_session_context(None);
        let base = || ObservedWriterContext {
            live_messages: context.messages.clone(),
            context_messages: context.messages.clone(),
            thinking_level: context.thinking_level.clone(),
            service_tier: context.service_tier.clone(),
            model: context.model.clone(),
        };
        assert!(session_projection(&manager, Some(base())).is_ok());
        let mut bad = base();
        bad.thinking_level = "off".into();
        assert!(session_projection(&manager, Some(bad)).is_err());
        let mut bad = base();
        bad.service_tier = Some(Some("priority".into()));
        assert!(session_projection(&manager, Some(bad)).is_err());
        let mut bad = base();
        bad.model = Some(SessionContextModel {
            provider: "different-provider".into(),
            model_id: "different-model".into(),
        });
        assert!(session_projection(&manager, Some(bad)).is_err());
        let mut bad = base();
        bad.live_messages.pop();
        assert!(session_projection(&manager, Some(bad)).is_err());
        let mut bad = base();
        bad.context_messages.pop();
        assert!(session_projection(&manager, Some(bad)).is_err());
        drop(manager);
    }

    #[test]
    fn strict_valid_reopen_preserves_four_stage_bytes_and_nonempty_root_inventory() {
        let case = seed_valid_case("valid-reopen");
        let original = std::fs::read(&case.session_file).expect("closed input bytes");
        let original_digest = sha256_hex(&original);
        let initial = snapshot_root_inventory(&case.work_root).expect("nonempty root inventory");
        assert!(initial.keys().any(|path| path.ends_with("semantic-edges.jsonl")));
        let output = case.work_root.join("verify-reopen.json");
        let cli = reopen_cli(&case);
        let exit = run_verify_reopen(&cli).expect("real production open/build/drop");
        assert_eq!(exit, 0);
        let report: Value = serde_json::from_str(
            &std::fs::read_to_string(&output).expect("reader report"),
        )
        .expect("reader report JSON");
        assert_eq!(report["schema"], COMMON_READER_SCHEMA);
        for stage in ["before_open", "immediately_after_open", "after_drop", "final"] {
            assert_eq!(report["input_bytes"][stage]["sha256"], original_digest);
            assert_eq!(report["input_bytes"][stage]["matches_before_open"], true);
        }
        assert_eq!(report["reader"]["pid"], std::process::id());
        assert!(report["reader"]["exe_sha256"]
            .as_str()
            .is_some_and(|hash| hash.len() == 64));
        assert_eq!(report["open_rebuild"]["additional_user_or_provider_turns"], 0);
        assert_eq!(
            report["open_rebuild"]["in_process_restore_substitute"],
            false
        );
        assert_eq!(
            report["session_projection"],
            case.projection,
            "writer/reader normalized bodies must match"
        );
        let final_inventory = snapshot_root_inventory(&case.work_root).expect("final root");
        assert_inventory_unchanged(
            &initial,
            &final_inventory,
            "verify-reopen.json",
            &[String::new()],
            true,
        )
        .expect("only declared report may be new");
        assert_eq!(
            std::fs::read(&case.session_file).expect("final input bytes"),
            original
        );

        // A key repeated in different JSON objects is legal. Exercise real
        // open/build/drop too, not just the visitor on an invented string.
        let legal = seed_valid_case("legal-keys-in-distinct-objects");
        let legal_compaction_index = legal
            .entries
            .iter()
            .position(|entry| entry["type"] == "compaction")
            .expect("legal protected entry index");
        mutate_jsonl(&legal, |rows| {
            let compaction = rows
                .iter_mut()
                .find(|row| row["type"] == "compaction")
                .expect("legal protected compaction");
            compaction["details"]["customProtected"]["legalObjects"] = json!([
                {"sameKey": "first"},
                {"sameKey": "second"},
            ]);
            compaction["details"]["customProtected"]["legalSibling"] =
                json!({"sameKey": "third"});
        });
        let legal_input = std::fs::read(&legal.session_file).expect("legal control input");
        let legal_input_hash = sha256_hex(&legal_input);
        let legal_output = legal.work_root.join("verify-reopen.json");
        assert_eq!(
            run_verify_reopen(&reopen_cli(&legal)).expect("legal distinct-object control"),
            0
        );
        let legal_report: Value = serde_json::from_str(
            &std::fs::read_to_string(&legal_output).expect("legal reader report"),
        )
        .expect("legal report JSON");
        assert_eq!(
            legal_report["session_projection"]["durable"]["entries"][legal_compaction_index]
                ["details"]["customProtected"]["legalObjects"],
            json!([{"sameKey": "first"}, {"sameKey": "second"}]),
            "per-object key scope and array order must both survive real reopen"
        );
        assert_eq!(
            legal_report["session_projection"]["durable"]["entries"][legal_compaction_index]
                ["details"]["customProtected"]["legalSibling"]["sameKey"],
            "third"
        );
        for stage in ["before_open", "immediately_after_open", "after_drop", "final"] {
            assert_eq!(legal_report["input_bytes"][stage]["sha256"], legal_input_hash);
        }

        // Existing output cannot be silently replaced during a second run.
        assert!(reopen_preflight(&cli).is_err());
        // This in-process test is not a fresh-PID or frozen-arm executable claim.
        assert_eq!(report["external_final_verification_required"], true);
    }

    #[test]
    fn malformed_migrating_missing_duplicate_and_nonrepair_inputs_fail_before_open() {
        for kind in [
            "empty",
            "headerless",
            "old-version-1",
            "old-version-2",
            "invalid-json",
            "invalid-utf8",
            "nul",
            "unterminated",
            "duplicate-header-current-version",
            "duplicate-nested-protected-key",
            "duplicate-id",
            "missing-id",
            "nonreplay",
            "unsafe-header-id",
            "parent-session-depth",
            "broken-parent-link",
            "missing-first-kept",
            "unsafe-compaction",
        ] {
            let case = seed_valid_case(&format!("negative-{kind}"));
            let duplicate_object_row = if matches!(
                kind,
                "duplicate-header-current-version" | "duplicate-nested-protected-key"
            ) {
                Some(inject_raw_duplicate_object_key(&case, kind))
            } else {
                None
            };
            match kind {
                "empty" => std::fs::write(&case.session_file, b"").expect("empty input"),
                "headerless" => mutate_jsonl(&case, |rows| {
                    rows.remove(0);
                }),
                "old-version-1" | "old-version-2" => mutate_jsonl(&case, |rows| {
                    rows[0]["version"] = json!(if kind.ends_with('1') { 1 } else { 2 });
                }),
                "invalid-json" => {
                    let mut bytes = std::fs::read(&case.session_file).expect("input");
                    bytes.extend_from_slice(b"{not-json}\n");
                    std::fs::write(&case.session_file, bytes).expect("invalid input");
                }
                "invalid-utf8" => {
                    std::fs::write(&case.session_file, b"\xff\n").expect("invalid UTF-8 input");
                }
                "nul" => {
                    let mut bytes = std::fs::read(&case.session_file).expect("input");
                    let last = bytes.len() - 1;
                    bytes.insert(last, 0);
                    std::fs::write(&case.session_file, bytes).expect("NUL input");
                }
                "unterminated" => {
                    let mut bytes = std::fs::read(&case.session_file).expect("input");
                    bytes.pop();
                    std::fs::write(&case.session_file, bytes).expect("unterminated input");
                }
                "duplicate-header-current-version" | "duplicate-nested-protected-key" => {
                    // Raw-byte mutation and ordinary-Value equivalence were completed above.
                }
                "duplicate-id" => mutate_jsonl(&case, |rows| {
                    let id = rows[1]["id"].clone();
                    rows[2]["id"] = id;
                }),
                "missing-id" => mutate_jsonl(&case, |rows| {
                    rows[2].as_object_mut().expect("entry").remove("id");
                }),
                "nonreplay" => mutate_jsonl(&case, |rows| {
                    for row in rows {
                        if row["message"]["role"] == "assistant" {
                            row["message"]["model"] = json!("production-model");
                        }
                    }
                }),
                "unsafe-header-id" => mutate_jsonl(&case, |rows| rows[0]["id"] = json!("../unsafe")),
                "parent-session-depth" => mutate_jsonl(&case, |rows| {
                    rows[0]["parentSessionId"] = json!("parent");
                }),
                "broken-parent-link" => mutate_jsonl(&case, |rows| {
                    rows[2]["parentId"] = json!("missing-parent");
                }),
                "missing-first-kept" => mutate_jsonl(&case, |rows| {
                    let compaction = rows
                        .iter_mut()
                        .find(|row| row["type"] == "compaction")
                        .expect("compaction");
                    compaction["firstKeptEntryId"] = json!("missing-first-kept");
                }),
                "unsafe-compaction" => mutate_jsonl(&case, |rows| {
                    let compaction = rows
                        .iter_mut()
                        .find(|row| row["type"] == "compaction")
                        .expect("compaction");
                    compaction.as_object_mut().expect("object").remove("summary");
                }),
                _ => unreachable!(),
            }
            let error = assert_rejected_before_open(&case, reopen_cli(&case));
            if let Some(row) = duplicate_object_row {
                assert!(
                    error.contains("DUPLICATE_OBJECT_KEY"),
                    "{kind}: wrong duplicate-key error class: {error}"
                );
                assert!(
                    error.contains(&format!("session JSON row {row}")),
                    "{kind}: expected duplicate at actual mutated row {row}: {error}"
                );
                assert!(!error.contains("DUPLICATE_REQUIRED_ENTRY_ID"));
            }
            if kind == "duplicate-id" {
                assert!(error.contains("DUPLICATE_REQUIRED_ENTRY_ID"), "{error}");
                assert!(!error.contains("DUPLICATE_OBJECT_KEY"));
            }
            if kind == "invalid-utf8" {
                assert!(
                    error.contains("session file is not UTF-8"),
                    "raw 0xff plus newline must reach UTF-8 validation: {error}"
                );
            }
        }
    }

    #[test]
    fn missing_directory_and_all_output_path_gates_fail_before_open() {
        let missing = seed_valid_case("path-missing");
        let mut cli = reopen_cli(&missing);
        let missing_path = missing.work_root.join("absent/input.jsonl");
        cli.options.insert(
            "session-file".into(),
            missing_path.to_string_lossy().to_string(),
        );
        assert_rejected_before_open(&missing, cli);

        let directory = seed_valid_case("path-directory");
        let directory_path = directory.work_root.join("directory-input");
        std::fs::create_dir(&directory_path).expect("directory negative input");
        let original = std::fs::read(&directory.session_file).expect("input bytes");
        let mut cli = reopen_cli(&directory);
        cli.options.insert(
            "session-file".into(),
            directory_path.to_string_lossy().to_string(),
        );
        assert_rejected_before_open(&directory, cli);
        assert_eq!(std::fs::read(&directory.session_file).expect("input"), original);

        for kind in [
            "relative-escape-input",
            "absolute-outside-input",
            "output-escape",
            "nonexistent-output-parent",
            "existing-output",
            "input-equals-output",
            "input-sibling-output",
            "unknown-option",
        ] {
            let case = seed_valid_case(&format!("path-{kind}"));
            let original = std::fs::read(&case.session_file).expect("original");
            let mut cli = reopen_cli(&case);
            match kind {
                "relative-escape-input" => {
                    cli.options.insert("session-file".into(), "../outside.jsonl".into());
                }
                "absolute-outside-input" => {
                    let outside = case._temp.path().join("outside.jsonl");
                    std::fs::copy(&case.session_file, &outside).expect("outside private copy");
                    cli.options.insert(
                        "session-file".into(),
                        outside.to_string_lossy().to_string(),
                    );
                }
                "output-escape" => {
                    cli.options.insert("out".into(), "../outside.json".into());
                }
                "nonexistent-output-parent" => {
                    cli.options.insert(
                        "out".into(),
                        case.work_root
                            .join("does-not-exist/out.json")
                            .to_string_lossy()
                            .to_string(),
                    );
                }
                "existing-output" => {
                    let output = case.work_root.join("verify-reopen.json");
                    std::fs::write(&output, "old report").expect("existing negative report");
                }
                "input-equals-output" => {
                    cli.options.insert(
                        "out".into(),
                        case.session_file.to_string_lossy().to_string(),
                    );
                }
                "input-sibling-output" => {
                    cli.options.insert(
                        "out".into(),
                        case.session_file
                            .parent()
                            .expect("input parent")
                            .join("verify-reopen.json")
                            .to_string_lossy()
                            .to_string(),
                    );
                }
                "unknown-option" => {
                    cli.options.insert("raw-samples".into(), String::new());
                }
                _ => unreachable!(),
            }
            let output = cli
                .options
                .get("out")
                .map(PathBuf::from)
                .expect("output argument");
            let called = Cell::new(false);
            assert!(verify_reopen_using(&cli, |_| {
                called.set(true);
                panic!("path negative must reject before open")
            })
            .is_err());
            assert!(!called.get());
            assert_eq!(std::fs::read(&case.session_file).expect("input"), original);
            if !matches!(kind, "existing-output" | "input-equals-output") {
                assert!(!output.exists());
            }
        }
    }

    #[cfg(unix)]
    fn make_file_symlink(target: &Path, link: &Path) {
        std::os::unix::fs::symlink(target, link).expect("symlink test capability");
    }

    #[cfg(windows)]
    fn make_file_symlink(target: &Path, link: &Path) {
        std::os::windows::fs::symlink_file(target, link).expect("symlink test capability");
    }

    #[cfg(unix)]
    fn make_dir_symlink(target: &Path, link: &Path) {
        std::os::unix::fs::symlink(target, link).expect("symlink test capability");
    }

    #[cfg(windows)]
    fn make_dir_symlink(target: &Path, link: &Path) {
        std::os::windows::fs::symlink_dir(target, link).expect("symlink test capability");
    }

    #[test]
    fn symlink_file_and_root_escape_fail_before_open() {
        let file_case = seed_valid_case("symlink-input");
        let link = file_case
            .session_file
            .parent()
            .expect("session parent")
            .join("linked-input.jsonl");
        make_file_symlink(&file_case.session_file, &link);
        let original = std::fs::read(&file_case.session_file).expect("target bytes");
        let mut cli = reopen_cli(&file_case);
        cli.options.insert(
            "session-file".into(),
            link.to_string_lossy().to_string(),
        );
        let called = Cell::new(false);
        assert!(verify_reopen_using(&cli, |_| {
            called.set(true);
            panic!("symlink file preflight must reject first")
        })
        .is_err());
        assert!(!called.get());
        assert_eq!(std::fs::read(&file_case.session_file).expect("target"), original);

        let root_case = seed_valid_case("symlink-root");
        let link_root = root_case._temp.path().join("root-link");
        make_dir_symlink(&root_case.work_root, &link_root);
        let mut cli = reopen_cli(&root_case);
        cli.options.insert(
            "work-root".into(),
            link_root.to_string_lossy().to_string(),
        );
        let called = Cell::new(false);
        assert!(verify_reopen_using(&cli, |_| {
            called.set(true);
            panic!("symlink root preflight must reject first")
        })
        .is_err());
        assert!(!called.get());

        let output_case = seed_valid_case("symlink-output");
        let outside_output = output_case._temp.path().join("outside-report.json");
        std::fs::write(&outside_output, "do not replace").expect("outside symlink target");
        let output_link = output_case.work_root.join("verify-reopen.json");
        make_file_symlink(&outside_output, &output_link);
        let cli = reopen_cli(&output_case);
        let called = Cell::new(false);
        assert!(verify_reopen_using(&cli, |_| {
            called.set(true);
            panic!("symlink output preflight must reject first")
        })
        .is_err());
        assert!(!called.get());
        assert_eq!(
            std::fs::read_to_string(&outside_output).expect("symlink target preserved"),
            "do not replace"
        );
    }

    #[test]
    fn open_error_after_mutation_is_reported_without_restoring_or_hiding_evidence() {
        let case = seed_valid_case("open-error-mutation");
        let before = std::fs::read(&case.session_file).expect("valid test input before error");
        let input = case.session_file.clone();
        let sidecar = case.work_root.join("unexpected-open-error-sidecar");
        let cli = reopen_cli(&case);
        let result = verify_reopen_using(&cli, |_| {
            std::fs::write(&input, b"intentional private failure fixture\n")
                .expect("mutate only the private negative fixture");
            std::fs::write(&sidecar, b"retain unexpected error evidence")
                .expect("negative unexpected sidecar");
            Err("injected open error after mutation".into())
        })
        .expect_err("post-error mutations fail closed");
        assert!(result.contains("post-error evidence check"));
        assert_ne!(
            std::fs::read(&case.session_file).expect("failure evidence bytes"),
            before,
            "reader must not restore over the altered input"
        );
        assert_eq!(
            std::fs::read(&sidecar).expect("unexpected sidecar retained"),
            b"retain unexpected error evidence"
        );
        assert!(!case.work_root.join("verify-reopen.json").exists());
    }

    #[test]
    fn post_open_side_effect_is_never_accepted_or_hidden() {
        let case = seed_valid_case("open-side-effect");
        let cli = reopen_cli(&case);
        let root = case.work_root.clone();
        let result = verify_reopen_using(&cli, |path| {
            std::fs::write(root.join("unexpected-sidecar"), b"test injected side effect")
                .expect("injected side effect");
            SessionManager::open(path, None, None)
        });
        assert!(result
            .expect_err("unexpected sibling must fail after open")
            .contains("unexpected-sidecar"));
        assert!(!case.work_root.join("verify-reopen.json").exists());
    }

    #[test]
    fn pending_replay_holders_do_not_claim_test_or_continuation_success() {
        let pending = replay_operation_evidence_pending();
        assert_eq!(pending["record_status"], "not_ready_not_a_complete_v2_pair_record");
        for field in [
            "protected_goals",
            "protected_constraints",
            "protected_evidence",
            "pending_subagent",
            "tool_result_identity",
        ] {
            assert_eq!(
                pending[field]["reason_code"],
                "PROTECTED_ITEM_ABSENT_FROM_FIXTURE"
            );
            assert!(pending[field]["verdict"]
                .as_str()
                .expect("pending verdict")
                .starts_with("not_ready"));
            assert!(pending[field]["value"].is_null());
            assert_eq!(pending[field]["supporting_test_refs"], json!([]));
        }
        let continuation = &pending["post_restart_provider_continuation"];
        assert_eq!(continuation["reason_code"], "REOPEN_HAS_NO_ADDITIONAL_TURNS");
        assert_eq!(continuation["additional_user_or_provider_turns_in_reader"], 0);
        assert_eq!(continuation["reader_fact_is_not_a_continuation_pass"], true);
        assert_eq!(continuation["supporting_test_refs"], json!([]));
        assert!(continuation["verdict"]
            .as_str()
            .expect("pending continuation")
            .starts_with("not_ready"));
        let source = include_str!("compaction_replay.rs");
        let projection_start = source.find("fn session_projection(").expect("projection source");
        let projection_end = source[projection_start..]
            .find("struct ReplayOptions")
            .map(|relative_end| projection_start + relative_end)
            .expect("ReplayOptions after projection source");
        let projection_source = &source[projection_start..projection_end];
        assert!(projection_source.starts_with("fn session_projection("));
        assert!(!projection_source.contains("DIRECT_SERIALIZER_NO_"));
        let mut disposable_projection_source = projection_source.to_string();
        disposable_projection_source.push_str(
            "// Mutation control: DIRECT_SERIALIZER_NO_ must be detected in this copy.\n",
        );
        assert!(disposable_projection_source.contains("DIRECT_SERIALIZER_NO_"));
    }
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
        // Synchronous by construction: no Tokio runtime or agent/provider stack.
        "verify-reopen" => run_verify_reopen(&cli),
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
