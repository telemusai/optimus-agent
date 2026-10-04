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
use std::collections::HashMap;
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
    CompactionSessionEntry, SUMMARY_UPDATE_POLICY_OFF, MAX_COMPACTION_CONTEXT_TOKENS,
};
use pi_coding_agent::core::compaction::utils::{
    serialize_conversation, SUMMARIZATION_SYSTEM_PROMPT,
};
use pi_coding_agent::core::jev_compaction::prepare_context;
use pi_coding_agent::core::messages::convert_to_llm;
use pi_coding_agent::core::model_registry::ModelRegistry;
use pi_coding_agent::core::model_tool_output_policy::ModelToolOutputPolicyOptions;
use pi_coding_agent::core::resource_loader::DefaultResourceLoaderOptions;
use pi_coding_agent::core::session_manager::SessionManager;
use pi_coding_agent::core::settings_manager::SettingsManager;
use pi_jev::compaction::CompactionConfig;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
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
// replay subcommand
// ---------------------------------------------------------------------------

struct ReplayOptions {
    provider_delay_ms: u64,
    summary_failure: Option<String>,
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
    // Per-call request payload digests, in call order, plus an aggregate: a
    // chunking/serialization regression that preserves chunk counts and sizes
    // still changes these.
    let provider_request_digests: Vec<String> = provider_calls
        .iter()
        .map(|call| {
            call.get("payload_sha256")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string()
        })
        .collect();
    let provider_requests_sha256 = sha256_hex(provider_request_digests.join("\n").as_bytes());

    session.dispose_async(Some(false)).await;

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

async fn run_replay(cli: &Cli) -> Result<i32, String> {
    let fixture_path = PathBuf::from(cli.required("fixture")?);
    let out = PathBuf::from(cli.required("out")?);
    let cases = discover_fixtures(&fixture_path)?;
    let options = ReplayOptions {
        provider_delay_ms: cli.number("provider-delay-ms", 0.0)? as u64,
        summary_failure: cli.optional("summary-failure"),
        context_window: cli.number("context-window", 1_000_000.0)?,
        keep_recent: match cli.number("keep-recent", f64::NAN) {
            Ok(value) if value.is_finite() => Some(value),
            Ok(_) => None,
            Err(error) => return Err(error),
        },
        work: cli
            .optional("work")
            .map(PathBuf::from)
            .unwrap_or_else(|| std::env::temp_dir().join("compaction-replay-work")),
        timeout_secs: cli.number("timeout-secs", 900.0)? as u64,
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
    let output = json!({
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
    json!({
        "name": name,
        "detail": detail,
        "warmup": warmup,
        "stats": timing_stats(samples),
        "allocs_first_iter": {
            "count": alloc_count,
            "bytes": alloc_bytes,
        },
        "checksum": checksum,
    })
}

fn run_bench_local(cli: &Cli) -> Result<i32, String> {
    let fixture_path = PathBuf::from(cli.required("fixture")?);
    let out = PathBuf::from(cli.required("out")?);
    let iters = cli.number("iters", 20.0)? as usize;
    let warmup = cli.number("warmup", 3.0)? as usize;
    let chunk_context_window = cli.number("chunk-context-window", 32_768.0)?;
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
            let preparation =
                prepare_compaction(&entries, &settings, &|_entries| messages.clone());
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
                let budget_chars = (((f64::min(
                    input_limit,
                    model.context_window - retry_max_tokens,
                ) - 1024.0)
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
                move || {
                    manager
                        .lock()
                        .unwrap()
                        .build_session_context(None)
                        .messages
                        .len() as f64
                },
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
    let output = json!({
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
    write_json(&out, &output)?;
    println!("wrote {}", out.display());
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
