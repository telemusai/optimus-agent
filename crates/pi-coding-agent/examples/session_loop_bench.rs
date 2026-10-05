//! Offline headless agent-session loop bench for hot-path profiling.
//!
//! Drives the REAL `AgentSession` + pi-agent-core agent loop in-process against
//! the faux provider (no network, no real provider). Purpose: attribute wall
//! time and allocations per turn to production code OUTSIDE the already-owned
//! areas (provider SSE/WS/parse: pi-ai/providers; compaction: core/compaction +
//! its restore path; disk writes: event log / session JSONL / atomic file).
//!
//! Subcommands (hand-rolled CLI; std + existing crate APIs only):
//!   run --out <json> [--turns N] [--prompt-chars N] [--reply-chars N]
//!       [--chunk-min N] [--chunk-max N] [--warmup N] [--alloc]
//!       [--stack-sample-rate 1/N] [--max-stack-samples N]
//!       [--snapshot-every N] [--work DIR] [--seed S]
//!       Startup, per-turn and snapshot timings; allocation counts/bytes per
//!       phase; optional sampled, symbolized allocation backtraces.
//!   startup --out <json> [--runs N] [--work DIR]
//!       Services + session creation cost (resource loading, settings, model
//!       registry) with allocation attribution.
//!
//! The counting allocator and the stack sampler are process-global and gated:
//! timing runs stay clean unless --alloc is passed. Everything here is
//! measurement tooling: no production file depends on it and it must keep
//! working unchanged BEFORE and AFTER optimization work.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use pi_ai::providers::faux::{
    faux_assistant_message, register_faux_provider, FauxAssistantContent, FauxResponseStep,
    RegisterFauxProviderOptions,
};
use pi_coding_agent::core::agent_session::AgentSession;
use pi_coding_agent::core::agent_session_services::{
    create_agent_session_from_services, create_agent_session_services,
    AgentSessionCreationOptions, CreateAgentSessionFromServicesOptions,
    CreateAgentSessionServicesOptions,
};
use pi_coding_agent::core::auth_storage::{AuthStorage, AuthStorageData};
use pi_coding_agent::core::model_registry::ModelRegistry;
use pi_coding_agent::core::resource_loader::DefaultResourceLoaderOptions;
use pi_coding_agent::core::session_manager::SessionManager;
use pi_coding_agent::core::settings_manager::SettingsManager;
use serde_json::Value;

// ---------------------------------------------------------------------------
// Counting global allocator (std-only; gated so timing runs stay clean)
// ---------------------------------------------------------------------------

static ALLOC_GATE: AtomicBool = AtomicBool::new(false);
static ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);
/// When true, every ~STACK_SAMPLE_RATE-th counted allocation also captures a
/// symbolized backtrace (bounded by MAX_STACK_SAMPLES). Reentrancy-safe.
static STACK_SAMPLING: AtomicBool = AtomicBool::new(false);
static STACK_SAMPLE_COUNTER: AtomicU64 = AtomicU64::new(0);
static STACK_SAMPLE_RATE: AtomicU64 = AtomicU64::new(1);
static STACK_SAMPLES: Mutex<Vec<String>> = Mutex::new(Vec::new());

thread_local! {
    static IN_ALLOC_HOOK: RefCell<bool> = const { RefCell::new(false) };
}

struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if ALLOC_GATE.load(Ordering::Relaxed) && !IN_ALLOC_HOOK.with(|f| *f.borrow()) {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
            maybe_sample_stack();
        }
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if ALLOC_GATE.load(Ordering::Relaxed) && !IN_ALLOC_HOOK.with(|f| *f.borrow()) {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
            maybe_sample_stack();
        }
        System.realloc(ptr, layout, new_size)
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if ALLOC_GATE.load(Ordering::Relaxed) && !IN_ALLOC_HOOK.with(|f| *f.borrow()) {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
            maybe_sample_stack();
        }
        System.alloc_zeroed(layout)
    }
}

fn maybe_sample_stack() {
    if !STACK_SAMPLING.load(Ordering::Relaxed) {
        return;
    }
    let rate = STACK_SAMPLE_RATE.load(Ordering::Relaxed).max(1);
    if STACK_SAMPLE_COUNTER.fetch_add(1, Ordering::Relaxed) % rate != 0 {
        return;
    }
    let mut samples = STACK_SAMPLES.lock().unwrap();
    if samples.len() >= 4096 {
        return;
    }
    // The backtrace machinery allocates; the thread-local guard keeps those
    // allocations out of the counters and out of recursion.
    IN_ALLOC_HOOK.with(|flag| {
        *flag.borrow_mut() = true;
    });
    let trace = std::backtrace::Backtrace::force_capture().to_string();
    IN_ALLOC_HOOK.with(|flag| {
        *flag.borrow_mut() = false;
    });
    samples.push(trace);
}

#[global_allocator]
static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

fn alloc_snapshot() -> (u64, u64) {
    (
        ALLOC_COUNT.load(Ordering::Relaxed),
        ALLOC_BYTES.load(Ordering::Relaxed),
    )
}

fn alloc_delta(before: (u64, u64)) -> (u64, u64) {
    let after = alloc_snapshot();
    (after.0 - before.0, after.1 - before.1)
}

fn drain_stack_samples() -> Vec<String> {
    let mut samples = STACK_SAMPLES.lock().unwrap();
    std::mem::take(&mut *samples)
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct Cli {
    subcommand: String,
    out: PathBuf,
    turns: usize,
    prompt_chars: usize,
    reply_chars: usize,
    thinking_chars: usize,
    chunk_min: f64,
    chunk_max: f64,
    warmup: usize,
    alloc: bool,
    stack_sample_rate: u64,
    max_stack_samples: usize,
    snapshot_every: usize,
    work: Option<PathBuf>,
    seed: u64,
    runs: usize,
}

fn usage() -> String {
    "usage: session_loop_bench run --out <json> [--turns N] [--prompt-chars N] \
     [--reply-chars N] [--thinking-chars N] [--chunk-min N] [--chunk-max N] \
     [--warmup N] [--alloc] [--stack-sample-rate N] [--max-stack-samples N] \
     [--snapshot-every N] [--work DIR] [--seed S]\n\
     session_loop_bench startup --out <json> [--runs N] [--work DIR]"
        .to_string()
}

fn parse_cli(args: &[String]) -> Result<Cli, String> {
    let mut cli = Cli {
        subcommand: String::new(),
        out: PathBuf::from("session_loop_bench.json"),
        turns: 200,
        prompt_chars: 600,
        reply_chars: 4000,
        thinking_chars: 0,
        chunk_min: 3.0,
        chunk_max: 5.0,
        warmup: 5,
        alloc: false,
        stack_sample_rate: 0,
        max_stack_samples: 2048,
        snapshot_every: 0,
        work: None,
        seed: 0x5EED_5EED,
        runs: 10,
    };
    let mut index = 0;
    while index < args.len() {
        let arg = args[index].as_str();
        let mut value = || -> Result<String, String> {
            index += 1;
            args.get(index)
                .cloned()
                .ok_or_else(|| format!("missing value for {arg}"))
        };
        match arg {
            "run" | "startup" => cli.subcommand = arg.to_string(),
            "--out" => cli.out = PathBuf::from(value()?),
            "--turns" => cli.turns = value()?.parse().map_err(|e| format!("turns: {e}"))?,
            "--prompt-chars" => {
                cli.prompt_chars = value()?.parse().map_err(|e| format!("prompt-chars: {e}"))?
            }
            "--reply-chars" => {
                cli.reply_chars = value()?.parse().map_err(|e| format!("reply-chars: {e}"))?
            }
            "--thinking-chars" => {
                cli.thinking_chars = value()?.parse().map_err(|e| format!("thinking: {e}"))?
            }
            "--chunk-min" => {
                cli.chunk_min = value()?.parse().map_err(|e| format!("chunk-min: {e}"))?
            }
            "--chunk-max" => {
                cli.chunk_max = value()?.parse().map_err(|e| format!("chunk-max: {e}"))?
            }
            "--warmup" => cli.warmup = value()?.parse().map_err(|e| format!("warmup: {e}"))?,
            "--alloc" => cli.alloc = true,
            "--stack-sample-rate" => {
                cli.stack_sample_rate = value()?
                    .parse()
                    .map_err(|e| format!("stack-sample-rate: {e}"))?
            }
            "--max-stack-samples" => {
                cli.max_stack_samples = value()?
                    .parse()
                    .map_err(|e| format!("max-stack-samples: {e}"))?
            }
            "--snapshot-every" => {
                cli.snapshot_every = value()?.parse().map_err(|e| format!("snapshot: {e}"))?
            }
            "--work" => cli.work = Some(PathBuf::from(value()?)),
            "--seed" => cli.seed = value()?.parse().map_err(|e| format!("seed: {e}"))?,
            "--runs" => cli.runs = value()?.parse().map_err(|e| format!("runs: {e}"))?,
            "--help" | "help" => return Err(usage()),
            other => return Err(format!("unknown argument: {other}\n{}", usage())),
        }
        index += 1;
    }
    if cli.subcommand.is_empty() {
        return Err(format!("missing subcommand\n{}", usage()));
    }
    Ok(cli)
}

// ---------------------------------------------------------------------------
// Deterministic content (same spirit as the other benches: content depends
// only on (index, size), never on timing)
// ---------------------------------------------------------------------------

fn sized_text(seed: u64, chars: usize) -> String {
    // Base-52 alphabet; index-independent words of ~7 chars.
    const WORDS: [&str; 12] = [
        "alpha", "bravo", "delta", "gamma", "kilo", "lima", "mike", "november", "oscar", "romeo",
        "sierra", "tango",
    ];
    let mut text = String::with_capacity(chars + 8);
    let mut state = seed | 1;
    let mut word_index = 0usize;
    while text.len() < chars {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let pick = ((state >> 33) as usize) % WORDS.len();
        if word_index > 0 {
            text.push(' ');
        }
        text.push_str(WORDS[pick]);
        word_index += 1;
    }
    text
}

fn prompt_text(turn: usize, chars: usize) -> String {
    let prefix = format!("bench turn {turn}: ");
    let body = sized_text(0x9E37_79B9 ^ (turn as u64), chars.saturating_sub(prefix.len()));
    format!("{prefix}{body}")
}

fn reply_text(turn: usize, chars: usize) -> String {
    let prefix = format!("bench reply {turn}: ");
    let body = sized_text(0x85EB_CA6B ^ (turn as u64), chars.saturating_sub(prefix.len()));
    format!("{prefix}{body}")
}

// ---------------------------------------------------------------------------
// Event accounting (the daemon/TUI subscription load, minus rendering)
// ---------------------------------------------------------------------------

/// Compact event codes (no per-event formatting).
fn event_code(name: &str) -> u32 {
    match name {
        "message_start" => 1,
        "message_update" => 2,
        "message_end" => 3,
        "agent_start" => 4,
        "agent_end" => 5,
        "tool_start" => 6,
        "tool_end" => 7,
        "session_action" => 8,
        "other" => 9,
        _ => 0,
    }
}

#[derive(Default)]
struct EventTally {
    counts: Mutex<HashMap<u32, u64>>,
}

impl EventTally {
    fn record(&self, code: u32) {
        *self.counts.lock().unwrap().entry(code).or_insert(0) += 1;
    }
}

// ---------------------------------------------------------------------------
// Session fixture (mirrors tests/long_session_soak.rs, minus the TUI render
// and connection-layer adapters: the daemon/TUI wire cost is out of scope and
// the soak already pins it)
// ---------------------------------------------------------------------------

const BENCH_API_KEY: &str = "synthetic-session-loop-bench-key";

fn no_resource_options(cwd: &str, agent_dir: &str) -> DefaultResourceLoaderOptions {
    DefaultResourceLoaderOptions {
        cwd: cwd.to_string(),
        agent_dir: agent_dir.to_string(),
        no_extensions: true,
        no_skills: true,
        no_prompt_templates: true,
        no_themes: true,
        no_context_files: true,
        bundled_skills_dir: Some(None),
        ..Default::default()
    }
}

struct Fixture {
    _root: tempfile::TempDir,
    session: Arc<AgentSession>,
    provider: pi_ai::providers::faux::FauxProviderRegistration,
    tally: Arc<EventTally>,
    _unsubscribe: Arc<dyn Fn() + Send + Sync>,
}

async fn build_fixture(work: Option<&Path>, chunk_min: f64, chunk_max: f64) -> Fixture {
    let (root, cwd, agent_dir) = match work {
        Some(dir) => {
            let cwd = dir.join("workspace");
            let agent_dir = dir.join("agent");
            std::fs::create_dir_all(&cwd).expect("work cwd");
            std::fs::create_dir_all(&agent_dir).expect("work agent dir");
            let owned = tempfile::Builder::new()
                .prefix("session-loop-bench-owned-")
                .tempdir_in(std::env::temp_dir())
                .expect("owned temp root");
            (owned, cwd, agent_dir)
        }
        None => {
            let root = tempfile::Builder::new()
                .prefix("session-loop-bench-")
                .tempdir_in(std::env::temp_dir())
                .expect("temp root");
            let cwd = root.path().join("workspace");
            let agent_dir = root.path().join("agent");
            std::fs::create_dir_all(&cwd).expect("cwd");
            std::fs::create_dir_all(&agent_dir).expect("agent dir");
            (root, cwd, agent_dir)
        }
    };
    let cwd_string = cwd.to_string_lossy().to_string();
    let agent_dir_string = agent_dir.to_string_lossy().to_string();

    let provider = register_faux_provider(Some(RegisterFauxProviderOptions {
        provider: Some(format!("bench-{}", uuid::Uuid::new_v4())),
        tokens_per_second: Some(0.0),
        token_size: Some(pi_ai::providers::faux::FauxTokenSize {
            min: Some(chunk_min),
            max: Some(chunk_max),
        }),
        ..Default::default()
    }));
    let model = provider.get_model();

    let settings = Arc::new(Mutex::new(SettingsManager::in_memory(
        serde_json::json!({
            "autoRefine": {"enabled": false},
            "retry": {"enabled": false},
            "compaction": {"enabled": false},
            "telemetryEnabled": false,
            "agentTracesEnabled": false,
            "quietStartup": true,
        })
        .as_object()
        .unwrap()
        .clone(),
    )));

    let session_manager = Arc::new(Mutex::new(
        SessionManager::in_memory(Some(&cwd_string), Some(&agent_dir_string)).expect("session manager"),
    ));
    let auth_storage = Arc::new(tokio::sync::Mutex::new(AuthStorage::in_memory(
        AuthStorageData::new(),
        None,
    )));
    let model_registry = Arc::new(Mutex::new(ModelRegistry::in_memory(
        AuthStorage::in_memory(AuthStorageData::new(), None),
    )));
    {
        auth_storage
            .lock()
            .await
            .set_runtime_api_key(&model.provider, BENCH_API_KEY);
        model_registry
            .lock()
            .unwrap()
            .set_runtime_api_key(&model.provider, BENCH_API_KEY);
    }

    let services = create_agent_session_services(CreateAgentSessionServicesOptions {
        cwd: cwd_string.clone(),
        agent_dir: Some(agent_dir_string.clone()),
        auth_storage: Some(auth_storage),
        settings_manager: Some(settings),
        model_registry: Some(model_registry),
        extension_flag_values: None,
        no_builtin_herdr_reporter: Some(true),
        telemetry_disabled: Some(true),
        resource_loader_options: Some(no_resource_options(&cwd_string, &agent_dir_string)),
    })
    .await
    .expect("services");

    let created = create_agent_session_from_services(CreateAgentSessionFromServicesOptions {
        services: Arc::new(services),
        session_manager,
        session_start_event: None,
        creation: AgentSessionCreationOptions {
            model: Some(model.clone()),
            no_tools: Some("all".to_string()),
            prewarm_ipython_kernel: Some(false),
            telemetry_disabled: Some(true),
            ..Default::default()
        },
    })
    .await
    .expect("agent session");
    let session = created.session;

    let tally = Arc::new(EventTally::default());
    let captured = Arc::clone(&tally);
    let unsubscribe = session.subscribe(Arc::new(move |event| {
        let name: &str = match &event {
            pi_coding_agent::core::agent_session::AgentSessionEvent::Agent(agent_event) => {
                use pi_agent_core::types::AgentEvent;
                match agent_event {
                    AgentEvent::MessageStart { .. } => "message_start",
                    AgentEvent::MessageUpdate { .. } => "message_update",
                    AgentEvent::MessageEnd { .. } => "message_end",
                    AgentEvent::AgentStart { .. } => "agent_start",
                    AgentEvent::AgentEnd { .. } => "agent_end",
                    AgentEvent::ToolExecutionStart { .. } => "tool_start",
                    AgentEvent::ToolExecutionEnd { .. } => "tool_end",
                    _ => "other",
                }
            }
            pi_coding_agent::core::agent_session::AgentSessionEvent::SessionActionUpdate { .. } => {
                "session_action"
            }
            _ => "other",
        };
        captured.record(event_code(name));
    }));

    Fixture {
        _root: root,
        session,
        provider,
        tally,
        _unsubscribe: unsubscribe,
    }
}

// ---------------------------------------------------------------------------
// run subcommand
// ---------------------------------------------------------------------------

async fn run_command(cli: &Cli) -> Result<(), String> {
    let turns = cli.turns;
    let warmup = cli.warmup.min(turns);
    let measured_turns = turns - warmup;

    let fixture = build_fixture(cli.work.as_deref(), cli.chunk_min, cli.chunk_max).await;

    // Scripted deterministic replies: one per turn.
    let replies: Vec<FauxResponseStep> = (0..turns)
        .map(|turn| {
            let mut message = faux_assistant_message(
                FauxAssistantContent::Text(reply_text(turn, cli.reply_chars)),
                None,
            );
            if cli.thinking_chars > 0 {
                message.content.insert(
                    0,
                    pi_ai::types::ContentBlock::Thinking(pi_ai::types::ThinkingContent::new(
                        sized_text(0xC0FF_EE01 ^ (turn as u64), cli.thinking_chars),
                    )),
                );
            }
            FauxResponseStep::Message(message)
        })
        .collect();
    fixture.provider.set_responses(replies);

    let mut turn_timings_ms: Vec<f64> = Vec::with_capacity(measured_turns);
    let mut turn_allocs: Vec<(u64, u64)> = Vec::with_capacity(measured_turns);
    let mut snapshot_timings_ms: Vec<(usize, f64, u64, u64)> = Vec::new();
    let mut prompt_accept_ms: Vec<f64> = Vec::with_capacity(measured_turns);

    if cli.alloc {
        ALLOC_GATE.store(true, Ordering::Relaxed);
    }
    if cli.stack_sample_rate > 0 {
        STACK_SAMPLE_RATE.store(cli.stack_sample_rate, Ordering::Relaxed);
        STACK_SAMPLING.store(true, Ordering::Relaxed);
    }

    let run_started = Instant::now();
    for turn in 0..turns {
        let prompt = prompt_text(turn, cli.prompt_chars);
        let before_turn = alloc_snapshot();
        let turn_started = Instant::now();

        let _accepted = tokio::time::timeout(Duration::from_secs(120), fixture.session.prompt(&prompt, None))
            .await
            .map_err(|_| format!("turn {turn}: prompt never returned"))?
            .map_err(|error| format!("turn {turn}: prompt rejected: {error}"))?;
        prompt_accept_ms.push(turn_started.elapsed().as_secs_f64() * 1000.0);

        tokio::time::timeout(Duration::from_secs(120), fixture.session.wait_for_idle())
            .await
            .map_err(|_| format!("turn {turn}: session never reached idle"))?
            .map_err(|error| format!("turn {turn}: idle wait failed: {error}"))?;

        let turn_elapsed = turn_started.elapsed();
        if turn >= warmup {
            turn_timings_ms.push(turn_elapsed.as_secs_f64() * 1000.0);
            turn_allocs.push(alloc_delta(before_turn));
        }

        if cli.snapshot_every > 0 && turn % cli.snapshot_every == 0 {
            let before = alloc_snapshot();
            let started = Instant::now();
            let messages = fixture.session.messages();
            let elapsed = started.elapsed().as_secs_f64() * 1000.0;
            let delta = alloc_delta(before);
            snapshot_timings_ms.push((messages.len(), elapsed, delta.0, delta.1));
        }
    }
    let run_elapsed = run_started.elapsed();

    let totals = alloc_snapshot();
    STACK_SAMPLING.store(false, Ordering::Relaxed);
    ALLOC_GATE.store(false, Ordering::Relaxed);

    let messages = fixture.session.messages();
    if messages.len() != 1 + 2 * turns {
        return Err(format!(
            "transcript invariant failed: {} messages for {turns} turns (expected {})",
            messages.len(),
            1 + 2 * turns
        ));
    }

    // p50/p95 helpers (min-of-none; simple sort)
    fn percentile(values: &mut Vec<f64>, fraction: f64) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let index = ((values.len() as f64 - 1.0) * fraction).round() as usize;
        values[index]
    }

    let mut sorted_turns = turn_timings_ms.clone();
    let turn_p50 = percentile(&mut sorted_turns, 0.5);
    let turn_p95 = percentile(&mut sorted_turns, 0.95);
    let mut sorted_accept = prompt_accept_ms[warmup.min(prompt_accept_ms.len())..].to_vec();
    let accept_p50 = percentile(&mut sorted_accept, 0.5);

    let mut event_counts: HashMap<String, u64> = HashMap::new();
    for (code, count) in fixture.tally.counts.lock().unwrap().iter() {
        let name = match code {
            1 => "message_start",
            2 => "message_update",
            3 => "message_end",
            4 => "agent_start",
            5 => "agent_end",
            6 => "tool_start",
            7 => "tool_end",
            8 => "session_action",
            9 => "other",
            _ => "unknown",
        };
        event_counts.insert(name.to_string(), *count);
    }

    let total_allocs_measured: u64 = turn_allocs.iter().map(|(count, _)| count).sum();
    let total_bytes_measured: u64 = turn_allocs.iter().map(|(_, bytes)| bytes).sum();

    let report = serde_json::json!({
        "kind": "session_loop_bench/run",
        "config": {
            "turns": turns,
            "warmup": warmup,
            "measured_turns": measured_turns,
            "prompt_chars": cli.prompt_chars,
            "reply_chars": cli.reply_chars,
            "thinking_chars": cli.thinking_chars,
            "chunk_min": cli.chunk_min,
            "chunk_max": cli.chunk_max,
            "alloc": cli.alloc,
            "stack_sample_rate": cli.stack_sample_rate,
            "snapshot_every": cli.snapshot_every,
            "workdir": cli.work,
        },
        "run_wall_ms": run_elapsed.as_secs_f64() * 1000.0,
        "turn_ms": {
            "p50": turn_p50,
            "p95": turn_p95,
            "mean": turn_timings_ms.iter().sum::<f64>() / measured_turns.max(1) as f64,
        },
        "prompt_accept_ms": {"p50": accept_p50},
        "turn_allocs_mean": {
            "count": total_allocs_measured as f64 / measured_turns.max(1) as f64,
            "bytes": total_bytes_measured as f64 / measured_turns.max(1) as f64,
        },
        "alloc_totals_while_gated": {"count": totals.0, "bytes": totals.1},
        "events": event_counts,
        "snapshots": snapshot_timings_ms,
        "final_message_count": messages.len(),
    });

    if let Some(parent) = cli.out.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    std::fs::write(&cli.out, serde_json::to_string_pretty(&report).unwrap())
        .map_err(|e| e.to_string())?;

    // Stack samples go next to the report (they are large).
    if cli.stack_sample_rate > 0 {
        let samples = drain_stack_samples();
        let mut path = cli.out.clone().into_os_string();
        path.push(".stacks");
        let mut text = String::new();
        for sample in &samples {
            text.push_str("==== SAMPLE ====\n");
            text.push_str(sample);
            text.push('\n');
        }
        std::fs::write(PathBuf::from(path), text).map_err(|e| e.to_string())?;
        println!("stack samples: {}", samples.len());
    }

    println!(
        "turn p50 {:.3} ms, p95 {:.3} ms, mean allocs/turn {:.0}, mean bytes/turn {:.0}",
        turn_p50,
        turn_p95,
        total_allocs_measured as f64 / measured_turns.max(1) as f64,
        total_bytes_measured as f64 / measured_turns.max(1) as f64,
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// startup subcommand
// ---------------------------------------------------------------------------

async fn startup_command(cli: &Cli) -> Result<(), String> {
    let mut services_ms: Vec<f64> = Vec::new();
    let mut session_ms: Vec<f64> = Vec::new();
    let mut services_allocs: Vec<(u64, u64)> = Vec::new();
    let mut session_allocs: Vec<(u64, u64)> = Vec::new();

    // One shared provider (models are per-registration); each run gets a fresh
    // session against the same faux api.
    let provider = register_faux_provider(Some(RegisterFauxProviderOptions {
        provider: Some(format!("bench-{}", uuid::Uuid::new_v4())),
        tokens_per_second: Some(0.0),
        ..Default::default()
    }));
    let model = provider.get_model();

    let root = tempfile::Builder::new()
        .prefix("session-loop-bench-startup-")
        .tempdir_in(std::env::temp_dir())
        .expect("temp root");
    let cwd = root.path().join("workspace");
    let agent_dir = root.path().join("agent");
    std::fs::create_dir_all(&cwd).expect("cwd");
    std::fs::create_dir_all(&agent_dir).expect("agent dir");
    let cwd_string = cwd.to_string_lossy().to_string();
    let agent_dir_string = agent_dir.to_string_lossy().to_string();

    ALLOC_GATE.store(true, Ordering::Relaxed);
    for run in 0..cli.runs {
        let settings = Arc::new(Mutex::new(SettingsManager::in_memory(
            serde_json::json!({
                "autoRefine": {"enabled": false},
                "retry": {"enabled": false},
                "compaction": {"enabled": false},
                "telemetryEnabled": false,
                "agentTracesEnabled": false,
                "quietStartup": true,
            })
            .as_object()
            .unwrap()
            .clone(),
        )));
        let auth_storage = Arc::new(tokio::sync::Mutex::new(AuthStorage::in_memory(
            AuthStorageData::new(),
            None,
        )));
        let model_registry = Arc::new(Mutex::new(ModelRegistry::in_memory(
            AuthStorage::in_memory(AuthStorageData::new(), None),
        )));
        {
            auth_storage
                .lock()
                .await
                .set_runtime_api_key(&model.provider, BENCH_API_KEY);
            model_registry
                .lock()
                .unwrap()
                .set_runtime_api_key(&model.provider, BENCH_API_KEY);
        }

        let before = alloc_snapshot();
        let started = Instant::now();
        let services = create_agent_session_services(CreateAgentSessionServicesOptions {
            cwd: cwd_string.clone(),
            agent_dir: Some(agent_dir_string.clone()),
            auth_storage: Some(auth_storage),
            settings_manager: Some(settings),
            model_registry: Some(model_registry),
            extension_flag_values: None,
            no_builtin_herdr_reporter: Some(true),
            telemetry_disabled: Some(true),
            resource_loader_options: Some(no_resource_options(&cwd_string, &agent_dir_string)),
        })
        .await
        .map_err(|e| format!("services: {e}"))?;
        services_ms.push(started.elapsed().as_secs_f64() * 1000.0);
        services_allocs.push(alloc_delta(before));

        let session_manager = Arc::new(Mutex::new(
            SessionManager::in_memory(Some(&cwd_string), Some(&agent_dir_string))
                .map_err(|e| format!("session manager: {e}"))?,
        ));

        let before = alloc_snapshot();
        let started = Instant::now();
        let created = create_agent_session_from_services(CreateAgentSessionFromServicesOptions {
            services: Arc::new(services),
            session_manager,
            session_start_event: None,
            creation: AgentSessionCreationOptions {
                model: Some(model.clone()),
                no_tools: Some("all".to_string()),
                prewarm_ipython_kernel: Some(false),
                telemetry_disabled: Some(true),
                ..Default::default()
            },
        })
        .await
        .map_err(|e| format!("session: {e}"))?;
        session_ms.push(started.elapsed().as_secs_f64() * 1000.0);
        session_allocs.push(alloc_delta(before));
        drop(created.session);
        if run + 1 < cli.runs {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }
    ALLOC_GATE.store(false, Ordering::Relaxed);

    fn stats(values: &[f64]) -> Value {
        if values.is_empty() {
            return serde_json::json!({});
        }
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        serde_json::json!({
            "p50": sorted[(sorted.len() as f64 - 1.0).round() as usize / 2],
            "mean": values.iter().sum::<f64>() / values.len() as f64,
            "min": sorted[0],
            "max": sorted[sorted.len() - 1],
        })
    }

    let report = serde_json::json!({
        "kind": "session_loop_bench/startup",
        "runs": cli.runs,
        "services_ms": stats(&services_ms),
        "session_ms": stats(&session_ms),
        "services_allocs_mean": {
            "count": services_allocs.iter().map(|(c, _)| c).sum::<u64>() as f64 / cli.runs as f64,
            "bytes": services_allocs.iter().map(|(_, b)| b).sum::<u64>() as f64 / cli.runs as f64,
        },
        "session_allocs_mean": {
            "count": session_allocs.iter().map(|(c, _)| c).sum::<u64>() as f64 / cli.runs as f64,
            "bytes": session_allocs.iter().map(|(_, b)| b).sum::<u64>() as f64 / cli.runs as f64,
        },
    });
    if let Some(parent) = cli.out.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    std::fs::write(&cli.out, serde_json::to_string_pretty(&report).unwrap())
        .map_err(|e| e.to_string())?;
    println!("startup report written");
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cli = match parse_cli(&args) {
        Ok(cli) => cli,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(2);
        }
    };
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .expect("tokio runtime");
    let code = runtime.block_on(async {
        match cli.subcommand.as_str() {
            "run" => run_command(&cli).await,
            "startup" => startup_command(&cli).await,
            other => Err(format!("unknown subcommand: {other}")),
        }
    });
    match code {
        Ok(()) => std::process::exit(0),
        Err(error) => {
            eprintln!("error: {error}");
            std::process::exit(1);
        }
    }
}
