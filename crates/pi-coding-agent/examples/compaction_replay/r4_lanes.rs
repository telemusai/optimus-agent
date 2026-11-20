//! R4 lane-separated benchmark plumbing. This module is example-only.

use super::*;
use std::sync::OnceLock;
use pi_coding_agent::core::compaction::compaction::{
    compact, default_compaction_settings, CompactionResult, ProviderRetryPolicy,
    SummaryCallRunner,
};

const SCHEMA: &str = "compaction-r4-lane-v1";
const WALL_FEATURE: &str = "compaction-r4-uninstrumented-wall";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lane {
    Wall,
    Allocation,
    ReplayPhase,
    MinimalProvider,
}

impl Lane {
    fn parse(raw: &str) -> Result<Self, String> {
        match raw {
            "wall" => Ok(Self::Wall),
            "allocation" => Ok(Self::Allocation),
            "replay-phase" => Ok(Self::ReplayPhase),
            "minimal-provider" => Ok(Self::MinimalProvider),
            _ => Err(stage_error(
                "r4-lane",
                "lane must be wall, allocation, replay-phase or minimal-provider",
            )),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Wall => "wall",
            Self::Allocation => "allocation",
            Self::ReplayPhase => "replay-phase",
            Self::MinimalProvider => "minimal-provider",
        }
    }

    fn provider_reporting(self) -> bool {
        matches!(self, Self::ReplayPhase | Self::MinimalProvider)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Injection {
    None,
    Failure,
    Cancelled,
}

impl Injection {
    fn parse(raw: Option<String>) -> Result<Self, String> {
        match raw.as_deref() {
            None | Some("none") => Ok(Self::None),
            Some("failure") => Ok(Self::Failure),
            Some("cancelled") => Ok(Self::Cancelled),
            _ => Err(stage_error(
                "r4-injection",
                "inject must be none, failure or cancelled",
            )),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Failure => "failure",
            Self::Cancelled => "cancelled",
        }
    }
}

struct Config {
    lane: Lane,
    fixture: PathBuf,
    work: PathBuf,
    out: PathBuf,
    wait_ms: u64,
    generation_units: usize,
    injection: Injection,
    #[cfg(not(feature = "compaction-r4-uninstrumented-wall"))]
    allocation_available: bool,
}

fn reject_unknown_options(cli: &Cli, allowed: &[&str]) -> Result<(), String> {
    let mut keys: Vec<&str> = cli.options.keys().map(String::as_str).collect();
    keys.sort_unstable();
    for key in keys {
        if !allowed.contains(&key) {
            return Err(stage_error(
                "r4-cli-options",
                format!("option --{key} is not valid for this lane"),
            ));
        }
    }
    Ok(())
}

fn config(cli: &Cli) -> Result<Config, String> {
    reject_unknown_options(
        cli,
        &[
            "lane",
            "fixture",
            "work",
            "out",
            "mock-wait-ms",
            "generation-units",
            "inject",
            "allocation-source",
        ],
    )?;
    let lane = Lane::parse(&cli.required("lane")?)?;
    let wall_build = cfg!(feature = "compaction-r4-uninstrumented-wall");
    if wall_build != (lane == Lane::Wall) {
        return Err(stage_error(
            "r4-build-lane-identity",
            format!(
                "lane {} requires build configuration {}",
                lane.as_str(),
                if lane == Lane::Wall {
                    "wall-uninstrumented"
                } else {
                    "instrumented"
                }
            ),
        ));
    }
    let wait_ms = cli.integer("mock-wait-ms", 0, 0, 60_000)? as u64;
    let generation_units = cli.integer("generation-units", 1, 1, 1_000_000)?;
    if !lane.provider_reporting()
        && (cli.options.contains_key("mock-wait-ms")
            || cli.options.contains_key("generation-units")
            || cli.options.contains_key("inject"))
    {
        return Err(stage_error(
            "r4-lane-exclusivity",
            "provider controls are valid only in replay-phase or minimal-provider",
        ));
    }
    #[cfg(not(feature = "compaction-r4-uninstrumented-wall"))]
    let allocation_available = match cli
        .optional("allocation-source")
        .as_deref()
    {
        None | Some("counting-global") if lane == Lane::Allocation => true,
        Some("unavailable") if lane == Lane::Allocation => false,
        None if lane != Lane::Allocation => false,
        _ => {
            return Err(stage_error(
                "r4-allocation-source",
                "allocation-source belongs to allocation and is counting-global or unavailable",
            ))
        }
    };
    Ok(Config {
        lane,
        fixture: PathBuf::from(cli.required("fixture")?),
        work: PathBuf::from(cli.required("work")?),
        out: PathBuf::from(cli.required("out")?),
        wait_ms,
        generation_units,
        injection: Injection::parse(cli.optional("inject"))?,
        #[cfg(not(feature = "compaction-r4-uninstrumented-wall"))]
        allocation_available,
    })
}

#[derive(Default)]
struct PhaseMarks {
    origin: OnceLock<Instant>,
    marks: Mutex<HashMap<String, f64>>,
}

impl PhaseMarks {
    fn new() -> Self {
        let marks = Self::default();
        let _ = marks.origin.set(Instant::now());
        marks
    }

    fn mark(&self, name: &str) {
        let elapsed = self
            .origin
            .get()
            .expect("phase origin")
            .elapsed()
            .as_secs_f64()
            * 1000.0;
        let previous = self
            .marks
            .lock()
            .unwrap()
            .insert(name.to_string(), elapsed);
        assert!(previous.is_none(), "duplicate phase mark {name}");
    }

    fn at(&self, name: &str) -> Result<f64, String> {
        self.marks
            .lock()
            .unwrap()
            .get(name)
            .copied()
            .ok_or_else(|| stage_error("r4-phase-boundary", format!("missing mark {name}")))
    }

    fn phase(&self, name: &str, start: &str, end: &str, boundary: &str) -> Result<Value, String> {
        let start_ms = self.at(start)?;
        let end_ms = self.at(end)?;
        let duration_ms = end_ms - start_ms;
        ensure_stage(
            duration_ms.is_finite() && duration_ms >= 0.0,
            "r4-phase-duration",
            format!("invalid measured phase {name}"),
        )?;
        Ok(json!({
            "name": name,
            "start_mark": start,
            "end_mark": end,
            "duration_ms": duration_ms,
            "clock": "std::time::Instant",
            "boundary": boundary,
            "overhead": "two monotonic timestamp reads plus mutex-protected mark insertion; unmeasured recorder overhead",
            "cpu_scope": "local-only; interval contains no mock wait or mock generation interval"
        }))
    }
}

fn optional_mark(marks: Option<&Arc<PhaseMarks>>, name: &str) {
    if let Some(marks) = marks {
        marks.mark(name);
    }
}

struct MockControl {
    wait_ms: u64,
    generation_units: usize,
    injection: Injection,
    calls: AtomicU64,
    marks: Option<Arc<PhaseMarks>>,
}

fn mock_message(model: &Model, failure: bool) -> AssistantMessage {
    let mut message = if failure {
        let mut message = text_message("", "error");
        message.error_message = Some("r4 synthetic provider failure".to_string());
        message
    } else {
        text_message(HARNESS_SUMMARY, "stop")
    };
    message.api = model.api.clone();
    message.provider = model.provider.clone();
    message.model = model.id.clone();
    message.usage = Usage {
        input: 1.0,
        output: 1.0,
        total_tokens: 2.0,
        ..Usage::zero()
    };
    message
}

fn register_mock_provider(control: Arc<MockControl>, model: &Model) {
    let stream_control = Arc::clone(&control);
    let stream_simple: SimpleStreamFunction = Arc::new(
        move |model: &Model, _context: &Context, _options: Option<&SimpleStreamOptions>| {
            let first_call = stream_control.calls.fetch_add(1, Ordering::SeqCst) == 0;
            if first_call {
                optional_mark(stream_control.marks.as_ref(), "mock_stream_entry");
            }
            // This loop makes generated work an explicit controlled input. It is
            // not converted into a duration or a provider-throughput estimate.
            let mut digest = 0usize;
            for index in 0..stream_control.generation_units {
                digest = digest.wrapping_add(
                    HARNESS_SUMMARY
                        .as_bytes()
                        .iter()
                        .enumerate()
                        .map(|(offset, byte)| (*byte as usize).wrapping_add(offset + index))
                        .sum::<usize>(),
                );
            }
            std::hint::black_box(digest);
            let reply = mock_message(model, stream_control.injection == Injection::Failure);
            let stream = create_assistant_message_event_stream();
            if first_call {
                optional_mark(stream_control.marks.as_ref(), "mock_generation_complete");
            }
            if stream_control.wait_ms == 0 {
                if first_call {
                    optional_mark(stream_control.marks.as_ref(), "mock_response_emit_entry");
                }
                for event in events_for(reply) {
                    stream.push(event);
                }
                stream.end(None);
                if first_call {
                    optional_mark(stream_control.marks.as_ref(), "mock_response_emit_exit");
                }
            } else {
                let wait_ms = stream_control.wait_ms;
                let task_stream = stream.clone();
                let response_marks = stream_control.marks.clone();
                tokio::spawn(async move {
                    tokio::time::sleep(Duration::from_millis(wait_ms)).await;
                    if first_call {
                        optional_mark(response_marks.as_ref(), "mock_response_emit_entry");
                    }
                    for event in events_for(reply) {
                        task_stream.push(event);
                    }
                    task_stream.end(None);
                    if first_call {
                        optional_mark(response_marks.as_ref(), "mock_response_emit_exit");
                    }
                });
            }
            if first_call {
                optional_mark(stream_control.marks.as_ref(), "mock_stream_return");
            }
            stream
        },
    );
    register_api_provider_simple(
        ApiProviderSimple {
            api: model.api.clone(),
            stream: Arc::new(|_, _, _| panic!("R4 uses only stream_simple")),
            stream_simple,
            compact: None,
            supports_compaction: Some(Arc::new(|_| false)),
        },
        Some("compaction-r4-mock".to_string()),
    );
}

struct Seeded {
    manager: SessionManager,
    messages: Vec<AgentMessage>,
    entries: Vec<CompactionSessionEntry>,
}

fn seed(config: &Config, fixture: &LoadedFixture, model: &Model) -> Result<Seeded, String> {
    let root = config.work.join(format!(
        "r4-{}-{}",
        config.lane.as_str(),
        std::process::id()
    ));
    let cwd = root.join("workspace");
    let sessions = root.join("sessions");
    std::fs::create_dir_all(&cwd).map_err(|error| stage_error("r4-seed", error.to_string()))?;
    std::fs::create_dir_all(&sessions)
        .map_err(|error| stage_error("r4-seed", error.to_string()))?;
    export_case_env(&root);
    let mut manager = SessionManager::create(
        &cwd.to_string_lossy(),
        Some(&sessions.to_string_lossy()),
    )?;
    let messages = fixture.to_agent_messages(model);
    for message in &messages {
        manager.append_message(message.clone())?;
    }
    manager.flush_now()?;
    let branch = manager.get_branch(None);
    let mut entries = Vec::with_capacity(branch.len());
    for entry in &branch {
        ensure_stage(
            entry.get("type").and_then(Value::as_str) == Some("message"),
            "r4-seed-entry",
            "seed unexpectedly contains a non-message entry",
        )?;
        let message = serde_json::from_value::<AgentMessage>(
            entry.get("message").cloned().unwrap_or(Value::Null),
        )
        .map_err(|error| stage_error("r4-seed-message", error.to_string()))?;
        entries.push(CompactionSessionEntry::Message {
            id: entry
                .get("id")
                .and_then(Value::as_str)
                .ok_or_else(|| stage_error("r4-seed-entry", "message id missing"))?
                .to_string(),
            parent_id: entry
                .get("parentId")
                .and_then(Value::as_str)
                .map(str::to_string),
            message,
        });
    }
    Ok(Seeded {
        manager,
        messages,
        entries,
    })
}

fn compact_with_mock(
    preparation: &pi_coding_agent::core::compaction::compaction::CompactionPreparation,
    model: &Model,
    config: &Config,
    marks: Option<Arc<PhaseMarks>>,
) -> pi_ai::types::BoxFuture<Result<CompactionResult, String>> {
    let preparation = preparation.clone();
    let model = model.clone();
    let injection = config.injection;
    Box::pin(async move {
        optional_mark(marks.as_ref(), "compact_entry");
        let runner_marks = marks.clone();
        let summary_call: SummaryCallRunner = Arc::new(move |call| {
            let runner_marks = runner_marks.clone();
            Box::pin(async move {
                optional_mark(runner_marks.as_ref(), "summary_runner_entry");
                let result = call(None).await;
                optional_mark(runner_marks.as_ref(), "summary_runner_return");
                result
            })
        });
        let token = tokio_util::sync::CancellationToken::new();
        if injection == Injection::Cancelled {
            token.cancel();
        }
        let one_attempt = ProviderRetryPolicy {
            enabled: false,
            max_retries: 0,
            base_delay_ms: 0.0,
            max_retry_delay_ms: 0.0,
        };
        let result = compact(
            &preparation,
            &model,
            HARNESS_API_KEY,
            None,
            Some(&token),
            None,
            summary_call,
            Some(&one_attempt),
            None,
        )
        .await;
        optional_mark(marks.as_ref(), "compact_exit");
        result
    })
}

fn continuation(
    manager: &mut SessionManager,
    result: &CompactionResult,
    marks: Option<&Arc<PhaseMarks>>,
) -> Result<Value, String> {
    optional_mark(marks, "append_entry");
    manager.append_compaction(
        &result.summary,
        &result.first_kept_entry_id,
        result.tokens_before,
        None,
        None,
        None,
        result.usage.as_ref(),
        None,
    )?;
    manager.flush_now()?;
    optional_mark(marks, "append_exit");
    optional_mark(marks, "continuation_entry");
    let context = manager.build_session_context(None);
    optional_mark(marks, "continuation_exit");
    let normalized = normalize_messages(&context.messages);
    Ok(json!({
        "status": "available",
        "boundary": "real append_compaction followed by build_session_context(None)",
        "message_count": context.messages.len(),
        "context_sha256": semantic_sha256(&normalized)?,
        "compaction_entries": 1
    }))
}

fn local_phases(marks: &PhaseMarks, lane: Lane, injection: Injection) -> Result<Vec<Value>, String> {
    if injection == Injection::Cancelled {
        return Ok(Vec::new());
    }
    let mut phases = vec![
        marks.phase(
            "prepare_compaction",
            "prepare_entry",
            "prepare_exit",
            "around the real prepare_compaction call; excludes seed and provider",
        )?,
        marks.phase(
            "compact_pre_provider",
            "compact_entry",
            "summary_runner_entry",
            "compact entry to summary runner entry; before mock dispatch/wait/generation",
        )?,
    ];
    if lane == Lane::MinimalProvider {
        phases.extend([
            marks.phase(
                "convert_to_llm",
                "convert_entry",
                "convert_exit",
                "around the real convert_to_llm request transformation",
            )?,
            marks.phase(
                "serialize_conversation",
                "serialize_entry",
                "serialize_exit",
                "around serde_json request-envelope serialization of the real serialized conversation",
            )?,
            marks.phase(
                "summary_dispatch_preparation",
                "summary_runner_entry",
                "mock_stream_entry",
                "summary runner entry to in-process mock stream entry; no provider wait/generation",
            )?,
            marks.phase(
                "response_parse",
                "mock_response_emit_exit",
                "summary_runner_return",
                "terminal mock event emission complete to summary runner return; local stream consumption/parsing after wait and generation",
            )?,
        ]);
    }
    phases.push(marks.phase(
        "compact_post_provider",
        "summary_runner_return",
        "compact_exit",
        "summary call return to compact exit; excludes mock stream interval",
    )?);
    if injection == Injection::None {
        phases.extend([
            marks.phase(
                "append_compaction",
                "append_entry",
                "append_exit",
                "real durable append_compaction and flush; local persistence only",
            )?,
            marks.phase(
                "build_continuation",
                "continuation_entry",
                "continuation_exit",
                "real build_session_context(None) after append; useful continuation boundary",
            )?,
        ]);
    }
    Ok(phases)
}

fn base_report(
    config: &Config,
    fixture: &LoadedFixture,
    exe_sha256: &str,
    transcript_sha256: &str,
) -> Value {
    let wall = config.lane == Lane::Wall;
    json!({
        "schema": SCHEMA,
        "lane": config.lane.as_str(),
        "build": {
            "configuration": if wall { "wall-uninstrumented" } else { "instrumented" },
            "feature": if wall { json!(WALL_FEATURE) } else { Value::Null },
            "profile": if cfg!(debug_assertions) { "debug" } else { "release" },
            "instrumentation": if wall { json!([]) } else { json!(["phase-marks", "counting-global-allocator"]) }
        },
        "run": {
            "pid": std::process::id(),
            "exe_sha256": exe_sha256,
            "process_state": "fresh-process"
        },
        "fixture": {
            "name": fixture.name,
            "record_count": fixture.records.len(),
            "transcript_sha256": transcript_sha256
        },
        "provider_network": "none; registered in-process deterministic mock only",
        "synthetic_durations": "non-decision"
    })
}

async fn execute(config: Config) -> Result<i32, String> {
    let fixture = load_fixture(&config.fixture)?;
    let transcript_sha256 = sha256_hex(
        &std::fs::read(config.fixture.join("transcript.json"))
            .map_err(|error| stage_error("r4-fixture-read", error.to_string()))?,
    );
    let exe_sha256 = exe_sha256()?;
    let marks = if config.lane == Lane::Wall {
        None
    } else {
        Some(Arc::new(PhaseMarks::new()))
    };
    let mut model = fixture_model_for_estimates();
    model.api = format!("r4-api-{}", config.lane.as_str());
    model.id = "r4-mock-model".to_string();
    model.provider = format!("r4-provider-{}", config.lane.as_str());
    model.base_url = "https://r4.fixture.invalid".to_string();
    model.context_window = 1_000_000.0;
    model.max_tokens = 64_000.0;
    let control = Arc::new(MockControl {
        wait_ms: config.wait_ms,
        generation_units: config.generation_units,
        injection: config.injection,
        calls: AtomicU64::new(0),
        marks: marks.clone(),
    });
    register_mock_provider(Arc::clone(&control), &model);
    let mut seeded = seed(&config, &fixture, &model)?;
    let mut settings = default_compaction_settings();
    settings.enabled = true;
    settings.keep_recent_tokens = 1_000.0;
    settings.reserve_tokens = 4_000.0;
    settings.summary_update_policy = Some(SUMMARY_UPDATE_POLICY_OFF.to_string());
    optional_mark(marks.as_ref(), "prepare_entry");
    let preparation = prepare_compaction(&seeded.entries, &settings, &|_entries| {
        estimate_context_tokens(&seeded.messages).tokens
    })
    .ok_or_else(|| stage_error("r4-prepare", "real prepare_compaction returned None"))?;
    optional_mark(marks.as_ref(), "prepare_exit");

    if config.lane == Lane::MinimalProvider {
        optional_mark(marks.as_ref(), "convert_entry");
        let llm = convert_to_llm(
            &preparation.messages_to_summarize,
            &ModelToolOutputPolicyOptions::default(),
        );
        optional_mark(marks.as_ref(), "convert_exit");
        optional_mark(marks.as_ref(), "serialize_entry");
        let serialized = serialize_conversation(&llm);
        let envelope = serde_json::to_vec(&json!({"messages": serialized}))
            .map_err(|error| stage_error("r4-serialize", error.to_string()))?;
        std::hint::black_box(envelope);
        optional_mark(marks.as_ref(), "serialize_exit");
    }

    #[cfg(not(feature = "compaction-r4-uninstrumented-wall"))]
    let alloc_before = if config.lane == Lane::Allocation && config.allocation_available {
        Some(alloc_snapshot())
    } else {
        None
    };

    let result = compact_with_mock(
        &preparation,
        &model,
        &config,
        marks.clone(),
    )
    .await;

    #[cfg(not(feature = "compaction-r4-uninstrumented-wall"))]
    let allocation = if config.lane == Lane::Allocation {
        match (config.allocation_available, alloc_before) {
            (true, Some(before)) => {
                let after = alloc_snapshot();
                json!({
                    "status": "available",
                    "source": "benchmark CountingAllocator GlobalAlloc counters",
                    "scope": "whole R4 lane operation from post-preparation to compact completion",
                    "count": after.0.saturating_sub(before.0),
                    "bytes_requested": after.1.saturating_sub(before.1),
                    "note": "real allocation events and requested sizes, not retained bytes or an estimate"
                })
            }
            _ => json!({
                "status": "unavailable",
                "source": Value::Null,
                "count": Value::Null,
                "bytes_requested": Value::Null,
                "reason": "allocation counters explicitly unavailable in this qualification run; zero and estimates are prohibited"
            }),
        }
    } else {
        Value::Null
    };

    let observed_provider_calls = control.calls.load(Ordering::SeqCst);
    let expected_provider_calls = if config.injection == Injection::Cancelled { 0 } else { 1 };
    ensure_stage(
        observed_provider_calls == expected_provider_calls,
        "r4-provider-call-accounting",
        format!(
            "minimal single-call fixture expected {expected_provider_calls} mock calls, observed {observed_provider_calls}; use a non-splitting fixture and one summary attempt"
        ),
    )?;
    let mut report = base_report(&config, &fixture, &exe_sha256, &transcript_sha256);
    match result {
        Ok(result) => {
            report["outcome"] = json!({"status": "success", "error": Value::Null});
            let continuation = continuation(&mut seeded.manager, &result, marks.as_ref())?;
            report["continuation"] = continuation;
        }
        Err(error)
            if (config.injection == Injection::Failure
                && error.contains("r4 synthetic provider failure"))
                || (config.injection == Injection::Cancelled && error.contains("Aborted")) =>
        {
            let status = if config.injection == Injection::Cancelled {
                "cancelled"
            } else {
                "failure"
            };
            report["outcome"] = json!({"status": status, "error": error});
            report["continuation"] = json!({
                "status": "unavailable",
                "message_count": Value::Null,
                "context_sha256": Value::Null,
                "reason": "compaction did not complete; no continuation was committed"
            });
        }
        Err(error) => return Err(stage_error("r4-compact", error)),
    }

    if config.lane.provider_reporting() {
        let phases = local_phases(
            marks
                .as_ref()
                .ok_or_else(|| stage_error("r4-phase-config", "instrumented marks missing"))?,
            config.lane,
            config.injection,
        )?;
        report["local_replay"] = json!({
            "schema": "r4-local-phase-v1",
            "phases": phases,
            "provider_time_policy": "no provider wait or generation value is subtracted, estimated or included in local phase arithmetic"
        });
        report["provider"] = json!({
            "schema": "r4-minimal-provider-v1",
            "kind": "deterministic-in-process-mock",
            "wait": {
                "requested_ms": config.wait_ms,
                "source": "controlled CLI input to tokio::time::sleep",
                "observed_duration_ms": Value::Null,
                "unavailable_reason": "only the controlled requested wait is admitted; no provider latency estimate"
            },
            "generation": {
                "summary_render_iterations": config.generation_units,
                "source": "controlled CLI input",
                "output_text": HARNESS_SUMMARY,
                "generated_units_are_not_duration": true
            },
            "call_count": control.calls.load(Ordering::SeqCst),
            "injection": config.injection.as_str()
        });
    }
    #[cfg(not(feature = "compaction-r4-uninstrumented-wall"))]
    if config.lane == Lane::Allocation {
        report["allocation"] = allocation;
    }
    write_new_json(&config.out, &report)?;
    Ok(0)
}

pub async fn run(cli: &Cli) -> Result<i32, String> {
    let config = config(cli)?;
    execute(config).await
}
