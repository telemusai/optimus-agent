//! Standalone timing harness for client-attributable token-delivery latency
//! through the REAL `openai-completions` provider code path.
//!
//! Drives `stream_openai_completions` against a deterministic local SSE fixture
//! server (loopback, precise pacing control) and records per-run timestamps:
//! request send start, response headers, every SSE payload at the network
//! reader (the `OnStreamObservation` seam), every consumer-visible TextDelta,
//! and stream end. Separates connection setup (the current production path
//! builds `reqwest::Client` per request, so every run includes a full connect)
//! from stream consumption (one connection reused within one stream).
//!
//! Subcommands (hand-rolled CLI; std + existing crate APIs only):
//!   bench --out <json> [--events N] [--delay-ms X] [--payload-bytes B]
//!       [--mode burst|segment] [--conn cold|warm] [--runs R] [--warmup W]
//!       [--alloc-runs A] [--control-runs C] [--port P]
//!   matrix --out <json> [--events N] [--runs R] [--warmup W]
//!       [--alloc-runs A] [--control-runs C]
//!
//! Determinism: token payloads depend only on (event index, payload size), so
//! timing varies but the concatenated token sequence must not; every run's
//! sequence digest is asserted against the expected digest.
//!
//! Everything here is measurement tooling: no production file depends on it and
//! it must keep working unchanged BEFORE and AFTER optimization work.

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::HashMap;
use std::net::Ipv4Addr;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use chrono::Utc;
use pi_ai::providers::openai_completions::{stream_openai_completions, OpenAICompletionsOptions};
use pi_ai::types::{
	AssistantMessage, AssistantMessageEvent, Context, InputModality, Message, Model, ModelCost,
	OnStreamObservation, StreamOptions, UserContent, UserMessage,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

// ---------------------------------------------------------------------------
// Counting global allocator (std-only; gated so timing runs stay clean)
// ---------------------------------------------------------------------------

static ALLOC_GATE: AtomicBool = AtomicBool::new(false);
static ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);

struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
	unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
		if ALLOC_GATE.load(Ordering::Relaxed) {
			ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
			ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
		}
		System.alloc(layout)
	}
	unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
		System.dealloc(ptr, layout)
	}
	unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
		if ALLOC_GATE.load(Ordering::Relaxed) {
			ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
			ALLOC_BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
		}
		System.realloc(ptr, layout, new_size)
	}
	unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
		if ALLOC_GATE.load(Ordering::Relaxed) {
			ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
			ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
		}
		System.alloc_zeroed(layout)
	}
}

#[global_allocator]
static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

fn alloc_snapshot() -> (u64, u64) {
	(ALLOC_COUNT.load(Ordering::Relaxed), ALLOC_BYTES.load(Ordering::Relaxed))
}

// ---------------------------------------------------------------------------
// Monotonic epoch shared by client and fixture-server timestamps
// ---------------------------------------------------------------------------

static EPOCH: OnceLock<Instant> = OnceLock::new();

fn now_ns() -> u64 {
	EPOCH.get_or_init(Instant::now).elapsed().as_nanos() as u64
}

/// Compact codes for the OnStreamObservation phase names (no per-event alloc).
fn phase_code(phase: &str) -> u8 {
	match phase {
		"generation_serialize_start" => 1,
		"generation_serialize_returned" => 2,
		"generation_send_start" => 3,
		"generation_headers_complete" => 4,
		"raw_event" => 5,
		"text" => 6,
		"thinking" => 7,
		"tool" => 8,
		"terminal" => 9,
		_ => 0,
	}
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

fn sha256_hex(bytes: &[u8]) -> String {
	let digest = Sha256::digest(bytes);
	digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn stats_f64_ms(samples: &[f64]) -> Value {
	if samples.is_empty() {
		return json!({"n": 0});
	}
	let mut sorted = samples.to_vec();
	sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
	let count = sorted.len();
	let mean = sorted.iter().sum::<f64>() / count as f64;
	let variance = sorted.iter().map(|value| (value - mean) * (value - mean)).sum::<f64>() / count as f64;
	let percentile = |fraction: f64| sorted[((fraction * (count - 1) as f64).round() as usize).min(count - 1)];
	json!({
		"n": count,
		"mean_ms": mean,
		"p50_ms": percentile(0.5),
		"p95_ms": percentile(0.95),
		"min_ms": sorted[0],
		"max_ms": sorted[count - 1],
		"stddev_ms": variance.sqrt(),
	})
}

fn stats_ms(samples_ns: &[u64]) -> Value {
	let samples: Vec<f64> = samples_ns.iter().map(|ns| *ns as f64 / 1e6).collect();
	stats_f64_ms(&samples)
}

fn write_json(path: &Path, value: &Value) -> Result<(), String> {
	if let Some(parent) = path.parent() {
		std::fs::create_dir_all(parent).map_err(|error| format!("mkdir: {error}"))?;
	}
	let body = serde_json::to_string_pretty(value).map_err(|error| error.to_string())?;
	std::fs::write(path, body).map_err(|error| format!("write {}: {error}", path.display()))?;
	Ok(())
}

fn exe_sha256() -> Result<String, String> {
	let exe = std::env::current_exe().map_err(|error| error.to_string())?;
	let bytes = std::fs::read(&exe).map_err(|error| format!("read exe: {error}"))?;
	Ok(sha256_hex(&bytes))
}

fn git_info() -> Value {
	let run = |args: &[&str]| -> Option<String> {
		let output = std::process::Command::new("git").args(args).output().ok()?;
		if !output.status.success() {
			return None;
		}
		Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
	};
	json!({
		"commit": run(&["rev-parse", "HEAD"]),
		"describe": run(&["describe", "--tags", "--always", "--dirty"]),
		"branch": run(&["rev-parse", "--abbrev-ref", "HEAD"]),
	})
}

// ---------------------------------------------------------------------------
// Fixture configuration and deterministic payloads
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum WriteMode {
	Burst,
	Segment,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ConnMode {
	Cold,
	Warm,
}

#[derive(Clone, Copy)]
struct FixtureConfig {
	events: usize,
	delay_ms: u64,
	payload_bytes: usize,
	mode: WriteMode,
	conn: ConnMode,
}

/// Token text for event `index`, `bytes` long. Depends only on (index, bytes):
/// identical across runs, invocations, and before/after optimization work.
fn deterministic_token(index: usize, bytes: usize) -> String {
	const ALPHABET: &[u8] = b"abcdefghijklmnopqrstuvwxyz0123456789";
	let mut out = String::with_capacity(bytes + 8);
	out.push_str(&format!("t{index:05x}"));
	let mut fill = index.wrapping_mul(7).wrapping_add(13);
	while out.len() < bytes {
		out.push(ALPHABET[fill % ALPHABET.len()] as char);
		fill = fill.wrapping_mul(31).wrapping_add(17);
	}
	out.truncate(bytes);
	out
}

/// The SSE `data:` payloads of one fixture response: N token events, one
/// finish/usage event, and the [DONE] sentinel.
fn build_sse_events(cfg: &FixtureConfig) -> Vec<Vec<u8>> {
	let mut events = Vec::with_capacity(cfg.events + 2);
	for index in 0..cfg.events {
		let payload = json!({
			"id": "bench-fixture",
			"object": "chat.completion.chunk",
			"created": 1772000000i64,
			"model": "bench-model",
			"choices": [{
				"index": 0,
				"delta": {"content": deterministic_token(index, cfg.payload_bytes)},
				"finish_reason": null,
			}],
		});
		events.push(format!("data: {payload}\n\n").into_bytes());
	}
	let finish = json!({
		"id": "bench-fixture",
		"object": "chat.completion.chunk",
		"created": 1772000000i64,
		"model": "bench-model",
		"choices": [{
			"index": 0,
			"delta": {},
			"finish_reason": "stop",
		}],
		"usage": {
			"prompt_tokens": 10,
			"completion_tokens": cfg.events,
			"total_tokens": 10 + cfg.events,
		},
	});
	events.push(format!("data: {finish}\n\n").into_bytes());
	events.push(b"data: [DONE]\n\n".to_vec());
	events
}

/// The token sequence the provider must deliver for `cfg`.
fn expected_content(cfg: &FixtureConfig) -> String {
	let mut content = String::with_capacity(cfg.events * cfg.payload_bytes.max(1));
	for index in 0..cfg.events {
		content.push_str(&deterministic_token(index, cfg.payload_bytes));
	}
	content
}

fn point_name(cfg: &FixtureConfig) -> String {
	format!(
		"e{}-d{}-p{}-{}-{}",
		cfg.events,
		cfg.delay_ms,
		cfg.payload_bytes,
		if cfg.mode == WriteMode::Burst { "burst" } else { "seg" },
		if cfg.conn == ConnMode::Cold { "cold" } else { "warm" }
	)
}

fn config_json(cfg: &FixtureConfig) -> Value {
	json!({
		"events": cfg.events,
		"delay_ms": cfg.delay_ms,
		"payload_bytes": cfg.payload_bytes,
		"mode": if cfg.mode == WriteMode::Burst { "burst" } else { "segment" },
		"conn": if cfg.conn == ConnMode::Cold { "cold" } else { "warm" },
	})
}

// ---------------------------------------------------------------------------
// Fixture SSE server
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct ServerRecord {
	seq: u64,
	kind: &'static str,
	headers_written_ns: u64,
	/// Timestamp taken after each write_all returned (one entry per event in
	/// segment mode, one entry for the whole body in burst mode).
	event_write_ns: Vec<u64>,
}

#[derive(Default)]
struct ServerState {
	next_seq: AtomicU64,
	requests_served: AtomicU64,
	connections_accepted: AtomicU64,
	records: Mutex<Vec<ServerRecord>>,
}

struct HttpRequest {
	method: String,
	path: String,
}

async fn read_request(socket: &mut TcpStream) -> std::io::Result<Option<HttpRequest>> {
	let mut buffer: Vec<u8> = Vec::with_capacity(4096);
	let mut chunk = [0u8; 4096];
	let header_end = loop {
		if let Some(end) = buffer.windows(4).position(|window| window == b"\r\n\r\n") {
			break end;
		}
		let count = socket.read(&mut chunk).await?;
		if count == 0 {
			return Ok(None);
		}
		buffer.extend_from_slice(&chunk[..count]);
	};
	let head = String::from_utf8_lossy(&buffer[..header_end]).to_string();
	let mut lines = head.lines();
	let request_line = lines.next().unwrap_or_default();
	let mut parts = request_line.split_whitespace();
	let method = parts.next().unwrap_or_default().to_string();
	let path = parts.next().unwrap_or_default().to_string();
	let mut content_length = 0usize;
	for line in lines {
		if let Some((name, value)) = line.split_once(':') {
			if name.trim().eq_ignore_ascii_case("content-length") {
				content_length = value.trim().parse().unwrap_or(0);
			}
		}
	}
	while buffer.len() < header_end + 4 + content_length {
		let count = socket.read(&mut chunk).await?;
		if count == 0 {
			return Ok(None);
		}
		buffer.extend_from_slice(&chunk[..count]);
	}
	Ok(Some(HttpRequest { method, path }))
}

async fn write_status(socket: &mut TcpStream, status: u16, text: &str) -> std::io::Result<()> {
	let head = format!("HTTP/1.1 {status} {text}\r\nContent-Type: text/plain\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
	socket.write_all(head.as_bytes()).await?;
	socket.flush().await
}

/// Streams the configured SSE body with the configured pacing.
async fn handle_stream(socket: &mut TcpStream, cfg: &FixtureConfig, seq: u64) -> ServerRecord {
	let events = build_sse_events(cfg);
	let body_len: usize = events.iter().map(|event| event.len()).sum();
	let connection = if cfg.conn == ConnMode::Cold { "close" } else { "keep-alive" };
	let head = format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {body_len}\r\nConnection: {connection}\r\n\r\n");
	let _ = socket.write_all(head.as_bytes()).await;
	let headers_written_ns = now_ns();
	let mut event_write_ns = Vec::with_capacity(events.len() + 1);
	match cfg.mode {
		WriteMode::Burst => {
			let mut body = Vec::with_capacity(body_len);
			for event in &events {
				body.extend_from_slice(event);
			}
			let _ = socket.write_all(&body).await;
			let _ = socket.flush().await;
			event_write_ns.push(now_ns());
		}
		WriteMode::Segment => {
			for (index, event) in events.iter().enumerate() {
				if index > 0 && cfg.delay_ms > 0 {
					tokio::time::sleep(Duration::from_millis(cfg.delay_ms)).await;
				}
				let _ = socket.write_all(event).await;
				let _ = socket.flush().await;
				event_write_ns.push(now_ns());
			}
		}
	}
	ServerRecord { seq, kind: "stream", headers_written_ns, event_write_ns }
}

/// Tiny immediate response used by the transport control measurement.
async fn handle_control(socket: &mut TcpStream, cfg: &FixtureConfig, seq: u64) -> ServerRecord {
	let body = b"{\"ok\":true}";
	let connection = if cfg.conn == ConnMode::Cold { "close" } else { "keep-alive" };
	let head = format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: {connection}\r\n\r\n", body.len());
	let _ = socket.write_all(head.as_bytes()).await;
	let headers_written_ns = now_ns();
	let _ = socket.write_all(body).await;
	let _ = socket.flush().await;
	let write_ns = now_ns();
	ServerRecord { seq, kind: "control", headers_written_ns, event_write_ns: vec![write_ns] }
}

async fn serve_connection(mut socket: TcpStream, cfg: FixtureConfig, state: Arc<ServerState>) {
	let _ = socket.set_nodelay(true);
	loop {
		let request = match tokio::time::timeout(Duration::from_secs(30), read_request(&mut socket)).await {
			Ok(Ok(Some(request))) => request,
			_ => return,
		};
		if request.method != "POST" || (request.path != "/chat/completions" && request.path != "/control") {
			let _ = write_status(&mut socket, 404, "Not Found").await;
			return;
		}
		let seq = state.next_seq.fetch_add(1, Ordering::Relaxed);
		let record = if request.path == "/control" {
			handle_control(&mut socket, &cfg, seq).await
		} else {
			handle_stream(&mut socket, &cfg, seq).await
		};
		state.records.lock().unwrap().push(record);
		state.requests_served.fetch_add(1, Ordering::Relaxed);
		if cfg.conn == ConnMode::Cold {
			let _ = socket.shutdown().await;
			return;
		}
	}
}

async fn wait_for_server_record(state: &ServerState, seq: u64, timeout: Duration) -> Option<ServerRecord> {
	let deadline = tokio::time::Instant::now() + timeout;
	loop {
		{
			let records = state.records.lock().unwrap();
			if let Some(record) = records.iter().rev().find(|record| record.seq == seq && record.kind == "stream") {
				return Some(record.clone());
			}
		}
		if tokio::time::Instant::now() >= deadline {
			return None;
		}
		tokio::time::sleep(Duration::from_millis(2)).await;
	}
}

// ---------------------------------------------------------------------------
// Real provider path
// ---------------------------------------------------------------------------

fn bench_model(base_url: &str) -> Model {
	Model {
		id: "bench-model".to_string(),
		name: "Bench Fixture Model".to_string(),
		api: "openai-completions".to_string(),
		provider: "bench-fixture".to_string(),
		base_url: base_url.to_string(),
		reasoning: false,
		input: vec![InputModality::Text],
		cost: ModelCost::zero(),
		context_window: 128_000.0,
		max_tokens: 4096.0,
		..Default::default()
	}
}

fn bench_context() -> Context {
	Context::new(
		None,
		vec![Message::user(UserMessage::new(UserContent::Text("bench fixture prompt".to_string()), 1))],
		None,
	)
}

struct RunData {
	request_start_ns: u64,
	ok: bool,
	stop_reason: String,
	error_message: Option<String>,
	usage: Option<(f64, f64)>,
	phases: Vec<(u64, u8)>,
	consumer_events: Vec<(u64, &'static str)>,
	consumer_text_ts: Vec<u64>,
	digest_hex: String,
	final_text: Option<String>,
	done_ns: Option<u64>,
	settled_completed: usize,
	settled_pending: usize,
	settled_failed: usize,
	alloc: Option<(u64, u64)>,
}

/// One `stream_openai_completions` call against the fixture, fully timestamped.
async fn run_provider_once(cfg: &FixtureConfig, base_url: &str, count_allocs: bool) -> Result<RunData, String> {
	let phases: Arc<Mutex<Vec<(u64, u8)>>> = Arc::new(Mutex::new(Vec::with_capacity(2 * cfg.events + 16)));
	let observer: OnStreamObservation = {
		let phases = phases.clone();
		Arc::new(move |phase: &str| {
			let _ = phases.lock().unwrap().push((now_ns(), phase_code(phase)));
		})
	};
	let model = bench_model(base_url);
	let context = bench_context();
	let options = OpenAICompletionsOptions {
		stream: StreamOptions {
			api_key: Some("local-fixture-key".to_string()),
			on_stream_observation: Some(observer),
			..Default::default()
		},
		..Default::default()
	};

	let alloc_before = if count_allocs {
		ALLOC_GATE.store(true, Ordering::Relaxed);
		Some(alloc_snapshot())
	} else {
		None
	};

	let request_start_ns = now_ns();
	let stream = stream_openai_completions(&model, &context, Some(options));
	let mut hasher = Sha256::new();
	let mut consumer_events: Vec<(u64, &'static str)> = Vec::with_capacity(cfg.events + 8);
	let mut consumer_text_ts: Vec<u64> = Vec::with_capacity(cfg.events);
	let mut final_message: Option<AssistantMessage> = None;
	let mut done_ns: Option<u64> = None;

	let consumed = tokio::time::timeout(Duration::from_secs(120), async {
		while let Some(event) = stream.next().await {
			let ts = now_ns();
			consumer_events.push((ts, event.event_type()));
			match event {
				AssistantMessageEvent::TextDelta { delta, .. } => {
					hasher.update(delta.as_bytes());
					consumer_text_ts.push(ts);
				}
				AssistantMessageEvent::Done { message, .. } => {
					done_ns = Some(ts);
					final_message = Some(message);
					break;
				}
				AssistantMessageEvent::Error { error, .. } => {
					done_ns = Some(ts);
					final_message = Some(error);
					break;
				}
				_ => {}
			}
		}
	})
	.await;

	let timed_out = consumed.is_err();
	if timed_out {
		stream.request_cancel();
	}
	let settled = stream.task_receipt().settle(Duration::from_secs(5)).await;

	let alloc = match (count_allocs, alloc_before) {
		(true, Some((count, bytes))) => {
			let (after_count, after_bytes) = alloc_snapshot();
			ALLOC_GATE.store(false, Ordering::Relaxed);
			Some((after_count.saturating_sub(count), after_bytes.saturating_sub(bytes)))
		}
		_ => None,
	};

	let phases = phases.lock().unwrap().clone();
	let stop_reason = final_message.as_ref().map(|message| message.stop_reason.clone()).unwrap_or_default();
	let digest_hex: String = {
		let digest = hasher.finalize();
		digest.iter().map(|byte| format!("{byte:02x}")).collect()
	};
	Ok(RunData {
		request_start_ns,
		ok: !timed_out && done_ns.is_some() && stop_reason == "stop",
		stop_reason,
		error_message: final_message.as_ref().and_then(|message| message.error_message.clone()),
		usage: final_message.as_ref().map(|message| (message.usage.input, message.usage.output)),
		phases,
		consumer_events,
		consumer_text_ts,
		digest_hex,
		final_text: final_message
			.as_ref()
			.and_then(|message| message.content.first())
			.and_then(|block| block.as_text())
			.map(|text| text.text.clone()),
		done_ns,
		settled_completed: settled.completed_tasks,
		settled_pending: settled.pending_tasks,
		settled_failed: settled.failed_tasks,
		alloc,
	})
}

#[allow(clippy::too_many_arguments)]
fn run_metrics(
	cfg: &FixtureConfig,
	data: &RunData,
	record: Option<&ServerRecord>,
	expected_digest: &str,
	expected_text: &str,
	run_index: usize,
	is_warmup: bool,
	is_alloc: bool,
) -> Value {
	let request_start = data.request_start_ns;
	let rel = |ts: u64| ts.saturating_sub(request_start);
	let phase_ts = |code: u8| data.phases.iter().find(|(_, c)| *c == code).map(|(ts, _)| rel(*ts));
	let serialize_start = phase_ts(1);
	let serialize_returned = phase_ts(2);
	let send_start = phase_ts(3);
	let headers_complete = phase_ts(4);
	let reader_text_ts: Vec<u64> = data.phases.iter().filter(|(_, c)| *c == 6).map(|(ts, _)| rel(*ts)).collect();
	let reader_raw_ts: Vec<u64> = data.phases.iter().filter(|(_, c)| *c == 5).map(|(ts, _)| rel(*ts)).collect();
	let consumer_text_ts: Vec<u64> = data.consumer_text_ts.iter().map(|ts| rel(*ts)).collect();
	let done_ts = data.done_ns.map(rel);

	let serialize_ns = serialize_returned.zip(serialize_start).map(|(end, start)| end - start);
	let setup_ns = headers_complete.zip(send_start).map(|(end, start)| end - start);
	let ttft_reader_ns = reader_text_ts.first().and_then(|ts| headers_complete.map(|headers| ts.saturating_sub(headers)));
	let ttft_consumer_ns = consumer_text_ts.first().and_then(|ts| headers_complete.map(|headers| ts.saturating_sub(headers)));
	let ttft_total_ns = consumer_text_ts.first().copied();
	let consume_ns = match (consumer_text_ts.last(), headers_complete) {
		(Some(ts), Some(headers)) => Some(ts.saturating_sub(headers)),
		_ => None,
	};
	let stream_end_ns = done_ts.zip(headers_complete).map(|(done, headers)| done.saturating_sub(headers));
	let wall_ns = done_ts;

	let consumer_gaps: Vec<u64> = consumer_text_ts.windows(2).map(|window| window[1] - window[0]).collect();
	let server_event_write_ts: Option<Vec<u64>> = record.map(|record| record.event_write_ns.iter().map(|ts| rel(*ts)).collect());
	let server_content_writes: Vec<u64> = server_event_write_ts
		.as_ref()
		.map(|ts| ts.iter().take(cfg.events).copied().collect())
		.unwrap_or_default();
	let server_gaps: Vec<u64> = server_content_writes.windows(2).map(|window| window[1] - window[0]).collect();

	// Client-attributable gap residual: consumer gap minus the server's actual
	// pacing (segment mode). In burst mode there is no pacing, so the gap itself
	// is the client cost.
	let segment_paced = cfg.mode == WriteMode::Segment
		&& server_gaps.len() + 1 == consumer_text_ts.len()
		&& !consumer_gaps.is_empty();
	let residual_gaps_ms: Vec<f64> = if segment_paced {
		consumer_gaps
			.iter()
			.zip(server_gaps.iter())
			.map(|(client, server)| (*client as f64 - *server as f64) / 1e6)
			.collect()
	} else {
		consumer_gaps.iter().map(|ns| *ns as f64 / 1e6).collect()
	};

	// Forwarding: reader parse of a content delta -> consumer-visible TextDelta.
	let forwarding_aligned = reader_text_ts.len() == consumer_text_ts.len() && !consumer_text_ts.is_empty();
	let forwarding_ns: Vec<u64> = if forwarding_aligned {
		consumer_text_ts
			.iter()
			.zip(reader_text_ts.iter())
			.map(|(consumer, reader)| consumer.saturating_sub(*reader))
			.collect()
	} else {
		Vec::new()
	};

	// Loopback + client I/O: server write -> reader parse (per content event,
	// segment mode only, where each event has its own server write timestamp).
	let net_reader_ns: Vec<u64> = if cfg.mode == WriteMode::Segment {
		let count = reader_raw_ts.len().min(server_content_writes.len()).min(cfg.events);
		reader_raw_ts
			.iter()
			.zip(server_content_writes.iter())
			.take(count)
			.map(|(reader, server)| reader.saturating_sub(*server))
			.collect()
	} else {
		Vec::new()
	};

	let consumer_gap_stats = stats_ms(&consumer_gaps);
	let residual_gap_stats = stats_f64_ms(&residual_gaps_ms);
	let forwarding_stats = stats_ms(&forwarding_ns);
	let net_reader_stats = if net_reader_ns.is_empty() { Value::Null } else { stats_ms(&net_reader_ns) };
	let server_pacing_stats = if server_gaps.is_empty() { Value::Null } else { stats_ms(&server_gaps) };
	let mut event_counts: HashMap<&'static str, u64> = HashMap::new();
	for (_, kind) in data.consumer_events.iter() {
		*event_counts.entry(*kind).or_insert(0) += 1;
	}
	let event_counts: serde_json::Map<String, Value> = event_counts
		.into_iter()
		.map(|(kind, count)| (kind.to_string(), json!(count)))
		.collect();

	json!({
		"run_index": run_index,
		"warmup": is_warmup,
		"alloc_run": is_alloc,
		"ok": data.ok,
		"stop_reason": data.stop_reason,
		"error": data.error_message,
		"usage": data.usage.map(|(input, output)| json!({"input": input, "output": output})),
		"timing": {
			"task_start_ns": serialize_start,
			"serialize_ns": serialize_ns,
			"send_start_ns": send_start,
			"headers_complete_ns": headers_complete,
			"setup_ns": setup_ns,
			"ttft_reader_ns": ttft_reader_ns,
			"ttft_consumer_ns": ttft_consumer_ns,
			"ttft_total_ns": ttft_total_ns,
			"consume_ns": consume_ns,
			"stream_end_ns": stream_end_ns,
			"wall_ns": wall_ns,
		},
		"arrays": {
			"server_headers_written_ns": record.map(|record| rel(record.headers_written_ns)),
			"consumer_text_ts_ns": consumer_text_ts,
			"reader_text_ts_ns": reader_text_ts,
			"reader_raw_ts_ns": reader_raw_ts,
			"server_event_write_ts_ns": server_event_write_ts,
			"consumer_gaps_ns": consumer_gaps,
			"forwarding_ns": forwarding_ns,
			"residual_gaps_ms": residual_gaps_ms,
		},
		"stats": {
			"consumer_gap": consumer_gap_stats,
			"residual_gap": residual_gap_stats,
			"forwarding": forwarding_stats,
			"net_reader": net_reader_stats,
			"server_pacing": server_pacing_stats,
		},
		"alignment": {
			"events_configured": cfg.events,
			"reader_text_events": reader_text_ts.len(),
			"reader_raw_events": reader_raw_ts.len(),
			"consumer_text_events": consumer_text_ts.len(),
			"forwarding_aligned": forwarding_aligned,
			"segment_paced": segment_paced,
		},
		"consumer_event_counts": event_counts,
		"parity": {
			"digest": data.digest_hex,
			"match_expected": data.digest_hex == expected_digest,
			"final_text_match": data.final_text.as_deref() == Some(expected_text),
			"final_text_len": data.final_text.as_ref().map(|text| text.len()),
		},
		"alloc": data.alloc.map(|(count, bytes)| json!({"count": count, "bytes": bytes})),
		"settle": {
			"completed": data.settled_completed,
			"pending": data.settled_pending,
			"failed": data.settled_failed,
		},
	})
}

// ---------------------------------------------------------------------------
// Transport control: fresh vs pooled reqwest client (harness-side reference)
// ---------------------------------------------------------------------------

fn control_variant_summary(runs: &[Value]) -> Value {
	let setups: Vec<f64> = runs.iter().filter_map(|run| run["setup_ns"].as_u64()).map(|ns| ns as f64 / 1e6).collect();
	if setups.is_empty() {
		return json!({});
	}
	let mut rest = setups[1..].to_vec();
	rest.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
	let warm_median = if rest.is_empty() { Value::Null } else { json!(rest[rest.len() / 2]) };
	json!({"first_ms": setups[0], "warm_median_ms": warm_median, "runs_ms": setups})
}

/// Isolates pure transport setup cost (client construction + TCP connect +
/// request write + response headers) at the /control endpoint: `fresh` builds a
/// `reqwest::Client` per request (mirrors today's production behavior),
/// `pooled` reuses one client (the future pooled-client reference).
async fn control_measurement(base_url: &str, cfg: &FixtureConfig, runs: usize) -> Result<Value, String> {
	let url = format!("{base_url}/control");
	let body = json!({"model": "bench-model", "stream": true});

	let mut fresh_runs: Vec<Value> = Vec::with_capacity(runs);
	for run in 0..runs {
		let started = now_ns();
		let client = reqwest::Client::new();
		let result = client.post(&url).json(&body).send().await;
		match result {
			Ok(response) => {
				let setup_ns = now_ns() - started;
				let status = response.status().as_u16();
				let _ = response.text().await;
				fresh_runs.push(json!({"run": run, "setup_ns": setup_ns, "status": status}));
			}
			Err(error) => return Err(format!("control fresh request failed: {error}")),
		}
	}

	let client = reqwest::Client::new();
	let mut pooled_runs: Vec<Value> = Vec::with_capacity(runs);
	for run in 0..runs {
		let started = now_ns();
		let result = client.post(&url).json(&body).send().await;
		match result {
			Ok(response) => {
				let setup_ns = now_ns() - started;
				let status = response.status().as_u16();
				let _ = response.text().await;
				pooled_runs.push(json!({"run": run, "setup_ns": setup_ns, "status": status}));
			}
			Err(error) => return Err(format!("control pooled request failed: {error}")),
		}
	}

	Ok(json!({
		"requests": runs,
		"conn": if cfg.conn == ConnMode::Cold { "cold" } else { "warm" },
		"fresh": control_variant_summary(&fresh_runs),
		"pooled": control_variant_summary(&pooled_runs),
		"fresh_runs": fresh_runs,
		"pooled_runs": pooled_runs,
	}))
}

// ---------------------------------------------------------------------------
// Point execution and reporting
// ---------------------------------------------------------------------------

fn summarize_point(cfg: &FixtureConfig, point_runs: &[Value], connections: u64, requests: u64) -> Value {
	let measured: Vec<&Value> = point_runs
		.iter()
		.filter(|run| !run["warmup"].as_bool().unwrap_or(false) && !run["alloc_run"].as_bool().unwrap_or(false))
		.collect();
	let collect = |path: &str| -> Vec<u64> {
		measured.iter().filter_map(|run| run.pointer(path).and_then(Value::as_u64)).collect()
	};
	let setup = collect("/timing/setup_ns");
	let task_start = collect("/timing/task_start_ns");
	let serialize = collect("/timing/serialize_ns");
	let ttft_consumer = collect("/timing/ttft_consumer_ns");
	let ttft_reader = collect("/timing/ttft_reader_ns");
	let consume = collect("/timing/consume_ns");
	let stream_end = collect("/timing/stream_end_ns");
	let wall = collect("/timing/wall_ns");
	let alloc: Vec<u64> = point_runs
		.iter()
		.filter(|run| run["alloc_run"].as_bool().unwrap_or(false))
		.filter_map(|run| run["alloc"]["count"].as_u64())
		.collect();
	let alloc_bytes: Vec<u64> = point_runs
		.iter()
		.filter(|run| run["alloc_run"].as_bool().unwrap_or(false))
		.filter_map(|run| run["alloc"]["bytes"].as_u64())
		.collect();

	let mut gaps_ms: Vec<f64> = Vec::new();
	let mut residual_ms: Vec<f64> = Vec::new();
	let mut forwarding_ns: Vec<u64> = Vec::new();
	for run in &measured {
		if let Some(list) = run["arrays"]["consumer_gaps_ns"].as_array() {
			gaps_ms.extend(list.iter().filter_map(Value::as_u64).map(|ns| ns as f64 / 1e6));
		}
		if let Some(list) = run["arrays"]["residual_gaps_ms"].as_array() {
			residual_ms.extend(list.iter().filter_map(Value::as_f64));
		}
		if let Some(list) = run["arrays"]["forwarding_ns"].as_array() {
			forwarding_ns.extend(list.iter().filter_map(Value::as_u64));
		}
	}

	let gap_p50 = stats_f64_ms(&gaps_ms)["p50_ms"].as_f64();
	json!({
		"measured_runs": measured.len(),
		"setup_ms": stats_ms(&setup),
		"task_start_ms": stats_ms(&task_start),
		"serialize_ms": stats_ms(&serialize),
		"ttft_consumer_ms": stats_ms(&ttft_consumer),
		"ttft_reader_ms": stats_ms(&ttft_reader),
		"consume_ms": stats_ms(&consume),
		"stream_end_ms": stats_ms(&stream_end),
		"wall_ms": stats_ms(&wall),
		"consumer_gap_ms": stats_f64_ms(&gaps_ms),
		"gap_p50_minus_nominal_delay_ms": gap_p50.map(|p50| p50 - cfg.delay_ms as f64),
		"residual_gap_ms": stats_f64_ms(&residual_ms),
		"forwarding_ms": stats_ms(&forwarding_ns),
		"alloc_count_mean": if alloc.is_empty() { Value::Null } else { json!(alloc.iter().sum::<u64>() / alloc.len() as u64) },
		"alloc_bytes_mean": if alloc_bytes.is_empty() { Value::Null } else { json!(alloc_bytes.iter().sum::<u64>() / alloc_bytes.len() as u64) },
		"server_connections_accepted": connections,
		"server_requests_served": requests,
	})
}

async fn run_point(
	cfg: &FixtureConfig,
	runs: usize,
	warmup: usize,
	alloc_runs: usize,
	control_runs: usize,
	port: u16,
) -> Result<Value, String> {
	let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, port))
		.await
		.map_err(|error| format!("bind: {error}"))?;
	let address = listener.local_addr().map_err(|error| format!("local_addr: {error}"))?;
	let base_url = format!("http://{address}");
	let state = Arc::new(ServerState::default());
	{
		let state = state.clone();
		let cfg = *cfg;
		tokio::spawn(async move {
			loop {
				let Ok((socket, _)) = listener.accept().await else { return };
				state.connections_accepted.fetch_add(1, Ordering::Relaxed);
				let state = state.clone();
				tokio::spawn(async move { serve_connection(socket, cfg, state).await });
			}
		});
	}

	let expected_text = expected_content(cfg);
	let expected_digest = sha256_hex(expected_text.as_bytes());
	let mut point_runs = Vec::with_capacity(warmup + runs + alloc_runs);
	let mut parity_failures: Vec<usize> = Vec::new();
	for run_index in 0..(warmup + runs + alloc_runs) {
		let is_warmup = run_index < warmup;
		let is_alloc = run_index >= warmup + runs;
		let data = run_provider_once(cfg, &base_url, is_alloc).await?;
		let record = wait_for_server_record(&state, run_index as u64, Duration::from_millis(500)).await;
		let metrics = run_metrics(cfg, &data, record.as_ref(), &expected_digest, &expected_text, run_index, is_warmup, is_alloc);
		if !metrics["parity"]["match_expected"].as_bool().unwrap_or(false) || !metrics["ok"].as_bool().unwrap_or(false) {
			parity_failures.push(run_index);
		}
		point_runs.push(metrics);
	}

	let control = if control_runs > 0 {
		control_measurement(&base_url, cfg, control_runs).await?
	} else {
		Value::Null
	};
	let connections = state.connections_accepted.load(Ordering::Relaxed);
	let requests = state.requests_served.load(Ordering::Relaxed);
	let summary = summarize_point(cfg, &point_runs, connections, requests);
	Ok(json!({
		"point_name": point_name(cfg),
		"config": config_json(cfg),
		"expected_digest": expected_digest,
		"expected_content_bytes": expected_text.len(),
		"runs": point_runs,
		"summary": summary,
		"control": control,
		"server": {
			"connections_accepted": connections,
			"requests_served": requests,
			"address": address.to_string(),
		},
		"parity": {"ok": parity_failures.is_empty(), "failed_runs": parity_failures},
	}))
}

fn build_report(subcommand: &str, points: Vec<Value>) -> Value {
	json!({
		"kind": "provider_stream_bench",
		"provider_path": "pi_ai::providers::openai_completions::stream_openai_completions",
		"subcommand": subcommand,
		"generated_utc": Utc::now().to_rfc3339(),
		"binary": {"sha256": exe_sha256().ok(), "git": git_info()},
		"host": {
			"os": std::env::consts::OS,
			"arch": std::env::consts::ARCH,
			"available_parallelism": std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
		},
		"notes": [
			"observer installed: raw_event/text phases add one JSON parse per SSE event on the reader path; identical before/after",
			"setup_ns = generation_send_start -> generation_headers_complete; includes TCP connect because the current path builds reqwest::Client per request",
			"alloc runs include same-process fixture-server allocations",
			"burst mode ignores --delay-ms (whole SSE body in one write)",
		],
		"points": points,
	})
}

fn report_parity_ok(report: &Value) -> bool {
	report["points"]
		.as_array()
		.map(|points| points.iter().all(|point| point["parity"]["ok"].as_bool().unwrap_or(false)))
		.unwrap_or(false)
}

// ---------------------------------------------------------------------------
// Table printing
// ---------------------------------------------------------------------------

fn print_table_header() {
	println!(
		"{:<26} {:>5} {:>10} {:>10} {:>10} {:>10} {:>10} {:>11} {:>11} {:>10} {:>10} {:>6} {:>13}",
		"point", "runs", "setup_p50", "ttft_p50", "gap_p50", "gap_p95", "resid_p50", "fwd_p50_us", "fwd_p95_us", "consume", "wall_ms", "conns", "digest12"
	);
	println!("{}", "-".repeat(166));
}

fn print_point_row(point: &Value) {
	let name = point["point_name"].as_str().unwrap_or("?");
	let summary = &point["summary"];
	let stat = |key: &str, field: &str| summary[key][field].as_f64();
	let fmt = |value: Option<f64>| match value {
		Some(value) => format!("{value:.3}"),
		None => "-".to_string(),
	};
	let micros = |value: Option<f64>| match value {
		Some(value) => format!("{:.1}", value * 1000.0),
		None => "-".to_string(),
	};
	let runs = summary["measured_runs"].as_u64().unwrap_or(0);
	let conns = point["server"]["connections_accepted"].as_u64().unwrap_or(0);
	let digest = point["expected_digest"].as_str().and_then(|d| d.get(..12)).unwrap_or("-");
	println!(
		"{:<26} {:>5} {:>10} {:>10} {:>10} {:>10} {:>10} {:>11} {:>11} {:>10} {:>10} {:>6} {:>13}",
		name,
		runs,
		fmt(stat("setup_ms", "p50_ms")),
		fmt(stat("ttft_consumer_ms", "p50_ms")),
		fmt(stat("consumer_gap_ms", "p50_ms")),
		fmt(stat("consumer_gap_ms", "p95_ms")),
		fmt(stat("residual_gap_ms", "p50_ms")),
		micros(stat("forwarding_ms", "p50_ms")),
		micros(stat("forwarding_ms", "p95_ms")),
		fmt(stat("consume_ms", "p50_ms")),
		fmt(stat("wall_ms", "p50_ms")),
		conns,
		digest,
	);
}

fn print_control(point: &Value) {
	let control = &point["control"];
	if control.is_null() {
		return;
	}
	let name = point["point_name"].as_str().unwrap_or("?");
	let conn = control["conn"].as_str().unwrap_or("?");
	for variant in ["fresh", "pooled"] {
		let first = control[variant]["first_ms"].as_f64();
		let warm = control[variant]["warm_median_ms"].as_f64();
		if let (Some(first), Some(warm)) = (first, warm) {
			println!("  control {name} conn={conn} {variant}: first(cold-connect)={first:.3}ms later-runs(median)={warm:.3}ms");
		}
	}
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
		self.options.get(key).cloned().filter(|value| !value.is_empty())
	}

	fn number_usize(&self, key: &str, default: usize) -> Result<usize, String> {
		match self.optional(key) {
			None => Ok(default),
			Some(raw) => raw.parse::<usize>().map_err(|error| format!("--{key}: {error}")),
		}
	}

	fn number_u64(&self, key: &str, default: u64) -> Result<u64, String> {
		match self.optional(key) {
			None => Ok(default),
			Some(raw) => raw.parse::<u64>().map_err(|error| format!("--{key}: {error}")),
		}
	}
}

fn usage() -> String {
	"usage:\n\
	provider_stream_bench bench --out <json> [--events N] [--delay-ms X] [--payload-bytes B]\n\
	[--mode burst|segment] [--conn cold|warm] [--runs R] [--warmup W] [--alloc-runs A]\n\
	[--control-runs C] [--port P]\n\
	provider_stream_bench matrix --out <json> [--events N] [--runs R] [--warmup W]\n\
	[--alloc-runs A] [--control-runs C]\n\
	defaults: events=200 delay-ms=0 payload-bytes=8 mode=segment conn=cold runs=5 warmup=1\n\
	alloc-runs=0(bench)/1(matrix) control-runs=5 port=ephemeral\n\
	mode burst: whole SSE body in one write (delay-ms ignored)\n\
	mode segment: each SSE event in its own TCP segment (small write + flush)\n\
	conn cold: fixture closes the connection after each response\n\
	conn warm: fixture keeps the connection open (keep-alive) for reuse"
		.to_string()
}

fn parse_mode(raw: &str) -> Result<WriteMode, String> {
	match raw {
		"burst" => Ok(WriteMode::Burst),
		"segment" => Ok(WriteMode::Segment),
		other => Err(format!("--mode: expected burst|segment, got {other}")),
	}
}

fn parse_conn(raw: &str) -> Result<ConnMode, String> {
	match raw {
		"cold" => Ok(ConnMode::Cold),
		"warm" => Ok(ConnMode::Warm),
		other => Err(format!("--conn: expected cold|warm, got {other}")),
	}
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

async fn bench_command(cli: &Cli) -> Result<(), String> {
	let events = cli.number_usize("events", 200)?;
	let delay_ms = cli.number_u64("delay-ms", 0)?;
	let payload_bytes = cli.number_usize("payload-bytes", 8)?;
	let mode = parse_mode(&cli.optional("mode").unwrap_or_else(|| "segment".to_string()))?;
	let conn = parse_conn(&cli.optional("conn").unwrap_or_else(|| "cold".to_string()))?;
	let runs = cli.number_usize("runs", 5)?;
	let warmup = cli.number_usize("warmup", 1)?;
	let alloc_runs = cli.number_usize("alloc-runs", 0)?;
	let control_runs = cli.number_usize("control-runs", 5)?;
	let port = cli.number_u64("port", 0)? as u16;
	let out = cli.required("out")?;
	if mode == WriteMode::Burst && delay_ms > 0 {
		eprintln!("note: --delay-ms {delay_ms} is ignored in burst mode (whole body in one write)");
	}
	let cfg = FixtureConfig { events, delay_ms, payload_bytes, mode, conn };
	let point = run_point(&cfg, runs, warmup, alloc_runs, control_runs, port).await?;
	print_table_header();
	print_point_row(&point);
	print_control(&point);
	let report = build_report("bench", vec![point]);
	write_json(Path::new(&out), &report)?;
	if !report_parity_ok(&report) {
		return Err("parity failure: token sequence digest mismatch (see report)".to_string());
	}
	println!("wrote {out}");
	Ok(())
}

async fn matrix_command(cli: &Cli) -> Result<(), String> {
	let events = cli.number_usize("events", 200)?;
	let runs = cli.number_usize("runs", 5)?;
	let warmup = cli.number_usize("warmup", 1)?;
	let alloc_runs = cli.number_usize("alloc-runs", 1)?;
	let control_runs = cli.number_usize("control-runs", 5)?;
	let out = cli.required("out")?;
	let started = Instant::now();
	let mut points: Vec<Value> = Vec::new();
	print_table_header();
	for conn in [ConnMode::Cold, ConnMode::Warm] {
		for payload_bytes in [8usize, 256] {
			for (mode, delay_ms) in [
				(WriteMode::Burst, 0u64),
				(WriteMode::Segment, 0),
				(WriteMode::Segment, 1),
				(WriteMode::Segment, 5),
			] {
				let cfg = FixtureConfig { events, delay_ms, payload_bytes, mode, conn };
				println!("# running {}", point_name(&cfg));
				let point = run_point(&cfg, runs, warmup, alloc_runs, control_runs, 0).await?;
				print_point_row(&point);
				points.push(point);
			}
		}
	}
	for point in &points {
		print_control(point);
	}
	let mut shapes: HashMap<(u64, u64), Vec<String>> = HashMap::new();
	for point in &points {
		let shape = (
			point["config"]["events"].as_u64().unwrap_or(0),
			point["config"]["payload_bytes"].as_u64().unwrap_or(0),
		);
		shapes.entry(shape).or_default().push(point["expected_digest"].as_str().unwrap_or("").to_string());
	}
	let shape_ok = shapes.values().all(|digests| digests.iter().all(|digest| digest == &digests[0]));
	let report = build_report("matrix", points);
	write_json(Path::new(&out), &report)?;
	println!("matrix complete in {:.1}s, wrote {out}", started.elapsed().as_secs_f64());
	if !report_parity_ok(&report) {
		return Err("parity failure: token sequence digest mismatch (see report)".to_string());
	}
	if !shape_ok {
		return Err("parity failure: digest differs across modes for the same (events, payload)".to_string());
	}
	Ok(())
}

#[tokio::main]
async fn main() {
	let args: Vec<String> = std::env::args().skip(1).collect();
	let cli = match Cli::parse(&args) {
		Ok(cli) => cli,
		Err(error) => {
			eprintln!("{error}");
			std::process::exit(2);
		}
	};
	let result = match cli.subcommand.as_str() {
		"bench" => bench_command(&cli).await,
		"matrix" => matrix_command(&cli).await,
		"help" => {
			println!("{}", usage());
			Ok(())
		}
		other => Err(format!("unknown subcommand: {other}\n{}", usage())),
	};
	if let Err(error) = result {
		eprintln!("error: {error}");
		std::process::exit(1);
	}
}
