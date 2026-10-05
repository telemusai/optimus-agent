//! Counting-alloc harness for the Codex WebSocket parse chain
//! (`native_web_socket` emit -> listener -> `parse_web_socket` -> yielded event).
//!
//! Drives the REAL native transport against a local tokio-tungstenite fixture
//! server: one WebSocket connection, then R request/response rounds of N JSON
//! frames each. Reports per-round allocation counts (std-only counting global
//! allocator, gated to the measured window), wall time, and a parity digest over
//! the delivered token sequence. Uses only public crate APIs, so the same file
//! measures the tree BEFORE and AFTER parse-chain changes.
//!
//! usage: cargo run --release --example codex_ws_parse_bench -- \
//!            --out <json> [--frames N] [--rounds R] [--warmup W] [--payload-bytes B]

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::{SinkExt, StreamExt};
use pi_ai::providers::openai_codex_responses::{
	get_web_socket_constructor, parse_web_socket, WebSocketLike,
};
use serde_json::{json, Value};
use tokio::net::TcpListener;
use tokio_tungstenite::tungstenite::Message as Frame;

// ---------------------------------------------------------------------------
// Counting global allocator (std-only; gated so connect/setup stays clean)
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
// Deterministic frames
// ---------------------------------------------------------------------------

fn deterministic_token(round: usize, index: usize, bytes: usize) -> String {
	const ALPHABET: &[u8] = b"abcdefghijklmnopqrstuvwxyz0123456789";
	let mut out = String::with_capacity(bytes + 12);
	out.push_str(&format!("r{round:03x}t{index:05x}"));
	let mut fill = round.wrapping_mul(131).wrapping_add(index.wrapping_mul(7)).wrapping_add(13);
	while out.len() < bytes {
		out.push(ALPHABET[fill % ALPHABET.len()] as char);
		fill = fill.wrapping_mul(31).wrapping_add(17);
	}
	out.truncate(bytes);
	out
}

/// The wire frames of one round: created + N deltas + completed.
fn round_frames(round: usize, frames: usize, payload_bytes: usize) -> Vec<String> {
	let mut events = Vec::with_capacity(frames + 2);
	events.push(
		json!({"type":"response.created","response":{"id": format!("bench-{round}")}}).to_string(),
	);
	for index in 0..frames {
		events.push(
			json!({"type":"response.output_text.delta","delta": deterministic_token(round, index, payload_bytes)})
				.to_string(),
		);
	}
	events.push(
		json!({
			"type":"response.completed",
			"response":{"id": format!("bench-{round}"), "status":"completed",
				"usage":{"input_tokens":5,"output_tokens":frames,"total_tokens":5+frames}}
		})
		.to_string(),
	);
	events
}

fn expected_digest_payload(rounds: usize, offset: usize, frames: usize, payload_bytes: usize) -> String {
	let mut content = String::new();
	for round in offset..offset + rounds {
		for index in 0..frames {
			content.push_str(&deterministic_token(round, index, payload_bytes));
		}
	}
	content
}

// ---------------------------------------------------------------------------
// Fixture WebSocket server
// ---------------------------------------------------------------------------

async fn serve_rounds(listener: TcpListener, total_rounds: usize, frames: usize, payload_bytes: usize) {
	let (tcp, _) = listener.accept().await.expect("fixture accept");
	let mut socket = tokio_tungstenite::accept_async(tcp).await.expect("fixture handshake");
	for round in 0..total_rounds {
		let request = tokio::time::timeout(Duration::from_secs(5), socket.next())
			.await
			.expect("request frame must arrive")
			.expect("stream open")
			.expect("request frame must be valid");
		assert!(matches!(request, Frame::Text(_)), "each round starts with one request frame");
		for payload in round_frames(round, frames, payload_bytes) {
			socket
				.send(Frame::Text(payload.into()))
				.await
				.expect("fixture frame write");
		}
	}
	let _ = tokio::time::timeout(Duration::from_secs(2), socket.close(None)).await;
}

// ---------------------------------------------------------------------------
// CLI (hand-rolled; std only)
// ---------------------------------------------------------------------------

struct Cli {
	options: HashMap<String, String>,
}

impl Cli {
	fn parse(args: &[String]) -> Result<Cli, String> {
		let mut options = HashMap::new();
		let mut index = 0usize;
		while index < args.len() {
			let arg = &args[index];
			index += 1;
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
		Ok(Cli { options })
	}

	fn optional(&self, key: &str) -> Option<String> {
		self.options.get(key).cloned().filter(|value| !value.is_empty())
	}

	fn required(&self, key: &str) -> Result<String, String> {
		self.optional(key).ok_or_else(|| format!("missing required option --{key}"))
	}

	fn number_usize(&self, key: &str, default: usize) -> Result<usize, String> {
		match self.optional(key) {
			None => Ok(default),
			Some(raw) => raw.parse::<usize>().map_err(|error| format!("--{key}: {error}")),
		}
	}
}

fn sha256_hex(bytes: &[u8]) -> String {
	use sha2::{Digest, Sha256};
	let digest = Sha256::digest(bytes);
	digest.iter().map(|byte| format!("{byte:02x}")).collect()
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

fn exe_sha256() -> Result<String, String> {
	let exe = std::env::current_exe().map_err(|error| error.to_string())?;
	let bytes = std::fs::read(&exe).map_err(|error| error.to_string())?;
	Ok(sha256_hex(&bytes))
}

// ---------------------------------------------------------------------------
// Bench
// ---------------------------------------------------------------------------

async fn wait_open(socket: &Arc<dyn WebSocketLike>) -> Result<(), String> {
	for _ in 0..500 {
		if socket.ready_state() == Some(1) {
			return Ok(());
		}
		tokio::time::sleep(Duration::from_millis(2)).await;
	}
	Err("socket never reached OPEN state".to_string())
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
	let out = cli.required("out").unwrap_or_else(|error| {
		eprintln!("{error}");
		std::process::exit(2);
	});
	let frames = cli.number_usize("frames", 200).expect("frames");
	let rounds = cli.number_usize("rounds", 5).expect("rounds");
	let warmup = cli.number_usize("warmup", 2).expect("warmup");
	let payload_bytes = cli.number_usize("payload-bytes", 96).expect("payload-bytes");
	let total_rounds = warmup + rounds;

	let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind fixture listener");
	let port = listener.local_addr().unwrap().port();
	let server = tokio::spawn(serve_rounds(listener, total_rounds, frames, payload_bytes));

	let constructor = get_web_socket_constructor().expect("native WebSocket constructor");
	let url = format!("ws://127.0.0.1:{port}/codex/responses");
	let mut headers = indexmap::IndexMap::new();
	headers.insert("session_id".to_string(), "codex-ws-parse-bench".to_string());
	let socket = constructor(&url, headers);
	wait_open(&socket).await.expect("socket open");

	let request = json!({"type":"response.create","model":"bench","input":[]}).to_string();
	let mut round_reports = Vec::new();
	let mut delivered = String::new();
	for round in 0..total_rounds {
		let measured = round >= warmup;
		if measured {
			ALLOC_GATE.store(true, Ordering::Relaxed);
		}
		let before = alloc_snapshot();
		let started = Instant::now();
		socket.send(&request);
		let mut events = 0usize;
		let mut stream = parse_web_socket(socket.clone(), None);
		let collected = tokio::time::timeout(Duration::from_secs(30), async {
			let mut deltas = String::new();
			while let Some(event) = stream.next().await {
				let event = event.expect("bench frames are valid JSON");
				events += 1;
				if event.get("type").and_then(Value::as_str) == Some("response.output_text.delta") {
					deltas.push_str(event.get("delta").and_then(Value::as_str).unwrap_or_default());
				}
			}
			deltas
		})
		.await
		.expect("round must settle");
		let duration = started.elapsed();
		let after = alloc_snapshot();
		ALLOC_GATE.store(false, Ordering::Relaxed);
		if measured {
			delivered.push_str(&collected);
			round_reports.push(json!({
				"round": round,
				"frames": frames + 2,
				"events": events,
				"allocs": after.0 - before.0,
				"alloc_bytes": after.1 - before.1,
				"duration_ms": duration.as_secs_f64() * 1000.0,
			}));
		}
	}

	let expected = expected_digest_payload(rounds, warmup, frames, payload_bytes);
	let digest_ok = delivered == expected;
	let measured_allocs: u64 = round_reports.iter().map(|r| r["allocs"].as_u64().unwrap_or(0)).sum();
	let measured_bytes: u64 = round_reports.iter().map(|r| r["alloc_bytes"].as_u64().unwrap_or(0)).sum();
	let measured_frames: u64 = (frames as u64 + 2) * rounds as u64;
	let report = json!({
		"git": git_info(),
		"exe_sha256": exe_sha256().unwrap_or_default(),
		"config": {"frames": frames, "rounds": rounds, "warmup": warmup, "payload_bytes": payload_bytes},
		"rounds": round_reports,
		"totals": {
			"frames": measured_frames,
			"allocs": measured_allocs,
			"alloc_bytes": measured_bytes,
			"allocs_per_frame": measured_allocs as f64 / measured_frames as f64,
			"alloc_bytes_per_frame": measured_bytes as f64 / measured_frames as f64,
		},
		"digest_ok": digest_ok,
	});
	if let Some(parent) = Path::new(&out).parent() {
		std::fs::create_dir_all(parent).map_err(|error| error.to_string()).unwrap();
	}
	std::fs::write(&out, serde_json::to_string_pretty(&report).unwrap()).expect("write report");
	println!("{}", serde_json::to_string_pretty(&report["totals"]).unwrap());
	if !digest_ok {
		eprintln!("parity failure: delivered token sequence differs from the expected fixture sequence");
		std::process::exit(1);
	}
	let _ = server.await;
}
