//! Parse-focused benchmark for the JSON/SSE/event-stream decode paths of the
//! providers touched by the J1 parse optimization (openai-completions,
//! anthropic, google, google-vertex, mistral, bedrock converse, kiro).
//!
//! Each provider path is driven end to end through its real public stream
//! entry point against a deterministic in-process fixture server: the body
//! depends only on the event index, so the consumer-visible event sequence
//! (and its SHA-256 digest) must be identical across builds. The digest is
//! the parity gate for the optimization work: run this example on the base
//! revision and on the optimized revision and compare the `digest` fields.
//!
//! Fixtures deliberately include the rare fields (usage, error-ish members,
//! reasoning/thinking, tool calls) so the typed fast paths are exercised
//! together with their Value fallbacks, not just the common text deltas.
//!
//! Subcommands (hand-rolled CLI; std + existing crate APIs only):
//!   bench --out <json> [--events N] [--runs R] [--warmup W]
//!        [--alloc-runs A] [--only name1,name2]
//!
//! Everything here is measurement tooling: no production file depends on it
//! and it must keep working unchanged BEFORE and AFTER optimization work.

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::BTreeMap;
use std::net::Ipv4Addr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use pi_ai::providers::amazon_bedrock::{stream_bedrock_with_options, BedrockOptions};
use pi_ai::providers::anthropic::{stream_anthropic, AnthropicOptions};
use pi_ai::providers::google::{stream_google, GoogleOptions};
use pi_ai::providers::google_vertex::{stream_google_vertex, GoogleVertexOptions};
use pi_ai::providers::kiro::stream_kiro;
use pi_ai::providers::mistral::{stream_mistral, MistralOptions};
use pi_ai::providers::openai_completions::{stream_openai_completions, OpenAICompletionsOptions};
use pi_ai::types::{
	AssistantMessage, AssistantMessageEvent, ContentBlock, Context, InputModality, Message, Model,
	ModelCost, StreamOptions, UserContent, UserMessage,
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
// Event-stream frame encoding (bedrock converse + kiro share the format)
// ---------------------------------------------------------------------------

fn crc32(bytes: &[u8]) -> u32 {
	let mut crc = !0u32;
	for &byte in bytes {
		crc ^= u32::from(byte);
		for _ in 0..8 {
			crc = (crc >> 1) ^ (0xedb88320 & (0u32.wrapping_sub(crc & 1)));
		}
	}
	!crc
}

/// One `vnd.amazon.eventstream` frame: string headers + JSON payload.
fn event_stream_frame(headers: &[(&str, &str)], payload: &str) -> Vec<u8> {
	let mut encoded_headers = Vec::new();
	for (name, value) in headers {
		encoded_headers.push(name.len() as u8);
		encoded_headers.extend_from_slice(name.as_bytes());
		encoded_headers.push(7);
		encoded_headers.extend_from_slice(&(value.len() as u16).to_be_bytes());
		encoded_headers.extend_from_slice(value.as_bytes());
	}
	let total = 16 + encoded_headers.len() + payload.len();
	let mut frame = Vec::with_capacity(total);
	frame.extend_from_slice(&(total as u32).to_be_bytes());
	frame.extend_from_slice(&(encoded_headers.len() as u32).to_be_bytes());
	frame.extend_from_slice(&crc32(&frame).to_be_bytes());
	frame.extend_from_slice(&encoded_headers);
	frame.extend_from_slice(payload.as_bytes());
	frame.extend_from_slice(&crc32(&frame).to_be_bytes());
	frame
}

// ---------------------------------------------------------------------------
// Deterministic fixture bodies
// ---------------------------------------------------------------------------

fn openai_completions_body(events: usize) -> Vec<u8> {
	let mut body = Vec::new();
	for index in 0..events {
		let delta = match index % 16 {
			// rare members: reasoning content, tool-call fragments
			3 => json!({"reasoning_content": format!("think {index}")}),
			5 => json!({"tool_calls": [{"index": 0, "id": "call-bench-1", "type": "function", "function": {"name": "read_file", "arguments": ""}}]}),
			6 => json!({"tool_calls": [{"index": 0, "function": {"arguments": "{\"path\":"}}]}),
			7 => json!({"tool_calls": [{"index": 0, "function": {"arguments": "\"/tmp/fixture\"}"}}]}),
			_ => json!({"content": format!("delta {index} ")}),
		};
		let chunk = json!({
			"id": "chatcmpl-bench",
			"object": "chat.completion.chunk",
			"model": "bench-model",
			"choices": [{"index": 0, "delta": delta}]
		});
		body.extend_from_slice(format!("data: {chunk}\n\n").as_bytes());
	}
	// final chunk: finish reason + usage (rare fields present)
	let final_chunk = json!({
		"id": "chatcmpl-bench",
		"object": "chat.completion.chunk",
		"model": "bench-model",
		"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
		"usage": {"prompt_tokens": 1000, "completion_tokens": events, "total_tokens": 1000 + events}
	});
	body.extend_from_slice(format!("data: {final_chunk}\n\n").as_bytes());
	body.extend_from_slice(b"data: [DONE]\n\n");
	body
}

fn anthropic_body(events: usize) -> Vec<u8> {
	let mut body = Vec::new();
	let event = |value: Value| format!("event: {}\ndata: {value}\n\n", value["type"].as_str().unwrap());
	body.extend_from_slice(
		event(json!({"type": "message_start", "message": {"id": "msg-bench", "usage": {"input_tokens": 12, "output_tokens": 0}}})).as_bytes(),
	);
	body.extend_from_slice(
		event(json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})).as_bytes(),
	);
	for index in 0..events {
		match index % 16 {
			// rare member: thinking delta with signature
			3 => body.extend_from_slice(
				event(json!({"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": format!("think {index} ")}})).as_bytes(),
			),
			_ => body.extend_from_slice(
				event(json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": format!("delta {index} ")}})).as_bytes(),
			),
		}
	}
	body.extend_from_slice(event(json!({"type": "content_block_stop", "index": 0})).as_bytes());
	// rare members: usage + stop reason on message_delta
	body.extend_from_slice(
		event(json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"input_tokens": 12, "output_tokens": events}})).as_bytes(),
	);
	body.extend_from_slice(event(json!({"type": "message_stop"})).as_bytes());
	body
}

fn google_body(events: usize) -> Vec<u8> {
	let mut body = Vec::new();
	let mut tool_call_seq = 0usize;
	for index in 0..events {
		let part = match index % 16 {
			3 => json!({"text": format!("think {index} "), "thought": true, "thoughtSignature": "sig-bench"}),
			5 => {
				tool_call_seq += 1;
				// unique ids: google regenerates colliding ids with a
				// time-based id, which would make the digest non-deterministic
				json!({"functionCall": {"id": format!("call-bench-{tool_call_seq}"), "name": "read_file", "args": {"path": "/tmp/fixture"}}})
			}
			_ => json!({"text": format!("delta {index} ")}),
		};
		let chunk = json!({
			"responseId": format!("resp-{index}"),
			"candidates": [{"content": {"parts": [part]}}]
		});
		body.extend_from_slice(format!("data: {chunk}\n\n").as_bytes());
	}
	// final chunk: finish reason + usage (rare fields present)
	let final_chunk = json!({
		"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "done"}]}}],
		"usageMetadata": {"promptTokenCount": 1000, "candidatesTokenCount": events, "totalTokenCount": 1000 + events}
	});
	body.extend_from_slice(format!("data: {final_chunk}\n\n").as_bytes());
	body
}

fn mistral_body(events: usize) -> Vec<u8> {
	let mut body = Vec::new();
	for index in 0..events {
		let delta = match index % 16 {
			5 => json!({"tool_calls": [{"index": 0, "id": "call-bench-1", "function": {"name": "read_file", "arguments": ""}}]}),
			6 => json!({"tool_calls": [{"index": 0, "function": {"arguments": "{\"path\":"}}]}),
			7 => json!({"tool_calls": [{"index": 0, "function": {"arguments": "\"/tmp/fixture\"}"}}]}),
			_ => json!({"content": format!("delta {index} ")}),
		};
		let chunk = json!({
			"id": "cmpl-bench",
			"model": "bench-model",
			"choices": [{"index": 0, "delta": delta}]
		});
		body.extend_from_slice(format!("data: {chunk}\n\n").as_bytes());
	}
	let final_chunk = json!({
		"id": "cmpl-bench",
		"model": "bench-model",
		"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
		"usage": {"prompt_tokens": 1000, "completion_tokens": events, "total_tokens": 1000 + events}
	});
	body.extend_from_slice(format!("data: {final_chunk}\n\n").as_bytes());
	body.extend_from_slice(b"data: [DONE]\n\n");
	body
}

fn bedrock_body(events: usize) -> Vec<u8> {
	let mut body = Vec::new();
	let frame = |event_type: &str, payload: Value| {
		event_stream_frame(&[(":message-type", "event"), (":event-type", event_type)], &payload.to_string())
	};
	body.extend(frame("messageStart", json!({"role": "assistant"})));
	body.extend(frame("contentBlockStart", json!({"contentBlockIndex": 0, "start": {}})));
	for index in 0..events {
		match index % 16 {
			5 => body.extend(frame(
				"contentBlockStart",
				json!({"contentBlockIndex": 1, "start": {"toolUse": {"toolUseId": "call-bench-1", "name": "read_file"}}}),
			)),
			6 | 7 => body.extend(frame(
				"contentBlockDelta",
				json!({"contentBlockIndex": 1, "delta": {"toolUse": {"input": if index == 6 { "{\"path\":" } else { "\"/tmp/fixture\"}" } }}}),
			)),
			3 => body.extend(frame(
				"contentBlockDelta",
				json!({"contentBlockIndex": 0, "delta": {"reasoningContent": {"text": format!("think {index} ")}}}),
			)),
			_ => body.extend(frame(
				"contentBlockDelta",
				json!({"contentBlockIndex": 0, "delta": {"text": format!("delta {index} ")}}),
			)),
		}
	}
	body.extend(frame("contentBlockStop", json!({"contentBlockIndex": 0})));
	body.extend(frame("contentBlockStop", json!({"contentBlockIndex": 1})));
	body.extend(frame(
		"metadata",
		json!({"usage": {"inputTokens": 1000, "outputTokens": events}}),
	));
	body.extend(frame("messageStop", json!({"stopReason": "tool_use"})));
	body
}

fn kiro_body(events: usize) -> Vec<u8> {
	let mut body = Vec::new();
	let frame = |payload: Value| event_stream_frame(&[(":message-type", "event"), (":event-type", "assistantResponseEvent")], &payload.to_string());
	for index in 0..events {
		// kiro completes at most one tool call per stream (the tool block is
		// only closed on the final `stop`), so the tool payload appears once
		let payload = match index {
			3 => json!({"content": format!("think {index} ")}),
			5 => json!({"toolUseId": "call-bench-1", "name": "read_file"}),
			6 => json!({"input": "{\"path\":"}),
			7 => json!({"input": "\"/tmp/fixture\"}"}),
			_ => json!({"content": format!("delta {index} ")}),
		};
		body.extend(frame(payload));
	}
	// rare fields: usage + context percentage, then stop
	body.extend(frame(json!({"usage": {"inputTokens": 1000, "outputTokens": events}})));
	body.extend(frame(json!({"contextUsagePercentage": 3.5})));
	body.extend(frame(json!({"stop": true})));
	body
}

// ---------------------------------------------------------------------------
// Fixture server (one connection per run; serves the same deterministic body)
// ---------------------------------------------------------------------------

async fn serve_once(content_type: &str, body: Arc<Vec<u8>>) -> Result<String, String> {
	let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.map_err(|error| error.to_string())?;
	let address = listener.local_addr().map_err(|error| error.to_string())?;
	let content_type = content_type.to_string();
	tokio::spawn(async move {
		let Ok((mut socket, _)) = listener.accept().await else {
			return;
		};
		if read_request(&mut socket).await.is_none() {
			return;
		}
		let header = format!(
			"HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nConnection: close\r\n\r\n"
		);
		if socket.write_all(header.as_bytes()).await.is_err() {
			return;
		}
		if socket.write_all(&body).await.is_err() {
			return;
		}
		let _ = socket.shutdown().await;
		// drain until the client closes so the socket does not reset
		let mut sink = [0u8; 4096];
		let _ = tokio::time::timeout(Duration::from_secs(5), socket.read(&mut sink)).await;
	});
	Ok(format!("http://{address}"))
}

async fn read_request(socket: &mut TcpStream) -> Option<()> {
	let mut request = Vec::new();
	loop {
		let mut buffer = [0u8; 4096];
		match socket.read(&mut buffer).await {
			Ok(0) | Err(_) => return None,
			Ok(count) => {
				request.extend_from_slice(&buffer[..count]);
				if let Some(end) = request.windows(4).position(|bytes| bytes == b"\r\n\r\n") {
					let headers = std::str::from_utf8(&request[..end]).ok()?;
					let length: usize = headers
						.lines()
						.filter_map(|line| line.split_once(':'))
						.find(|(key, _)| key.eq_ignore_ascii_case("content-length"))
						.map(|(_, value)| value.trim().parse().unwrap_or(0))
						.unwrap_or(0);
					if request.len() >= end + 4 + length {
						return Some(());
					}
				}
			}
		}
	}
}

// ---------------------------------------------------------------------------
// Digest of the consumer-visible event sequence
// ---------------------------------------------------------------------------

fn hash_message(hasher: &mut Sha256, message: &AssistantMessage) {
	hasher.update(message.stop_reason.as_bytes());
	hasher.update(&[0]);
	if let Some(error) = &message.error_message {
		hasher.update(error.as_bytes());
	}
	hasher.update(&[0]);
	hasher.update(&message.usage.input.to_le_bytes());
	hasher.update(&message.usage.output.to_le_bytes());
	hasher.update(&message.usage.total_tokens.to_le_bytes());
	for block in &message.content {
		match block {
			ContentBlock::Text(text) => {
				hasher.update(b"T");
				hasher.update(text.text.as_bytes());
			}
			ContentBlock::Thinking(thinking) => {
				hasher.update(b"H");
				hasher.update(thinking.thinking.as_bytes());
			}
			ContentBlock::ToolCall(call) => {
				hasher.update(b"C");
				hasher.update(call.id.as_bytes());
				hasher.update(call.name.as_bytes());
				hasher.update(serde_json::to_string(&call.arguments).unwrap_or_default().as_bytes());
			}
		}
		hasher.update(&[0]);
	}
}

fn hash_event(hasher: &mut Sha256, event: &AssistantMessageEvent) {
	hasher.update(event.event_type().as_bytes());
	hasher.update(&[0]);
	match event {
		AssistantMessageEvent::TextDelta { delta, .. } => hasher.update(delta.as_bytes()),
		AssistantMessageEvent::ThinkingDelta { delta, .. } => hasher.update(delta.as_bytes()),
		AssistantMessageEvent::ToolCallDelta { delta, .. } => hasher.update(delta.as_bytes()),
		AssistantMessageEvent::ToolCallEnd { tool_call, .. } => {
			hasher.update(tool_call.id.as_bytes());
			hasher.update(tool_call.name.as_bytes());
			hasher.update(serde_json::to_string(&tool_call.arguments).unwrap_or_default().as_bytes());
		}
		AssistantMessageEvent::Done { message, .. } => hash_message(hasher, message),
		AssistantMessageEvent::Error { error, .. } => hash_message(hasher, error),
		_ => {}
	}
	hasher.update(&[0]);
}

// ---------------------------------------------------------------------------
// Per-provider wiring
// ---------------------------------------------------------------------------

struct RunOutcome {
	ok: bool,
	digest_hex: String,
	events: usize,
	per_event_ns: Vec<u64>,
}

fn bench_model(api: &str, provider: &str, base_url: &str) -> Model {
	Model {
		id: "bench-model".to_string(),
		name: "Bench Fixture Model".to_string(),
		api: api.to_string(),
		provider: provider.to_string(),
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

fn api_key_options() -> StreamOptions {
	StreamOptions {
		api_key: Some("local-fixture-key".to_string()),
		..Default::default()
	}
}

async fn consume_stream(stream: pi_ai::types::AssistantMessageEventStreamReExport) -> RunOutcome {
	let mut hasher = Sha256::new();
	let mut events = 0usize;
	let mut per_event_ns = Vec::new();
	let mut ok = false;
	let mut last_ts: Option<Instant> = None;
	let consumed = tokio::time::timeout(Duration::from_secs(120), async {
		while let Some(event) = stream.next().await {
			let now = Instant::now();
			if let Some(last) = last_ts {
				per_event_ns.push(now.duration_since(last).as_nanos().min(u64::MAX as u128) as u64);
			}
			last_ts = Some(now);
			hash_event(&mut hasher, &event);
			events += 1;
			match event {
				AssistantMessageEvent::Done { .. } => {
					ok = true;
					break;
				}
				AssistantMessageEvent::Error { .. } => break,
				_ => {}
			}
		}
	})
	.await;
	if consumed.is_err() {
		stream.request_cancel();
	}
	let _ = stream.task_receipt().settle(Duration::from_secs(5)).await;
	let digest = hasher.finalize();
	RunOutcome {
		ok: ok && consumed.is_ok(),
		digest_hex: digest.iter().map(|byte| format!("{byte:02x}")).collect(),
		events,
		per_event_ns,
	}
}

async fn run_openai_completions(base_url: String) -> Result<RunOutcome, String> {
	let model = bench_model("openai-completions", "bench-fixture", &base_url);
	let options = OpenAICompletionsOptions {
		stream: api_key_options(),
		..Default::default()
	};
	Ok(consume_stream(stream_openai_completions(&model, &bench_context(), Some(options))).await)
}

async fn run_anthropic(base_url: String) -> Result<RunOutcome, String> {
	let model = bench_model("anthropic-messages", "bench-fixture", &base_url);
	let options = AnthropicOptions {
		stream: api_key_options(),
		..Default::default()
	};
	Ok(consume_stream(stream_anthropic(&model, &bench_context(), Some(options))).await)
}

async fn run_google(base_url: String) -> Result<RunOutcome, String> {
	let model = bench_model("google-generative-ai", "bench-fixture", &base_url);
	let options = GoogleOptions {
		stream: api_key_options(),
		..Default::default()
	};
	Ok(consume_stream(stream_google(&model, &bench_context(), Some(options))).await)
}

async fn run_google_vertex(base_url: String) -> Result<RunOutcome, String> {
	let model = bench_model("google-vertex", "bench-fixture", &base_url);
	let options = GoogleVertexOptions {
		stream: api_key_options(),
		project: Some("bench-project".to_string()),
		location: Some("us-east1".to_string()),
		..Default::default()
	};
	Ok(consume_stream(stream_google_vertex(&model, &bench_context(), Some(options))).await)
}

async fn run_mistral(base_url: String) -> Result<RunOutcome, String> {
	let model = bench_model("mistral-conversations", "bench-fixture", &base_url);
	let options = MistralOptions {
		stream: api_key_options(),
		..Default::default()
	};
	Ok(consume_stream(stream_mistral(&model, &bench_context(), Some(options))).await)
}

async fn run_bedrock(base_url: String) -> Result<RunOutcome, String> {
	// dummy credentials, no signing, no profile resolution
	std::env::set_var("AWS_BEDROCK_SKIP_AUTH", "1");
	let model = bench_model("bedrock-converse-stream", "bench-fixture", &base_url);
	let options = BedrockOptions {
		stream: api_key_options(),
		region: Some("us-east-1".to_string()),
		..Default::default()
	};
	Ok(consume_stream(stream_bedrock_with_options(&model, &bench_context(), Some(options))).await)
}

async fn run_kiro(base_url: String) -> Result<RunOutcome, String> {
	let model = bench_model("kiro-api", "bench-fixture", &base_url);
	// API-key mode: the OAuth envelope form is restricted to regional
	// endpoints outside `cfg(test)`, and the fixture is a loopback server.
	std::env::set_var("KIRO_API_REGION", "us-east-1");
	let options = StreamOptions {
		api_key: Some("ksk_bench-fixture-key".to_string()),
		..Default::default()
	};
	Ok(consume_stream(stream_kiro(&model, &bench_context(), Some(&options))).await)
}

// ---------------------------------------------------------------------------
// Bench driver
// ---------------------------------------------------------------------------

struct ProviderBench {
	name: &'static str,
	content_type: &'static str,
	fixture: fn(usize) -> Vec<u8>,
	run: fn(String) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<RunOutcome, String>> + Send>>,
}

fn provider_benches() -> Vec<ProviderBench> {

fn boxed_run<F>(future: F) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<RunOutcome, String>> + Send>>
where
	F: std::future::Future<Output = Result<RunOutcome, String>> + Send + 'static,
{
	Box::pin(future)
}

vec![
		ProviderBench {
			name: "openai_completions",
			content_type: "text/event-stream",
			fixture: openai_completions_body,
			run: |base_url: String| boxed_run(run_openai_completions(base_url)),
		},
		ProviderBench {
			name: "anthropic",
			content_type: "text/event-stream",
			fixture: anthropic_body,
			run: |base_url: String| boxed_run(run_anthropic(base_url)),
		},
		ProviderBench {
			name: "google",
			content_type: "text/event-stream",
			fixture: google_body,
			run: |base_url: String| boxed_run(run_google(base_url)),
		},
		ProviderBench {
			name: "google_vertex",
			content_type: "text/event-stream",
			fixture: google_body,
			run: |base_url: String| boxed_run(run_google_vertex(base_url)),
		},
		ProviderBench {
			name: "mistral",
			content_type: "text/event-stream",
			fixture: mistral_body,
			run: |base_url: String| boxed_run(run_mistral(base_url)),
		},
		ProviderBench {
			name: "bedrock",
			content_type: "application/vnd.amazon.eventstream",
			fixture: bedrock_body,
			run: |base_url: String| boxed_run(run_bedrock(base_url)),
		},
		ProviderBench {
			name: "kiro",
			content_type: "application/vnd.amazon.eventstream",
			fixture: kiro_body,
			run: |base_url: String| boxed_run(run_kiro(base_url)),
		},
	]
}

fn percentile(mut values: Vec<u64>, fraction: f64) -> u64 {
	if values.is_empty() {
		return 0;
	}
	values.sort_unstable();
	let index = ((values.len() as f64 - 1.0) * fraction).round() as usize;
	values[index.min(values.len() - 1)]
}

async fn bench_provider(spec: &ProviderBench, events: usize, warmup: usize, timed_runs: usize, alloc_runs: usize) -> Result<Value, String> {
	let body = Arc::new((spec.fixture)(events));

	// warmup (no timing, no alloc counting)
	for _ in 0..warmup {
		let base_url = serve_once(spec.content_type, body.clone()).await?;
		let outcome = (spec.run)(base_url).await?;
		if !outcome.ok {
			return Err(format!("{}: warmup run failed", spec.name));
		}
	}

	// allocation measurement (separate from timing: the gate adds atomics)
	let mut allocs_per_run: Vec<u64> = Vec::with_capacity(alloc_runs);
	let mut alloc_bytes_per_run: Vec<u64> = Vec::with_capacity(alloc_runs);
	for _ in 0..alloc_runs {
		let base_url = serve_once(spec.content_type, body.clone()).await?;
		ALLOC_GATE.store(true, Ordering::Relaxed);
		let before = alloc_snapshot();
		let outcome = (spec.run)(base_url).await?;
		let after = alloc_snapshot();
		ALLOC_GATE.store(false, Ordering::Relaxed);
		if !outcome.ok {
			return Err(format!("{}: alloc run failed", spec.name));
		}
		allocs_per_run.push(after.0.saturating_sub(before.0));
		alloc_bytes_per_run.push(after.1.saturating_sub(before.1));
	}

	// timed runs
	let mut stream_ns: Vec<u64> = Vec::with_capacity(timed_runs);
	let mut per_event_ns: Vec<u64> = Vec::new();
	let mut digest_hex: Option<String> = None;
	let mut events_seen: Option<usize> = None;
	for _ in 0..timed_runs {
		let base_url = serve_once(spec.content_type, body.clone()).await?;
		let start = Instant::now();
		let outcome = (spec.run)(base_url).await?;
		let elapsed = start.elapsed();
		if !outcome.ok {
			return Err(format!("{}: timed run failed", spec.name));
		}
		stream_ns.push(elapsed.as_nanos().min(u64::MAX as u128) as u64);
		per_event_ns.extend(outcome.per_event_ns.iter().copied());
		match (&digest_hex, &events_seen) {
			(None, _) => {
				digest_hex = Some(outcome.digest_hex.clone());
				events_seen = Some(outcome.events);
			}
			(Some(seen), _) => {
				if seen != &outcome.digest_hex {
					return Err(format!("{}: digest drifted across runs", spec.name));
				}
			}
		}
	}

	let allocs: u64 = allocs_per_run.iter().sum::<u64>() / alloc_runs.max(1) as u64;
	let alloc_bytes: u64 = alloc_bytes_per_run.iter().sum::<u64>() / alloc_runs.max(1) as u64;
	let events_per_run = events_seen.unwrap_or(0);
	Ok(json!({
		"provider": spec.name,
		"events_per_run": events_per_run,
		"warmup_runs": warmup,
		"timed_runs": timed_runs,
		"alloc_runs": alloc_runs,
		"allocs_per_run": allocs,
		"alloc_bytes_per_run": alloc_bytes,
		"allocs_per_event": if events_per_run > 0 { allocs / events_per_run as u64 } else { 0 },
		"stream_ns_p50": percentile(stream_ns.clone(), 0.50),
		"stream_ns_p95": percentile(stream_ns.clone(), 0.95),
		"per_event_ns_p50": percentile(per_event_ns.clone(), 0.50),
		"per_event_ns_p95": percentile(per_event_ns.clone(), 0.95),
		"digest": digest_hex,
	}))
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct Cli {
	subcommand: String,
	out: String,
	events: usize,
	runs: usize,
	warmup: usize,
	alloc_runs: usize,
	only: Option<Vec<String>>,
}

fn parse_cli(args: &[String]) -> Result<Cli, String> {
	let mut cli = Cli {
		subcommand: "bench".to_string(),
		out: "json_parse_bench.json".to_string(),
		events: 256,
		runs: 30,
		warmup: 3,
		alloc_runs: 5,
		only: None,
	};
	let mut index = 0;
	while index < args.len() {
		let arg = args[index].as_str();
		let mut value = || -> Result<String, String> {
			index += 1;
			args.get(index).cloned().ok_or_else(|| format!("missing value for --{arg}"))
		};
		match arg {
			"bench" => {}
			"--out" => cli.out = value()?,
			"--events" => cli.events = value()?.parse().map_err(|_| "--events needs a number")?,
			"--runs" => cli.runs = value()?.parse().map_err(|_| "--runs needs a number")?,
			"--warmup" => cli.warmup = value()?.parse().map_err(|_| "--warmup needs a number")?,
			"--alloc-runs" => cli.alloc_runs = value()?.parse().map_err(|_| "--alloc-runs needs a number")?,
			"--only" => {
				cli.only = Some(
					value()?
						.split(',')
						.map(|name| name.trim().to_string())
						.filter(|name| !name.is_empty())
						.collect(),
				);
			}
			other => return Err(format!("unknown argument {other}")),
		}
		index += 1;
	}
	Ok(cli)
}

async fn bench_command(cli: &Cli) -> Result<(), String> {
	let benches = provider_benches();
	let mut results = Vec::new();
	let mut failed = BTreeMap::new();
	for spec in &benches {
		if let Some(only) = &cli.only {
			if !only.iter().any(|name| name == spec.name) {
				continue;
			}
		}
		println!("bench {} (events={}, runs={}, warmup={}, alloc-runs={})", spec.name, cli.events, cli.runs, cli.warmup, cli.alloc_runs);
		match bench_provider(spec, cli.events, cli.warmup, cli.runs, cli.alloc_runs).await {
			Ok(value) => {
				println!(
					"  allocs/run={} allocs/event={} stream p50={}ns p95={}ns digest={}",
					value["allocs_per_run"], value["allocs_per_event"], value["stream_ns_p50"], value["stream_ns_p95"], value["digest"].as_str().unwrap_or("")
				);
				results.push(value);
			}
			Err(error) => {
				failed.insert(spec.name.to_string(), error.clone());
				results.push(json!({"provider": spec.name, "error": error}));
			}
		}
	}
	let payload = json!({
		"events": cli.events,
		"runs": cli.runs,
		"warmup": cli.warmup,
		"alloc_runs": cli.alloc_runs,
		"providers": results,
		"failed": failed,
	});
	std::fs::write(&cli.out, serde_json::to_string_pretty(&payload).map_err(|error| error.to_string())?)
		.map_err(|error| error.to_string())?;
	println!("wrote {}", cli.out);
	if !failed.is_empty() {
		Err(format!("failed providers: {}", failed.keys().cloned().collect::<Vec<_>>().join(", ")))
	} else {
		Ok(())
	}
}

#[tokio::main]
async fn main() {
	let args: Vec<String> = std::env::args().skip(1).collect();
	let cli = match parse_cli(&args) {
		Ok(cli) => cli,
		Err(error) => {
			eprintln!("{error}");
			std::process::exit(2);
		}
	};
	let result = match cli.subcommand.as_str() {
		"bench" => bench_command(&cli).await,
		other => Err(format!("unknown subcommand {other}")),
	};
	if let Err(error) = result {
		eprintln!("{error}");
		std::process::exit(1);
	}
}
