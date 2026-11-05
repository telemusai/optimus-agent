//! OCC-3 retained-state offline integration proof: six families through the
//! real Agent, isolated native family controller, `Occ3NativeSession`,
//! `SglangOcc3Connector` and the OpenAI-completions provider path against a
//! synthetic loopback HTTP fixture. Deterministic; no GPU, daemon,
//! credential, or network beyond 127.0.0.1. No live-semantics claim.
//!
//! Families (see V3 retained-state proof scope):
//! 1. POSITIVE: sender capture -> handle -> the receiver's actual next
//!    provider request carries the typed attachment through the
//!    `StreamOptions` inference path, and the bound consumption receipt is
//!    enforced field-by-field (request/message/session/fingerprint/epoch/
//!    handle) before any provider event is exposed.
//! 2. NO-STATE: identical wiring without capture: no attachment, no
//!    receipt, and no OCC-3 claim, while the receiver resolver really runs.
//! 3. STALE-NEGATIVE: an expired handle or an evicted receiver session:
//!    consumption is refused fail-closed, the refusal is recorded, and no
//!    text fallback or clean success is possible.
//! 4. ONCE-COUNTER: a harness-owned unconditional single-use pre-request
//!    ceiling that fires before every provider request, independent of the
//!    capture sentinel (absent, malformed, or present) and of the per-prompt
//!    request-index reset; a second request is refused pre-dispatch.
//! 5. RAW-RETENTION: the private retained record keeps the RAW source_handle
//!    and the FULL capture receipt in memory (data-only), never the
//!    hash-only `audit()` projection.
//! 6. WRONG-MODEL/WRONG-EPOCH: the same-model/same-epoch rule is enforced at
//!    consume time on the receiver's real resolution path.
//!
//! Production source is untouched: only public Agent/session/connector APIs
//! and test-local doubles are used (same style as occ3_source_handoff.rs).

use pi_agent_core::agent::{Agent, AgentOptions, PromptInput};
use pi_agent_core::types::{
    AgentMessage, AgentState, AgentTool, AgentToolResult, ContentBlock, ResolveInferenceState,
    SourceCaptureHooks, ThinkingLevel,
};
use pi_ai::inference_attachment::INFERENCE_RECEIPT_HEADER;
use pi_ai::source_capture::{
    canonical_json_bytes, sha256, SourceCaptureIntent, SourceCaptureLease, SourceCaptureProfile,
    SourceCaptureReceipt, SourceCaptureState, SourceToolBinding,
};
use pi_ai::types::{Model, API_OPENAI_COMPLETIONS};
use pi_coding_agent::core::agent_messages::*;
use pi_coding_agent::core::agent_session::occ3_state::{
    NativeSourceBudget, Occ3NativeSession, NATIVE_HANDOFF_TOOL,
};
use pi_coding_agent::core::occ::occ3::connector::Occ3Connector;
use pi_coding_agent::core::occ::occ3::inference::permits_text_fallback;
use pi_coding_agent::core::occ::occ3::types::{Occ3DeliveryStatus, Occ3ModelIdentity};
use pi_coding_agent::core::occ3_connectors::sglang::{
    NativeExperimentalPermit, SglangOcc3Config, SglangOcc3Connector,
};
use pi_coding_agent::modes::daemon::daemon_mode::occ3_native::{
    IsolatedNativeFamily, IsolatedNativeMember,
};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

const METHOD: &str = "statebridge.glm53.final-prenorm-predictor.continuous-prefix.v1";
/// Raw private source handle issued by the synthetic runtime fixture. The
/// retained record must keep this RAW value; only the audit hashes it.
const RAW_SOURCE_HANDLE: &str = "retained-raw-source-handle-fixture";

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

fn identity() -> Occ3ModelIdentity {
    Occ3ModelIdentity {
        model_id: "fixture-model".into(),
        architecture: "fixture-arch".into(),
        checkpoint_revision: "fixture-revision".into(),
        weight_fingerprint: "metadata-fixture-not-weights".into(),
        tokenizer_fingerprint: "fixture-tokenizer".into(),
        chat_template_fingerprint: "fixture-template".into(),
        quantisation: "fixture-bf16".into(),
        tensor_geometry: "fixture-d8".into(),
        kv_layout: "fixture-tp8".into(),
        latent_method: METHOD.into(),
        runtime: "synthetic-local-fixture".into(),
    }
}

// ---------------------------------------------------------------------------
// Synthetic loopback fixture
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ReceiptMode {
    Exact,
    /// The source fixture answers the default v2 arming with the stale bare
    /// `occ3.source.v1` wire (no span block): the pre-fix live shape.
    StaleV1,
    WrongHandle,
    WrongRequest,
    WrongMessage,
    WrongSession,
    WrongFingerprint,
    WrongExpiry,
    NotConsumed,
    FallbackUsed,
    NotInjected,
    Duplicate,
    Missing,
}

impl ReceiptMode {
    fn name(self) -> &'static str {
        match self {
            ReceiptMode::Exact => "exact",
            ReceiptMode::StaleV1 => "stale_v1",
            ReceiptMode::WrongHandle => "wrong_handle",
            ReceiptMode::WrongRequest => "wrong_request",
            ReceiptMode::WrongMessage => "wrong_message",
            ReceiptMode::WrongSession => "wrong_session",
            ReceiptMode::WrongFingerprint => "wrong_fingerprint",
            ReceiptMode::WrongExpiry => "wrong_expiry",
            ReceiptMode::NotConsumed => "not_consumed",
            ReceiptMode::FallbackUsed => "fallback_used",
            ReceiptMode::NotInjected => "not_injected",
            ReceiptMode::Duplicate => "duplicate",
            ReceiptMode::Missing => "missing",
        }
    }
}

#[derive(Clone)]
struct FixtureMode {
    receipt: ReceiptMode,
    /// Seconds from now used as the source receipt expiry (stale-handle case).
    short_receipt_expiry: Option<u64>,
}

impl Default for FixtureMode {
    fn default() -> Self {
        Self {
            receipt: ReceiptMode::Exact,
            short_receipt_expiry: None,
        }
    }
}

#[derive(Default)]
struct FixtureRecords {
    requests: Vec<(String, Value)>,
    publication: Option<Value>,
    published_request: Option<Value>,
    source_intent: Option<Value>,
    receiver_receipts: Vec<Value>,
}

struct FixtureState {
    records: Mutex<FixtureRecords>,
    /// Flipped by the harness to simulate a runtime restart: later enroll
    /// replies report a different runtime epoch/generation.
    restarted: AtomicBool,
}

impl Default for FixtureState {
    fn default() -> Self {
        Self {
            records: Mutex::new(FixtureRecords::default()),
            restarted: AtomicBool::new(false),
        }
    }
}

fn sse(id: &str, tool: bool) -> String {
    let mut chunks = vec![
        json!({"id":id,"choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"retained native reasoning"},"finish_reason":null}]}),
    ];
    if tool {
        chunks.push(json!({"id":id,"choices":[{"index":0,"delta":{"tool_calls":[
            {"index":0,"id":"actual-tool-1","type":"function","function":{"name":"occ3_handoff","arguments":"{\"receiver_role\":\"child\",\"receiver_name\":\"receiver\"}"}}]},
            "finish_reason":null}]}));
    } else {
        chunks.push(json!({"id":id,"choices":[{"index":0,"delta":{"content":"synthetic fixture completion"},"finish_reason":null}]}));
    }
    chunks.push(json!({"id":id,"choices":[{"index":0,"delta":{},"finish_reason":if tool {"tool_calls"}else{"stop"}}]}));
    let mut body = chunks
        .into_iter()
        .map(|v| format!("data: {v}\n\n"))
        .collect::<String>();
    body.push_str("data: [DONE]\n\n");
    body
}

async fn read_request(socket: &mut tokio::net::TcpStream) -> (String, Value) {
    let mut bytes = Vec::new();
    let end;
    loop {
        let mut part = [0u8; 4096];
        let n = socket.read(&mut part).await.unwrap();
        assert!(n > 0);
        bytes.extend_from_slice(&part[..n]);
        if let Some(i) = bytes.windows(4).position(|v| v == b"\r\n\r\n") {
            end = i + 4;
            break;
        }
    }
    let headers = String::from_utf8(bytes[..end].to_vec()).unwrap();
    assert!(headers
        .to_lowercase()
        .contains("authorization: bearer synthetic-fixture-key"));
    let size = headers
        .lines()
        .find_map(|l| {
            l.to_ascii_lowercase()
                .strip_prefix("content-length: ")
                .map(|v| v.trim().parse::<usize>().unwrap())
        })
        .unwrap();
    while bytes.len() < end + size {
        let mut p = [0u8; 4096];
        let n = socket.read(&mut p).await.unwrap();
        assert!(n > 0);
        bytes.extend_from_slice(&p[..n]);
    }
    (
        headers
            .lines()
            .next()
            .unwrap()
            .split_whitespace()
            .nth(1)
            .unwrap()
            .into(),
        serde_json::from_slice(&bytes[end..end + size]).unwrap(),
    )
}

async fn send(socket: &mut tokio::net::TcpStream, body: String, extra: Vec<(&str, String)>) {
    let mut headers = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n",
        body.len()
    );
    for (k, v) in extra {
        headers.push_str(&format!("{k}: {v}\r\n"));
    }
    headers.push_str("\r\n");
    socket.write_all(headers.as_bytes()).await.unwrap();
    socket.write_all(body.as_bytes()).await.unwrap();
    socket.shutdown().await.unwrap();
}

async fn serve(
    mode: FixtureMode,
    state: Arc<FixtureState>,
) -> (String, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        loop {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mode = mode.clone();
            let state = state.clone();
            tokio::spawn(async move {
                let (path, body) = read_request(&mut socket).await;
                state
                    .records
                    .lock()
                    .unwrap()
                    .requests
                    .push((path.clone(), body.clone()));
                let result = match path.as_str() {
                    "/occ3/v1/native/sessions/enroll" => {
                        let (runtime_id, generation) = if state.restarted.load(Ordering::SeqCst) {
                            (json!("runtime-epoch-2"), json!(2))
                        } else {
                            (body["runtime_id"].clone(), body["runtime_generation"].clone())
                        };
                        json!({"schema":"occ3.sessions.enrolled.v1","operation_id":body["operation_id"],"session_id":body["session_id"],
                            "principal_digest":"a".repeat(64),"runtime_id":runtime_id,"runtime_generation":generation,"model_fingerprint":body["model_fingerprint"],"expires_at_epoch":body["expires_at_epoch"],
                            "readiness":{"runtime_epoch":runtime_id,"runtime_generation":generation,"model_fingerprint":body["model_fingerprint"],"method_id":METHOD,
                                "expected_ranks":8,"ready_ranks":8,"tp_layout_digest":"b".repeat(64),"hidden_dim":8,"layer_index":3,"dtype":"bfloat16","profile_digest":"c".repeat(64),"source_capture":true,"continuous_prefix":true,"terminal_journal":true}})
                    }
                    "/occ3/v1/native/publish" => {
                        let source = state.records.lock().unwrap().source_intent.clone().unwrap();
                        assert_eq!(body["source_session_id"], source["source_session_id"]);
                        assert_eq!(body["source_turn_nonce"], source["turn_nonce"]);
                        assert_eq!(body["tool_call_id"], "actual-tool-1");
                        assert_eq!(body["source_handle"], RAW_SOURCE_HANDLE);
                        let mut binding = body.clone();
                        binding.as_object_mut().unwrap().remove("schema");
                        binding["principal_digest"] = json!("a".repeat(64));
                        let reply = json!({"schema":"occ3.publication.v1","publication_id":body["publication_id"],
                            "binding_digest":sha256(&canonical_json_bytes(&binding).unwrap()),
                            "state":"published","source_handle":body["source_handle"],"expires_at_epoch":body["expires_at_epoch"]});
                        {
                            let mut records = state.records.lock().unwrap();
                            records.publication = Some(reply.clone());
                            records.published_request = Some(body.clone());
                        }
                        reply
                    }
                    "/occ3/v1/native/publication/status" => {
                        state.records.lock().unwrap().publication.clone().unwrap_or_else(|| {
                            json!({"schema":"occ3.publication.v1","publication_id":body["publication_id"],"state":"absent"})
                        })
                    }
                    "/occ3/v1/native/retire" => json!({"schema":"occ3.retired.v1","operation_id":body["operation_id"],
                        "source_handle":body["source_handle"],"revoked":true,"gpu_freed":false,"state":"retired"}),
                    "/v1/chat/completions" => {
                        assert_eq!(
                            body.get("max_tokens").or_else(|| body.get("max_completion_tokens")),
                            Some(&json!(131072)),
                            "real native default is preserved; fixture output is short"
                        );
                        assert!(
                            body.get("reasoning_effort").and_then(Value::as_str).is_some_and(|v| matches!(v, "high"|"xhigh"|"max")),
                            "MAX goes through the existing model-specific conversion"
                        );
                        if let Some(intent) = body.get("occ3_capture") {
                            state.records.lock().unwrap().source_intent = Some(intent.clone());
                            let output = sse("retained-source-rid", true);
                            let expires = mode
                                .short_receipt_expiry
                                .map(|secs| now() + secs)
                                .unwrap_or_else(|| intent["expires_at_epoch"].as_u64().unwrap());
                            let stale_v1 = mode.receipt == ReceiptMode::StaleV1;
                            let mut receipt = json!({"schema":if stale_v1 {"occ3.source.v1"} else {"occ3.source.v2"},
                                "intent_digest":sha256(&canonical_json_bytes(intent).unwrap()),
                                "source_handle":RAW_SOURCE_HANDLE,"native_rid":"retained-source-rid","response_id":"retained-source-rid",
                                "capture_commit_id":"retained-proof-capture-commit",
                                "accepted_span_start":0,"accepted_span_end":8,"retained_count":8,"predictor_position_offset":-1,
                                "representation":"final_layer_prenorm_predictor","layer_index":3,"hidden_dim":8,"dtype":"bfloat16",
                                "layout_digest":"b".repeat(64),"ready_ranks":8,"expected_ranks":8,
                                "accepted_sequence_digest":"d".repeat(64),"output_sha256":sha256(output.as_bytes()),
                                "expires_at_epoch":expires,"byte_count":128});
                            // The default arming is occ3.capture.v2, so the
                            // intent is answered on the v2 wire with the
                            // occ3.span.v1 block. The counts model this
                            // fixture's 8-row window the way the pi-ai
                            // canonical v2 wire and the lifecycle fixture do:
                            // the reasoning region, the thinking_end
                            // terminator marker row and the role-token stop
                            // row (both CONTROL), the single occ3_handoff
                            // tool-call row and its argument rows. The
                            // captured turn emits no content delta, so the
                            // window carries no CONTENT row.
                            let span = json!({"span_schema":"occ3.span.v1","span_policy":intent["span_policy"],
                                "span_marker_profile":"glm45_glm47_xml_kv.v1","span_marker_schema":"occ3.span.markers.v1",
                                "span_reasoning_terminator":"thinking_end","span_turn_end_attested":true,
                                "span_thinking_count":2,"span_content_count":0,"span_tool_call_count":1,
                                "span_tool_arguments_count":3,"span_control_count":2,"span_unknown_count":0,
                                "span_window_start_position":0,"span_window_end_position":8,
                                "span_contiguous":true,"span_contains_final_content":true,
                                "span_contains_tool_call":true,"span_message_truncated":false});
                            if !stale_v1 {
                                for (key, value) in span.as_object().unwrap() {
                                    receipt[key.clone()] = value.clone();
                                }
                            }
                            // StaleV1 answers the default v2 arming with the
                            // pre-fix live shape: the bare occ3.source.v1 wire
                            // with no span block. The client must refuse it.
                            let headers = vec![("x-optimus-occ3-source", receipt.to_string())];
                            send(&mut socket, output, headers).await;
                            return;
                        }
                        let mut headers = Vec::new();
                        if let Some(attachment) = body.get("occ3") {
                            let publication =
                                state.records.lock().unwrap().published_request.clone().unwrap();
                            // The attachment reached the request builder: bind it
                            // against the publication on the real request.
                            assert_eq!(attachment["message_id"], publication["message_id"]);
                            assert_eq!(attachment["receiver_session_id"], publication["receiver_session_id"]);
                            assert_eq!(attachment["state_handle"], publication["source_handle"]);
                            assert_eq!(attachment["runtime_id"], "runtime-epoch-1");
                            assert_eq!(attachment["runtime_generation"], 1);
                            assert_eq!(attachment["model_fingerprint"], identity().fingerprint());
                            assert_eq!(attachment["method"], METHOD);
                            assert_eq!(attachment["connector_id"], "sglang-occ3");
                            assert_eq!(attachment["protocol_version"], 1);
                            assert_eq!(attachment["expires_at_epoch"], publication["expires_at_epoch"]);
                            assert!(attachment["request_id"].as_str().is_some_and(|v| !v.is_empty()));
                            let mut receipt_attachment = attachment.clone();
                            let (mut consumed, mut injected, mut fallback) = (true, true, false);
                            match mode.receipt {
                                ReceiptMode::WrongHandle => {
                                    receipt_attachment["state_handle"] = json!("wrong-handle-value")
                                }
                                ReceiptMode::WrongRequest => {
                                    receipt_attachment["request_id"] = json!("different-request-id")
                                }
                                ReceiptMode::WrongMessage => {
                                    receipt_attachment["message_id"] = json!("different-message-id")
                                }
                                ReceiptMode::WrongSession => {
                                    receipt_attachment["receiver_session_id"] = json!("unrelated-session")
                                }
                                ReceiptMode::WrongFingerprint => {
                                    receipt_attachment["model_fingerprint"] = json!("0".repeat(64))
                                }
                                ReceiptMode::WrongExpiry => {
                                    receipt_attachment["expires_at_epoch"] =
                                        json!(attachment["expires_at_epoch"].as_u64().unwrap() + 4096)
                                }
                                ReceiptMode::NotConsumed => consumed = false,
                                ReceiptMode::FallbackUsed => fallback = true,
                                ReceiptMode::NotInjected => injected = false,
                                ReceiptMode::Exact | ReceiptMode::Duplicate | ReceiptMode::Missing => {}
                                // StaleV1 only shapes the source receipt; a
                                // receiver attachment is never reached in that
                                // scenario (the source turn fails first).
                                ReceiptMode::StaleV1 => {}
                            }
                            let receipt = json!({"attachment":receipt_attachment,"consumed":consumed,
                                "fallback_used":fallback,"injected_state":injected});
                            state.records.lock().unwrap().receiver_receipts.push(receipt.clone());
                            match mode.receipt {
                                ReceiptMode::Missing => {}
                                ReceiptMode::Duplicate => {
                                    headers.push((INFERENCE_RECEIPT_HEADER, receipt.to_string()));
                                    headers.push((INFERENCE_RECEIPT_HEADER, receipt.to_string()));
                                }
                                _ => headers.push((INFERENCE_RECEIPT_HEADER, receipt.to_string())),
                            }
                        }
                        send(&mut socket, sse("retained-completion-rid", false), headers).await;
                        return;
                    }
                    _ => panic!("Unexpected fixture route: {path}"),
                };
                send(&mut socket, result.to_string(), Vec::new()).await;
            });
        }
    });
    (endpoint, task)
}

// ---------------------------------------------------------------------------
// Harness doubles: once-counter, retained record, wrapped hooks, stub tool
// ---------------------------------------------------------------------------

type BeforeRequestHook = Arc<
    dyn Fn(
            u64,
            Option<tokio_util::sync::CancellationToken>,
        ) -> futures::future::BoxFuture<'static, anyhow::Result<()>>
        + Send
        + Sync,
>;

/// Harness-owned unconditional single-use pre-request ceiling.
///
/// The refusal decision is ONLY `seen > allowed`; it never consults the
/// capture sentinel (that cell is telemetry the hook observes and records)
/// and it is independent of the per-prompt request index the loop passes in.
struct OnceCounter {
    allowed: u64,
    seen: AtomicU64,
    fired: AtomicU64,
    indices: Mutex<Vec<u64>>,
    /// Telemetry only. Absent, malformed, or well-formed values must all be
    /// refused identically.
    sentinel: Mutex<Option<Value>>,
    refusal_sentinels: Mutex<Vec<Option<String>>>,
}

impl OnceCounter {
    fn new(allowed: u64) -> Arc<Self> {
        Arc::new(Self {
            allowed,
            seen: AtomicU64::new(0),
            fired: AtomicU64::new(0),
            indices: Mutex::new(Vec::new()),
            sentinel: Mutex::new(None),
            refusal_sentinels: Mutex::new(Vec::new()),
        })
    }

    fn hook(self: &Arc<Self>) -> BeforeRequestHook {
        let counter = self.clone();
        Arc::new(move |index, _signal| {
            let counter = counter.clone();
            Box::pin(async move {
                counter.indices.lock().unwrap().push(index);
                let n = counter.seen.fetch_add(1, Ordering::SeqCst) + 1;
                if n > counter.allowed {
                    counter.fired.fetch_add(1, Ordering::SeqCst);
                    let observed = counter.sentinel.lock().unwrap().clone();
                    counter
                        .refusal_sentinels
                        .lock()
                        .unwrap()
                        .push(observed.map(|v| v.to_string()));
                    return Err(anyhow::anyhow!(
                        "retained-state once-counter: provider request {} exceeds the single-use ceiling of {}",
                        n,
                        counter.allowed
                    ));
                }
                Ok(())
            })
        })
    }
}

/// PRIVATE harness-side retained record. In-memory, data-only: keeps the RAW
/// source_handle and the FULL capture receipt for the eventual private
/// artifact. Never written to a durable file by this proof, and never reduced
/// to the hash-only `audit()` projection.
#[derive(Clone)]
struct RetainedRecord {
    raw_source_handle: String,
    receipt: SourceCaptureReceipt,
    intent: SourceCaptureIntent,
    bindings: Vec<SourceToolBinding>,
    tool_call: (String, String, Value),
    captured_at_epoch: u64,
    reason: &'static str,
}

/// Wrap the session's own source hooks: the ORIGINAL validation runs first;
/// only after it succeeds is the data-only raw record extracted. The wrapped
/// resolve also refuses every later resolution instead of declining quietly.
fn retained_source_hooks(
    session: Arc<Occ3NativeSession>,
    retained: Arc<Mutex<Option<RetainedRecord>>>,
    capture_sentinel: Arc<AtomicBool>,
    resolve_calls: Arc<AtomicUsize>,
    completed_calls: Arc<AtomicUsize>,
    lease_slot: Arc<Mutex<Option<SourceCaptureLease>>>,
) -> SourceCaptureHooks {
    let original = session.source_hooks();
    let original_resolve = original.resolve.clone();
    let original_completed = original.completed.clone();
    let used = Arc::new(AtomicBool::new(false));
    SourceCaptureHooks {
        resolve: Arc::new(move |request| {
            let original_resolve = original_resolve.clone();
            let used = used.clone();
            let calls = resolve_calls.clone();
            Box::pin(async move {
                calls.fetch_add(1, Ordering::SeqCst);
                if used.swap(true, Ordering::SeqCst) {
                    return Err(anyhow::anyhow!(
                        "retention ceiling: no second source resolution"
                    ));
                }
                match original_resolve(request).await {
                    Ok(Some(lease)) => Ok(Some(lease)),
                    Ok(None) => Err(anyhow::anyhow!(
                        "retention ceiling: armed source resolution must not decline"
                    )),
                    Err(error) => Err(error),
                }
            })
        }),
        completed: Arc::new(move |completion| {
            let original_completed = original_completed.clone();
            let retained = retained.clone();
            let sentinel = capture_sentinel.clone();
            let calls = completed_calls.clone();
            let lease_slot = lease_slot.clone();
            Box::pin(async move {
                // Original session validation (Captured + single handoff binding).
                original_completed(completion.clone()).await?;
                calls.fetch_add(1, Ordering::SeqCst);
                let lease = completion.lease.clone();
                let receipt = lease.receipt().ok_or_else(|| {
                    anyhow::anyhow!("retained record requires a captured receipt")
                })?;
                let tool_call = completion
                    .assistant_message
                    .content
                    .iter()
                    .find_map(|block| match block {
                        pi_ai::types::ContentBlock::ToolCall(call) => Some((
                            call.id.clone(),
                            call.name.clone(),
                            Value::Object(call.arguments.clone()),
                        )),
                        _ => None,
                    })
                    .ok_or_else(|| {
                        anyhow::anyhow!("retained record requires the handoff tool call")
                    })?;
                // Non-consuming binding validation before recording.
                lease.validate_tool(&tool_call.0, &tool_call.1, &tool_call.2)?;
                *retained.lock().unwrap() = Some(RetainedRecord {
                    raw_source_handle: receipt.source_handle.clone(),
                    receipt,
                    intent: lease.intent().clone(),
                    bindings: lease.tool_bindings(),
                    tool_call,
                    captured_at_epoch: now(),
                    reason: "planned_restart",
                });
                *lease_slot.lock().unwrap() = Some(lease);
                sentinel.store(true, Ordering::SeqCst);
                Ok(())
            })
        }),
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum StubBehavior {
    Terminate,
    NoTerminate,
    Error,
}

/// Replace ONLY `execute` on the already-installed handoff tool: the exact
/// name/label/description/parameters/prepare_arguments/execution_mode are
/// preserved, and the original execute closure is dropped, never callable.
fn retained_stub_tool(
    real: AgentTool,
    behavior: StubBehavior,
    lease_slot: Arc<Mutex<Option<SourceCaptureLease>>>,
    executed: Arc<AtomicUsize>,
    invocation: Arc<Mutex<Option<(String, Value)>>>,
) -> AgentTool {
    let mut stub = real.clone();
    stub.execute = Arc::new(move |tool_id, args, _signal, _update| {
        let lease_slot = lease_slot.clone();
        let executed = executed.clone();
        let invocation = invocation.clone();
        Box::pin(async move {
            executed.fetch_add(1, Ordering::SeqCst);
            let lease = lease_slot.lock().unwrap().take().ok_or_else(|| {
                anyhow::anyhow!("retained stub: the completed hook never handed over the lease")
            })?;
            if lease.state() != SourceCaptureState::Captured {
                return Err(anyhow::anyhow!("retained stub: lease is not Captured"));
            }
            if lease.receipt().is_none() {
                return Err(anyhow::anyhow!("retained stub: receipt missing"));
            }
            lease.validate_tool(&tool_id, NATIVE_HANDOFF_TOOL, &args)?;
            let object = args
                .as_object()
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("retained stub: arguments must be an object"))?;
            if object.len() != 2
                || object.get("receiver_role").and_then(Value::as_str) != Some("child")
                || object.get("receiver_name").and_then(Value::as_str) != Some("receiver")
            {
                return Err(anyhow::anyhow!(
                    "retained stub: unexpected receiver arguments"
                ));
            }
            *invocation.lock().unwrap() = Some((tool_id, args.clone()));
            match behavior {
                StubBehavior::Terminate => Ok(AgentToolResult {
                    content: vec![ContentBlock::text(
                        "retained negative control: deliberate stop",
                    )],
                    details: json!({"retained_stop": true}),
                    is_error: None,
                    terminate: Some(true),
                }),
                StubBehavior::NoTerminate => Ok(AgentToolResult {
                    content: vec![ContentBlock::text(
                        "retained control: stub forgot terminate",
                    )],
                    details: json!({"retained_stop": false}),
                    is_error: None,
                    terminate: None,
                }),
                StubBehavior::Error => Err(anyhow::anyhow!("retained control: stub error")),
            }
        })
    });
    stub
}

// ---------------------------------------------------------------------------
// Family builder
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum SourceTool {
    /// Keep the real family-controller tool (real dispatch path).
    Real,
    /// Replace the source tool with one bound to the queue-only proof
    /// controller: admission is real, the consumption attempt is scheduled
    /// by the test itself.
    QueueController,
    /// Retained negative control: wrapped hooks + execute-replaced stub.
    RetainedStub(StubBehavior),
}

struct FamilyConfig {
    mode: FixtureMode,
    source_tool: SourceTool,
    source_ceiling: Option<u64>,
    count_receiver_resolver: bool,
}

impl Default for FamilyConfig {
    fn default() -> Self {
        Self {
            mode: FixtureMode::default(),
            source_tool: SourceTool::Real,
            source_ceiling: None,
            count_receiver_resolver: false,
        }
    }
}

struct ProofFamily {
    family: Arc<IsolatedNativeFamily>,
    source: Arc<Occ3NativeSession>,
    receiver: Arc<Occ3NativeSession>,
    source_agent: Arc<Agent>,
    receiver_agent: Arc<Agent>,
    connector: Arc<SglangOcc3Connector>,
    state: Arc<FixtureState>,
    server: tokio::task::JoinHandle<()>,
    endpoint: String,
    retained_record: Arc<Mutex<Option<RetainedRecord>>>,
    capture_sentinel: Arc<AtomicBool>,
    resolve_calls: Arc<AtomicUsize>,
    completed_calls: Arc<AtomicUsize>,
    stub_executed: Arc<AtomicUsize>,
    stub_invocation: Arc<Mutex<Option<(String, Value)>>>,
    queued_delivery: Arc<Mutex<Option<(AgentSessionMessagePayload, String)>>>,
    source_ceiling: Option<Arc<OnceCounter>>,
    receiver_resolver_calls: Arc<AtomicUsize>,
    receiver_leased: Arc<AtomicUsize>,
}

impl ProofFamily {
    fn requests(&self) -> Vec<(String, Value)> {
        self.state.records.lock().unwrap().requests.clone()
    }

    fn chat_bodies(&self) -> Vec<Value> {
        self.requests()
            .into_iter()
            .filter(|(path, _)| path == "/v1/chat/completions")
            .map(|(_, body)| body)
            .collect()
    }

    fn occ3_bodies(&self) -> Vec<Value> {
        self.chat_bodies()
            .into_iter()
            .filter(|body| body.get("occ3").is_some())
            .collect()
    }
}

/// Queue-only admission double: the real receiver inbox admission, without
/// scheduling the consumption. The recorded payload lets the test drive the
/// receiver's actual prompt itself (after a delay, restart, or model switch).
struct ProofController {
    inner: Arc<dyn AgentSessionMessageController>,
    receiver: Arc<Occ3NativeSession>,
    last_delivery: Arc<Mutex<Option<(AgentSessionMessagePayload, String)>>>,
}

impl AgentSessionMessageController for ProofController {
    fn roster(&self) -> pi_ai::types::BoxFuture<Result<AgentFamilyRosterResult, String>> {
        self.inner.roster()
    }
    fn await_pending_child_publication(
        &self,
        selector: String,
    ) -> pi_ai::types::BoxFuture<Result<Option<String>, String>> {
        self.inner.await_pending_child_publication(selector)
    }
    fn send_agent_message(
        &self,
        input: AgentSessionMessageSendInput,
    ) -> pi_ai::types::BoxFuture<Result<AgentSessionMessageReceipt, String>> {
        self.inner.send_agent_message(input)
    }
    fn authorize_native_handoff(
        &self,
        relationship: String,
        selector: String,
    ) -> pi_ai::types::BoxFuture<Result<AuthorizedNativeHandoff, String>> {
        self.inner.authorize_native_handoff(relationship, selector)
    }
    fn send_prepared_native_handoff(
        &self,
        prepared: PreparedNativeHandoff,
    ) -> pi_ai::types::BoxFuture<Result<AgentSessionMessageReceipt, String>> {
        let receiver = self.receiver.clone();
        let last_delivery = self.last_delivery.clone();
        Box::pin(async move {
            if !receiver.admit(&prepared)? {
                return Err("duplicate queued proof admission".into());
            }
            let payload = prepared.control_payload();
            *last_delivery.lock().unwrap() = Some((payload.clone(), "source-session".into()));
            Ok(create_agent_session_message_receipt(
                &payload,
                &DELIVERY_STATUS_QUEUED.into(),
                &chrono::Utc::now().to_rfc3339(),
            ))
        })
    }
}

fn proof_model(endpoint: &str, id: &str) -> Model {
    let mut model = Model::new(
        id,
        "Fixture",
        API_OPENAI_COMPLETIONS,
        "native-fixture",
        format!("{endpoint}/v1"),
    );
    model.max_tokens = 262144.;
    model.default_max_tokens = Some(131072.);
    model.context_window = 1048576.;
    model.reasoning = true;
    model
}

fn native_permit(endpoint: &str) -> NativeExperimentalPermit {
    NativeExperimentalPermit {
        attempt_id: "offline-retained-proof".into(),
        identity: identity(),
        target: pi_ai::inference_attachment::InferenceStateTarget {
            provider: "native-fixture".into(),
            model_id: "fixture-model".into(),
            base_url: format!("{endpoint}/v1"),
        },
        runtime_id: "runtime-epoch-1".into(),
        runtime_generation: 1,
        access_domain: "isolated-family".into(),
        expires_at_epoch: now() + 240,
        source_profile: SourceCaptureProfile {
            layer_index: 3,
            hidden_dim: 8,
            dtype: "bfloat16".into(),
            layout_digest: "b".repeat(64),
            expected_ranks: 8,
        },
        profile_digest: "c".repeat(64),
        max_capture_bytes: 1048576,
    }
}

fn proof_options(model: Model) -> AgentOptions {
    AgentOptions {
        initial_state: Some(AgentState {
            model,
            thinking_level: ThinkingLevel::Max,
            system_prompt:
                "Native retained-proof fixture. Use occ3_handoff once when armed, then stop.".into(),
            ..Default::default()
        }),
        get_api_key: Some(Arc::new(|_| {
            Box::pin(async { Some("synthetic-fixture-key".into()) })
        })),
        ..Default::default()
    }
}

async fn build_family(config: FamilyConfig) -> ProofFamily {
    let state = Arc::new(FixtureState::default());
    let (endpoint, server) = serve(config.mode.clone(), state.clone()).await;
    let model = proof_model(&endpoint, "fixture-model");
    let connector = Arc::new(
        SglangOcc3Connector::new_experimental(
            SglangOcc3Config {
                endpoint: url::Url::parse(&endpoint).unwrap(),
                api_key: Some("synthetic-fixture-key".into()),
                control_headers: None,
            },
            native_permit(&endpoint),
        )
        .expect("connector"),
    );
    let source = Occ3NativeSession::enroll("source-session", &model, connector.clone())
        .await
        .expect("source enrollment");
    let receiver = Occ3NativeSession::enroll("receiver-session", &model, connector.clone())
        .await
        .expect("receiver enrollment");

    let retained_record = Arc::new(Mutex::new(None));
    let capture_sentinel = Arc::new(AtomicBool::new(false));
    let resolve_calls = Arc::new(AtomicUsize::new(0));
    let completed_calls = Arc::new(AtomicUsize::new(0));
    let stub_executed = Arc::new(AtomicUsize::new(0));
    let stub_invocation = Arc::new(Mutex::new(None));
    let queued_delivery = Arc::new(Mutex::new(None));
    let receiver_resolver_calls = Arc::new(AtomicUsize::new(0));
    let receiver_leased = Arc::new(AtomicUsize::new(0));
    let source_ceiling = config.source_ceiling.map(OnceCounter::new);

    let mut source_options = proof_options(model.clone());
    if let Some(ceiling) = &source_ceiling {
        source_options.before_request = Some(ceiling.hook());
    }
    let source_agent = Agent::new(source_options);
    let receiver_agent = Agent::new(proof_options(model));

    let family = IsolatedNativeFamily::new(vec![
        IsolatedNativeMember {
            catalog: AgentFamilyCatalogEntry {
                id: "source-session".into(),
                name: Some("source".into()),
                depth: 0.,
                ..Default::default()
            },
            endpoint: AgentSessionMessageEndpoint {
                active_session_id: "source-active".into(),
                session_id: "source-session".into(),
                session_name: Some("source".into()),
                ..Default::default()
            },
            agent: source_agent.clone(),
            native: source.clone(),
        },
        IsolatedNativeMember {
            catalog: AgentFamilyCatalogEntry {
                id: "receiver-session".into(),
                name: Some("receiver".into()),
                depth: 1.,
                parent_session_id: Some("source-session".into()),
                ..Default::default()
            },
            endpoint: AgentSessionMessageEndpoint {
                active_session_id: "receiver-active".into(),
                session_id: "receiver-session".into(),
                session_name: Some("receiver".into()),
                ..Default::default()
            },
            agent: receiver_agent.clone(),
            native: receiver.clone(),
        },
    ])
    .expect("isolated family");
    family.install().expect("family install");

    if config.count_receiver_resolver {
        let inner = receiver.receiver_resolver();
        let calls = receiver_resolver_calls.clone();
        let leased = receiver_leased.clone();
        let wrapped: ResolveInferenceState = Arc::new(move |request| {
            let inner = inner.clone();
            let calls = calls.clone();
            let leased = leased.clone();
            Box::pin(async move {
                calls.fetch_add(1, Ordering::SeqCst);
                let result = inner(request).await;
                if matches!(&result, Ok(Some(_))) {
                    leased.fetch_add(1, Ordering::SeqCst);
                }
                result
            })
        });
        receiver_agent.set_resolve_inference_state(Some(wrapped));
    }

    match config.source_tool {
        SourceTool::Real => {}
        SourceTool::QueueController => {
            let controller: Arc<dyn AgentSessionMessageController> = Arc::new(ProofController {
                inner: family
                    .controller("source-session")
                    .expect("source controller"),
                receiver: receiver.clone(),
                last_delivery: queued_delivery.clone(),
            });
            let mut agent_state = source_agent.state();
            agent_state.tools = Some(vec![source.handoff_tool(controller)]);
            source_agent.set_state(agent_state);
        }
        SourceTool::RetainedStub(behavior) => {
            let lease_slot = Arc::new(Mutex::new(None));
            source_agent.set_source_capture_hooks(Some(retained_source_hooks(
                source.clone(),
                retained_record.clone(),
                capture_sentinel.clone(),
                resolve_calls.clone(),
                completed_calls.clone(),
                lease_slot.clone(),
            )));
            // Replace ONLY execute on the already-installed tool.
            let mut agent_state = source_agent.state();
            let tools = agent_state.tools.take().unwrap_or_default();
            let real = tools
                .into_iter()
                .find(|tool| tool.name == NATIVE_HANDOFF_TOOL)
                .expect("handoff tool installed");
            let stub = retained_stub_tool(
                real,
                behavior,
                lease_slot,
                stub_executed.clone(),
                stub_invocation.clone(),
            );
            agent_state.tools = Some(vec![stub]);
            source_agent.set_state(agent_state);
        }
    }

    ProofFamily {
        family,
        source,
        receiver,
        source_agent,
        receiver_agent,
        connector,
        state,
        server,
        endpoint,
        retained_record,
        capture_sentinel,
        resolve_calls,
        completed_calls,
        stub_executed,
        stub_invocation,
        queued_delivery,
        source_ceiling,
        receiver_resolver_calls,
        receiver_leased,
    }
}

async fn arm_and_prompt_source(pf: &ProofFamily) {
    pf.source
        .arm_source(NativeSourceBudget {
            retained_count_limit: 128,
            max_capture_bytes: 1048576,
            deadline: Instant::now() + Duration::from_secs(30),
            max_response_bytes: 65536,
        })
        .expect("arm source");
    pf.source_agent
        .prompt(PromptInput::Text {
            input: "Retained-source fixture prompt. Call occ3_handoff to child receiver. Arguments contain only role and name.".into(),
            images: Vec::new(),
        })
        .await
        .expect("source prompt completes");
}

/// Prompt the receiver with the recorded delivery, exactly like the isolated
/// controller does, but at a moment the test chooses.
async fn prompt_receiver_with_delivery(pf: &ProofFamily) {
    let (payload, speaker) = pf
        .queued_delivery
        .lock()
        .unwrap()
        .clone()
        .expect("queued delivery");
    let delivery = create_agent_session_message_delivery(
        &payload,
        chrono::Utc::now().timestamp_millis(),
        &Default::default(),
        &speaker,
        "receiver-session",
        &mut Occ2StrikeLedger::new(),
        chrono::Utc::now().timestamp_millis() as u64,
    );
    pf.receiver_agent
        .prompt(PromptInput::Messages(vec![AgentMessage::Custom(
            delivery.record,
        )]))
        .await
        .expect("receiver prompt returns");
}

fn last_assistant(agent: &Agent) -> Option<pi_ai::types::AssistantMessage> {
    agent
        .state()
        .messages
        .iter()
        .rev()
        .find_map(|message| match message {
            AgentMessage::Message(pi_ai::types::Message::Assistant(assistant)) => {
                Some(assistant.clone())
            }
            _ => None,
        })
}

async fn cleanup(pf: &ProofFamily) {
    let _ = pf.family.close().await;
    let _ = pf.connector.shutdown().await;
    pf.server.abort();
}

// ---------------------------------------------------------------------------
// Family 1: POSITIVE
// ---------------------------------------------------------------------------

#[tokio::test]
async fn retained_positive_flow_carries_consumes_and_binds_the_real_request() {
    let pf = build_family(FamilyConfig::default()).await;
    arm_and_prompt_source(&pf).await;

    let chat = pf.chat_bodies();
    assert_eq!(
        chat.len(),
        3,
        "source capture, receiver attach, source continuation"
    );
    let occ3 = pf.occ3_bodies();
    assert_eq!(occ3.len(), 1, "exactly one attachment-carrying request");
    let attachment = &occ3[0]["occ3"];
    let publication = pf
        .state
        .records
        .lock()
        .unwrap()
        .published_request
        .clone()
        .unwrap();
    // The attachment reached the request builder and is exactly the published
    // binding: request/message/session/fingerprint/epoch/handle.
    assert_eq!(attachment["state_handle"], json!(RAW_SOURCE_HANDLE));
    assert_eq!(attachment["message_id"], publication["message_id"]);
    assert_eq!(attachment["receiver_session_id"], json!("receiver-session"));
    assert_eq!(attachment["runtime_id"], json!("runtime-epoch-1"));
    assert_eq!(attachment["runtime_generation"], json!(1));
    assert_eq!(
        attachment["model_fingerprint"],
        json!(identity().fingerprint())
    );
    assert_eq!(attachment["method"], json!(METHOD));
    assert_eq!(attachment["connector_id"], json!("sglang-occ3"));
    assert_eq!(attachment["protocol_version"], json!(1));
    assert_eq!(
        attachment["expires_at_epoch"],
        publication["expires_at_epoch"]
    );
    let request_id = attachment["request_id"]
        .as_str()
        .expect("request id")
        .to_string();
    assert!(!request_id.is_empty());
    assert_eq!(
        chat.iter()
            .filter(|body| body.get("occ3_capture").is_some())
            .count(),
        1,
        "exactly one capture request"
    );

    // The bound consumption receipt was returned and echoes the exact attachment.
    let receipts = pf.state.records.lock().unwrap().receiver_receipts.clone();
    assert_eq!(receipts.len(), 1);
    assert_eq!(receipts[0]["attachment"], *attachment);
    assert_eq!(receipts[0]["consumed"], json!(true));
    assert_eq!(receipts[0]["injected_state"], json!(true));
    assert_eq!(receipts[0]["fallback_used"], json!(false));

    // The receiver result reflects use: clean completion, consumed lease.
    assert!(pf.receiver_agent.state().error_message.is_none());
    let last = last_assistant(&pf.receiver_agent).expect("receiver assistant message");
    assert_eq!(last.stop_reason, "stop");
    let audit = pf.receiver.audit();
    assert_eq!(audit["receiver"].as_array().unwrap().len(), 1);
    assert_eq!(audit["receiver"][0]["lease_state"], json!("Consumed"));
    assert_eq!(audit["receiver"][0]["status"], json!("CONSUMED"));
    assert_eq!(audit["receiver"][0]["request_id"], json!(request_id));
    assert_eq!(
        audit["receiver"][0]["handle_sha256"],
        json!(sha256(RAW_SOURCE_HANDLE.as_bytes()))
    );
    let message_id = publication["message_id"].as_str().unwrap().to_string();
    assert_eq!(
        pf.receiver.outcome(&message_id).expect("outcome"),
        Occ3DeliveryStatus::Consumed
    );
    let receipts_journal = pf.family.receipts();
    assert_eq!(receipts_journal.len(), 1);
    assert_eq!(receipts_journal[0].id, message_id);
    assert_eq!(receipts_journal[0].delivery_status, "delivered");

    // No replay and no re-attachment on a later ordinary receiver turn.
    pf.receiver_agent
        .prompt(PromptInput::Text {
            input: "ordinary follow-up after consumption".into(),
            images: Vec::new(),
        })
        .await
        .expect("follow-up prompt");
    assert!(pf.receiver_agent.state().error_message.is_none());
    assert_eq!(pf.occ3_bodies().len(), 1, "the attachment is single-use");
    assert_eq!(pf.chat_bodies().len(), 4);
    assert_eq!(
        pf.receiver
            .outcome(&message_id)
            .expect("outcome after follow-up"),
        Occ3DeliveryStatus::Consumed
    );

    cleanup(&pf).await;
}

#[tokio::test]
async fn retained_receipt_binding_is_enforced_field_by_field_on_the_real_request() {
    for mode in [
        ReceiptMode::WrongHandle,
        ReceiptMode::WrongRequest,
        ReceiptMode::WrongMessage,
        ReceiptMode::WrongSession,
        ReceiptMode::WrongFingerprint,
        ReceiptMode::WrongExpiry,
        ReceiptMode::NotConsumed,
        ReceiptMode::FallbackUsed,
        ReceiptMode::NotInjected,
        ReceiptMode::Duplicate,
        ReceiptMode::Missing,
    ] {
        let pf = build_family(FamilyConfig {
            mode: FixtureMode {
                receipt: mode,
                short_receipt_expiry: None,
            },
            source_tool: SourceTool::QueueController,
            ..Default::default()
        })
        .await;
        arm_and_prompt_source(&pf).await;
        assert_eq!(
            pf.receiver.audit()["receiver"][0]["status"],
            json!("ADMITTED"),
            "{}: admission happened before the consumption attempt",
            mode.name()
        );
        prompt_receiver_with_delivery(&pf).await;
        // Fail-closed: no clean success, no consumption, no fallback.
        assert!(
            pf.receiver_agent.state().error_message.is_some(),
            "{}: receiver error must be recorded",
            mode.name()
        );
        assert_eq!(
            pf.occ3_bodies().len(),
            1,
            "{}: one attachment request reached the provider",
            mode.name()
        );
        let audit = pf.receiver.audit();
        assert_eq!(
            audit["receiver"][0]["lease_state"],
            json!("Uncertain"),
            "{}",
            mode.name()
        );
        assert_eq!(
            audit["receiver"][0]["status"],
            json!("UNCERTAIN"),
            "{}",
            mode.name()
        );
        let message_id = pf
            .state
            .records
            .lock()
            .unwrap()
            .published_request
            .clone()
            .unwrap()["message_id"]
            .as_str()
            .unwrap()
            .to_string();
        assert_eq!(
            pf.receiver.outcome(&message_id).expect("outcome recorded"),
            Occ3DeliveryStatus::Uncertain,
            "{}",
            mode.name()
        );
        assert!(
            !permits_text_fallback(Occ3DeliveryStatus::Uncertain),
            "{}",
            mode.name()
        );
        assert!(
            pf.family.receipts().is_empty(),
            "{}: no delivered receipt",
            mode.name()
        );
        assert!(
            pf.source_agent.state().error_message.is_none(),
            "{}: the source side completed",
            mode.name()
        );
        if mode == ReceiptMode::WrongHandle {
            // A retry of the same delivery never silently becomes ordinary text.
            prompt_receiver_with_delivery(&pf).await;
            assert!(
                pf.receiver_agent.state().error_message.is_some(),
                "retry refused"
            );
            assert_eq!(pf.occ3_bodies().len(), 1, "no second attachment request");
        }
        cleanup(&pf).await;
    }
}

/// The stale-fixture regression this campaign's baseline hit live: the default
/// `arm_source` arming is `occ3.capture.v2` + `content_tail_v2`, so the bare
/// span-less `occ3.source.v1` wire a pre-fix server answered is refused by the
/// pi-ai verifier. The source turn fails closed and nothing is published,
/// queued or consumed: the refusal is never a silent fallback to text.
#[tokio::test]
async fn stale_v1_source_receipt_is_refused_under_default_v2_arming() {
    let pf = build_family(FamilyConfig {
        mode: FixtureMode {
            receipt: ReceiptMode::StaleV1,
            short_receipt_expiry: None,
        },
        source_tool: SourceTool::QueueController,
        ..Default::default()
    })
    .await;
    pf.source
        .arm_source(NativeSourceBudget {
            retained_count_limit: 128,
            max_capture_bytes: 1048576,
            deadline: Instant::now() + Duration::from_secs(30),
            max_response_bytes: 65536,
        })
        .expect("arm source");
    pf.source_agent
        .prompt(PromptInput::Text {
            input: "Retained-source fixture prompt. Call occ3_handoff to child receiver. Arguments contain only role and name.".into(),
            images: Vec::new(),
        })
        .await
        .expect("source prompt returns");
    assert!(
        pf.source_agent.state().error_message.is_some(),
        "a span-less occ3.source.v1 receipt under v2 arming is refused"
    );
    assert!(
        !pf
            .requests()
            .iter()
            .any(|(path, _)| path.ends_with("/publish")),
        "no publication attempt happened"
    );
    assert!(
        pf.state.records.lock().unwrap().published_request.is_none(),
        "nothing was published"
    );
    assert!(
        pf.queued_delivery.lock().unwrap().is_none(),
        "no handoff delivery was queued for the receiver"
    );
    assert!(
        pf.state.records.lock().unwrap().receiver_receipts.is_empty(),
        "the receiver was never prompted"
    );
    let audit = pf.receiver.audit();
    assert!(
        audit["receiver"].as_array().is_none_or(|rows| rows.is_empty()),
        "receiver inbox stayed empty"
    );
    cleanup(&pf).await;
}

// ---------------------------------------------------------------------------
// Family 2: NO-STATE control
// ---------------------------------------------------------------------------

#[tokio::test]
async fn retained_no_state_control_has_no_attachment_receipt_or_claim() {
    let pf = build_family(FamilyConfig {
        count_receiver_resolver: true,
        ..Default::default()
    })
    .await;
    // Same wiring, but the source is never armed: no capture budget exists.
    pf.source_agent
        .prompt(PromptInput::Text {
            input: "Ordinary fixture prompt without armed capture.".into(),
            images: Vec::new(),
        })
        .await
        .expect("source prompt");
    assert!(pf.source_agent.state().error_message.is_none());
    assert!(
        pf.source.audit()["source"].is_null(),
        "no capture lease exists"
    );

    let controller = pf.family.controller("source-session").expect("controller");
    controller
        .send_agent_message(AgentSessionMessageSendInput {
            target: "receiver".into(),
            message: "ordinary native-family message without retained state".into(),
            receiver_role: None,
        })
        .await
        .expect("normal delivery");

    let chat = pf.chat_bodies();
    assert_eq!(chat.len(), 2, "one source and one receiver request");
    assert!(
        chat.iter()
            .all(|body| body.get("occ3").is_none() && body.get("occ3_capture").is_none()),
        "no attachment and no capture on any request"
    );
    // The receiver resolver really ran and declined without a lease.
    assert_eq!(pf.receiver_resolver_calls.load(Ordering::SeqCst), 1);
    assert_eq!(pf.receiver_leased.load(Ordering::SeqCst), 0);
    assert!(pf.receiver_agent.state().error_message.is_none());
    assert!(
        pf.receiver.audit()["receiver"]
            .as_array()
            .unwrap()
            .is_empty(),
        "no OCC-3 claim recorded"
    );
    assert!(
        pf.state
            .records
            .lock()
            .unwrap()
            .receiver_receipts
            .is_empty(),
        "no receipt was needed or sent"
    );
    assert!(pf.family.receipts().is_empty());
    assert!(
        !pf.requests()
            .iter()
            .any(|(path, _)| path.ends_with("/publish")),
        "no publication control-plane call"
    );
    cleanup(&pf).await;
}

// ---------------------------------------------------------------------------
// Family 3: STALE-NEGATIVE controls
// ---------------------------------------------------------------------------

#[tokio::test]
async fn retained_stale_expired_handle_refuses_consumption_fail_closed() {
    let pf = build_family(FamilyConfig {
        mode: FixtureMode {
            receipt: ReceiptMode::Exact,
            short_receipt_expiry: Some(5),
        },
        source_tool: SourceTool::QueueController,
        ..Default::default()
    })
    .await;
    arm_and_prompt_source(&pf).await;
    assert_eq!(
        pf.receiver.audit()["receiver"][0]["status"],
        json!("ADMITTED")
    );

    // The handle expires before the receiver ever consumes it.
    tokio::time::sleep(Duration::from_secs(7)).await;
    prompt_receiver_with_delivery(&pf).await;
    assert!(
        pf.receiver_agent.state().error_message.is_some(),
        "fail-closed refusal"
    );
    assert_eq!(
        pf.occ3_bodies().len(),
        0,
        "no provider request carried the stale handle"
    );
    let audit = pf.receiver.audit();
    assert_eq!(
        audit["receiver"][0]["status"],
        json!("EXPIRED"),
        "refusal recorded"
    );
    assert!(
        audit["receiver"][0]["lease_state"].is_null(),
        "no lease was ever bound"
    );
    assert!(!permits_text_fallback(Occ3DeliveryStatus::Expired));
    assert!(pf.family.receipts().is_empty());

    // A retry of the same delivery is refused again: no silent text fallback.
    prompt_receiver_with_delivery(&pf).await;
    assert!(pf.receiver_agent.state().error_message.is_some());
    assert_eq!(pf.occ3_bodies().len(), 0);
    assert_eq!(
        pf.chat_bodies().len(),
        2,
        "only the source capture and continuation ran"
    );
    // An unrelated fresh user turn recovers: BOTH expired delivery records
    // stay in the retained history, none is freshly supplied, so the turn
    // proceeds as an ordinary chat request with no attachment and no error.
    pf.receiver_agent
        .prompt(PromptInput::Text {
            input: "Unrelated follow-up after the expired delivery.".into(),
            images: Vec::new(),
        })
        .await
        .expect("the unrelated turn must complete");
    assert!(
        pf.receiver_agent.state().error_message.is_none(),
        "history-only expired records never fail a later turn"
    );
    assert_eq!(
        pf.occ3_bodies().len(),
        0,
        "no provider request ever carried the stale handle"
    );
    assert_eq!(
        pf.chat_bodies().len(),
        3,
        "the unrelated turn ran as an ordinary chat request"
    );
    assert_eq!(
        pf.receiver.audit()["receiver"][0]["status"],
        json!("EXPIRED"),
        "the expired record is preserved terminal"
    );
    cleanup(&pf).await;
}

#[tokio::test]
async fn retained_stale_evicted_receiver_session_refuses_consumption_fail_closed() {
    let pf = build_family(FamilyConfig {
        source_tool: SourceTool::QueueController,
        ..Default::default()
    })
    .await;
    arm_and_prompt_source(&pf).await;
    assert_eq!(
        pf.receiver.audit()["receiver"][0]["status"],
        json!("ADMITTED")
    );

    // Eviction/restart simulation: the receiver's process-local native
    // session state is invalidated before consumption.
    pf.receiver.invalidate();
    prompt_receiver_with_delivery(&pf).await;
    assert!(
        pf.receiver_agent.state().error_message.is_some(),
        "fail-closed refusal"
    );
    assert_eq!(
        pf.occ3_bodies().len(),
        0,
        "no provider request carried the handle"
    );
    let audit = pf.receiver.audit();
    assert_eq!(
        audit["receiver"][0]["status"],
        json!("FAILED"),
        "refusal recorded"
    );
    assert!(!permits_text_fallback(Occ3DeliveryStatus::Failed));
    assert!(pf.family.receipts().is_empty());
    cleanup(&pf).await;
}

// ---------------------------------------------------------------------------
// Family 4: ONCE-COUNTER
// ---------------------------------------------------------------------------

#[tokio::test]
async fn retained_once_counter_is_unconditional_and_survives_prompt_index_reset() {
    // Sentinel absent, malformed, or present: the refusal must be identical.
    // The per-prompt request index resets to 0 on every prompt; the harness
    // counter, not the index, is the authority.
    for variant in ["absent", "malformed", "present", "counting"] {
        let allowed: u64 = if variant == "counting" { 2 } else { 1 };
        let state = Arc::new(FixtureState::default());
        let (endpoint, server) = serve(FixtureMode::default(), state.clone()).await;
        let counter = OnceCounter::new(allowed);
        match variant {
            "malformed" => {
                *counter.sentinel.lock().unwrap() = Some(json!({"sentinel": "garbage-not-a-bool"}))
            }
            "present" => *counter.sentinel.lock().unwrap() = Some(json!(true)),
            _ => {}
        }
        let mut options = proof_options(proof_model(&endpoint, "fixture-model"));
        options.before_request = Some(counter.hook());
        let agent = Agent::new(options);

        let prompts = allowed + 1;
        for index in 0..prompts {
            agent
                .prompt(PromptInput::Text {
                    input: format!("once-counter probe {variant} {index}"),
                    images: Vec::new(),
                })
                .await
                .expect("prompt returns even when refused pre-dispatch");
        }
        let chat: Vec<Value> = state
            .records
            .lock()
            .unwrap()
            .requests
            .iter()
            .filter(|(path, _)| path == "/v1/chat/completions")
            .map(|(_, body)| body.clone())
            .collect();
        assert_eq!(
            chat.len() as u64,
            allowed,
            "{variant}: only allowed requests reached the provider"
        );
        assert_eq!(
            counter.fired.load(Ordering::SeqCst),
            1,
            "{variant}: exactly one refusal"
        );
        let indices = counter.indices.lock().unwrap().clone();
        assert_eq!(
            indices,
            vec![0u64; (allowed + 1) as usize],
            "{variant}: index resets per prompt; the counter refused anyway"
        );
        assert!(
            agent.state().error_message.is_some(),
            "{variant}: not a clean success"
        );
        let refusals = counter.refusal_sentinels.lock().unwrap().clone();
        assert_eq!(refusals.len(), 1);
        match variant {
            "absent" => assert_eq!(refusals[0], None),
            "malformed" => assert!(refusals[0]
                .as_deref()
                .unwrap()
                .contains("garbage-not-a-bool")),
            "present" => assert_eq!(refusals[0].as_deref(), Some("true")),
            _ => {}
        }
        server.abort();
    }
}

#[tokio::test]
async fn retained_once_counter_ceiling_blocks_second_request_after_stub_failure() {
    for behavior in [StubBehavior::NoTerminate, StubBehavior::Error] {
        let pf = build_family(FamilyConfig {
            source_tool: SourceTool::RetainedStub(behavior),
            source_ceiling: Some(1),
            ..Default::default()
        })
        .await;
        arm_and_prompt_source(&pf).await;
        pf.source_agent.wait_for_idle().await;

        let chat = pf.chat_bodies();
        assert_eq!(
            chat.len(),
            1,
            "{behavior:?}: the ceiling refused the second provider request pre-dispatch"
        );
        assert!(
            chat[0].get("occ3_capture").is_some(),
            "the one request was the capture"
        );
        let ceiling = pf.source_ceiling.as_ref().expect("ceiling");
        assert_eq!(
            ceiling.fired.load(Ordering::SeqCst),
            1,
            "{behavior:?}: ceiling fired exactly once"
        );
        assert_eq!(
            ceiling.indices.lock().unwrap().len(),
            2,
            "{behavior:?}: it fired before EVERY request attempt"
        );
        assert_eq!(
            pf.resolve_calls.load(Ordering::SeqCst),
            1,
            "{behavior:?}: no second resolution"
        );
        // The capture sentinel WAS set; the refusal is still unconditional.
        assert!(pf.capture_sentinel.load(Ordering::SeqCst));
        assert!(pf.completed_calls.load(Ordering::SeqCst) == 1);
        assert!(
            pf.source_agent.state().error_message.is_some(),
            "{behavior:?}: never a clean success"
        );
        assert_eq!(pf.stub_executed.load(Ordering::SeqCst), 1);
        // The retained capture is still Captured and unpublished.
        assert_eq!(pf.source.audit()["source"]["state"], json!("Captured"));
        assert!(!pf
            .requests()
            .iter()
            .any(|(path, _)| path.ends_with("/publish")));
        cleanup(&pf).await;
    }
}

// ---------------------------------------------------------------------------
// Family 5: RAW-RETENTION
// ---------------------------------------------------------------------------

#[tokio::test]
async fn retained_record_keeps_raw_handle_and_full_receipt_in_memory_only() {
    let pf = build_family(FamilyConfig {
        source_tool: SourceTool::RetainedStub(StubBehavior::Terminate),
        source_ceiling: Some(1),
        ..Default::default()
    })
    .await;
    arm_and_prompt_source(&pf).await;
    pf.source_agent.wait_for_idle().await;

    // Deliberate stop: exactly one provider request (the capture), and the
    // loop ended by terminate, not by exhaustion or an error relabeled success.
    let chat = pf.chat_bodies();
    assert_eq!(chat.len(), 1);
    assert!(chat[0].get("occ3_capture").is_some());
    assert!(pf.source_agent.state().error_message.is_none());
    assert_eq!(pf.stub_executed.load(Ordering::SeqCst), 1);
    let (tool_id, args) = pf
        .stub_invocation
        .lock()
        .unwrap()
        .clone()
        .expect("stub invocation");
    assert_eq!(tool_id, "actual-tool-1");
    assert_eq!(
        args,
        json!({"receiver_role": "child", "receiver_name": "receiver"})
    );
    assert_eq!(pf.completed_calls.load(Ordering::SeqCst), 1);
    assert!(pf.capture_sentinel.load(Ordering::SeqCst));
    assert_eq!(pf.resolve_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        pf.source_ceiling
            .as_ref()
            .unwrap()
            .fired
            .load(Ordering::SeqCst),
        0
    );

    // Zero publication or retirement: the retained control is unpublished.
    assert!(!pf
        .requests()
        .iter()
        .any(|(path, _)| path.ends_with("/publish") || path.ends_with("/retire")));

    // The lease stays Captured, unclaimed, unpublished.
    let audit = pf.source.audit();
    assert_eq!(audit["source"]["state"], json!("Captured"));
    assert_eq!(audit["source"]["tool_claimed"], json!(false));
    assert!(audit["source"]["published_message"].is_null());

    // RAW retention: the private record keeps the raw handle and the full
    // receipt, not the hash-only audit projection.
    let record = pf
        .retained_record
        .lock()
        .unwrap()
        .clone()
        .expect("retained record");
    assert_eq!(record.raw_source_handle, RAW_SOURCE_HANDLE);
    let receipt = &record.receipt;
    assert_eq!(receipt.source_handle, RAW_SOURCE_HANDLE);
    assert_eq!(receipt.schema, "occ3.source.v2");
    let span = receipt.span.as_ref().expect("v2 receipt carries the span block");
    assert_eq!(span.span_policy, record.intent.span_policy);
    assert_eq!(span.span_reasoning_terminator, "thinking_end");
    assert_eq!(span.span_tool_call_count, 1);
    assert_eq!(span.span_control_count, 2);
    assert!(span.span_contains_tool_call && span.span_contains_final_content);
    assert_eq!(receipt.native_rid, "retained-source-rid");
    assert_eq!(receipt.response_id, "retained-source-rid");
    assert_eq!(receipt.capture_commit_id, "retained-proof-capture-commit");
    assert_eq!(
        receipt.intent_digest,
        record.intent.digest().expect("intent digest")
    );
    assert_eq!(receipt.output_sha256.len(), 64);
    assert_eq!(receipt.retained_count, 8);
    assert_eq!(receipt.layer_index, 3);
    assert_eq!(receipt.hidden_dim, 8);
    assert_eq!(receipt.dtype, "bfloat16");
    assert!(receipt.expires_at_epoch > now());
    assert_eq!(record.bindings.len(), 1);
    assert_eq!(record.bindings[0].tool_name, NATIVE_HANDOFF_TOOL);
    assert_eq!(record.bindings[0].tool_call_id, "actual-tool-1");
    assert_eq!(record.tool_call.0, "actual-tool-1");
    assert_eq!(record.tool_call.1, NATIVE_HANDOFF_TOOL);
    assert_eq!(record.reason, "planned_restart");
    assert!(record.captured_at_epoch > 0 && record.captured_at_epoch <= now());
    let record_json = serde_json::to_string(&record.receipt).expect("receipt serializes");
    assert!(
        record_json.contains(RAW_SOURCE_HANDLE),
        "raw handle, not a hash substitute"
    );

    // The audit projection is hash-only and never carries the raw value.
    let audit_string = audit.to_string();
    assert!(!audit_string.contains(RAW_SOURCE_HANDLE));
    assert_eq!(
        audit["source"]["receipt"]["handle_sha256"],
        json!(sha256(RAW_SOURCE_HANDLE.as_bytes()))
    );
    assert_ne!(
        record.raw_source_handle,
        sha256(RAW_SOURCE_HANDLE.as_bytes())
    );

    // Retention holds through the end of the proof: still Captured, and this
    // harness never closes, cancels, or remotely retires the retained family.
    assert_eq!(pf.source.audit()["source"]["state"], json!("Captured"));
    pf.server.abort();
}

// ---------------------------------------------------------------------------
// Family 6: WRONG-MODEL / WRONG-EPOCH at consume time
// ---------------------------------------------------------------------------

#[tokio::test]
async fn retained_wrong_model_is_refused_at_consume_time() {
    let pf = build_family(FamilyConfig {
        source_tool: SourceTool::QueueController,
        ..Default::default()
    })
    .await;
    arm_and_prompt_source(&pf).await;
    assert_eq!(
        pf.receiver.audit()["receiver"][0]["status"],
        json!("ADMITTED")
    );

    // The receiver switches to a different model before consuming the handle.
    let other_model = proof_model(&pf.endpoint, "fixture-model-b");
    let mut state = pf.receiver_agent.state();
    state.model = other_model;
    pf.receiver_agent.set_state(state);

    prompt_receiver_with_delivery(&pf).await;
    assert!(
        pf.receiver_agent.state().error_message.is_some(),
        "fail-closed refusal"
    );
    assert_eq!(
        pf.occ3_bodies().len(),
        0,
        "no provider request carried the handle"
    );
    let audit = pf.receiver.audit();
    assert_eq!(
        audit["receiver"][0]["status"],
        json!("FAILED"),
        "the session invalidated itself"
    );
    assert!(audit["receiver"][0]["lease_state"].is_null());
    assert!(!permits_text_fallback(Occ3DeliveryStatus::Failed));
    assert!(pf.family.receipts().is_empty());
    cleanup(&pf).await;
}

#[tokio::test]
async fn retained_wrong_epoch_runtime_restart_is_refused_at_consume_time() {
    let pf = build_family(FamilyConfig {
        source_tool: SourceTool::QueueController,
        ..Default::default()
    })
    .await;
    arm_and_prompt_source(&pf).await;
    assert_eq!(
        pf.receiver.audit()["receiver"][0]["status"],
        json!("ADMITTED")
    );

    // The runtime restarts between publication and consumption: the handle's
    // epoch no longer matches the runtime the receiver re-verifies.
    pf.state.restarted.store(true, Ordering::SeqCst);
    prompt_receiver_with_delivery(&pf).await;
    assert!(
        pf.receiver_agent.state().error_message.is_some(),
        "fail-closed refusal"
    );
    assert_eq!(
        pf.occ3_bodies().len(),
        0,
        "no provider request carried the stale-epoch handle"
    );
    let audit = pf.receiver.audit();
    assert!(
        audit["receiver"][0]["lease_state"].is_null(),
        "no lease was bound"
    );
    assert_eq!(
        audit["receiver"][0]["status"],
        json!("DELIVERED"),
        "claimed but never consumable"
    );

    // A retry is refused as uncertain: no silent text fallback or regeneration.
    prompt_receiver_with_delivery(&pf).await;
    assert!(pf.receiver_agent.state().error_message.is_some());
    assert_eq!(pf.occ3_bodies().len(), 0);
    assert!(pf.family.receipts().is_empty());
    cleanup(&pf).await;
}
