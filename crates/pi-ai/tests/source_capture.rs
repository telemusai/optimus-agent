//! Offline real-provider fixtures. These do not prove GPU state capture or use.
use futures::FutureExt;
use pi_ai::inference_attachment::InferenceStateTarget;
use pi_ai::providers::openai_completions::{
    stream_openai_completions, stream_simple_openai_completions, OpenAICompletionsOptions,
};
use pi_ai::source_capture::*;
use pi_ai::types::*;
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

fn epoch() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}
fn intent() -> SourceCaptureIntent {
    SourceCaptureIntent {
        schema: SOURCE_CAPTURE_SCHEMA.into(),
        capture_id: "capture-1".into(),
        source_session_id: "source-1".into(),
        source_request_id: "client-request-1".into(),
        turn_nonce: "turn-1".into(),
        model_fingerprint: "a".repeat(64),
        runtime_id: "runtime-1".into(),
        runtime_generation: 7,
        method: "statebridge-prenorm-v1".into(),
        retained_count_limit: 128,
        max_capture_bytes: 4096,
        expires_at_epoch: epoch() + 300,
        span_policy: SOURCE_SPAN_POLICY.into(),
        capture_mode: String::new(),
        label_anchor_alpha: None,
    }
}
fn profile() -> SourceCaptureProfile {
    SourceCaptureProfile {
        layer_index: 3,
        hidden_dim: 8,
        dtype: "bfloat16".into(),
        layout_digest: "b".repeat(64),
        expected_ranks: 8,
    }
}
fn body() -> String {
    let chunks = [
        json!({"id":"native-1","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"real native reasoning"},"finish_reason":null}]}),
        json!({"id":"native-1","choices":[{"index":0,"delta":{"content":"source-only fact S47"},"finish_reason":null}]}),
        json!({"id":"native-1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call-handoff","type":"function","function":{"name":"agent_message","arguments":"{\"control\":\"continue\"}"}}]},"finish_reason":null}]}),
        json!({"id":"native-1","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}),
        json!({"id":"native-1","choices":[],"usage":{"prompt_tokens":9,"completion_tokens":4,"total_tokens":13}}),
    ];
    let mut result = chunks
        .iter()
        .map(|v| format!("data: {v}\r\n\r\n"))
        .collect::<String>();
    result.push_str("data: [DONE]\r\n\r\n");
    result
}
fn receipt(intent: &SourceCaptureIntent, body: &str) -> SourceCaptureReceipt {
    SourceCaptureReceipt {
        schema: SOURCE_RECEIPT_SCHEMA.into(),
        intent_digest: intent.digest().unwrap(),
        source_handle: "owner-only-handle".into(),
        native_rid: "native-1".into(),
        response_id: "native-1".into(),
        capture_commit_id: "commit-1".into(),
        accepted_span_start: 2,
        accepted_span_end: 4,
        retained_count: 2,
        predictor_position_offset: -1,
        representation: SOURCE_REPRESENTATION.into(),
        layer_index: 3,
        hidden_dim: 8,
        dtype: "bfloat16".into(),
        layout_digest: "b".repeat(64),
        ready_ranks: 8,
        expected_ranks: 8,
        accepted_sequence_digest: "c".repeat(64),
        label_anchor_alpha: Default::default(),
        output_sha256: sha256(body.as_bytes()),
        expires_at_epoch: intent.expires_at_epoch,
        byte_count: 32,
        span: None,
    }
}
#[derive(Clone)]
enum Reply {
    Good,
    Missing,
    Duplicate,
    Oversized,
    Field(&'static str, Value),
    Body(String),
    Encoding,
    Disconnect,
    Status,
    HoldEof,
    /// Pre-event refusal: status line, body and extra header lines. No receipt.
    Refusal(&'static str, String, &'static str),
    /// Busy 503 whose body never finishes: headers plus a partial body, then
    /// the connection is held open.
    StalledRefusal,
    /// One reply per accepted request, in order; the last one repeats.
    Script(Vec<Reply>),
}
struct Fixture {
    model: Model,
    requests: Arc<Mutex<Vec<(String, Value)>>>,
    seen: Arc<tokio::sync::Notify>,
    release: Arc<tokio::sync::Notify>,
    task: tokio::task::JoinHandle<()>,
}
impl Drop for Fixture {
    fn drop(&mut self) {
        self.task.abort();
    }
}
impl Fixture {
    async fn new(script: Reply) -> Self {
        let listener = tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
            .await
            .unwrap();
        let address = listener.local_addr().unwrap();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let capture = requests.clone();
        let seen = Arc::new(tokio::sync::Notify::new());
        let notify = seen.clone();
        let release = Arc::new(tokio::sync::Notify::new());
        let gate = release.clone();
        let task = tokio::spawn(async move {
            let mut served = 0usize;
            loop {
                let (mut socket, _) = listener.accept().await.unwrap();
                let reply = match &script {
                    Reply::Script(steps) => steps[served.min(steps.len() - 1)].clone(),
                    other => other.clone(),
                };
                served += 1;
                let mut bytes = Vec::new();
                let (head, payload) = loop {
                    let mut chunk = [0; 4096];
                    let n = socket.read(&mut chunk).await.unwrap();
                    if n == 0 {
                        return;
                    }
                    bytes.extend_from_slice(&chunk[..n]);
                    assert!(bytes.len() < 262144);
                    if let Some(end) = bytes.windows(4).position(|v| v == b"\r\n\r\n") {
                        let head = String::from_utf8(bytes[..end].to_vec()).unwrap();
                        let len: usize = head
                            .lines()
                            .filter_map(|l| l.split_once(':'))
                            .find(|(k, _)| k.eq_ignore_ascii_case("content-length"))
                            .unwrap()
                            .1
                            .trim()
                            .parse()
                            .unwrap();
                        if bytes.len() >= end + 4 + len {
                            break (
                                head,
                                serde_json::from_slice::<Value>(&bytes[end + 4..end + 4 + len])
                                    .unwrap(),
                            );
                        }
                    }
                };
                capture.lock().unwrap().push((head, payload.clone()));
                notify.notify_one();
                if matches!(reply, Reply::Disconnect) {
                    continue;
                }
                if matches!(reply, Reply::StalledRefusal) {
                    let partial = "{\"error\":{\"message\":\"scheduler_bu";
                    let wire = format!("HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\nContent-Length: 4000\r\n\r\n{partial}");
                    let _ = socket.write_all(wire.as_bytes()).await;
                    tokio::time::sleep(std::time::Duration::from_secs(30)).await;
                    continue;
                }
                if let Reply::Refusal(status, body, extra) = &reply {
                    let wire = format!("HTTP/1.1 {status}\r\nContent-Type: application/json\r\n{extra}Content-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
                    let _ = socket.write_all(wire.as_bytes()).await;
                    continue;
                }
                let output = match &reply {
                    Reply::Body(v) => v.clone(),
                    _ => body(),
                };
                let mut headers = String::new();
                if let Some(value) = payload.get("occ3_capture") {
                    let intent: SourceCaptureIntent =
                        serde_json::from_value(value.clone()).unwrap();
                    let mut receipt = serde_json::to_value(receipt(&intent, &output)).unwrap();
                    if let Reply::Field(field, value) = &reply {
                        receipt[*field] = value.clone();
                    }
                    if !matches!(reply, Reply::Missing) {
                        let value = if matches!(reply, Reply::Oversized) {
                            "x".repeat(2049)
                        } else {
                            receipt.to_string()
                        };
                        headers.push_str(&format!("{SOURCE_RECEIPT_HEADER}: {value}\r\n"));
                        if matches!(reply, Reply::Duplicate) {
                            headers.push_str(&format!("{SOURCE_RECEIPT_HEADER}: {value}\r\n"));
                        }
                    }
                }
                if matches!(reply, Reply::Encoding) {
                    headers.push_str("Content-Encoding: gzip\r\n");
                }
                let status = if matches!(reply, Reply::Status) {
                    "503 Unavailable"
                } else {
                    "200 OK"
                };
                let wire = if matches!(reply, Reply::HoldEof) {
                    format!("HTTP/1.1 {status}\r\n{headers}Content-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n{:x}\r\n{output}\r\n",output.len())
                } else {
                    format!("HTTP/1.1 {status}\r\n{headers}Content-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{output}",output.len())
                };
                let _ = socket.write_all(wire.as_bytes()).await;
                if matches!(reply, Reply::HoldEof) {
                    gate.notified().await;
                    let _ = socket.write_all(b"0\r\n\r\n").await;
                }
            }
        });
        Self {
            model: Model {
                id: "fixture-model".into(),
                name: "fixture".into(),
                api: API_OPENAI_COMPLETIONS.into(),
                provider: "fixture-provider".into(),
                base_url: format!("http://{address}/v1"),
                reasoning: true,
                max_tokens: 4096.0,
                input: vec![InputModality::Text],
                ..Default::default()
            },
            requests,
            seen,
            release,
            task,
        }
    }
    fn count(&self) -> usize {
        self.requests.lock().unwrap().len()
    }
    fn lease(&self) -> SourceCaptureLease {
        // Long enough for one resend: delay + the 25 s admission headroom.
        self.lease_with(Duration::from_secs(40))
    }
    fn lease_with(&self, budget: Duration) -> SourceCaptureLease {
        SourceCaptureLease::new(
            InferenceStateTarget {
                provider: self.model.provider.clone(),
                model_id: self.model.id.clone(),
                base_url: self.model.base_url.clone(),
            },
            intent(),
            profile(),
            SourceCaptureLimits {
                deadline: Instant::now() + budget,
                max_response_bytes: 65536,
            },
        )
        .unwrap()
    }
}
fn context() -> Context {
    Context::new(
        Some("native system".into()),
        vec![Message::User(UserMessage::new(
            UserContent::Text("native source task".into()),
            1,
        ))],
        Some(vec![Tool {
            name: "agent_message".into(),
            description: "native handoff".into(),
            parameters: json!({"type":"object","properties":{"control":{"type":"string"}}}),
        }]),
    )
}
fn options(lease: Option<SourceCaptureLease>) -> OpenAICompletionsOptions {
    OpenAICompletionsOptions {
        stream: StreamOptions {
            api_key: Some("synthetic-fixture-key".into()),
            max_tokens: Some(800.0),
            timeout_ms: Some(3000.0),
            cache_retention: Some("none".into()),
            source_capture: lease,
            ..Default::default()
        },
        reasoning_effort: Some("high".into()),
        ..Default::default()
    }
}
async fn collect(
    stream: &pi_ai::utils::event_stream::AssistantMessageEventStream,
) -> Vec<AssistantMessageEvent> {
    let mut result = Vec::new();
    while let Some(event) = tokio::time::timeout(Duration::from_secs(10), stream.next())
        .await
        .expect("fixture deadline")
    {
        result.push(event);
    }
    result
}
fn assert_uncertain(events: &[AssistantMessageEvent], lease: &SourceCaptureLease) {
    assert_eq!(lease.state(), SourceCaptureState::Uncertain);
    assert!(lease.receipt().is_none());
    assert_eq!(events.len(), 1, "no partial output before proof");
    let AssistantMessageEvent::Error { error, .. } = &events[0] else {
        panic!("expected failure")
    };
    let wire = serde_json::to_string(error).unwrap();
    assert!(!wire.contains("owner-only-handle"));
    assert!(wire.contains("request_interrupted"));
}
#[tokio::test]
async fn source_actual_normal_and_simple_native_request_preserves_tools_reasoning_and_max() {
    for simple in [false, true] {
        let fixture = Fixture::new(Reply::Good).await;
        let lease = fixture.lease();
        let mut opts = options(Some(lease.clone()));
        opts.stream.on_payload = Some(Arc::new(|value, _| {
            assert!(value.get("occ3_capture").is_none());
            Box::pin(async move { Some(value) })
        }));
        opts.stream.on_response = Some(Arc::new(|response, _| {
            assert!(!response.headers.contains_key(SOURCE_RECEIPT_HEADER));
            Box::pin(async {})
        }));
        let stream = if simple {
            stream_simple_openai_completions(
                &fixture.model,
                &context(),
                Some(SimpleStreamOptions {
                    stream: opts.stream,
                    reasoning: Some("high".into()),
                    ..Default::default()
                }),
            )
        } else {
            stream_openai_completions(&fixture.model, &context(), Some(opts))
        };
        let events = collect(&stream).await;
        assert!(
            matches!(events.last(), Some(AssistantMessageEvent::Done { .. })),
            "{events:?}"
        );
        assert_eq!(lease.state(), SourceCaptureState::Captured);
        let receipt = lease.receipt().unwrap();
        assert_eq!(receipt.native_rid, "native-1");
        assert_ne!(receipt.native_rid, lease.intent().source_request_id);
        assert_eq!(receipt.output_sha256, sha256(body().as_bytes()));
        assert_eq!(lease.tool_bindings().len(), 1);
        assert!(lease
            .validate_tool(
                "call-handoff",
                "agent_message",
                &json!({"control":"continue"})
            )
            .is_ok());
        assert!(lease
            .validate_tool("other", "agent_message", &json!({"control":"continue"}))
            .is_err());
        assert!(lease
            .validate_tool(
                "call-handoff",
                "agent_message",
                &json!({"control":"forged"})
            )
            .is_err());
        let requests = fixture.requests.lock().unwrap();
        let (head, payload) = &requests[0];
        // The occ3 server contract (extensions/sglang-occ3 chat_adapter
        // handle_around) admits capture-armed source requests only with
        // exactly one accept-encoding header whose value is identity.
        let encodings: Vec<String> = head
            .lines()
            .filter_map(|line| {
                line.to_ascii_lowercase()
                    .strip_prefix("accept-encoding:")
                    .map(|value| value.trim().to_string())
            })
            .collect();
        assert_eq!(
            encodings,
            vec!["identity".to_string()],
            "guarded native request must carry exactly one accept-encoding: identity header"
        );
        assert_eq!(
            payload["occ3_capture"],
            serde_json::to_value(lease.intent()).unwrap()
        );
        assert_eq!(payload["messages"][0]["content"], "native system");
        assert_eq!(payload["tools"][0]["function"]["name"], "agent_message");
        assert_eq!(payload["max_completion_tokens"], 800);
        assert_eq!(payload["reasoning_effort"], "high");
        assert!(payload.get("occ3").is_none());
        assert_eq!(requests.len(), 1);
    }
}
#[tokio::test]
async fn source_digest_receipt_encoding_identity_shape_fail_before_output_and_never_replay() {
    let mut replies = vec![
        Reply::Missing,
        Reply::Duplicate,
        Reply::Oversized,
        Reply::Encoding,
        Reply::Disconnect,
        Reply::Status,
    ];
    for (field, value) in [
        ("intent_digest", json!("0".repeat(64))),
        ("output_sha256", json!("0".repeat(64))),
        ("response_id", json!("other")),
        ("ready_ranks", json!(7)),
        ("layout_digest", json!("0".repeat(64))),
        ("byte_count", json!(256)),
        ("accepted_span_end", json!(2)),
        ("predictor_position_offset", json!(0)),
        ("source_text", json!("forbidden")),
    ] {
        replies.push(Reply::Field(field, value));
    }
    for reply in replies {
        let fixture = Fixture::new(reply).await;
        let lease = fixture.lease();
        let stream = stream_openai_completions(
            &fixture.model,
            &context(),
            Some(options(Some(lease.clone()))),
        );
        assert_uncertain(&collect(&stream).await, &lease);
        assert_eq!(fixture.count(), 1);
        let replay = stream_openai_completions(
            &fixture.model,
            &context(),
            Some(options(Some(lease.clone()))),
        );
        assert_uncertain(&collect(&replay).await, &lease);
        assert_eq!(fixture.count(), 1);
    }
}
#[tokio::test]
async fn source_exact_finite_sse_body_requires_finish_done_clean_eof_and_consistent_ids() {
    for invalid in [
        String::new(),
        "data: [DONE]\n\n".into(),
        body().replace("data: [DONE]\r\n\r\n", ""),
        format!("{}data: {{\"error\":\"late\"}}\n\n", body()),
        format!("{}: trailing protocol\n", body()),
        body().replace("native-1", "other"),
        body().replace("tool_calls\"}]", "content_filter\"}]"),
        body().replace("[DONE]", "[DONE]forged"),
    ] {
        let fixture = Fixture::new(Reply::Body(invalid)).await;
        let lease = fixture.lease();
        let stream = stream_openai_completions(
            &fixture.model,
            &context(),
            Some(options(Some(lease.clone()))),
        );
        assert_uncertain(&collect(&stream).await, &lease);
    }
}
#[tokio::test]
async fn source_done_before_eof_never_exposes_or_publishes_lease() {
    let fixture = Fixture::new(Reply::HoldEof).await;
    let lease = fixture.lease();
    let stream = stream_openai_completions(
        &fixture.model,
        &context(),
        Some(options(Some(lease.clone()))),
    );
    fixture.seen.notified().await;
    assert_eq!(lease.state(), SourceCaptureState::Submitted);
    assert!(stream.next().now_or_never().is_none());
    assert!(lease.receipt().is_none());
    fixture.release.notify_one();
    let events = collect(&stream).await;
    assert!(matches!(
        events.last(),
        Some(AssistantMessageEvent::Done { .. })
    ));
    assert_eq!(lease.state(), SourceCaptureState::Captured);
}
#[tokio::test]
async fn source_abort_waiting_for_eof_is_uncertain_and_blocks_replay() {
    let fixture = Fixture::new(Reply::HoldEof).await;
    let lease = fixture.lease();
    let signal = tokio_util::sync::CancellationToken::new();
    let mut opts = options(Some(lease.clone()));
    opts.stream.signal = Some(signal.clone());
    let stream = stream_openai_completions(&fixture.model, &context(), Some(opts));
    fixture.seen.notified().await;
    signal.cancel();
    let events = collect(&stream).await;
    assert_eq!(lease.state(), SourceCaptureState::Uncertain);
    assert!(lease.receipt().is_none());
    assert!(
        matches!(events.as_slice(), [AssistantMessageEvent::Error { error, .. }] if error.stop_reason == STOP_REASON_ABORTED)
    );
    assert_eq!(fixture.count(), 1);
    let replay = stream_openai_completions(
        &fixture.model,
        &context(),
        Some(options(Some(lease.clone()))),
    );
    assert_uncertain(&collect(&replay).await, &lease);
    assert_eq!(fixture.count(), 1);
}
#[tokio::test]
async fn source_target_payload_authority_and_prepared_cancel_refuse_without_send() {
    let fixture = Fixture::new(Reply::Good).await;
    for field in [
        "provider",
        "api",
        "model",
        "endpoint",
        "reserved",
        "payload_model",
        "host",
        "encoding",
        "cancel",
    ] {
        let lease = fixture.lease();
        let mut model = fixture.model.clone();
        let mut opts = options(Some(lease.clone()));
        match field {
            "provider" => model.provider = "other".into(),
            "api" => model.api = API_OPENAI_RESPONSES.into(),
            "model" => model.id = "other".into(),
            "endpoint" => model.base_url.push_str("/wrong"),
            "host" => {
                opts.stream.headers = Some(
                    [("Host".into(), "other.invalid".into())]
                        .into_iter()
                        .collect(),
                )
            }
            "encoding" => {
                opts.stream.headers = Some(
                    [("Accept-Encoding".into(), "gzip".into())]
                        .into_iter()
                        .collect(),
                )
            }
            "cancel" => {
                lease.cancel();
            }
            _ => {
                opts.stream.on_payload = Some(Arc::new(move |mut value, _| {
                    if field == "reserved" {
                        value["occ3_capture"] = json!({});
                    } else {
                        value["model"] = json!("other");
                    }
                    Box::pin(async move { Some(value) })
                }));
            }
        }
        let stream = stream_openai_completions(&model, &context(), Some(opts));
        let events = collect(&stream).await;
        assert!(matches!(
            events.as_slice(),
            [AssistantMessageEvent::Error { .. }]
        ));
        assert_eq!(fixture.count(), 0);
        assert!(matches!(
            lease.state(),
            SourceCaptureState::Prepared | SourceCaptureState::Cancelled
        ));
    }
}
#[tokio::test]
async fn source_clone_fence_one_actual_generation_no_recapture_from_text() {
    let fixture = Fixture::new(Reply::Good).await;
    let lease = fixture.lease();
    let one = stream_openai_completions(
        &fixture.model,
        &context(),
        Some(options(Some(lease.clone()))),
    );
    let two = stream_openai_completions(
        &fixture.model,
        &context(),
        Some(options(Some(lease.clone()))),
    );
    let (one, two) = tokio::join!(collect(&one), collect(&two));
    assert_eq!(
        one.iter()
            .chain(two.iter())
            .filter(|e| matches!(e, AssistantMessageEvent::Done { .. }))
            .count(),
        1
    );
    assert_eq!(fixture.count(), 1);
    assert_eq!(lease.state(), SourceCaptureState::Captured);
}
#[tokio::test]
async fn ordinary_no_source_request_preserves_done_without_eof_behavior_and_no_metadata() {
    let fixture = Fixture::new(Reply::HoldEof).await;
    let stream = stream_openai_completions(&fixture.model, &context(), Some(options(None)));
    let events = collect(&stream).await;
    assert!(matches!(
        events.last(),
        Some(AssistantMessageEvent::Done { .. })
    ));
    assert!(fixture.requests.lock().unwrap()[0]
        .1
        .get("occ3_capture")
        .is_none());
}
#[test]
fn source_contract_canonical_digest_bounds_storage_units_and_nonserialization() {
    let intent = intent();
    let profile = profile();
    let receipt = receipt(&intent, &body());
    validate_receipt(&receipt, &intent, &profile, epoch()).unwrap();
    assert_eq!(receipt.byte_count, 32);
    assert_eq!(intent.max_capture_bytes, 4096);
    let bytes = intent.canonical_bytes().unwrap();
    let value: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value, serde_json::to_value(&intent).unwrap());
    assert_eq!(intent.digest().unwrap(), sha256(&bytes));
    assert!(std::str::from_utf8(&bytes)
        .unwrap()
        .starts_with("{\"capture_id\":"));
    for invalid in [
        json!(true),
        json!(-1),
        json!(1.5),
        json!("18446744073709551616"),
    ] {
        let mut value = serde_json::to_value(&intent).unwrap();
        value["max_capture_bytes"] = invalid;
        assert!(serde_json::from_value::<SourceCaptureIntent>(value).is_err());
    }
    let mut value = serde_json::to_value(&intent).unwrap();
    value["unknown"] = json!(true);
    assert!(serde_json::from_value::<SourceCaptureIntent>(value).is_err());
    assert!(serde_json::from_str::<SourceCaptureIntent>(&format!(
        "{{\"capture_id\":\"duplicate\",{}",
        &String::from_utf8(bytes).unwrap()[1..]
    ))
    .is_err());
    let options: StreamOptions =
        serde_json::from_value(json!({"sourceCapture":{"source_handle":"forged"}})).unwrap();
    assert!(options.source_capture.is_none());
}

#[test]
fn shared_python_source_v1_fixture_matches_exact_canonical_bytes_hashes_and_negative_cases() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../extensions/sglang-occ3/tests/fixtures/occ3_source_v1.json"
    ))
    .unwrap();
    let intent: SourceCaptureIntent = serde_json::from_value(fixture["intent"].clone()).unwrap();
    intent.validate(1).unwrap();
    assert_eq!(
        String::from_utf8(intent.canonical_bytes().unwrap()).unwrap(),
        fixture["intent_canonical"]
    );
    assert_eq!(intent.digest().unwrap(), fixture["intent_sha256"]);
    assert_eq!(
        sha256(fixture["response_body_utf8"].as_str().unwrap().as_bytes()),
        fixture["response_body_sha256"]
    );
    let receipt: SourceCaptureReceipt = serde_json::from_value(fixture["receipt"].clone()).unwrap();
    let expected = SourceCaptureProfile {
        layer_index: receipt.layer_index,
        hidden_dim: receipt.hidden_dim,
        dtype: receipt.dtype.clone(),
        layout_digest: receipt.layout_digest.clone(),
        expected_ranks: receipt.expected_ranks,
    };
    validate_receipt(&receipt, &intent, &expected, 1).unwrap();
    let bytes = canonical_json_bytes(&serde_json::to_value(&receipt).unwrap()).unwrap();
    assert_eq!(
        String::from_utf8(bytes.clone()).unwrap(),
        fixture["receipt_canonical"]
    );
    assert_eq!(sha256(&bytes), fixture["receipt_sha256"]);
    let probe = canonical_json_bytes(&fixture["canonical_probe"]).unwrap();
    assert_eq!(
        String::from_utf8(probe.clone()).unwrap(),
        fixture["canonical_probe_bytes"]
    );
    assert_eq!(sha256(&probe), fixture["canonical_probe_sha256"]);
    assert_eq!(
        canonical_json_bytes(&fixture["max_valid_receipt"])
            .unwrap()
            .len() as u64,
        fixture["max_valid_receipt_bytes"].as_u64().unwrap()
    );
    for negative in fixture["negative_intents"].as_array().unwrap() {
        if let Ok(value) = serde_json::from_value::<SourceCaptureIntent>(negative["wire"].clone()) {
            assert!(value.validate(1).is_err(), "{}", negative["name"]);
        }
    }
    for negative in fixture["negative_receipts"].as_array().unwrap() {
        if let Ok(value) = serde_json::from_value::<SourceCaptureReceipt>(negative["wire"].clone())
        {
            assert!(
                validate_receipt(&value, &intent, &expected, 1).is_err(),
                "{}",
                negative["name"]
            );
        }
    }
    assert!(serde_json::from_str::<SourceCaptureIntent>(
        fixture["duplicate_intent_json"].as_str().unwrap()
    )
    .is_err());
    assert!(serde_json::from_str::<SourceCaptureReceipt>(
        fixture["duplicate_receipt_json"].as_str().unwrap()
    )
    .is_err());
}

fn busy_body(code: &str) -> String {
    format!("{{\"error\":{{\"message\":\"{code}\",\"type\":\"occ3_attachment_error\",\"code\":503}}}}")
}
fn failure_details(events: &[AssistantMessageEvent]) -> serde_json::Map<String, Value> {
    let AssistantMessageEvent::Error { error, .. } = events.last().expect("terminal event") else {
        panic!("expected failure")
    };
    error
        .diagnostics
        .as_ref()
        .unwrap()
        .iter()
        .find(|d| d.type_ == "provider_stream_failure")
        .unwrap()
        .details
        .clone()
        .unwrap()
}

/// A pre-event 503 with a recognised admission-busy code on an armed SOURCE
/// capture is a distinct retryable non-use outcome: the lease closes as
/// `Cancelled` with a busy refusal record, never `Uncertain`, and nothing is
/// resent or replayable.
#[tokio::test]
async fn source_pre_event_busy_is_retryable_non_use_without_resend() {
    for code in ["scheduler_busy", "admission_capacity", "admission_timeout", "occ3_busy_batch"] {
        let fixture = Fixture::new(Reply::Refusal(
            "503 Service Unavailable",
            busy_body(code),
            "Retry-After: 5\r\n",
        ))
        .await;
        // Retry-After 5 s outlasts this lease, so no same-request resend fits.
        let lease = fixture.lease_with(Duration::from_secs(3));
        let stream =
            stream_openai_completions(&fixture.model, &context(), Some(options(Some(lease.clone()))));
        let events = collect(&stream).await;
        assert_eq!(events.len(), 1, "{code}: no partial output");
        let details = failure_details(&events);
        assert_eq!(details["providerErrorType"], "occ3_source_busy", "{code}");
        assert_eq!(details["kind"], "overloaded", "{code}: retryable, not request_interrupted");
        assert_eq!(details["status"], 503);
        assert_eq!(details["retryAfterMs"], 5000.0, "{code}");
        assert!(details.get("occ3Cause").is_none());
        assert_eq!(lease.state(), SourceCaptureState::Cancelled, "{code}: proven non-use");
        let refusal = lease.busy_refusal().expect("busy refusal recorded");
        assert_eq!(refusal.retry_after_secs, Some(5));
        assert!(refusal.refused_at_epoch > 0);
        assert!(lease.receipt().is_none());
        let wire = serde_json::to_string(&events).unwrap();
        assert!(wire.contains("not delivered"), "{code}: truthful message");
        assert_eq!(fixture.count(), 1, "{code}: no resend past the lease deadline");
        let replay =
            stream_openai_completions(&fixture.model, &context(), Some(options(Some(lease.clone()))));
        let _ = collect(&replay).await;
        assert_eq!(fixture.count(), 1, "{code}: the closed lease cannot be replayed");
        assert_eq!(lease.state(), SourceCaptureState::Cancelled);
    }
}

/// Retry-After is honoured only as one plain delta-seconds value in 1..=30.
#[tokio::test]
async fn source_busy_retry_after_is_bounded() {
    for (extra, expected) in [
        ("Retry-After: 1\r\n", Some(1u64)),
        ("Retry-After: 30\r\n", Some(30)),
        ("", None),
        ("Retry-After: 0\r\n", None),
        ("Retry-After: 31\r\n", None),
        ("Retry-After: -1\r\n", None),
        ("Retry-After: abc\r\n", None),
        ("Retry-After: 2.5\r\n", None),
        ("Retry-After: 99999999999999999999999\r\n", None),
        ("Retry-After: Wed, 21 Oct 2015 07:28:00 GMT\r\n", None),
        ("Retry-After: 5\r\nRetry-After: 6\r\n", None),
    ] {
        let fixture =
            Fixture::new(Reply::Refusal("503 Service Unavailable", busy_body("scheduler_busy"), extra))
                .await;
        // Shorter than any delay plus the 25 s resend headroom: the refusal
        // closes without a resend.
        let lease = fixture.lease_with(Duration::from_secs(10));
        let stream =
            stream_openai_completions(&fixture.model, &context(), Some(options(Some(lease.clone()))));
        let details = failure_details(&collect(&stream).await);
        assert_eq!(details["providerErrorType"], "occ3_source_busy", "{extra:?}");
        assert_eq!(lease.busy_refusal().unwrap().retry_after_secs, expected, "{extra:?}");
        assert_eq!(
            details.get("retryAfterMs").and_then(Value::as_f64),
            expected.map(|s| (s * 1000) as f64),
            "{extra:?}"
        );
    }
}

/// Unknown 503 codes, oversized or non-JSON bodies, and busy bodies on any
/// other status keep the existing uncertain, no-replay source contract.
#[tokio::test]
async fn source_unrecognised_refusals_stay_uncertain() {
    let oversized = format!(
        "{{\"error\":{{\"message\":\"scheduler_busy\",\"pad\":\"{}\"}}}}",
        "x".repeat(5000)
    );
    let cases: Vec<(&'static str, String)> = vec![
        ("503 Service Unavailable", busy_body("native_completion_failed")),
        ("503 Service Unavailable", busy_body("unclassified")),
        ("503 Service Unavailable", "scheduler_busy".into()),
        ("503 Service Unavailable", oversized),
        ("502 Bad Gateway", busy_body("scheduler_busy")),
        ("504 Gateway Timeout", busy_body("admission_timeout")),
        ("429 Too Many Requests", busy_body("scheduler_busy")),
    ];
    for (status, body) in cases {
        let fixture = Fixture::new(Reply::Refusal(status, body, "Retry-After: 5\r\n")).await;
        let lease = fixture.lease();
        let stream =
            stream_openai_completions(&fixture.model, &context(), Some(options(Some(lease.clone()))));
        let events = collect(&stream).await;
        assert_uncertain(&events, &lease);
        assert!(lease.busy_refusal().is_none(), "{status}");
        assert_eq!(fixture.count(), 1);
        let details = failure_details(&events);
        assert_eq!(details["providerErrorType"], "occ3_inference_uncertain", "{status}");
        assert_eq!(details["occ3Cause"], "http_non_success", "{status}");
    }
}

/// A busy 503 whose body stalls is not classified: the bounded read gives
/// up quickly and the existing uncertain path applies, without waiting for
/// the lease deadline.
#[tokio::test]
async fn source_stalled_refusal_body_stays_uncertain_and_returns_quickly() {
    let fixture = Fixture::new(Reply::StalledRefusal).await;
    let lease = fixture.lease();
    let started = std::time::Instant::now();
    let stream = stream_openai_completions(&fixture.model, &context(), Some(options(Some(lease.clone()))));
    let events = tokio::time::timeout(std::time::Duration::from_secs(10), collect(&stream))
        .await
        .expect("bounded body read");
    assert!(started.elapsed() < std::time::Duration::from_secs(8));
    assert_uncertain(&events, &lease);
    assert!(lease.busy_refusal().is_none());
    assert_eq!(fixture.count(), 1);
}

fn busy(code: &'static str, extra: &'static str) -> Reply {
    Reply::Refusal("503 Service Unavailable", busy_body(code), extra)
}
fn sent_bodies(fixture: &Fixture) -> Vec<Value> {
    fixture.requests.lock().unwrap().iter().map(|(_, body)| body.clone()).collect()
}

/// A pre-admission busy 503 is followed by one resend of the IDENTICAL
/// request after Retry-After; the resend's success captures normally.
#[tokio::test]
async fn source_busy_then_success_resends_identical_request() {
    let fixture = Fixture::new(Reply::Script(vec![busy("scheduler_busy", "Retry-After: 1\r\n"), Reply::Good])).await;
    let lease = fixture.lease();
    let started = std::time::Instant::now();
    let stream = stream_openai_completions(&fixture.model, &context(), Some(options(Some(lease.clone()))));
    let events = collect(&stream).await;
    assert!(started.elapsed() >= Duration::from_secs(1), "Retry-After honoured");
    assert!(matches!(events.last(), Some(AssistantMessageEvent::Done { .. })), "{events:?}");
    assert_eq!(lease.state(), SourceCaptureState::Captured);
    assert!(lease.receipt().is_some());
    assert!(lease.busy_refusal().is_none());
    let bodies = sent_bodies(&fixture);
    assert_eq!(bodies.len(), 2);
    assert_eq!(bodies[0], bodies[1], "the resend is byte-for-byte the same body");
    let heads: Vec<String> = fixture.requests.lock().unwrap().iter().map(|(head, _)| head.to_ascii_lowercase()).collect();
    assert!(heads.iter().all(|head| head.contains("accept-encoding: identity")));
}

/// Three busy refusals: two resends, then the lease closes as proven
/// non-use with the busy record of the final refusal.
#[tokio::test]
async fn source_busy_three_times_refuses_before_admission() {
    let fixture = Fixture::new(Reply::Script(vec![
        busy("scheduler_busy", "Retry-After: 1\r\n"),
        busy("admission_capacity", "Retry-After: 1\r\n"),
        busy("occ3_busy_batch", "Retry-After: 2\r\n"),
    ]))
    .await;
    let lease = fixture.lease();
    let stream = stream_openai_completions(&fixture.model, &context(), Some(options(Some(lease.clone()))));
    let events = collect(&stream).await;
    assert_eq!(fixture.count(), 3, "exactly two resends");
    assert_eq!(events.len(), 1);
    let details = failure_details(&events);
    assert_eq!(details["providerErrorType"], "occ3_source_busy");
    assert_eq!(details["retryAfterMs"], 2000.0);
    assert_eq!(lease.state(), SourceCaptureState::Cancelled);
    assert_eq!(lease.busy_refusal().expect("busy refusal").retry_after_secs, Some(2));
    let bodies = sent_bodies(&fixture);
    assert!(bodies.iter().all(|body| body == &bodies[0]));
}

/// A server without the fence release answers the resend with 409
/// replay_forbidden: that closes as the original busy refusal, not uncertain.
#[tokio::test]
async fn source_busy_then_replay_forbidden_is_busy_not_uncertain() {
    let fixture = Fixture::new(Reply::Script(vec![
        busy("scheduler_busy", "Retry-After: 1\r\n"),
        Reply::Refusal("409 Conflict", busy_body("replay_forbidden"), ""),
        Reply::Good,
    ]))
    .await;
    let lease = fixture.lease();
    let stream = stream_openai_completions(&fixture.model, &context(), Some(options(Some(lease.clone()))));
    let events = collect(&stream).await;
    assert_eq!(fixture.count(), 2, "no retry after replay_forbidden");
    let details = failure_details(&events);
    assert_eq!(details["providerErrorType"], "occ3_source_busy");
    assert_eq!(lease.state(), SourceCaptureState::Cancelled);
    assert_eq!(lease.busy_refusal().expect("busy refusal").retry_after_secs, Some(1));
}

/// A first-attempt 409 replay_forbidden, any other 409 on a resend, and a
/// 429 on a resend are never retried and keep the uncertain contract.
#[tokio::test]
async fn source_busy_resend_never_retries_other_refusals() {
    let cases: Vec<Reply> = vec![
        Reply::Script(vec![Reply::Refusal("409 Conflict", busy_body("replay_forbidden"), ""), Reply::Good]),
        Reply::Script(vec![busy("scheduler_busy", "Retry-After: 1\r\n"), Reply::Refusal("409 Conflict", busy_body("unsupported_profile"), ""), Reply::Good]),
        Reply::Script(vec![busy("scheduler_busy", "Retry-After: 1\r\n"), Reply::Refusal("429 Too Many Requests", busy_body("quota_exceeded"), "Retry-After: 1\r\n"), Reply::Good]),
        Reply::Script(vec![busy("scheduler_busy", "Retry-After: 1\r\n"), Reply::Refusal("503 Service Unavailable", busy_body("native_completion_failed"), ""), Reply::Good]),
        Reply::Script(vec![Reply::Refusal("429 Too Many Requests", busy_body("scheduler_busy"), "Retry-After: 1\r\n"), Reply::Good]),
    ];
    for (index, script) in cases.into_iter().enumerate() {
        let expected = match &script { Reply::Script(steps) => steps.len() - 1, _ => unreachable!() };
        let fixture = Fixture::new(script).await;
        let lease = fixture.lease();
        let stream = stream_openai_completions(&fixture.model, &context(), Some(options(Some(lease.clone()))));
        let events = collect(&stream).await;
        assert_eq!(fixture.count(), expected, "case {index}");
        assert_uncertain(&events, &lease);
        assert!(lease.busy_refusal().is_none(), "case {index}");
    }
}

/// An in-stream failure after a 200 is never resent.
#[tokio::test]
async fn source_in_stream_failure_is_never_resent() {
    let fixture = Fixture::new(Reply::Script(vec![Reply::Disconnect, Reply::Good])).await;
    let lease = fixture.lease();
    let stream = stream_openai_completions(&fixture.model, &context(), Some(options(Some(lease.clone()))));
    let events = collect(&stream).await;
    assert_eq!(fixture.count(), 1);
    assert_uncertain(&events, &lease);
}

/// Cancelling during the Retry-After sleep stops at once without a resend.
#[tokio::test]
async fn source_busy_cancel_during_resend_sleep_stops_without_resend() {
    let fixture = Fixture::new(Reply::Script(vec![busy("scheduler_busy", "Retry-After: 5\r\n"), Reply::Good])).await;
    let lease = fixture.lease();
    let signal = tokio_util::sync::CancellationToken::new();
    let mut opts = options(Some(lease.clone()));
    opts.stream.signal = Some(signal.clone());
    let started = std::time::Instant::now();
    let stream = stream_openai_completions(&fixture.model, &context(), Some(opts));
    fixture.seen.notified().await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    signal.cancel();
    let events = collect(&stream).await;
    assert!(started.elapsed() < Duration::from_secs(4), "cancel interrupts the sleep");
    assert_eq!(fixture.count(), 1, "no resend after cancel");
    assert!(lease.receipt().is_none());
    // Nothing was in flight and the refusal was proven pre-use: the lease
    // closes as that busy refusal, never Uncertain.
    assert_eq!(lease.state(), SourceCaptureState::Cancelled);
    assert_eq!(lease.busy_refusal().expect("busy refusal").retry_after_secs, Some(5));
    assert!(matches!(events.last(), Some(AssistantMessageEvent::Error { .. })));
}

/// A resend is attempted only when the delay PLUS the admission headroom
/// fits the lease, so it can never expire in flight as Uncertain.
#[tokio::test]
async fn source_busy_resend_requires_admission_headroom() {
    let fixture = Fixture::new(Reply::Script(vec![busy("scheduler_busy", "Retry-After: 1\r\n"), Reply::Good])).await;
    // 1 s delay fits a 10 s lease, but 1 s + 25 s headroom does not.
    let lease = fixture.lease_with(Duration::from_secs(10));
    let stream = stream_openai_completions(&fixture.model, &context(), Some(options(Some(lease.clone()))));
    let events = collect(&stream).await;
    assert_eq!(fixture.count(), 1, "no resend without headroom");
    assert_eq!(failure_details(&events)["providerErrorType"], "occ3_source_busy");
    assert_eq!(lease.state(), SourceCaptureState::Cancelled);
    assert_eq!(lease.busy_refusal().unwrap().retry_after_secs, Some(1));
}

/// A busy refusal without a usable Retry-After resends after the 5 s default.
#[tokio::test]
async fn source_busy_without_retry_after_uses_default_delay() {
    let fixture = Fixture::new(Reply::Script(vec![busy("scheduler_busy", ""), Reply::Good])).await;
    let lease = fixture.lease();
    let started = std::time::Instant::now();
    let stream = stream_openai_completions(&fixture.model, &context(), Some(options(Some(lease.clone()))));
    let events = collect(&stream).await;
    assert!(started.elapsed() >= Duration::from_secs(5), "default delay honoured");
    assert!(matches!(events.last(), Some(AssistantMessageEvent::Done { .. })), "{events:?}");
    assert_eq!(fixture.count(), 2);
    assert_eq!(lease.state(), SourceCaptureState::Captured);
}

/// A resend whose delay would cross the lease deadline is not attempted:
/// the refusal closes at once as proven non-use.
#[tokio::test]
async fn source_busy_resend_respects_lease_deadline() {
    let fixture = Fixture::new(Reply::Script(vec![
        busy("scheduler_busy", "Retry-After: 1\r\n"),
        busy("scheduler_busy", "Retry-After: 3\r\n"),
        Reply::Good,
    ]))
    .await;
    // First resend: 1 s + 25 s headroom fits 28 s. Second, at ~1 s:
    // 3 s + 25 s crosses it.
    let lease = fixture.lease_with(Duration::from_secs(28));
    let started = std::time::Instant::now();
    let stream = stream_openai_completions(&fixture.model, &context(), Some(options(Some(lease.clone()))));
    let events = collect(&stream).await;
    assert!(started.elapsed() < Duration::from_secs(3));
    assert_eq!(fixture.count(), 2, "second delay plus headroom crosses the deadline");
    assert_eq!(failure_details(&events)["providerErrorType"], "occ3_source_busy");
    assert_eq!(lease.state(), SourceCaptureState::Cancelled);
    assert_eq!(lease.busy_refusal().unwrap().retry_after_secs, Some(3));
}

#[tokio::test]
async fn deferred_source_busy_is_proven_non_use() {
    for code in ["scheduler_busy", "admission_capacity", "occ3_busy_batch", "admission_timeout"] {
        let body = format!(": occ3-keepalive\n\ndata: {}\n\n", json!({"error":{"type":"occ3_attachment_error","code":503,"message":code,"retryAfterSeconds":5}}));
        let fixture = Fixture::new(Reply::Refusal("200 OK", body,
            "x-optimus-occ3-deferred-receipt: deferred-v1\r\n")).await;
        let lease = fixture.lease();
        let stream = stream_openai_completions(&fixture.model, &context(), Some(options(Some(lease.clone()))));
        let events = collect(&stream).await;
        assert_eq!(failure_details(&events)["providerErrorType"], "occ3_source_busy");
        assert_eq!(lease.state(), SourceCaptureState::Cancelled);
        assert_eq!(lease.busy_refusal().unwrap().retry_after_secs, Some(5));
        assert!(lease.receipt().is_none());
        assert_eq!(fixture.count(), 1);
    }
}

#[tokio::test]
async fn default_source_offer_declined_uses_header_receipt() {
    // ScopedEnv is crate-private and cfg(test), unavailable to integration tests.
    // This tests the unset default; run with OCC3_KEEPALIVE_STREAM unset rather
    // than mutating process-global env while sibling async tests are running.
    assert!(std::env::var_os("OCC3_KEEPALIVE_STREAM").is_none());
    let fixture = Fixture::new(Reply::Good).await;
    let lease = fixture.lease();
    let stream = stream_openai_completions(&fixture.model, &context(), Some(options(Some(lease.clone()))));
    let _ = collect(&stream).await;
    assert!(lease.receipt().is_some());
    assert!(fixture.requests.lock().unwrap()[0].0.to_lowercase().contains("x-optimus-occ3-keepalive: keepalive-v1"));
}

#[tokio::test]
async fn deferred_source_error_after_data_stays_uncertain() {
    let busy = json!({"error":{"type":"occ3_attachment_error","code":503,"message":"scheduler_busy"}});
    let body = format!("data: {{\"choices\":[]}}\n\ndata: {busy}\n\n");
    let fixture = Fixture::new(Reply::Refusal("200 OK", body,
        "x-optimus-occ3-deferred-receipt: deferred-v1\r\n")).await;
    let lease = fixture.lease();
    let stream = stream_openai_completions(&fixture.model, &context(), Some(options(Some(lease.clone()))));
    let _ = collect(&stream).await;
    assert_eq!(lease.state(), SourceCaptureState::Uncertain);
    assert!(lease.busy_refusal().is_none());
}

#[tokio::test]
async fn deferred_source_unknown_or_unaccepted_error_stays_uncertain() {
    for (code, headers) in [("unknown", "x-optimus-occ3-deferred-receipt: deferred-v1\r\n"), ("scheduler_busy", "")] {
        let body = format!("data: {}\n\n", json!({"error":{"type":"occ3_attachment_error","code":503,"message":code}}));
        let fixture = Fixture::new(Reply::Refusal("200 OK", body, headers)).await;
        let lease = fixture.lease();
        let stream = stream_openai_completions(&fixture.model, &context(), Some(options(Some(lease.clone()))));
        let _ = collect(&stream).await;
        assert_eq!(lease.state(), SourceCaptureState::Uncertain);
        assert!(lease.busy_refusal().is_none());
    }
}

#[test]
fn label_anchor_selection_serialization_and_receipt_binding() {
    let mut intent = intent();
    let default_wire = serde_json::to_value(&intent).unwrap();
    assert!(default_wire.get("label_anchor_alpha").is_none());
    assert_eq!(serde_json::from_value::<SourceCaptureIntent>(default_wire).unwrap().label_anchor_alpha, None);
    for alpha in [0.0, 0.3, 0.5, 0.7, 1.0] {
        let selected = LabelAnchorAlpha::try_from(alpha).unwrap();
        intent.label_anchor_alpha = Some(selected);
        let wire = serde_json::to_value(&intent).unwrap();
        assert_eq!(wire["label_anchor_alpha"], json!(alpha));
        assert_eq!(serde_json::from_value::<SourceCaptureIntent>(wire).unwrap().label_anchor_alpha, Some(selected));
        let mut receipt = receipt(&intent, "body");
        receipt.label_anchor_alpha = selected;
        assert!(validate_receipt(&receipt, &intent, &profile(), epoch()).is_ok());
        receipt.label_anchor_alpha = LabelAnchorAlpha::try_from(if alpha == 0.0 { 1.0 } else { 0.0 }).unwrap();
        assert!(validate_receipt(&receipt, &intent, &profile(), epoch()).is_err());
        intent.label_anchor_alpha = None;
        receipt.intent_digest = intent.digest().unwrap();
        assert!(validate_receipt(&receipt, &intent, &profile(), epoch()).is_ok());
    }
    for invalid in [json!(0.4), json!("0.3"), json!(-1), json!(null), json!(true)] {
        let mut wire = serde_json::to_value(&intent).unwrap();
        wire["label_anchor_alpha"] = invalid;
        assert!(serde_json::from_value::<SourceCaptureIntent>(wire).is_err());
    }
}

#[test]
fn label_anchor_selection_matches_python_canonical_digest_vectors() {
    // Python json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=True), SHA-256.
    let mut intent = intent();
    intent.expires_at_epoch = 2_000_000_000;
    for (alpha, expected) in [
        (None, "5de0308e3e394cd7e5275a5f7c35704853935b63b3eac78dec6617f123b98065"),
        (Some(0.0), "4eb17b21225a458fe37d74ff24638047d183fbbd1124ddda8b51c7fb84d68dae"),
        (Some(0.3), "b85e8d6b733560ce31d25a6a79b9d0aa75ebfa28c65c3b6b68a58d0bdf9b7042"),
        (Some(0.7), "39be23526a02271d94083bcab9296ffff4d627ccb39e8b196b0ce6ca75ccf402"),
        (Some(1.0), "87d087f82898938d6c1dbd59601928b932bb1b9c9c087e2642d57f9edd01b1b0"),
    ] {
        intent.label_anchor_alpha = alpha.map(|v| LabelAnchorAlpha::try_from(v).unwrap());
        assert_eq!(intent.digest().unwrap(), expected);
    }
    assert!(canonical_json_bytes(&json!({"unrelated": 0.3})).is_err());
}
