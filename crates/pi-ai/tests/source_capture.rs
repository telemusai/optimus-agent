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
    async fn new(reply: Reply) -> Self {
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
            loop {
                let (mut socket, _) = listener.accept().await.unwrap();
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
        SourceCaptureLease::new(
            InferenceStateTarget {
                provider: self.model.provider.clone(),
                model_id: self.model.id.clone(),
                base_url: self.model.base_url.clone(),
            },
            intent(),
            profile(),
            SourceCaptureLimits {
                deadline: Instant::now() + Duration::from_secs(8),
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
