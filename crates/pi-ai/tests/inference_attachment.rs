//! Local contract fixtures, not evidence of a native engine importing state.
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use pi_ai::inference_attachment::{
    InferenceAttachment, InferenceStateLease, InferenceStateLeaseState, InferenceStateTarget,
    INFERENCE_RECEIPT_HEADER, MAX_INFERENCE_RECEIPT_BYTES,
};
use pi_ai::providers::openai_completions::{
    stream_openai_completions, stream_simple_openai_completions, OpenAICompletionsOptions,
};
use pi_ai::types::*;
use pi_ai::utils::event_stream::AssistantMessageEventStream;
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

const COMPLETE: &str = concat!(
    "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"fixture reasoning\"}}]}\n\n",
    "data: {\"choices\":[{\"delta\":{\"content\":\"native-fixture-answer\"}}]}\n\n",
    "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call-out\",\"function\":{\"name\":\"fixture_tool\",\"arguments\":\"{\\\"value\\\":1}\"}}]}}]}\n\n",
    "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
    "data: [DONE]\n\n",
);
const PARTIAL: &str = "data: {\"choices\":[{\"delta\":{\"content\":\"partial-fixture\"}}]}\n\n";

#[derive(Clone)]
enum Reply {
    Good,
    Missing,
    Mismatch(&'static str),
    Flag(&'static str),
    Unknown,
    Oversized,
    Duplicate,
    Disconnect,
    Status,
    /// Pre-event admission-unavailability 503 with this bounded code.
    Busy(&'static str),
    Redirect(String),
    Incomplete,
    Body(&'static str),
    Error,
    Stall,
}

struct Fixture {
    model: Model,
    requests: Arc<Mutex<Vec<(String, Value)>>>,
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
        let task = tokio::spawn(async move {
            loop {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut bytes = Vec::new();
                let (head, body) = loop {
                    let mut chunk = [0; 4096];
                    let count = socket.read(&mut chunk).await.unwrap();
                    assert_ne!(count, 0);
                    bytes.extend_from_slice(&chunk[..count]);
                    assert!(bytes.len() < 262144);
                    if let Some(end) = bytes.windows(4).position(|v| v == b"\r\n\r\n") {
                        let head = String::from_utf8(bytes[..end].to_vec()).unwrap();
                        let length: usize = head
                            .lines()
                            .filter_map(|line| line.split_once(':'))
                            .find(|(name, _)| name.eq_ignore_ascii_case("content-length"))
                            .unwrap()
                            .1
                            .trim()
                            .parse()
                            .unwrap();
                        if bytes.len() >= end + 4 + length {
                            break (
                                head,
                                serde_json::from_slice::<Value>(&bytes[end + 4..end + 4 + length])
                                    .unwrap(),
                            );
                        }
                    }
                };
                capture.lock().unwrap().push((head, body.clone()));
                if matches!(reply, Reply::Disconnect) {
                    continue;
                }
                let mut receipt = json!({"attachment":body["occ3"], "consumed":true, "fallback_used":false, "injected_state":true});
                match &reply {
                    Reply::Mismatch(field) => {
                        receipt["attachment"][*field] = if receipt["attachment"][*field].is_string()
                        {
                            json!("other-binding")
                        } else {
                            json!(0)
                        };
                    }
                    Reply::Flag(field) => {
                        receipt[*field] = json!(*field == "fallback_used");
                    }
                    Reply::Unknown => {
                        receipt["unexpected"] = json!(true);
                    }
                    _ => {}
                }
                let mut headers = String::new();
                if body.get("occ3").is_some() && !matches!(reply, Reply::Missing) {
                    let encoded = if matches!(reply, Reply::Oversized) {
                        "x".repeat(MAX_INFERENCE_RECEIPT_BYTES + 1)
                    } else {
                        receipt.to_string()
                    };
                    headers.push_str(&format!("{INFERENCE_RECEIPT_HEADER}: {encoded}\r\n"));
                    if matches!(reply, Reply::Duplicate) {
                        headers.push_str(&format!("{INFERENCE_RECEIPT_HEADER}: {encoded}\r\n"));
                    }
                }
                let status = match &reply {
                    Reply::Status | Reply::Busy(_) => "503 Service Unavailable",
                    Reply::Redirect(destination) => {
                        headers.push_str(&format!("Location: {destination}/chat/completions\r\n"));
                        "307 Temporary Redirect"
                    }
                    _ => "200 OK",
                };
                let busy_body;
                let body = match reply {
                    Reply::Busy(code) => {
                        busy_body = format!("{{\"error\":{{\"message\":\"{code}\",\"type\":\"occ3_attachment_error\",\"code\":503}}}}");
                        busy_body.as_str()
                    }
                    Reply::Body(body) => body,
                    Reply::Incomplete | Reply::Stall => PARTIAL,
                    Reply::Error => "data: {\"error\":{\"message\":\"raw-error-secret\"}}\n\n",
                    _ => COMPLETE,
                };
                let response = if matches!(reply, Reply::Stall) {
                    format!("HTTP/1.1 {status}\r\nContent-Type: text/event-stream\r\n{headers}Transfer-Encoding: chunked\r\n\r\n{:x}\r\n{body}\r\n", body.len())
                } else {
                    format!("HTTP/1.1 {status}\r\nContent-Type: text/event-stream\r\n{headers}Content-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len())
                };
                // Refusing a bad receipt can close before the fixture body is written.
                let _ = socket.write_all(response.as_bytes()).await;
                if matches!(reply, Reply::Stall) {
                    let mut byte = [0];
                    let _ = socket.read(&mut byte).await;
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
                max_tokens: 1024.0,
                input: vec![InputModality::Text],
                ..Default::default()
            },
            requests,
            task,
        }
    }

    fn count(&self) -> usize {
        self.requests.lock().unwrap().len()
    }
    fn body(&self) -> Value {
        self.requests.lock().unwrap()[0].1.clone()
    }
}

fn epoch() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

fn lease_with_expiry(model: &Model, expires_at_epoch: u64) -> InferenceStateLease {
    InferenceStateLease::new(
        InferenceStateTarget {
            provider: model.provider.clone(),
            model_id: model.id.clone(),
            base_url: model.base_url.clone(),
        },
        InferenceAttachment {
            protocol_version: 1,
            request_id: "request-1".into(),
            message_id: "message-1".into(),
            connector_id: "generic-fixture".into(),
            receiver_session_id: "receiver-1".into(),
            model_fingerprint: "sha256:fixture-model".into(),
            runtime_id: "runtime-1".into(),
            runtime_generation: 4,
            state_handle: "opaque-handle-not-source-text".into(),
            expires_at_epoch,
            method: "native-state-v1".into(),
        },
    )
    .unwrap()
}

fn lease(model: &Model) -> InferenceStateLease {
    lease_with_expiry(model, epoch() + 300)
}

fn options(inference: Option<InferenceStateLease>) -> OpenAICompletionsOptions {
    OpenAICompletionsOptions {
        stream: StreamOptions {
            api_key: Some("synthetic-fixture-key".into()),
            timeout_ms: Some(3000.0),
            cache_retention: Some("none".into()),
            inference_state: inference,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn context() -> Context {
    Context::new(
        None,
        vec![Message::User(UserMessage::new(
            UserContent::Text("fixture question".into()),
            1,
        ))],
        None,
    )
}

async fn events(stream: &AssistantMessageEventStream) -> Vec<AssistantMessageEvent> {
    let mut events = Vec::new();
    while let Some(event) = tokio::time::timeout(Duration::from_secs(5), stream.next())
        .await
        .expect("stream deadline")
    {
        events.push(event);
    }
    events
}

fn error(events: &[AssistantMessageEvent]) -> &AssistantMessage {
    events
        .iter()
        .find_map(|event| match event {
            AssistantMessageEvent::Error { error, .. } => Some(error),
            _ => None,
        })
        .expect("terminal failure")
}

fn assert_no_replay_diagnostic(message: &AssistantMessage) {
    let details = message
        .diagnostics
        .as_ref()
        .unwrap()
        .iter()
        .find(|d| d.type_ == "provider_stream_failure")
        .unwrap()
        .details
        .as_ref()
        .unwrap();
    assert_eq!(details["kind"], "request_interrupted");
    assert_eq!(details["providerErrorType"], "occ3_inference_uncertain");
    assert_eq!(message.error_message.as_deref(), Some(
        "Inference state request did not complete; automatic replay is disabled."
    ));
    assert!(!serde_json::to_string(message)
        .unwrap()
        .contains("opaque-handle"));
    assert!(!serde_json::to_string(message)
        .unwrap()
        .contains("raw-error-secret"));
}

/// The bounded `occ3Cause` retained for an OCC interruption; the HTTP status
/// survives only on the labelled non-success response.
fn assert_occ3_cause(message: &AssistantMessage, cause: &str, status: Option<u64>) {
    let details = message
        .diagnostics
        .as_ref()
        .unwrap()
        .iter()
        .find(|d| d.type_ == "provider_stream_failure")
        .unwrap()
        .details
        .as_ref()
        .unwrap();
    assert_eq!(details["occ3Cause"], cause);
    assert_eq!(details.get("status").and_then(Value::as_u64), status);
}

#[tokio::test]
async fn ordinary_payload_is_unchanged_and_has_no_attachment_field() {
    let fixture = Fixture::new(Reply::Good).await;
    let stream = stream_openai_completions(&fixture.model, &context(), Some(options(None)));
    let output = stream.result().await;
    assert_eq!(output.stop_reason, "toolUse");
    assert_eq!(
        fixture.body(),
        json!({
            "model":"fixture-model", "messages":[{"role":"user","content":"fixture question"}],
            "stream":true, "stream_options":{"include_usage":true}, "store":false,
        })
    );
    assert_eq!(fixture.count(), 1);
}

#[tokio::test]
async fn attached_normal_and_simple_requests_preserve_actual_chat_tools_and_reasoning() {
    for simple in [false, true] {
        let fixture = Fixture::new(Reply::Good).await;
        let lease = lease(&fixture.model);
        let mut ctx = context();
        ctx.system_prompt = Some("fixture system".into());
        let mut thinking = ThinkingContent::new("history reasoning");
        thinking.thinking_signature = Some("reasoning_content".into());
        let mut assistant = AssistantMessage::new(
            API_OPENAI_COMPLETIONS,
            &fixture.model.provider,
            &fixture.model.id,
            2,
        );
        assistant.content = vec![
            ContentBlock::Thinking(thinking),
            ContentBlock::ToolCall(ToolCall::new(
                "call-in",
                "fixture_tool",
                json!({"value":0}).as_object().unwrap().clone(),
            )),
        ];
        assistant.stop_reason = STOP_REASON_TOOL_USE.into();
        ctx.messages.push(Message::Assistant(assistant));
        ctx.messages
            .push(Message::ToolResult(ToolResultMessage::new(
                "call-in",
                "fixture_tool",
                vec![ImageOrTextContent::Text(TextContent::new(
                    "ordinary tool data",
                ))],
                false,
                3,
            )));
        ctx.messages.push(Message::User(UserMessage::new(
            UserContent::Text("continue".into()),
            4,
        )));
        ctx.tools = Some(vec![Tool {
            name: "fixture_tool".into(),
            description: "fixture tool".into(),
            parameters: json!({"type":"object","properties":{"value":{"type":"integer"}}}),
        }]);
        let mut opts = options(Some(lease.clone()));
        opts.reasoning_effort = Some("high".into());
        opts.stream.on_payload = Some(Arc::new(|mut body, _| {
            assert!(
                body.get("occ3").is_none(),
                "opaque handles must not enter generic payload hooks"
            );
            body.as_object_mut().unwrap().remove("occ3");
            body["temperature"] = json!(0.25);
            Box::pin(async move { Some(body) })
        }));
        let inspected_lease = lease.clone();
        opts.stream.on_response = Some(Arc::new(move |response, _| {
            assert_eq!(inspected_lease.state(), InferenceStateLeaseState::Submitted);
            assert!(!response.headers.contains_key(INFERENCE_RECEIPT_HEADER));
            Box::pin(async {})
        }));
        let stream = if simple {
            stream_simple_openai_completions(
                &fixture.model,
                &ctx,
                Some(SimpleStreamOptions {
                    stream: opts.stream,
                    reasoning: Some("high".into()),
                    ..Default::default()
                }),
            )
        } else {
            stream_openai_completions(&fixture.model, &ctx, Some(opts))
        };
        let output = events(&stream).await;
        assert!(matches!(
            output.last(),
            Some(AssistantMessageEvent::Done { .. })
        ));
        assert_eq!(lease.state(), InferenceStateLeaseState::Consumed);
        let body = fixture.body();
        assert_eq!(
            body["occ3"],
            serde_json::to_value(lease.attachment()).unwrap()
        );
        assert_eq!(
            body["messages"][0],
            json!({"role":"developer","content":"fixture system"})
        );
        assert_eq!(
            body["messages"][2]["reasoning_content"],
            "history reasoning"
        );
        assert_eq!(body["messages"][2]["tool_calls"][0]["id"], "call-in");
        assert_eq!(body["messages"][3]["content"], "ordinary tool data");
        assert_eq!(body["messages"][4]["content"], "continue");
        assert_eq!(body["tools"][0]["function"]["name"], "fixture_tool");
        assert_eq!(body["reasoning_effort"], "high");
        assert_eq!(body["temperature"], 0.25);
        assert!(body["occ3"].get("base_url").is_none());
        assert!(body["occ3"].get("source_text").is_none());
        assert!(output.iter().any(|e| matches!(e, AssistantMessageEvent::TextDelta {delta,..} if delta == "native-fixture-answer")));
        assert!(output.iter().any(|e| matches!(e, AssistantMessageEvent::ToolCallEnd {tool_call,..} if tool_call.name == "fixture_tool" && tool_call.arguments.get("value") == Some(&json!(1)))));
        assert!(fixture.requests.lock().unwrap()[0]
            .0
            .starts_with("POST /v1/chat/completions HTTP/1.1"));
        // The guarded native one-shot transport must satisfy the occ3 server
        // contract: exactly one accept-encoding header, value identity.
        assert_single_identity_accept_encoding(&fixture.requests.lock().unwrap()[0].0);
        assert_eq!(fixture.count(), 1);
    }
}

/// The occ3 server contract (extensions/sglang-occ3 chat_adapter handle_around)
/// admits guarded native generation requests only with exactly one
/// accept-encoding header whose value is identity; the no-compression one-shot
/// client sends none by default, so the header must be forced on the wire.
fn assert_single_identity_accept_encoding(head: &str) {
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
}

#[tokio::test]
async fn missing_mismatched_or_untrusted_receipts_emit_no_start_content_tools_or_hooks() {
    for reply in [
        Reply::Missing,
        Reply::Mismatch("request_id"),
        Reply::Mismatch("runtime_generation"),
        Reply::Mismatch("state_handle"),
        Reply::Flag("consumed"),
        Reply::Flag("fallback_used"),
        Reply::Flag("injected_state"),
        Reply::Unknown,
        Reply::Oversized,
        Reply::Duplicate,
    ] {
        let fixture = Fixture::new(reply).await;
        let lease = lease(&fixture.model);
        let mut opts = options(Some(lease.clone()));
        let hooks = Arc::new(AtomicUsize::new(0));
        let seen = hooks.clone();
        opts.stream.on_response = Some(Arc::new(move |_, _| {
            seen.fetch_add(1, Ordering::SeqCst);
            Box::pin(async {})
        }));
        let raw_events = Arc::new(AtomicUsize::new(0));
        let seen = raw_events.clone();
        opts.stream.on_stream_observation = Some(Arc::new(move |phase| {
            if phase == "raw_event" {
                seen.fetch_add(1, Ordering::SeqCst);
            }
        }));
        let stream = stream_openai_completions(&fixture.model, &context(), Some(opts));
        let output = events(&stream).await;
        assert_eq!(output.len(), 1);
        assert_eq!(lease.state(), InferenceStateLeaseState::Uncertain);
        assert_eq!(hooks.load(Ordering::SeqCst), 0);
        assert_eq!(raw_events.load(Ordering::SeqCst), 0);
        assert_no_replay_diagnostic(error(&output));
        assert_occ3_cause(error(&output), "receipt_invalid", None);
        assert_eq!(fixture.count(), 1);
    }
}

#[tokio::test]
async fn wrong_provider_model_endpoint_api_and_queued_expiry_refuse_before_network() {
    let fixture = Fixture::new(Reply::Good).await;
    for field in ["provider", "model", "endpoint", "api"] {
        let lease = lease(&fixture.model);
        let mut wrong = fixture.model.clone();
        match field {
            "provider" => wrong.provider = "other-provider".into(),
            "model" => wrong.id = "other-model".into(),
            "endpoint" => wrong.base_url.push_str("/wrong"),
            _ => wrong.api = API_OPENAI_RESPONSES.into(),
        }
        let stream =
            stream_openai_completions(&wrong, &context(), Some(options(Some(lease.clone()))));
        let output = events(&stream).await;
        assert_eq!(output.len(), 1);
        assert!(error(&output)
            .error_message
            .as_ref()
            .unwrap()
            .contains("does not match"));
        assert_eq!(lease.state(), InferenceStateLeaseState::Prepared);
        assert_eq!(fixture.count(), 0);
    }
    let lease = lease_with_expiry(&fixture.model, epoch() + 1);
    let mut opts = options(Some(lease.clone()));
    opts.stream.on_payload = Some(Arc::new(|body, _| {
        Box::pin(async move {
            tokio::time::sleep(Duration::from_secs(2)).await;
            Some(body)
        })
    }));
    let stream = stream_openai_completions(&fixture.model, &context(), Some(opts));
    let output = events(&stream).await;
    assert!(error(&output)
        .error_message
        .as_ref()
        .unwrap()
        .contains("expired"));
    assert_eq!(lease.state(), InferenceStateLeaseState::Prepared);
    assert_eq!(fixture.count(), 0);
}

#[tokio::test]
async fn payload_hooks_cannot_forge_metadata_or_change_target() {
    let fixture = Fixture::new(Reply::Good).await;
    for field in ["occ3", "model", "stream", "object"] {
        let lease = lease(&fixture.model);
        let mut opts = options(Some(lease.clone()));
        opts.stream.on_payload = Some(Arc::new(move |mut body, _| {
            match field {
                "occ3" => body["occ3"] = json!({"state_handle":"forged"}),
                "model" => body["model"] = json!("wrong-model"),
                "stream" => body["stream"] = json!(false),
                _ => body = json!([]),
            }
            Box::pin(async move { Some(body) })
        }));
        let stream = stream_openai_completions(&fixture.model, &context(), Some(opts));
        assert_eq!(events(&stream).await.len(), 1);
        assert_eq!(lease.state(), InferenceStateLeaseState::Prepared);
        assert_eq!(fixture.count(), 0);
    }
    let mut opts = options(None);
    opts.stream.on_payload = Some(Arc::new(|mut body, _| {
        body["occ3"] = json!({"state_handle":"forged"});
        Box::pin(async move { Some(body) })
    }));
    let stream = stream_openai_completions(&fixture.model, &context(), Some(opts));
    assert!(error(&events(&stream).await)
        .error_message
        .as_ref()
        .unwrap()
        .contains("reserved"));
    assert_eq!(fixture.count(), 0);
}

#[tokio::test]
async fn shared_clones_allow_only_one_actual_send_and_never_replay_after_success() {
    let fixture = Fixture::new(Reply::Good).await;
    let lease = lease(&fixture.model);
    let first = stream_openai_completions(
        &fixture.model,
        &context(),
        Some(options(Some(lease.clone()))),
    );
    let second = stream_openai_completions(
        &fixture.model,
        &context(),
        Some(options(Some(lease.clone()))),
    );
    let (first, second) = tokio::join!(events(&first), events(&second));
    assert_eq!(
        first
            .iter()
            .chain(second.iter())
            .filter(|e| matches!(e, AssistantMessageEvent::Done { .. }))
            .count(),
        1
    );
    assert_eq!(lease.state(), InferenceStateLeaseState::Consumed);
    assert_eq!(fixture.count(), 1);
    let third = stream_openai_completions(
        &fixture.model,
        &context(),
        Some(options(Some(lease.clone()))),
    );
    assert_no_replay_diagnostic(error(&events(&third).await));
    assert_eq!(fixture.count(), 1);
}

#[tokio::test]
async fn ambiguous_send_http_error_and_incomplete_stream_never_retry() {
    for (reply, cause, status) in [
        (Reply::Disconnect, "send_transport_error", None),
        (Reply::Status, "http_non_success", Some(503)),
        (Reply::Incomplete, "incomplete_stream", None),
        (Reply::Error, "unknown", None),
    ] {
        let fixture = Fixture::new(reply).await;
        let lease = lease(&fixture.model);
        let stream = stream_openai_completions(
            &fixture.model,
            &context(),
            Some(options(Some(lease.clone()))),
        );
        let output = events(&stream).await;
        assert_no_replay_diagnostic(error(&output));
        assert_occ3_cause(error(&output), cause, status);
        assert_eq!(lease.state(), InferenceStateLeaseState::Uncertain);
        assert_eq!(fixture.count(), 1);
        let replay = stream_openai_completions(
            &fixture.model,
            &context(),
            Some(options(Some(lease.clone()))),
        );
        assert_no_replay_diagnostic(error(&events(&replay).await));
        assert_eq!(fixture.count(), 1);
    }
}

#[tokio::test]
async fn attached_http_failure_retains_bounded_cause_without_replay() {
    let fixture = Fixture::new(Reply::Status).await;
    let lease = lease(&fixture.model);
    let stream = stream_openai_completions(
        &fixture.model, &context(), Some(options(Some(lease.clone()))),
    );
    let output = events(&stream).await;
    assert_eq!(output.len(), 1);
    let message = error(&output);
    assert_no_replay_diagnostic(message);
    assert_eq!(message.error_message.as_deref(), Some(
        "Inference state request did not complete; automatic replay is disabled."
    ));
    let details = message.diagnostics.as_ref().unwrap().iter()
        .find(|d| d.type_ == "provider_stream_failure").unwrap()
        .details.as_ref().unwrap();
    assert_eq!(details["occ3Cause"], "http_non_success");
    assert_eq!(details["status"], 503);
    assert_eq!(lease.state(), InferenceStateLeaseState::Uncertain);
    assert_eq!(fixture.count(), 1);
    let replay = stream_openai_completions(
        &fixture.model, &context(), Some(options(Some(lease.clone()))),
    );
    assert_no_replay_diagnostic(error(&events(&replay).await));
    assert_eq!(fixture.count(), 1);

    let ordinary = stream_openai_completions(&fixture.model, &context(), Some(options(None)));
    let ordinary_output = events(&ordinary).await;
    let ordinary_details = error(&ordinary_output).diagnostics.as_ref().unwrap().iter()
        .find(|d| d.type_ == "provider_stream_failure").unwrap()
        .details.as_ref().unwrap();
    assert_eq!(ordinary_details["kind"], "server_error");
    assert_eq!(ordinary_details["status"], 503);
    assert!(ordinary_details.get("occ3Cause").is_none());
}

#[tokio::test]
async fn attached_requests_do_not_follow_redirects_or_forward_state() {
    let redirected = Fixture::new(Reply::Good).await;
    let fixture = Fixture::new(Reply::Redirect(redirected.model.base_url.clone())).await;
    let lease = lease(&fixture.model);
    let stream = stream_openai_completions(
        &fixture.model,
        &context(),
        Some(options(Some(lease.clone()))),
    );
    assert_no_replay_diagnostic(error(&events(&stream).await));
    assert_eq!(fixture.count(), 1);
    assert_eq!(redirected.count(), 0);
    assert_eq!(lease.state(), InferenceStateLeaseState::Uncertain);
}

#[tokio::test]
async fn drop_explicit_cancel_and_signal_abort_after_headers_remain_uncertain() {
    for method in ["drop", "lease", "signal"] {
        let fixture = Fixture::new(Reply::Stall).await;
        let lease = lease(&fixture.model);
        let signal = tokio_util::sync::CancellationToken::new();
        let mut opts = options(Some(lease.clone()));
        opts.stream.signal = Some(signal.clone());
        let stream = stream_openai_completions(&fixture.model, &context(), Some(opts));
        loop {
            let event = tokio::time::timeout(Duration::from_secs(5), stream.next())
                .await
                .unwrap()
                .unwrap();
            if matches!(event, AssistantMessageEvent::TextDelta { .. }) {
                break;
            }
        }
        assert_eq!(lease.state(), InferenceStateLeaseState::Submitted);
        let tasks = stream.task_receipt();
        match method {
            "lease" => {
                lease.cancel();
                let output = events(&stream).await;
                assert_no_replay_diagnostic(error(&output));
            }
            "signal" => {
                signal.cancel();
                assert_eq!(
                    error(&events(&stream).await).stop_reason,
                    STOP_REASON_ABORTED
                );
            }
            _ => {
                drop(stream);
            }
        }
        let status = tasks.settle(Duration::from_secs(5)).await;
        assert_eq!(status.pending_tasks, 0);
        assert_eq!(lease.state(), InferenceStateLeaseState::Uncertain);
        assert_eq!(fixture.count(), 1);
    }
}

#[tokio::test]
async fn cancelled_or_invalid_header_requests_remain_unsent() {
    let fixture = Fixture::new(Reply::Good).await;
    let lease = lease(&fixture.model);
    lease.cancel();
    let stream = stream_openai_completions(
        &fixture.model,
        &context(),
        Some(options(Some(lease.clone()))),
    );
    assert_eq!(events(&stream).await.len(), 1);
    assert_eq!(lease.state(), InferenceStateLeaseState::Cancelled);
    for (name, value) in [("x-fixture", "invalid\r\nvalue"), ("Host", "other.invalid")] {
        let lease = lease_with_expiry(&fixture.model, epoch() + 300);
        let mut opts = options(Some(lease.clone()));
        opts.stream.headers = Some([(name.into(), value.into())].into_iter().collect());
        let stream = stream_openai_completions(&fixture.model, &context(), Some(opts));
        assert_eq!(events(&stream).await.len(), 1);
        assert_eq!(lease.state(), InferenceStateLeaseState::Prepared);
        assert_eq!(fixture.count(), 0);
    }
}

#[tokio::test]
async fn attached_stream_requires_successful_finish_and_exact_done_even_with_matching_receipt() {
    let finish_only = "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n";
    let done_only = "data: [DONE]\n\n";
    for body in [
        "",
        done_only,
        finish_only,
        "data: {\"error\":{\"message\":\"raw-error-secret\"}}\n\ndata: [DONE]\n\n",
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]garbage\n\n",
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"content_filter\"}]}\n\ndata: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n",
    ] {
        let fixture = Fixture::new(Reply::Body(body)).await;
        let lease = lease(&fixture.model);
        let stream = stream_openai_completions(&fixture.model, &context(), Some(options(Some(lease.clone()))));
        let output = events(&stream).await;
        assert!(!output.iter().any(|event| matches!(event, AssistantMessageEvent::Done { .. } | AssistantMessageEvent::ToolCallEnd { .. })));
        assert_no_replay_diagnostic(error(&output));
        assert_eq!(lease.state(), InferenceStateLeaseState::Uncertain);
        assert_eq!(fixture.count(), 1);
    }
    // The ordinary provider's EOF-or-DONE compatibility stays unchanged.
    for body in [done_only, finish_only] {
        let fixture = Fixture::new(Reply::Body(body)).await;
        let stream = stream_openai_completions(&fixture.model, &context(), Some(options(None)));
        assert!(matches!(
            events(&stream).await.last(),
            Some(AssistantMessageEvent::Done { .. })
        ));
    }
}


/// Fix 1 (receiver busy): a pre-event 503 `scheduler_busy` /
/// `admission_capacity` (`occ3_busy_batch` / `occ3_busy` server-side) is a
/// distinct retryable `occ3_receiver_busy` failure. The lease closes as
/// proven non-use (`Cancelled`, never `Uncertain`), the request is not
/// resent, and the attachment cannot be replayed.
#[tokio::test]
async fn pre_event_receiver_busy_is_retryable_non_use_without_resend() {
    for code in ["scheduler_busy", "admission_capacity", "occ3_busy_batch"] {
        let fixture = Fixture::new(Reply::Busy(code)).await;
        let lease = lease(&fixture.model);
        let stream = stream_openai_completions(
            &fixture.model, &context(), Some(options(Some(lease.clone()))),
        );
        let output = events(&stream).await;
        let message = error(&output);
        let details = message.diagnostics.as_ref().unwrap().iter()
            .find(|d| d.type_ == "provider_stream_failure").unwrap()
            .details.as_ref().unwrap();
        assert_eq!(details["providerErrorType"], "occ3_receiver_busy", "{code}");
        assert_eq!(details["kind"], "overloaded", "{code}: retryable, not request_interrupted");
        assert_eq!(details["status"], 503);
        assert!(details.get("occ3Cause").is_none());
        assert_eq!(lease.state(), InferenceStateLeaseState::Cancelled, "{code}: proven non-use");
        assert!(lease.refused_busy(), "{code}: the lease records the busy refusal");
        assert_eq!(fixture.count(), 1, "{code}: no automatic resend");
        let replay = stream_openai_completions(
            &fixture.model, &context(), Some(options(Some(lease.clone()))),
        );
        let _ = events(&replay).await;
        assert_eq!(fixture.count(), 1, "{code}: the closed lease cannot be replayed");
    }
}

/// An unrecognised 503 body on an attached request keeps the existing
/// uncertain, no-replay contract.
#[tokio::test]
async fn attached_503_with_unknown_code_stays_uncertain() {
  // `admission_timeout` is busy only for a SOURCE capture: on the receiver
  // path the attempt fence may be charged, so attachment non-use is unproven.
  for code in ["native_completion_failed", "admission_timeout"] {
    let fixture = Fixture::new(Reply::Busy(code)).await;
    let lease = lease(&fixture.model);
    let stream = stream_openai_completions(
        &fixture.model, &context(), Some(options(Some(lease.clone()))),
    );
    let output = events(&stream).await;
    assert_no_replay_diagnostic(error(&output));
    assert_occ3_cause(error(&output), "http_non_success", Some(503));
    assert_eq!(lease.state(), InferenceStateLeaseState::Uncertain);
    assert!(!lease.refused_busy(), "an unknown 503 is not a busy refusal");
    assert_eq!(fixture.count(), 1);
  }
}
