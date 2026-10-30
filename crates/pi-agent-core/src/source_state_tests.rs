//! Actual native Agent source request and tool provenance; localhost fixtures only.
use crate::agent::{Agent, AgentOptions, PromptInput};
use crate::types::{AgentState, AgentTool, AgentToolResult, SourceCaptureHooks, StreamFn};
use pi_ai::inference_attachment::InferenceStateTarget;
use pi_ai::source_capture::*;
use pi_ai::types::{
    AssistantMessage, AssistantMessageEvent, ContentBlock, Model, ToolCall, UserContent,
    UserMessage, API_OPENAI_COMPLETIONS,
};
use pi_ai::utils::event_stream::AssistantMessageEventStream;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

fn lease(model: &Model) -> SourceCaptureLease {
    SourceCaptureLease::new(
        InferenceStateTarget {
            provider: model.provider.clone(),
            model_id: model.id.clone(),
            base_url: model.base_url.clone(),
        },
        SourceCaptureIntent {
            schema: SOURCE_CAPTURE_SCHEMA.into(),
            capture_id: "source-capture".into(),
            source_session_id: "source-session".into(),
            source_request_id: "client-attempt".into(),
            turn_nonce: "native-turn".into(),
            model_fingerprint: "a".repeat(64),
            runtime_id: "runtime-1".into(),
            runtime_generation: 1,
            method: "statebridge-v1".into(),
            retained_count_limit: 128,
            max_capture_bytes: 4096,
            expires_at_epoch: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs()
                + 300,
            span_policy: SOURCE_SPAN_POLICY.into(),
        },
        SourceCaptureProfile {
            layer_index: 3,
            hidden_dim: 8,
            dtype: "bfloat16".into(),
            layout_digest: "b".repeat(64),
            expected_ranks: 8,
        },
        SourceCaptureLimits {
            deadline: Instant::now() + Duration::from_secs(10),
            max_response_bytes: 65536,
        },
    )
    .unwrap()
}
fn tool(calls: Arc<AtomicUsize>, captured: Arc<Mutex<Option<SourceCaptureLease>>>) -> AgentTool {
    AgentTool {
        name: "agent_message".into(),
        description: "handoff fixture".into(),
        label: "handoff".into(),
        parameters: json!({"type":"object","properties":{"control":{"type":"string"}},"required":["control"]}),
        prepare_arguments: None,
        execution_mode: None,
        execute: Arc::new(move |id, args, _, _| {
            let source = captured
                .lock()
                .unwrap()
                .clone()
                .expect("native completed hook precedes tools");
            assert!(source.validate_tool(&id, "agent_message", &args).is_ok());
            calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { Ok(AgentToolResult::new(Vec::new(), json!({"fixture":true}))) })
        }),
    }
}
fn options(model: Model) -> AgentOptions {
    AgentOptions {
        initial_state: Some(AgentState {
            model,
            system_prompt: "native source fixture".into(),
            ..Default::default()
        }),
        get_api_key: Some(Arc::new(|_| {
            Box::pin(async { Some("synthetic-fixture-key".into()) })
        })),
        // Other crate tests reset the global provider registry concurrently.
        // Pin the real provider function, not a mock and not shared test state.
        stream_fn: Some(Arc::new(|model, context, options| {
            Box::pin(async move {
                pi_ai::providers::openai_completions::stream_simple_openai_completions(
                    &model,
                    &context,
                    Some(options),
                )
            })
        })),
        ..Default::default()
    }
}
async fn server() -> (Model, Arc<Mutex<Vec<Value>>>, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let model = Model::new(
        "fixture-model",
        "fixture",
        API_OPENAI_COMPLETIONS,
        "native-fixture",
        format!("http://{}/v1", listener.local_addr().unwrap()),
    );
    let requests = Arc::new(Mutex::new(Vec::new()));
    let seen = requests.clone();
    let task = tokio::spawn(async move {
        loop {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut bytes = Vec::new();
            let body = loop {
                let mut chunk = [0; 4096];
                let n = socket.read(&mut chunk).await.unwrap();
                assert!(n > 0);
                bytes.extend_from_slice(&chunk[..n]);
                assert!(bytes.len() < 1048576);
                if let Some(end) = bytes.windows(4).position(|v| v == b"\r\n\r\n") {
                    let head = String::from_utf8_lossy(&bytes[..end]);
                    assert!(head.starts_with("POST /v1/chat/completions"));
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
                        break serde_json::from_slice::<Value>(&bytes[end + 4..end + 4 + len])
                            .unwrap();
                    }
                }
            };
            seen.lock().unwrap().push(body.clone());
            let captured = body.get("occ3_capture");
            let (delta, reason) = if captured.is_some() {
                (
                    json!({"role":"assistant","reasoning_content":"source private reasoning","tool_calls":[{"index":0,"id":"handoff-call","type":"function","function":{"name":"agent_message","arguments":"{\"control\":\"continue\"}"}}]}),
                    "tool_calls",
                )
            } else {
                (
                    json!({"role":"assistant","content":"ordinary continuation"}),
                    "stop",
                )
            };
            let output = format!(
                "data: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
                json!({"id":"actual-native-rid","choices":[{"index":0,"delta":delta,"finish_reason":null}]}),
                json!({"id":"actual-native-rid","choices":[{"index":0,"delta":{},"finish_reason":reason}]})
            );
            let header = if let Some(intent) = captured {
                let intent: SourceCaptureIntent = serde_json::from_value(intent.clone()).unwrap();
                format!(
                    "{SOURCE_RECEIPT_HEADER}: {}\r\n",
                    serde_json::to_string(&SourceCaptureReceipt {
                        schema: SOURCE_RECEIPT_SCHEMA.into(),
                        intent_digest: intent.digest().unwrap(),
                        source_handle: "private-owner-handle".into(),
                        native_rid: "actual-native-rid".into(),
                        response_id: "actual-native-rid".into(),
                        capture_commit_id: "capture-commit".into(),
                        accepted_span_start: 1,
                        accepted_span_end: 3,
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
                        output_sha256: sha256(output.as_bytes()),
                        expires_at_epoch: intent.expires_at_epoch,
                        byte_count: 32,
                        span: None,
                    })
                    .unwrap()
                )
            } else {
                String::new()
            };
            let response=format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n{header}Content-Length: {}\r\nConnection: close\r\n\r\n{output}",output.len());
            socket.write_all(response.as_bytes()).await.unwrap();
            socket.shutdown().await.unwrap();
        }
    });
    (model, requests, task)
}
fn hooks(
    source: SourceCaptureLease,
    captured: Arc<Mutex<Option<SourceCaptureLease>>>,
    reject: bool,
) -> SourceCaptureHooks {
    let used = Arc::new(AtomicBool::new(false));
    SourceCaptureHooks {
        resolve: Arc::new(move |request| {
            assert!(request
                .context
                .tools
                .as_ref()
                .unwrap()
                .iter()
                .any(|t| t.name == "agent_message"));
            let result = (!used.swap(true, Ordering::SeqCst)).then(|| source.clone());
            Box::pin(async move { Ok(result) })
        }),
        completed: Arc::new(move |result| {
            assert_eq!(result.lease.state(), SourceCaptureState::Captured);
            assert_eq!(
                result.assistant_message.response_id.as_deref(),
                Some("actual-native-rid")
            );
            assert!(result
                .lease
                .validate_tool(
                    "handoff-call",
                    "agent_message",
                    &json!({"control":"continue"})
                )
                .is_ok());
            let captured = captured.clone();
            Box::pin(async move {
                if reject {
                    return Err(anyhow::anyhow!("ambiguous handoff fixture"));
                }
                *captured.lock().unwrap() = Some(result.lease);
                Ok(())
            })
        }),
    }
}
#[tokio::test]
async fn native_source_actual_request_binds_completed_tool_then_no_capture_on_continuation() {
    let (model, requests, server) = server().await;
    let source = lease(&model);
    let captured = Arc::new(Mutex::new(None));
    let calls = Arc::new(AtomicUsize::new(0));
    let mut opts = options(model);
    opts.initial_state.as_mut().unwrap().tools = Some(vec![tool(calls.clone(), captured.clone())]);
    opts.source_capture = Some(hooks(source.clone(), captured.clone(), false));
    let agent = Agent::new(opts);
    agent
        .prompt(PromptInput::Text {
            input: "private source task".into(),
            images: Vec::new(),
        })
        .await
        .unwrap();
    assert!(
        agent.state().error_message.is_none(),
        "{:?}",
        agent.state().error_message
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(source.state(), SourceCaptureState::Captured);
    assert!(captured.lock().unwrap().is_some());
    agent
        .prompt(PromptInput::Text {
            input: "later independent request".into(),
            images: Vec::new(),
        })
        .await
        .unwrap();
    assert_eq!(
        requests
            .lock()
            .unwrap()
            .iter()
            .map(|v| v.get("occ3_capture").is_some())
            .collect::<Vec<_>>(),
        [true, false, false]
    );
    assert_eq!(
        requests.lock().unwrap()[0]["occ3_capture"]["source_request_id"],
        "client-attempt"
    );
    server.abort();
}
#[tokio::test]
async fn native_source_provenance_callback_failure_prevents_tools_and_retires_lease() {
    let (model, requests, server) = server().await;
    let source = lease(&model);
    let captured = Arc::new(Mutex::new(None));
    let calls = Arc::new(AtomicUsize::new(0));
    let mut opts = options(model);
    opts.initial_state.as_mut().unwrap().tools = Some(vec![tool(calls.clone(), captured.clone())]);
    opts.source_capture = Some(hooks(source.clone(), captured, true));
    let agent = Agent::new(opts);
    agent
        .prompt(PromptInput::Text {
            input: "source".into(),
            images: Vec::new(),
        })
        .await
        .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(source.state(), SourceCaptureState::Retired);
    assert_eq!(requests.lock().unwrap().len(), 1);
    assert!(agent.state().error_message.unwrap().contains("provenance"));
    // TS14 F5 regression, red-first: a settlement refusal must surface the
    // completed callback's bounded reason so the live sub-cause (which check
    // rejected the captured turn) is diagnosable from the harness evidence
    // alone. The public provider code stays the fixed generic
    // occ3_source_uncertain.
    assert!(agent
        .state()
        .error_message
        .unwrap()
        .contains("ambiguous handoff fixture"));
    server.abort();
}
#[tokio::test]
async fn native_source_ignored_by_stream_cannot_run_handoff_tools() {
    let model = Model::new(
        "fixture",
        "fixture",
        API_OPENAI_COMPLETIONS,
        "fixture",
        "http://127.0.0.1:1/v1",
    );
    let source = lease(&model);
    let calls = Arc::new(AtomicUsize::new(0));
    let captured = Arc::new(Mutex::new(None));
    let mut opts = options(model);
    opts.initial_state.as_mut().unwrap().tools = Some(vec![tool(calls.clone(), captured.clone())]);
    opts.source_capture = Some(hooks(source.clone(), captured, false));
    opts.stream_fn = Some(Arc::new(
        |model: Model, _, _| -> futures::future::BoxFuture<'static, AssistantMessageEventStream> {
            let stream = AssistantMessageEventStream::new();
            let mut message = AssistantMessage::new(model.api, model.provider, model.id, 1);
            message.stop_reason = "toolUse".into();
            message.content.push(ContentBlock::ToolCall(ToolCall::new(
                "handoff-call",
                "agent_message",
                json!({"control":"continue"}).as_object().unwrap().clone(),
            )));
            stream.push(AssistantMessageEvent::Done {
                reason: message.stop_reason.clone(),
                message,
            });
            stream.end(None);
            Box::pin(async move { stream })
        },
    ) as StreamFn);
    let agent = Agent::new(opts);
    agent
        .prompt(PromptInput::Text {
            input: "source".into(),
            images: Vec::new(),
        })
        .await
        .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(source.state(), SourceCaptureState::Cancelled);
    assert!(agent.state().error_message.unwrap().contains("provenance"));
    // TS14 F5 regression, red-first: the pre-callback branch must name the
    // lease state that blocked settlement. The fixture stream never submits
    // the armed lease, so settlement sees Prepared (the guard cancels it only
    // after the turn ends), and the refusal must say so.
    assert!(agent
        .state()
        .error_message
        .unwrap()
        .contains("occ3_capture_state_prepared"));
}
#[tokio::test]
async fn native_source_cancel_after_resolution_never_dispatches() {
    let model = Model::new(
        "fixture",
        "fixture",
        API_OPENAI_COMPLETIONS,
        "fixture",
        "http://127.0.0.1:1/v1",
    );
    let source = lease(&model);
    let sent = Arc::new(AtomicUsize::new(0));
    let calls = sent.clone();
    let mut opts = options(model);
    opts.stream_fn = Some(Arc::new(move |_, _, _| {
        calls.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { AssistantMessageEventStream::new() })
    }));
    opts.source_capture = Some(SourceCaptureHooks {
        resolve: Arc::new({
            let source = source.clone();
            move |request| {
                request.signal.unwrap().cancel();
                let source = source.clone();
                Box::pin(async move { Ok(Some(source)) })
            }
        }),
        completed: Arc::new(|_| Box::pin(async { panic!("must not complete") })),
    });
    let agent = Agent::new(opts);
    agent
        .prompt(PromptInput::Messages(vec![UserMessage::new(
            UserContent::Text("source".into()),
            1,
        )
        .into()]))
        .await
        .unwrap();
    assert_eq!(sent.load(Ordering::SeqCst), 0);
    assert_eq!(source.state(), SourceCaptureState::Cancelled);
}
