use std::sync::{Arc, Mutex};
use std::time::Duration;

use pi_agent_core::types::{AgentMessage, ThinkingLevel};
use pi_ai::types::{AssistantMessage, ContentBlock, Model, TextContent, UserContent, UserMessage};
use pi_coding_agent::core::compaction::compaction::{
    default_summary_call_runner, generate_summary, ProviderRetryPolicy, SummaryCallRunner,
    SummarySlice,
};
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::task::JoinHandle;

const VALID: &str = "## Goal\nComplete.\n## Constraints & Preferences\nNone.\n## Progress\nDone.\n## Key Decisions\nWait.\n## Next Steps\nReview.\n## Critical Context\nSaved.";

struct Fixture {
    model: Model,
    requests: Arc<Mutex<Vec<Value>>>,
    server: JoinHandle<()>,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        self.server.abort();
    }
}

impl Fixture {
    async fn new(finishes: Vec<&'static str>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}/v1", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let seen = requests.clone();
        let server = tokio::spawn(async move {
            for finish in finishes {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut bytes = Vec::new();
                let (body_start, body_len) = loop {
                    let mut buffer = [0; 4096];
                    let count = socket.read(&mut buffer).await.unwrap();
                    assert_ne!(count, 0, "request ended before its headers");
                    bytes.extend_from_slice(&buffer[..count]);
                    assert!(
                        bytes.len() < 1_000_000,
                        "unexpectedly large fixture request"
                    );
                    if let Some(end) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
                        let headers = std::str::from_utf8(&bytes[..end]).unwrap();
                        assert!(headers.starts_with("POST /v1/chat/completions HTTP/1.1"));
                        let length = headers
                            .lines()
                            .find_map(|line| {
                                let (key, value) = line.split_once(':')?;
                                key.eq_ignore_ascii_case("content-length")
                                    .then(|| value.trim().parse::<usize>().unwrap())
                            })
                            .expect("request content length");
                        assert!(length < 1_000_000);
                        break (end + 4, length);
                    }
                };
                while bytes.len() < body_start + body_len {
                    let mut buffer = [0; 4096];
                    let count = socket.read(&mut buffer).await.unwrap();
                    assert_ne!(count, 0, "request ended before its body");
                    bytes.extend_from_slice(&buffer[..count]);
                }
                let request: Value =
                    serde_json::from_slice(&bytes[body_start..body_start + body_len]).unwrap();
                seen.lock().unwrap().push(request);
                let chunk = json!({
                    "id": "fixture-summary", "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {"content": if finish == "stop" { VALID } else { "PARTIAL-HANDOFF" }}, "finish_reason": finish}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
                });
                let body = format!("data: {chunk}\n\ndata: [DONE]\n\n");
                let reply = format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
                socket.write_all(reply.as_bytes()).await.unwrap();
            }
        });
        let mut model = Model::new(
            "fixture-summary",
            "Fixture",
            "openai-completions",
            "fixture",
            endpoint,
        );
        model.context_window = 128_000.0;
        model.max_tokens = 32_000.0;
        model.reasoning = true;
        Self {
            model,
            requests,
            server,
        }
    }
}

async fn summarize(model: &Model, runner: SummaryCallRunner) -> Result<SummarySlice, String> {
    let messages = [AgentMessage::from(UserMessage::new(
        UserContent::Text("Preserve the original history and its acceptance gates.".into()),
        0,
    ))];
    let retry = ProviderRetryPolicy {
        enabled: false,
        max_retries: 0,
        base_delay_ms: 0.0,
        max_retry_delay_ms: 0.0,
    };
    tokio::time::timeout(
        Duration::from_secs(5),
        generate_summary(
            &messages,
            model,
            16_384.0,
            "synthetic-fixture-key",
            None,
            None,
            None,
            Some(&ThinkingLevel::Low),
            Some(&retry),
            runner,
            &"off".into(),
        ),
    )
    .await
    .expect("local summarization fixture timed out")
}

fn output_budget(request: &Value) -> f64 {
    request
        .get("max_completion_tokens")
        .or_else(|| request.get("max_tokens"))
        .and_then(Value::as_f64)
        .expect("wire output budget")
}

#[tokio::test]
async fn chat_finish_length_recovers_with_a_larger_wire_budget_and_identical_history() {
    let fixture = Fixture::new(vec!["length", "stop"]).await;
    let result = summarize(&fixture.model, default_summary_call_runner(None))
        .await
        .unwrap();
    assert_eq!(result.summary, VALID);
    assert_eq!(
        result.usage.unwrap().output,
        40.0,
        "count both summary attempts"
    );
    let requests = fixture.requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(output_budget(&requests[0]), 13_107.0);
    assert_eq!(output_budget(&requests[1]), 26_214.0);
    let mut initial = requests[0].clone();
    let mut retried = requests[1].clone();
    for request in [&mut initial, &mut retried] {
        request.as_object_mut().unwrap().remove("max_tokens");
        request
            .as_object_mut()
            .unwrap()
            .remove("max_completion_tokens");
        assert_eq!(request["reasoning_effort"], "low");
        assert!(request.get("tools").is_none());
    }
    assert_eq!(
        initial, retried,
        "only the output budget changes, never the transcript"
    );
}

#[tokio::test]
async fn chat_repeated_length_preserves_history_without_a_third_attempt() {
    let fixture = Fixture::new(vec!["length", "length"]).await;
    let error = summarize(&fixture.model, default_summary_call_runner(None))
        .await
        .unwrap_err();
    assert!(error.contains("stop_reason=length"), "{error}");
    assert!(error.contains("existing conversation preserved"), "{error}");
    assert_eq!(fixture.requests.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn chat_output_ceiling_and_filtering_do_not_trigger_another_wire_call() {
    for (finish, max_tokens) in [("length", 13_107.0), ("content_filter", 32_000.0)] {
        let mut fixture = Fixture::new(vec![finish]).await;
        fixture.model.max_tokens = max_tokens;
        assert!(summarize(&fixture.model, default_summary_call_runner(None))
            .await
            .is_err());
        assert_eq!(fixture.requests.lock().unwrap().len(), 1);
    }
}

#[tokio::test]
async fn budget_retry_requires_provider_specific_exhaustion_evidence() {
    for (api, raw, calls) in [
        ("openai-completions", None, 2),
        ("openai-completions", Some("length"), 2),
        ("openai-completions", Some("content_filter"), 1),
        ("openai-completions", Some("refusal"), 1),
        ("openai-completions", Some("unknown"), 1),
        ("openai-responses", None, 1),
        ("azure-openai-responses", None, 1),
        ("openai-codex-responses", None, 1),
        ("custom-summary-api", None, 1),
    ] {
        let mut model = Model::new("fixture", "Fixture", api, "faux", "https://fixture.invalid");
        model.context_window = 128_000.0;
        model.max_tokens = 32_000.0;
        let count = Arc::new(Mutex::new(0));
        let seen = count.clone();
        let runner: SummaryCallRunner = Arc::new(move |_| {
            *seen.lock().unwrap() += 1;
            Box::pin(async move {
                Ok(AssistantMessage {
                    stop_reason: "length".into(),
                    stop_reason_raw: raw.map(str::to_string),
                    content: vec![ContentBlock::Text(TextContent::new(VALID))],
                    ..Default::default()
                })
            })
        });
        let error = summarize(&model, runner).await.unwrap_err();
        assert!(error.contains("existing conversation preserved"), "{error}");
        assert_eq!(*count.lock().unwrap(), calls, "{api}, raw={raw:?}");
    }
}
