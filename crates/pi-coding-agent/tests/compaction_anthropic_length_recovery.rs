use std::sync::{Arc, Mutex};
use std::time::Duration;

use pi_agent_core::types::{AgentMessage, ThinkingLevel};
use pi_ai::types::{Model, UserContent, UserMessage};
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
    async fn new(id: &str, finishes: Vec<&'static str>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
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
                        assert!(headers.starts_with("POST /v1/messages HTTP/1.1"));
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
                let events = [
                    json!({"type":"message_start","message":{"id":"fixture","usage":{"input_tokens":100,"output_tokens":0}}}),
                    json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}),
                    json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":if finish == "end_turn" { VALID } else { "PARTIAL-HANDOFF" }}}),
                    json!({"type":"content_block_stop","index":0}),
                    json!({"type":"message_delta","delta":{"stop_reason":finish},"usage":{"output_tokens":20}}),
                    json!({"type":"message_stop"}),
                ];
                let body: String = events
                    .iter()
                    .map(|event| {
                        format!(
                            "event: {}\ndata: {event}\n\n",
                            event["type"].as_str().unwrap()
                        )
                    })
                    .collect();
                let reply = format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
                socket.write_all(reply.as_bytes()).await.unwrap();
            }
        });
        let mut model = Model::new(id, "Fixture", "anthropic-messages", "fixture", endpoint);
        model.context_window = 1_000_000.0;
        model.max_tokens = 128_000.0;
        model.thinking_level_map =
            Some(serde_json::from_value(json!({"low":"low","high":"high","max":"max"})).unwrap());
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
            Some(&ThinkingLevel::Max),
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
async fn adaptive_claude_recovers_length_preserving_history_and_effort() {
    for id in ["claude-opus-5-5", "claude-sonnet-5", "claude-sonnet-4-6"] {
        let fixture = Fixture::new(id, vec!["max_tokens", "end_turn"]).await;
        let result = summarize(&fixture.model, default_summary_call_runner(None))
            .await
            .unwrap();
        assert_eq!(result.summary, VALID);
        assert_eq!(result.usage.unwrap().output, 40.0);
        let requests = fixture.requests.lock().unwrap();
        assert_eq!(requests.len(), 2, "{id}");
        assert_eq!(output_budget(&requests[0]), 29_491.0);
        assert_eq!(output_budget(&requests[1]), 58_982.0);
        let mut initial = requests[0].clone();
        let mut retried = requests[1].clone();
        for request in [&mut initial, &mut retried] {
            assert!(request["max_tokens"].is_u64());
            request.as_object_mut().unwrap().remove("max_tokens");
            assert_eq!(request["thinking"]["type"], "adaptive");
            assert_eq!(request["output_config"]["effort"], "max");
            assert!(request.get("tools").is_none());
        }
        assert_eq!(initial, retried);
    }
}

#[tokio::test]
async fn repeated_exhaustion_preserves_conversation_and_reports_length() {
    let fixture = Fixture::new("claude-opus-5-5", vec!["max_tokens", "max_tokens"]).await;
    let error = summarize(&fixture.model, default_summary_call_runner(None))
        .await
        .unwrap_err();
    assert!(error.contains("existing conversation preserved"), "{error}");
    assert_eq!(
        pi_coding_agent::core::compaction::compaction::classify_summary_failure(&error),
        pi_coding_agent::core::compaction::compaction::SummaryFailureKind::LengthExhausted
    );
    assert_eq!(fixture.requests.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn refusal_unknown_stop_and_output_ceiling_do_not_retry() {
    for (finish, ceiling) in [
        ("refusal", 128_000.0),
        ("unknown", 128_000.0),
        ("max_tokens", 20_000.0),
    ] {
        let mut fixture = Fixture::new("claude-opus-5-5", vec![finish]).await;
        fixture.model.max_tokens = ceiling;
        assert!(summarize(&fixture.model, default_summary_call_runner(None))
            .await
            .is_err());
        assert_eq!(fixture.requests.lock().unwrap().len(), 1);
    }
}

#[tokio::test]
async fn legacy_claude_thinking_budget_is_added_only_by_adapter() {
    let fixture = Fixture::new("claude-haiku-4-5", vec!["end_turn"]).await;
    summarize(&fixture.model, default_summary_call_runner(None))
        .await
        .unwrap();
    let requests = fixture.requests.lock().unwrap();
    assert_eq!(output_budget(&requests[0]), 29_491.0);
    assert_eq!(requests[0]["thinking"]["budget_tokens"], 16_384);
}
