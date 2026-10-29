use std::sync::{Arc, Mutex};
use std::time::Duration;

use pi_agent_core::types::{AgentMessage, ThinkingLevel};
use pi_ai::types::{Model, UserContent, UserMessage};
use pi_coding_agent::core::compaction::compaction::{
    classify_summary_failure, default_summary_call_runner, generate_summary, ProviderRetryPolicy,
    SummaryCallRunner, SummaryFailureKind, SummarySlice,
};
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::task::JoinHandle;

const VALID: &str = "## Goal\nComplete.\n## Constraints & Preferences\nNone.\n## Progress\nDone.\n## Key Decisions\nWait.\n## Next Steps\nReview.\n## Critical Context\nSaved.";

struct Fixture {
    model: Model,
    requests: Arc<Mutex<Vec<Value>>>,
    headers: Arc<Mutex<Vec<String>>>,
    server: JoinHandle<()>,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        self.server.abort();
    }
}

impl Fixture {
    async fn new(id: &str, replies: Vec<(&'static str, Option<&'static str>)>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let seen = requests.clone();
        let headers = Arc::new(Mutex::new(Vec::new()));
        let seen_headers = headers.clone();
        let server = tokio::spawn(async move {
            for (finish, recovery_model) in replies {
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
                        seen_headers.lock().unwrap().push(headers.to_string());
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
                let mut events = vec![
                    json!({"type":"message_start","message":{"id":"fixture","model":"claude-opus-5-5","usage":{"input_tokens":100,"output_tokens":0}}}),
                    json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}),
                    json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":if finish == "end_turn" { VALID } else { "PARTIAL-HANDOFF" }}}),
                    json!({"type":"content_block_stop","index":0}),
                    json!({"type":"message_delta","delta":{"stop_reason":finish,"stop_details":if finish == "refusal" { json!({"type":"refusal","category":"reasoning_extraction","explanation":"fixture refusal"}) } else { Value::Null }},"usage":{"output_tokens":20}}),
                    json!({"type":"message_stop"}),
                ];
                if let Some(served) = recovery_model {
                    events.insert(1, json!({"type":"content_block_start","index":8,"content_block":{"type":"fallback","from":{"model":"claude-opus-5-5"},"to":{"model":served},"trigger":{"type":"refusal","category":"cyber"}}}));
                    events.insert(2, json!({"type":"content_block_stop","index":8}));
                    let terminal = events
                        .iter_mut()
                        .find(|event| event["type"] == "message_delta")
                        .unwrap();
                    terminal["usage"]["iterations"] = json!([
                        {"type":"message","model":"claude-opus-5-5","input_tokens":100,"output_tokens":3},
                        {"type":"fallback_message","model":served,"input_tokens":105,"output_tokens":20}
                    ]);
                }
                let body: String = events
                    .iter()
                    .map(|event| {
                        format!(
                            "event: {}\ndata: {event}\n\n",
                            event["type"].as_str().unwrap()
                        )
                    })
                    .collect();
                let reply = format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nrequest-id: req_fixture_refusal\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
                socket.write_all(reply.as_bytes()).await.unwrap();
            }
        });
        let mut model = Model::new(id, "Fixture", "anthropic-messages", "anthropic", endpoint);
        model.context_window = 1_000_000.0;
        model.max_tokens = 128_000.0;
        model.thinking_level_map =
            Some(serde_json::from_value(json!({"low":"low","high":"high","max":"max"})).unwrap());
        model.reasoning = true;
        Self {
            model,
            requests,
            headers,
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
            "sk-ant-oat01-synthetic-fixture",
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

#[tokio::test]
async fn oauth_summary_uses_provider_selected_claude_without_changing_chat_model() {
    let fixture = Fixture::new(
        "claude-opus-5-5",
        vec![("end_turn", Some("claude-opus-4-8"))],
    )
    .await;
    let seen = Arc::new(Mutex::new(Vec::new()));
    let captured = seen.clone();
    let runner: SummaryCallRunner = Arc::new(move |call| {
        let captured = captured.clone();
        Box::pin(async move {
            let message = call(None).await?;
            captured.lock().unwrap().push(message.clone());
            Ok(message)
        })
    });
    let original = fixture.model.clone();
    let result = summarize(&fixture.model, runner).await.unwrap();
    assert_eq!(result.summary, VALID);
    assert_eq!(fixture.model, original);
    let usage = result.usage.unwrap();
    assert_eq!(usage.input, 205.0);
    assert_eq!(usage.output, 23.0);
    assert_eq!(usage.total_tokens, 228.0);
    assert_eq!(seen.lock().unwrap()[0].model, "claude-opus-4-8");
    let requests = fixture.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0]["model"], "claude-opus-5-5");
    assert_eq!(requests[0]["fallbacks"], "default");
    assert_eq!(requests[0]["output_config"]["effort"], "max");
    assert!(requests[0].get("tools").is_none());
    assert_eq!(
        requests[0]["system"][0]["text"],
        "You are Claude Code, Anthropic's official CLI for Claude."
    );
    let headers = fixture.headers.lock().unwrap();
    let beta = headers[0]
        .lines()
        .find(|line| line.to_ascii_lowercase().starts_with("anthropic-beta:"))
        .unwrap();
    for value in [
        "claude-code-20250219",
        "oauth-2025-04-20",
        "server-side-fallback-2026-07-01",
    ] {
        assert!(beta.contains(value), "{beta}");
    }
    assert_eq!(
        headers[0]
            .to_ascii_lowercase()
            .matches("anthropic-beta:")
            .count(),
        1
    );
    assert!(headers[0]
        .to_ascii_lowercase()
        .contains("authorization: bearer sk-ant-oat01-synthetic-fixture"));
}

#[tokio::test]
async fn terminal_claude_refusal_preserves_history_category_and_request_id() {
    let fixture = Fixture::new("claude-opus-5-5", vec![("refusal", None)]).await;
    let error = summarize(&fixture.model, default_summary_call_runner(None))
        .await
        .unwrap_err();
    assert_eq!(
        classify_summary_failure(&error),
        SummaryFailureKind::FilteredOrRefused,
        "{error}"
    );
    assert!(error.contains("existing conversation preserved"), "{error}");
    assert!(error.contains("reasoning_extraction"), "{error}");
    assert!(error.contains("req_fixture_refusal"), "{error}");
    assert!(!error.contains("PARTIAL-HANDOFF"));
    assert_eq!(fixture.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn old_claude_and_kiro_routes_do_not_enable_anthropic_beta() {
    for (id, provider) in [
        ("claude-sonnet-4-6", "anthropic"),
        ("claude-opus-5-5", "kiro"),
    ] {
        let mut fixture = Fixture::new(id, vec![("end_turn", None)]).await;
        fixture.model.provider = provider.into();
        summarize(&fixture.model, default_summary_call_runner(None))
            .await
            .unwrap();
        assert!(fixture.requests.lock().unwrap()[0]
            .get("fallbacks")
            .is_none());
        assert!(!fixture.headers.lock().unwrap()[0].contains("server-side-fallback"));
    }
}

#[tokio::test]
async fn ordinary_claude_chat_does_not_enable_summary_recovery() {
    let fixture = Fixture::new("claude-opus-5-5", vec![("end_turn", None)]).await;
    let context = pi_ai::types::Context {
        system_prompt: None,
        messages: vec![pi_ai::types::Message::User(UserMessage::new(
            UserContent::Text("hello".into()),
            0,
        ))],
        tools: None,
    };
    let mut options = pi_ai::types::SimpleStreamOptions::default();
    options.stream.api_key = Some("sk-ant-oat01-synthetic-fixture".into());
    let response = pi_ai::stream::complete_simple(&fixture.model, &context, Some(&options)).await;
    assert_eq!(response.stop_reason, "stop");
    assert!(fixture.requests.lock().unwrap()[0]
        .get("fallbacks")
        .is_none());
    assert!(!fixture.headers.lock().unwrap()[0].contains("server-side-fallback"));
}
