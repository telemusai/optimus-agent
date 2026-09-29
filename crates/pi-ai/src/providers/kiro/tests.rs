use super::*;
use serde_json::json;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
};
use tokio_util::sync::CancellationToken;

mod request_history;

fn model() -> Model {
    catalog::built_in_models()
        .into_iter()
        .find(|m| m.id == "claude-haiku-4.5")
        .unwrap()
}
fn context() -> Context {
    Context::new(
        Some("You are Optimus.".into()),
        vec![Message::User(UserMessage::new(
            UserContent::Text("Hello".into()),
            1,
        ))],
        None,
    )
}
fn options() -> SimpleStreamOptions {
    SimpleStreamOptions {
        stream: StreamOptions {
            api_key: Some("ksk_synthetic".into()),
            timeout_ms: Some(2000.0),
            ..Default::default()
        },
        ..Default::default()
    }
}
fn frame_type(message_type: &str, value: Value) -> Vec<u8> {
    let mut headers = vec![];
    for (name, value) in [
        (":message-type", message_type),
        (":event-type", "assistantResponseEvent"),
    ] {
        headers.push(name.len() as u8);
        headers.extend_from_slice(name.as_bytes());
        headers.push(7);
        headers.extend_from_slice(&(value.len() as u16).to_be_bytes());
        headers.extend_from_slice(value.as_bytes());
    }
    let payload = value.to_string();
    let total = 16 + headers.len() + payload.len();
    let mut frame = vec![];
    frame.extend_from_slice(&(total as u32).to_be_bytes());
    frame.extend_from_slice(&(headers.len() as u32).to_be_bytes());
    frame.extend_from_slice(&event::crc32(&frame).to_be_bytes());
    frame.extend(headers);
    frame.extend_from_slice(payload.as_bytes());
    frame.extend_from_slice(&event::crc32(&frame).to_be_bytes());
    frame
}
fn frame(value: Value) -> Vec<u8> {
    frame_type("event", value)
}
async fn read_request(socket: &mut tokio::net::TcpStream) -> (String, Value) {
    let mut request = vec![];
    let mut temp = [0; 2048];
    let split = loop {
        let n = socket.read(&mut temp).await.unwrap();
        assert!(n > 0);
        request.extend_from_slice(&temp[..n]);
        if let Some(index) = request.windows(4).position(|b| b == b"\r\n\r\n") {
            break index + 4;
        }
    };
    let headers = String::from_utf8(request[..split].to_vec()).unwrap();
    let length: usize = headers
        .lines()
        .find_map(|line| {
            line.to_ascii_lowercase()
                .strip_prefix("content-length: ")
                .map(str::to_owned)
        })
        .unwrap_or_else(|| "0".into())
        .parse()
        .unwrap();
    while request.len() < split + length {
        let n = socket.read(&mut temp).await.unwrap();
        assert!(n > 0);
        request.extend_from_slice(&temp[..n]);
    }
    let payload: Value = if length == 0 {
        Value::Null
    } else {
        serde_json::from_slice(&request[split..split + length]).unwrap()
    };
    (headers, payload)
}

async fn fixture(
    body: Vec<u8>,
    status: u16,
    hang: bool,
) -> (String, tokio::task::JoinHandle<(String, Value)>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let task = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let (headers, payload) = read_request(&mut socket).await;
        socket.write_all(format!("HTTP/1.1 {status} Fixture\r\nContent-Type: application/vnd.amazon.eventstream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",if hang {body.len()+100} else {body.len()}).as_bytes()).await.unwrap();
        for chunk in body.chunks(3) {
            if socket.write_all(chunk).await.is_err() {
                break;
            }
        }
        if hang {
            let mut temp = [0; 1];
            let _ = tokio::time::timeout(Duration::from_secs(5), socket.read(&mut temp)).await;
        }
        (headers, payload)
    });
    (format!("http://{address}/"), task)
}

#[tokio::test]
async fn streams_fragmented_unicode_repeated_text_and_exact_usage() {
    let bytes = [
        frame(json!({"content":"Hi 🦀"})),
        frame(json!({"content":"Hi 🦀"})),
        frame(json!({"usage":{"inputTokens":11,"outputTokens":7}})),
        frame(json!({"contextUsagePercentage":0.1})),
    ]
    .concat();
    let (url, server) = fixture(bytes, 200, false).await;
    let mut m = model();
    m.base_url = url;
    let result = stream_simple_kiro(&m, &context(), Some(&options()))
        .result()
        .await;
    assert_eq!(result.stop_reason, "stop");
    assert_eq!(result.content[0].as_text().unwrap().text, "Hi 🦀Hi 🦀");
    assert_eq!(result.usage.input, 11.0);
    assert_eq!(result.usage.output, 7.0);
    assert_eq!(result.usage.total_tokens, 18.0);
    let (headers, body) = server.await.unwrap();
    assert!(headers.starts_with("POST / HTTP/1.1"));
    assert!(headers.contains("tokentype: API_KEY"));
    assert!(!headers.contains("app/AmazonQ-For-CLI"));
    assert_eq!(
        body["conversationState"]["currentMessage"]["userInputMessage"]["origin"],
        "AI_EDITOR"
    );
}

#[tokio::test]
async fn oauth_uses_own_profile_without_api_key_identity() {
    let (url, server) = fixture(
        [
            frame(json!({"content":"OK"})),
            frame(json!({"contextUsagePercentage":1})),
        ]
        .concat(),
        200,
        false,
    )
    .await;
    let mut m = model();
    m.base_url = url;
    let mut o = options();
    o.stream.api_key = Some(
        Credential {
            access: "synthetic-oauth".into(),
            region: "us-east-1".into(),
            profile_arn: Some("synthetic-profile".into()),
        }
        .encode(),
    );
    assert_eq!(
        stream_simple_kiro(&m, &context(), Some(&o))
            .result()
            .await
            .stop_reason,
        "stop"
    );
    let (headers, body) = server.await.unwrap();
    assert!(headers.starts_with("POST /generateAssistantResponse HTTP/1.1"));
    assert!(!headers.contains("tokentype"));
    assert!(!headers.contains("x-amz-target"));
    assert!(headers.contains("content-type: application/json"));
    assert!(headers.contains("user-agent: Optimus-Agent/"));
    let expected_agent = concat!(
        "Optimus-Agent/",
        env!("CARGO_PKG_VERSION"),
        " app/AmazonQ-For-CLI"
    );
    for name in ["user-agent", "x-amz-user-agent"] {
        assert_eq!(
            headers
                .lines()
                .filter(|line| line.starts_with(&format!("{name}:")))
                .count(),
            1
        );
        assert!(headers
            .lines()
            .any(|line| line == format!("{name}: {expected_agent}")));
    }
    assert!(headers.contains("authorization: Bearer synthetic-oauth"));
    assert_eq!(body["profileArn"], "synthetic-profile");
}

#[tokio::test]
async fn streamed_tool_call_round_trips_result_and_current_system_prompt() {
    let (url, server) = fixture(
        [
            frame(json!({"name":"inspect","toolUseId":"call-1","input":"{\"path\":"})),
            frame(json!({"input":"\"x.rs\"}","stop":true})),
        ]
        .concat(),
        200,
        false,
    )
    .await;
    let mut m = model();
    m.base_url = url;
    let mut ctx = context();
    ctx.tools = Some(vec![Tool {
        name: "inspect".into(),
        description: "Inspect a file".into(),
        parameters: json!({"type":"object","properties":{"path":{"type":"string"}}}),
    }]);
    let output = stream_simple_kiro(&m, &ctx, Some(&options()))
        .result()
        .await;
    server.await.unwrap();
    assert_eq!(output.stop_reason, "toolUse");
    let call = output.content[0].as_tool_call().unwrap();
    assert_eq!(call.arguments["path"], "x.rs");
    ctx.messages.push(Message::Assistant(output));
    ctx.messages
        .push(Message::ToolResult(ToolResultMessage::new(
            "call-1",
            "inspect",
            vec![
                TextContent::new("result").into(),
                ImageContent::new("aW1hZ2U=", "image/png").into(),
            ],
            false,
            2,
        )));
    ctx.system_prompt = Some("Return JavaScript in Node mode.".into());
    let body = request::build(&m, &ctx, &options(), "KIRO_CLI", None).unwrap();
    assert!(
        body["conversationState"]["history"][0]["userInputMessage"]["content"]
            .as_str()
            .unwrap()
            .starts_with("Return JavaScript")
    );
    let current = &body["conversationState"]["currentMessage"]["userInputMessage"];
    assert_eq!(
        current["userInputMessageContext"]["toolResults"][0]["toolUseId"],
        "call-1"
    );
    assert_eq!(current["images"][0]["source"]["bytes"], "aW1hZ2U=");
    assert_eq!(
        current["userInputMessageContext"]["tools"][0]["toolSpecification"]["name"],
        "inspect"
    );
    let (url, server) = fixture(
        [
            frame(json!({"content":"Inspected x.rs"})),
            frame(json!({"contextUsagePercentage":1})),
        ]
        .concat(),
        200,
        false,
    )
    .await;
    m.base_url = url;
    let result = stream_simple_kiro(&m, &ctx, Some(&options()))
        .result()
        .await;
    assert_eq!(result.stop_reason, "stop");
    assert_eq!(result.content[0].as_text().unwrap().text, "Inspected x.rs");
    let (_, submitted) = server.await.unwrap();
    assert_eq!(
        submitted["conversationState"]["currentMessage"]["userInputMessage"]
            ["userInputMessageContext"]["toolResults"][0]["toolUseId"],
        "call-1"
    );
}

#[tokio::test]
async fn thinking_tags_split_between_events_keep_stable_block_indices() {
    let bytes = ["<thi", "nking>plan", "</think", "ing>Answer"]
        .map(|s| frame(json!({"content":s})))
        .into_iter()
        .chain([frame(json!({"contextUsagePercentage":2}))])
        .flatten()
        .collect();
    let (url, server) = fixture(bytes, 200, false).await;
    let mut m = model();
    m.base_url = url;
    let stream = stream_simple_kiro(&m, &context(), Some(&options()));
    let mut ends = vec![];
    while let Some(event) = stream.next().await {
        if let AssistantMessageEvent::ThinkingEnd { content_index, .. }
        | AssistantMessageEvent::TextEnd { content_index, .. } = event
        {
            ends.push(content_index);
        }
    }
    let result = stream.result().await;
    server.await.unwrap();
    assert_eq!(result.stop_reason, "stop");
    assert_eq!(result.content[0].as_thinking().unwrap().thinking, "plan");
    assert_eq!(result.content[1].as_text().unwrap().text, "Answer");
    assert_eq!(ends, vec![0, 1]);
}

#[tokio::test]
async fn access_denied_is_not_retried_or_reported_as_success_and_redacts_token() {
    let (url, server) = fixture(
        br#"{"message":"Your subscription does not support this application: ksk_synthetic"}"#
            .to_vec(),
        403,
        false,
    )
    .await;
    let mut m = model();
    m.base_url = url;
    let result = stream_simple_kiro(&m, &context(), Some(&options()))
        .result()
        .await;
    server.await.unwrap();
    assert_eq!(result.stop_reason, "error");
    let error = result.error_message.unwrap();
    assert!(error.contains("HTTP 403"));
    assert!(error.contains("subscription"));
    assert!(!error.contains("ksk_synthetic"));
    assert!(!error.contains("Check this account's API entitlement"));
}

#[tokio::test]
async fn cancellation_before_and_during_stream_finishes_promptly() {
    let mut o = options();
    let cancel = CancellationToken::new();
    o.stream.signal = Some(cancel.clone());
    cancel.cancel();
    let mut m = model();
    m.base_url = "http://127.0.0.1:1/".into();
    assert_eq!(
        stream_simple_kiro(&m, &context(), Some(&o))
            .result()
            .await
            .stop_reason,
        "aborted"
    );
    let (url, server) = fixture(frame(json!({"content":"partial"})), 200, true).await;
    m.base_url = url;
    let cancel = CancellationToken::new();
    o.stream.signal = Some(cancel.clone());
    let stream = stream_simple_kiro(&m, &context(), Some(&o));
    loop {
        if matches!(
            stream.next().await,
            Some(AssistantMessageEvent::TextDelta { .. })
        ) {
            break;
        }
    }
    cancel.cancel();
    let result = tokio::time::timeout(Duration::from_millis(500), stream.result())
        .await
        .unwrap();
    assert_eq!(result.stop_reason, "aborted");
    server.await.unwrap();
}

#[tokio::test]
async fn idle_timeout_preserves_partial_output_as_error() {
    let (url, server) = fixture(frame(json!({"content":"partial"})), 200, true).await;
    let mut m = model();
    m.base_url = url;
    let mut o = options();
    o.stream.timeout_ms = Some(100.0);
    let result = stream_simple_kiro(&m, &context(), Some(&o)).result().await;
    server.await.unwrap();
    assert_eq!(result.stop_reason, "error");
    assert!(result.error_message.unwrap().contains("timeout"));
}

#[tokio::test]
async fn unfinished_or_invalid_tool_arguments_never_finish_successfully() {
    for bytes in [
        frame(json!({"name":"inspect","toolUseId":"x","input":"{"})),
        frame(json!({"name":"inspect","toolUseId":"x","input":"{","stop":true})),
    ] {
        let (url, server) = fixture(bytes, 200, false).await;
        let mut m = model();
        m.base_url = url;
        let result = stream_simple_kiro(&m, &context(), Some(&options()))
            .result()
            .await;
        server.await.unwrap();
        assert_eq!(result.stop_reason, "error");
    }
}

#[tokio::test]
async fn provider_exception_truncated_frame_and_empty_response_are_errors() {
    let mut short = frame(json!({"content":"partial"}));
    short.pop();
    for bytes in [
        frame_type("exception", json!({"message":"fixture failure"})),
        short,
        vec![],
    ] {
        let (url, server) = fixture(bytes, 200, false).await;
        let mut m = model();
        m.base_url = url;
        let result = stream_simple_kiro(&m, &context(), Some(&options()))
            .result()
            .await;
        server.await.unwrap();
        assert_eq!(result.stop_reason, "error");
    }
}

#[test]
fn malformed_frames_are_bounded_and_checksums_are_validated() {
    assert!(event::decode(&mut vec![0; 12]).is_err());
    let mut valid = frame(json!({"content":"ok"}));
    valid[15] ^= 1;
    assert!(event::decode(&mut valid).is_err());
    let mut truncated = frame(json!({"content":"ok"}));
    let last = truncated.pop().unwrap();
    assert!(event::decode(&mut truncated).unwrap().is_empty());
    truncated.push(last);
    assert_eq!(event::decode(&mut truncated).unwrap().len(), 1);
    assert_eq!(event::crc32(b"123456789"), 0xcbf43926);
}

#[test]
fn history_merges_adjacent_messages_preserves_images_and_ignores_failed_assistants() {
    let mut ctx = context();
    ctx.messages.push(Message::User(UserMessage::new(
        UserContent::Blocks(vec![ImageContent::new("cG5n", "image/png").into()]),
        2,
    )));
    ctx.messages.push(Message::Assistant(AssistantMessage {
        stop_reason: "error".into(),
        content: vec![ContentBlock::Text(TextContent::new("bad response"))],
        ..Default::default()
    }));
    let body = request::build(&model(), &ctx, &options(), "KIRO_CLI", None).unwrap();
    let message = &body["conversationState"]["currentMessage"]["userInputMessage"];
    assert!(message["content"].as_str().unwrap().contains("Hello"));
    assert_eq!(message["images"][0]["format"], "png");
    assert!(!body.to_string().contains("bad response"));
}

#[test]
fn model_ids_and_million_token_contexts_are_not_rewritten() {
    let all = catalog::built_in_models();
    assert_eq!(all.len(), 20);
    assert!(all
        .iter()
        .any(|m| m.id == "claude-opus-5.5" && m.context_window == 1_000_000.0));
    let models=catalog::parse_models(&json!({"models":[{"modelId":"future-6.1","tokenLimits":{"maxInputTokens":1_000_000,"maxOutputTokens":128000},"supportedInputTypes":["TEXT","IMAGE"]}]}),DEFAULT_ENDPOINT).unwrap();
    assert_eq!(models[0].id, "future-6.1");
    assert_eq!(models[0].context_window, 1_000_000.0);
    assert_eq!(models[0].max_tokens, 128000.0);
    assert!(models[0].input.contains(&InputModality::Image));
    assert!(catalog::parse_models(&json!({"models":[{}]}), DEFAULT_ENDPOINT).is_err());
}

#[tokio::test]
async fn discovery_returns_only_reported_models_and_fails_on_denial() {
    let body = json!({"models":[{"modelId":"model-1","tokenLimits":{"maxInputTokens":65536}}]})
        .to_string()
        .into_bytes();
    let (url, server) = fixture(body, 200, false).await;
    let result = catalog::discover("ksk_synthetic", Some(&url))
        .await
        .unwrap();
    let (headers, _) = server.await.unwrap();
    assert!(headers.contains("AmazonCodeWhispererService.ListAvailableModels"));
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].id, "model-1");
    let (url, server) = fixture(b"{}".to_vec(), 403, false).await;
    assert!(catalog::discover("ksk_synthetic", Some(&url))
        .await
        .is_err());
    server.await.unwrap();
}

#[test]
fn cli_login_reader_is_read_only_and_uses_profile_region() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("data.sqlite3");
    let conn = rusqlite::Connection::open(&path).unwrap();
    conn.execute_batch("CREATE TABLE auth_kv (key TEXT PRIMARY KEY,value TEXT); CREATE TABLE state (key TEXT PRIMARY KEY,value BLOB);").unwrap();
    let token=json!({"access_token":"synthetic-access","refresh_token":"must-stay-in-cli","region":"ap-southeast-2","expires_at":"2099-01-01T00:00:00Z"}).to_string();
    conn.execute(
        "INSERT INTO auth_kv VALUES (?1,?2)",
        rusqlite::params!["kirocli:odic:token", &token],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO state VALUES (?1,?2)",
        rusqlite::params![
            "api.codewhisperer.profile",
            json!({"arn":"arn:aws:codewhisperer:eu-central-1:synthetic:profile/test"}).to_string()
        ],
    )
    .unwrap();
    drop(conn);
    let before = std::fs::read(&path).unwrap();
    let snapshot = auth::read_cli(&path).unwrap().unwrap();
    assert_eq!(snapshot.credential.region, "eu-central-1");
    assert!(!snapshot.credential.encode().contains("must-stay-in-cli"));
    assert_eq!(std::fs::read(&path).unwrap(), before);
    assert!(auth::read_cli(&dir.path().join("missing"))
        .unwrap()
        .is_none());
    assert!(!dir.path().join("missing").exists());
}

#[test]
fn endpoint_and_auth_validation_fail_without_exposing_secret() {
    assert!(Access::parse("not-a-key").is_err());
    let oauth = Access::parse(
        &Credential {
            access: "synthetic-oauth".into(),
            region: "us-east-1".into(),
            profile_arn: None,
        }
        .encode(),
    )
    .unwrap();
    assert!(oauth.root("https://example.com/").is_err());
    assert!(oauth.root(DEFAULT_ENDPOINT).is_ok());
    let access = Access::parse("ksk_synthetic").unwrap();
    assert!(access.root("http://example.com/").is_err());
    assert!(access.root("https://user:pass@example.com/").is_err());
    assert!(Access::parse(&format!(
        "{ENVELOPE}{{\"access\":\"secret\",\"region\":\"evil.com/\"}}"
    ))
    .is_err());
}

#[tokio::test]
async fn discovery_paginates_deduplicates_and_rejects_repeated_page_tokens() {
    for repeat in [false, true] {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            for page in 0..2 {
                let (mut socket, _) = listener.accept().await.unwrap();
                let (_, payload) = read_request(&mut socket).await;
                assert_eq!(
                    payload.get("nextToken").and_then(Value::as_str),
                    if page == 0 { None } else { Some("page-2") }
                );
                let mut response = if page == 0 {
                    json!({"models":[{"modelId":"one"}],"nextToken":"page-2"})
                } else {
                    json!({"models":[{"modelId":"one"},{"modelId":"two"}]})
                };
                if repeat {
                    response["nextToken"] = json!("page-2");
                }
                let body = response.to_string();
                socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).as_bytes()).await.unwrap();
            }
        });
        let result = catalog::discover("ksk_synthetic", Some(&url)).await;
        server.await.unwrap();
        if repeat {
            assert!(result.unwrap_err().contains("repeated"));
        } else {
            assert_eq!(
                result
                    .unwrap()
                    .iter()
                    .map(|m| m.id.as_str())
                    .collect::<Vec<_>>(),
                ["one", "two"]
            );
        }
    }
}

#[test]
fn oauth_defaults_to_us_east_1_and_uses_the_configured_service_region() {
    let access = Access::parse(
        &Credential {
            access: "fixture".into(),
            region: "eu-central-1".into(),
            profile_arn: None,
        }
        .encode(),
    )
    .unwrap();
    assert_eq!(
        access.root(DEFAULT_ENDPOINT).unwrap().as_str(),
        "https://runtime.us-east-1.kiro.dev/"
    );
    assert_eq!(
        access.management_root(DEFAULT_ENDPOINT).unwrap().as_str(),
        "https://management.us-east-1.kiro.dev/"
    );
    let configured = endpoint_for_region("eu-central-1").unwrap();
    assert_eq!(
        access.root(&configured).unwrap().as_str(),
        "https://runtime.eu-central-1.kiro.dev/"
    );
    assert_eq!(
        access.management_root(&configured).unwrap().as_str(),
        "https://management.eu-central-1.kiro.dev/"
    );
    assert!(access
        .root("https://runtime.us-east-1.kiro.dev.evil.test/")
        .is_err());
    assert!(access
        .root("https://q.eu-central-1.amazonaws.com/")
        .is_err());
}

#[test]
fn configured_api_key_region_overrides_credential_default_for_both_services() {
    let access = Access {
        credential: Credential {
            access: "ksk_fixture".into(),
            region: "eu-central-1".into(),
            profile_arn: None,
        },
        api_key: true,
    };
    let endpoint = endpoint_for_region("us-east-1").unwrap();
    assert_eq!(
        access.root(&endpoint).unwrap().as_str(),
        "https://q.us-east-1.amazonaws.com/"
    );
    assert_eq!(
        access.management_root(&endpoint).unwrap().as_str(),
        "https://q.us-east-1.amazonaws.com/"
    );
    for invalid in ["", "US-EAST-1", "us-east-1.evil.test", "us-east-1/path"] {
        assert!(endpoint_for_region(invalid).is_err());
    }
}

#[tokio::test]
async fn oauth_catalog_uses_management_get_and_preserves_runtime_model_urls() {
    let (url, server) = fixture(json!({"models":[{"modelId":"claude-opus-5.5","tokenLimits":{"maxInputTokens":1000000,"maxOutputTokens":128000},"supportedInputTypes":["TEXT","IMAGE"]}]}).to_string().into_bytes(),200,false).await;
    let key = Credential {
        access: "synthetic-oauth".into(),
        region: "us-east-1".into(),
        profile_arn: Some("arn:aws:kiro:us-east-1:test:profile/example".into()),
    }
    .encode();
    let models = catalog::discover(&key, Some(&url)).await.unwrap();
    let (headers, body) = server.await.unwrap();
    assert!(headers.starts_with("GET /List-Available-Models?"));
    assert!(headers.contains("authorization: Bearer synthetic-oauth"));
    assert!(!headers.contains("x-amz-target"));
    assert!(body.is_null());
    let path = headers
        .lines()
        .next()
        .unwrap()
        .split_whitespace()
        .nth(1)
        .unwrap();
    let request = url::Url::parse(&format!("http://fixture{path}")).unwrap();
    let query = request
        .query_pairs()
        .collect::<std::collections::HashMap<_, _>>();
    assert_eq!(query.get("origin").map(|v| v.as_ref()), Some("KIRO_CLI"));
    assert_eq!(
        query.get("profileArn").map(|v| v.as_ref()),
        Some("arn:aws:kiro:us-east-1:test:profile/example")
    );
    assert_eq!(models[0].base_url, url);
    assert_eq!(models[0].max_tokens, 128000.0);
}
