//! Offline-only transport/body/header/error fixtures. No real keys or profiles.
use std::collections::BTreeMap;
use std::ffi::OsString;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use indexmap::IndexMap;
use pi_ai::providers::openai_completions::{stream_openai_completions, stream_simple_openai_completions, OpenAICompletionsOptions};
use pi_ai::types::{AssistantMessage, AssistantMessageEvent, Compat, Context, Model, OpenAICompletionsCompat, SimpleStreamOptions, StreamOptions};
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

struct Env(Vec<(&'static str, Option<OsString>)>);
impl Env {
    fn local_policy() -> Self {
        let names = ["ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "NO_PROXY", "no_proxy", "REQUEST_METHOD", "SSL_CERT_FILE", "SSL_CERT_DIR", "OPENSSL_CONF", "OPENSSL_MODULES", "OPTIMUS_K3_REQUEST_AUDIT_PATH", "PI_CACHE_RETENTION"];
        let previous = names.iter().map(|name| (*name, std::env::var_os(name))).collect();
        for name in names { std::env::remove_var(name); }
        // Fully observed policy. hyper-util checks numeric hosts only against IP
        // exclusions, so its domain wildcard alone does not bypass loopback IPs.
        std::env::set_var("HTTP_PROXY", "http://fixture-one:fixture-secret@127.0.0.1:1");
        std::env::set_var("HTTPS_PROXY", "http://fixture-one:fixture-secret@127.0.0.1:1");
        std::env::set_var("NO_PROXY", "127.0.0.1,::1,*");
        Self(previous)
    }
}
impl Drop for Env {
    fn drop(&mut self) {
        for (name, value) in &self.0 {
            match value { Some(value) => std::env::set_var(name, value), None => std::env::remove_var(name) }
        }
    }
}

#[derive(Debug)]
struct Request { connection: usize, path: String, headers: BTreeMap<String, String>, bytes: Vec<u8>, body: Value }

async fn read_request(socket: &mut tokio::net::TcpStream, buffer: &mut Vec<u8>, connection: usize) -> Option<Request> {
    loop {
        if let Some(end) = buffer.windows(4).position(|w| w == b"\r\n\r\n") {
            let head = std::str::from_utf8(&buffer[..end]).unwrap();
            let path = head.lines().next().unwrap().split_whitespace().nth(1).unwrap().to_string();
            let headers: BTreeMap<_, _> = head.lines().skip(1).map(|line| {
                let (key, value) = line.split_once(':').unwrap(); (key.to_ascii_lowercase(), value.trim().to_string())
            }).collect();
            let size = headers.get("content-length").map(|v| v.parse::<usize>().unwrap()).unwrap_or(0);
            if buffer.len() >= end + 4 + size {
                let bytes = buffer[end + 4..end + 4 + size].to_vec();
                buffer.drain(..end + 4 + size);
                let body = if bytes.is_empty() { Value::Null } else { serde_json::from_slice(&bytes).unwrap() };
                return Some(Request { connection, path, headers, bytes, body });
            }
        }
        let mut chunk = [0; 8192];
        let count = socket.read(&mut chunk).await.unwrap();
        if count == 0 { return None; }
        buffer.extend_from_slice(&chunk[..count]);
    }
}

struct Fixture {
    address: std::net::SocketAddr,
    received: mpsc::UnboundedReceiver<Request>,
    stop: CancellationToken,
    server: tokio::task::JoinHandle<()>,
}
impl Fixture {
    async fn start(reply: Arc<dyn Fn(&Request) -> (u16, Vec<(&'static str, String)>, String) + Send + Sync>) -> Self {
        let listener = tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).await.unwrap();
        let address = listener.local_addr().unwrap();
        let stop = CancellationToken::new();
        let stopped = stop.clone();
        let (sender, received) = mpsc::unbounded_channel();
        let server = tokio::spawn(async move {
            let mut workers = tokio::task::JoinSet::new();
            let mut connection = 0;
            loop {
                tokio::select! { biased;
                    _ = stopped.cancelled() => break,
                    accepted = listener.accept() => {
                        let (mut socket, _) = accepted.unwrap();
                        connection += 1;
                        let id = connection;
                        let reply = reply.clone(); let sender = sender.clone(); let stop = stopped.clone();
                        workers.spawn(async move {
                            let mut buffer = Vec::new();
                            loop {
                                let request = tokio::select! { biased;
                                    _ = stop.cancelled() => break,
                                    request = read_request(&mut socket, &mut buffer, id) => request,
                                };
                                let Some(request) = request else { break; };
                                let (status, headers, body) = reply(&request);
                                sender.send(request).unwrap();
                                let mut head = format!("HTTP/1.1 {status} Fixture\r\nContent-Length: {}\r\n", body.len());
                                for (name, value) in headers { head.push_str(&format!("{name}: {value}\r\n")); }
                                head.push_str("\r\n");
                                head.push_str(&body);
                                socket.write_all(head.as_bytes()).await.unwrap();
                            }
                        });
                    }
                }
            }
            while let Some(result) = workers.join_next().await { result.unwrap(); }
        });
        Self { address, received, stop, server }
    }
    async fn next(&mut self) -> Request { tokio::time::timeout(Duration::from_secs(5), self.received.recv()).await.unwrap().unwrap() }
    async fn finish(self) {
        self.stop.cancel();
        tokio::time::timeout(Duration::from_secs(5), self.server).await.unwrap().unwrap();
    }
}

fn model(address: std::net::SocketAddr) -> Model {
    Model {
        id: "kimi-k3".into(), provider: "offline-fixture".into(), api: "openai-completions".into(),
        base_url: format!("http://{address}/v1"), reasoning: true,
        context_window: 1_048_576.0, max_tokens: 262_144.0, max_tokens_explicit: true, default_max_tokens: Some(131_072.0),
        thinking_level_map: Some([("max".into(), Some("max".into()))].into_iter().collect()),
        compat: Some(Compat::Completions(OpenAICompletionsCompat {
            sglang_token_budget: Some(true), supports_store: Some(false),
            max_tokens_field: Some("max_tokens".into()), supports_reasoning_effort: Some(true),
            send_session_affinity_headers: Some(true), ..Default::default()
        })), ..Default::default()
    }
}

fn options(key: &str, session: &str) -> SimpleStreamOptions {
    let headers: IndexMap<_, _> = [("X-Prime-Team-ID".into(), format!("team-{session}")), ("X-Rotation".into(), key.to_string())].into_iter().collect();
    SimpleStreamOptions {
        reasoning: Some("max".into()),
        stream: StreamOptions { api_key: Some(key.into()), session_id: Some(session.into()), headers: Some(headers),
            cache_retention: Some("short".into()), timeout_ms: Some(5000.0), ..Default::default() },
        ..Default::default()
    }
}

fn done_body() -> String {
    "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":106,\"completion_tokens\":1}}\n\ndata: [DONE]\n\n".into()
}

async fn run(model: &Model, options: SimpleStreamOptions) -> AssistantMessage {
    let stream = stream_simple_openai_completions(model, &Context::default(), Some(options));
    let output = tokio::time::timeout(Duration::from_secs(5), stream.result()).await.unwrap();
    let joined = stream.task_receipt().settle(Duration::from_secs(5)).await;
    assert_eq!(joined.pending_tasks + joined.failed_tasks + joined.cancelled_tasks, 0);
    let mut terminals = 0;
    while let Some(event) = stream.next().await {
        if matches!(event, AssistantMessageEvent::Done { .. } | AssistantMessageEvent::Error { .. }) { terminals += 1; }
    }
    assert_eq!(terminals, 1);
    output
}

fn details(message: &AssistantMessage) -> &serde_json::Map<String, Value> {
    message.diagnostics.as_ref().unwrap().iter().find(|d| d.type_ == "provider_stream_failure").unwrap().details.as_ref().unwrap()
}

async fn reuse_headers_body_and_rotation() {
    let mut fixture = Fixture::start(Arc::new(|request| {
        if request.path.ends_with("/tokenize") {
            (200, vec![], json!({"count":106,"tokens":vec![1;106],"max_model_len":1048576}).to_string())
        } else { (200, vec![("Content-Type", "text/event-stream".into())], done_body()) }
    })).await;
    let model = model(fixture.address);
    let mut pairs = Vec::new();
    for (key, session) in [("key-one", "session-one"), ("key-two", "session-two")] {
        let mut opts = options(key, session);
        // A None payload hook keeps the exact body; Some uses the same body too.
        opts.stream.on_payload = Some(Arc::new(move |body, _| Box::pin(async move {
            assert_eq!(body["max_tokens"], 131072);
            assert_eq!(body["reasoning_effort"], "max");
            if key == "key-one" { None } else { Some(body) }
        })));
        let phases = Arc::new(Mutex::new(Vec::new()));
        let captured = phases.clone();
        opts.stream.on_stream_observation = Some(Arc::new(move |phase| captured.lock().unwrap().push(phase.to_string())));
        let output = run(&model, opts).await;
        assert_eq!(output.stop_reason, "stop");
        let client_phase = if key == "key-one" { "http_client_created" } else { "http_client_reused" };
        assert_eq!(*phases.lock().unwrap(), vec!["count_start", client_phase, "count_serialize_start", "count_serialize_complete", "count_send_start", "count_headers_complete", "count_complete", "fresh_generation_policy", "generation_serialize_start", "generation_serialize_complete", "generation_send_start", "generation_headers_complete", "raw_event", "text", "raw_event", "terminal"]);
        let count = fixture.next().await; let generation = fixture.next().await;
        assert_eq!(count.path, "/v1/tokenize");
        assert_eq!(generation.path, "/v1/chat/completions");
        for request in [&count, &generation] {
            assert_eq!(request.headers["authorization"], format!("Bearer {key}"));
            assert_eq!(request.headers["x-prime-team-id"], format!("team-{session}"));
            assert_eq!(request.headers["session_id"], session);
            assert_eq!(request.headers["x-client-request-id"], session);
            assert_eq!(request.headers["x-session-affinity"], session);
            assert_eq!(request.headers["x-rotation"], key);
        }
        let expected_generation = json!({"model":"kimi-k3","messages":[],"stream":true,"stream_options":{"include_usage":true},"max_tokens":131072,"reasoning_effort":"max"});
        // Mirror the frozen count guard's clone/remove/insert recipe. Map
        // removals affect preserve_order bytes, not just the remaining fields.
        let mut expected_count = expected_generation.clone();
        let count_object = expected_count.as_object_mut().unwrap();
        count_object.remove("stream");
        count_object.remove("stream_options");
        count_object.remove("max_completion_tokens");
        count_object.insert("max_tokens".into(), json!(0));
        assert_eq!(generation.bytes, serde_json::to_vec(&expected_generation).unwrap());
        assert_eq!(count.bytes, serde_json::to_vec(&expected_count).unwrap());
        assert_ne!(count.connection, generation.connection, "NO-REDIRECT count must not share generation transport");
        pairs.push((count.connection, generation.connection));
    }
    assert_eq!(pairs[0].0, pairs[1].0, "each fresh exact count reuses only the NO-REDIRECT HTTP transport");
    assert_ne!(pairs[0].1, pairs[1].1, "generation keeps fresh native TLS/default redirect policy");
    // Changing process proxy credentials invalidates transport reuse, not request credentials.
    std::env::set_var("HTTP_PROXY", "http://fixture-two:rotated-secret@127.0.0.1:1");
    let output = run(&model, options("key-three", "session-three")).await;
    assert_eq!(output.stop_reason, "stop");
    let count = fixture.next().await; let generation = fixture.next().await;
    assert_ne!(count.connection, pairs[0].0);
    assert_ne!(generation.connection, pairs[0].1);
    assert_eq!(count.headers["authorization"], "Bearer key-three");
    assert_eq!(generation.headers["authorization"], "Bearer key-three");
    fixture.finish().await;
}

async fn concurrent_header_isolation() {
    let mut fixture = Fixture::start(Arc::new(|request| {
        if request.path.ends_with("/tokenize") {
            (200, vec![], json!({"count":2,"tokens":[1,2],"max_model_len":1048576}).to_string())
        } else { (200, vec![], done_body()) }
    })).await;
    let model = model(fixture.address);
    let (a,b) = tokio::join!(run(&model, options("concurrent-a", "a")), run(&model, options("concurrent-b", "b")));
    assert_eq!(a.stop_reason, "stop"); assert_eq!(b.stop_reason, "stop");
    let mut counts = BTreeMap::new();
    for _ in 0..4 {
        let request = fixture.next().await;
        let session = &request.headers["session_id"];
        assert!(session == "a" || session == "b");
        assert_eq!(request.headers["authorization"], format!("Bearer concurrent-{session}"));
        assert_eq!(request.headers["x-prime-team-id"], format!("team-{session}"));
        *counts.entry((session.clone(), request.path)).or_insert(0) += 1;
    }
    assert_eq!(counts.len(), 4); assert!(counts.values().all(|n| *n == 1));
    fixture.finish().await;
}

async fn count_http_semantics_and_redirect_rejection() {
    for (status, kind) in [(302,"unknown"),(400,"invalid_request"),(404,"invalid_request"),(401,"auth"),(403,"permission"),(429,"rate_limit"),(503,"server_error")] {
        let target = tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).await.unwrap();
        let redirect = format!("http://{}/credential-trap", target.local_addr().unwrap());
        let stop_trap = CancellationToken::new();
        let stopped = stop_trap.clone();
        let trap = tokio::spawn(async move {
            tokio::select! {
                accepted = target.accept() => {
                    let (mut socket, _) = accepted.unwrap();
                    let body = b"{}";
                    socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n").await.unwrap();
                    socket.write_all(body).await.unwrap();
                    true
                }
                _ = stopped.cancelled() => false,
            }
        });
        let mut fixture = Fixture::start(Arc::new(move |_| (status,
            vec![("Location", redirect.clone()), ("Retry-After", "3".into()), ("X-Secret", "must-not-copy".into())],
            "must-not-copy provider-body".into()))).await;
        let output = run(&model(fixture.address), options("key-error", "error")).await;
        assert_eq!(output.stop_reason, "error");
        let info = details(&output);
        assert_eq!(info["kind"], kind); assert_eq!(info["status"], status);
        assert_eq!(info["retryAfterMs"], 3000.0);
        assert!(!serde_json::to_string(&output).unwrap().contains("must-not-copy"));
        assert!(!serde_json::to_string(&output).unwrap().contains("credential-trap"));
        assert_eq!(fixture.next().await.path, "/v1/tokenize");
        assert!(fixture.received.try_recv().is_err(), "count failure must prevent generation");
        stop_trap.cancel();
        assert!(!tokio::time::timeout(Duration::from_secs(5), trap).await.unwrap().unwrap(), "count followed redirect");
        fixture.finish().await;
    }
}

async fn raw_image_near_ceiling_and_unknown_remote_errors() {
    for (response, expected_kind, expected_output) in [
        (json!({"usage":{"prompt_tokens":103,"completion_tokens":0},"choices":[{"message":{"content":""},"prompt_token_ids":vec![1;106]}]}).to_string(), None, Some(3)),
        (json!({"usage":{"prompt_tokens":103,"completion_tokens":0},"choices":[{"message":{"content":""},"prompt_token_ids":vec![1;109]}]}).to_string(), Some("invalid_request"), None),
        (json!({"usage":{"prompt_tokens":103,"completion_tokens":0},"choices":[{"message":{"content":""}}]}).to_string(), Some("unknown"), None),
        ("malformed JSON".into(), Some("unknown"), None),
    ] {
        let response = Arc::new(response);
        let mut fixture = Fixture::start(Arc::new(move |request| {
            if request.body["max_tokens"] == 0 { (200, vec![], (*response).clone()) }
            else { (200, vec![], done_body()) }
        })).await;
        let mut model = model(fixture.address); model.context_window = 110.0;
        let mut opts = options("image-key", "image");
        opts.stream.on_payload = Some(Arc::new(|mut body, _| Box::pin(async move {
            body["messages"] = json!([{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,fixture"}}]}]);
            Some(body)
        })));
        let output = run(&model, opts).await;
        let count = fixture.next().await;
        assert_eq!(count.path, "/v1/chat/completions");
        assert_eq!(count.body["stream"], false); assert_eq!(count.body["max_tokens"], 0);
        assert_eq!(count.body["return_prompt_token_ids"], true); assert_eq!(count.body["reasoning_effort"], "max");
        if let Some(expected) = expected_output {
            assert_eq!(output.stop_reason, "stop");
            let generation = fixture.next().await;
            assert_eq!(generation.body["max_tokens"], expected);
            assert_eq!(generation.body["messages"], count.body["messages"]);
            assert!(generation.body.get("return_prompt_token_ids").is_none());
            assert_eq!(generation.body["reasoning_effort"], "max");
        } else {
            assert_eq!(output.stop_reason, "error"); assert_eq!(details(&output)["kind"], expected_kind.unwrap());
            assert!(fixture.received.try_recv().is_err());
        }
        fixture.finish().await;
    }
}

async fn partial_errors_preserve_non_replay_evidence() {
    for delta in [json!({"content":"partial"}),json!({"reasoning_content":"partial thought"}),json!({"tool_calls":[{"id":"c","index":0,"function":{"name":"read","arguments":"{}"}}]})] {
        let body = format!("data: {{\"choices\":[{{\"delta\":{delta}}}]}}\n\ndata: {{\"error\":{{\"type\":\"server_error\",\"message\":\"fixture failure\"}}}}\n\ndata: [DONE]\n\n");
        let body = Arc::new(body);
        let mut fixture = Fixture::start(Arc::new(move |_| (200, vec![], (*body).clone()))).await;
        let mut model = model(fixture.address);
        model.compat.as_mut().and_then(|c| if let Compat::Completions(c) = c { Some(c) } else { None }).unwrap().sglang_token_budget = Some(false);
        let options = OpenAICompletionsOptions { stream: options("partial-key", "partial").stream, ..Default::default() };
        let stream = stream_openai_completions(&model, &Context::default(), Some(options));
        let output = tokio::time::timeout(Duration::from_secs(5), stream.result()).await.unwrap();
        assert_eq!(output.stop_reason, "error"); assert_eq!(details(&output)["kind"], "server_error");
        assert_eq!(output.content.len(), 1);
        assert!(output.content[0].as_text().map(|c| !c.text.is_empty()).unwrap_or(false)
            || output.content[0].as_thinking().map(|c| !c.thinking.is_empty()).unwrap_or(false)
            || output.content[0].as_tool_call().is_some());
        assert_eq!(stream.task_receipt().settle(Duration::from_secs(5)).await.pending_tasks, 0);
        assert_eq!(fixture.next().await.path, "/v1/chat/completions");
        assert!(fixture.received.try_recv().is_err());
        fixture.finish().await;
    }
}

async fn ordinary_generation_keeps_upstream_pool_and_per_request_headers() {
    let mut fixture = Fixture::start(Arc::new(|_| (200, vec![], done_body()))).await;
    let mut model = model(fixture.address);
    if let Some(Compat::Completions(compat)) = &mut model.compat {
        compat.sglang_token_budget = Some(false);
    }
    let mut connections = Vec::new();
    for (key, session) in [("ordinary-one", "one"), ("ordinary-two", "two")] {
        let mut opts = options(key, session);
        let phases = Arc::new(Mutex::new(Vec::new()));
        let captured = phases.clone();
        opts.stream.on_stream_observation = Some(Arc::new(move |phase| captured.lock().unwrap().push(phase.to_string())));
        assert_eq!(run(&model, opts).await.stop_reason, "stop");
        let request = fixture.next().await;
        assert_eq!(request.path, "/v1/chat/completions");
        assert_eq!(request.headers["authorization"], format!("Bearer {key}"));
        assert_eq!(request.headers["x-prime-team-id"], format!("team-{session}"));
        assert_eq!(request.headers["session_id"], session);
        assert_eq!(request.headers["x-client-request-id"], session);
        assert_eq!(request.headers["x-session-affinity"], session);
        assert_eq!(request.headers["x-rotation"], key);
        assert_eq!(*phases.lock().unwrap(), vec![
            "generation_serialize_start", "generation_serialize_returned", "generation_send_start",
            "generation_headers_complete", "raw_event", "text", "raw_event", "terminal",
        ]);
        connections.push(request.connection);
        assert!(fixture.received.try_recv().is_err(), "ordinary models must not tokenize");
    }
    assert_eq!(connections[0], connections[1], "ordinary generation keeps upstream connection reuse");
    fixture.finish().await;
}

async fn guarded_dispatch_audit_matches_actual_wire() {
    use sha2::Digest;
    let directory = tempfile::tempdir().unwrap();
    let audit_path = directory.path().join("request-audit.jsonl");
    std::env::set_var("OPTIMUS_K3_REQUEST_AUDIT_PATH", &audit_path);
    let mut fixture = Fixture::start(Arc::new(|request| {
        if request.path.ends_with("/tokenize") {
            (200, vec![], json!({"count":106,"tokens":vec![1;106],"max_model_len":110}).to_string())
        } else { (200, vec![], done_body()) }
    })).await;
    let mut opts = options("audit-key", "audit-session");
    opts.stream.on_payload = Some(Arc::new(|mut body, _| Box::pin(async move {
        body["messages"] = json!([{"role":"user","content":"private-after-hook-fixture"}]);
        Some(body)
    })));
    assert_eq!(run(&model(fixture.address), opts).await.stop_reason, "stop");
    let count = fixture.next().await;
    let generation = fixture.next().await;
    assert_eq!(count.path, "/v1/tokenize");
    assert_eq!(generation.path, "/v1/chat/completions");
    assert_eq!(count.body["messages"], generation.body["messages"]);
    assert_eq!(generation.body["messages"][0]["content"], "private-after-hook-fixture");
    assert_eq!(generation.body["max_tokens"], 3);
    let audit = std::fs::read_to_string(&audit_path).unwrap();
    assert_eq!(audit.lines().count(), 1);
    assert!(!audit.contains("private-after-hook-fixture"));
    assert!(!audit.contains("audit-key"));
    let receipt: Value = serde_json::from_str(audit.trim()).unwrap();
    assert_eq!(receipt["serialized_body_sha256"], format!("{:x}", sha2::Sha256::digest(&generation.bytes)));
    assert_eq!(receipt["serialized_body_bytes"], generation.bytes.len());
    assert_eq!(receipt["prompt_tokens"], 106);
    assert_eq!(receipt["max_tokens"], 3);
    assert_eq!(receipt["count_http_requests"], 1);
    std::env::remove_var("OPTIMUS_K3_REQUEST_AUDIT_PATH");
    fixture.finish().await;
}

async fn cancelled_count_preflight_prevents_network() {
    let mut fixture = Fixture::start(Arc::new(|_| (200, vec![], "{}".into()))).await;
    let signal = CancellationToken::new();
    let cancel = signal.clone();
    let mut opts = options("cancel-key", "cancel-session");
    opts.stream.signal = Some(signal);
    opts.stream.on_stream_observation = Some(Arc::new(move |phase| {
        if phase == "count_serialize_complete" { cancel.cancel(); }
    }));
    let output = run(&model(fixture.address), opts).await;
    assert_eq!(output.stop_reason, "aborted");
    assert!(fixture.received.try_recv().is_err(), "cancelled preflight must not send a count or generation request");
    fixture.finish().await;
}

#[tokio::test]
async fn offline_speed_regressions() {
    // One test owns process-local synthetic policy throughout this binary.
    let _environment = Env::local_policy();
    ordinary_generation_keeps_upstream_pool_and_per_request_headers().await;
    reuse_headers_body_and_rotation().await;
    guarded_dispatch_audit_matches_actual_wire().await;
    cancelled_count_preflight_prevents_network().await;
    concurrent_header_isolation().await;
    count_http_semantics_and_redirect_rejection().await;
    raw_image_near_ceiling_and_unknown_remote_errors().await;
    partial_errors_preserve_non_replay_evidence().await;
}
