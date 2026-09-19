//! Account/session-scoped Responses transport. Never replays a sent WS request.
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use futures::{SinkExt, StreamExt};
use indexmap::IndexMap;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, AUTHORIZATION};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use tokio::net::TcpStream;
use tokio::sync::{Mutex, OwnedMutexGuard};
use tokio_tungstenite::{
    connect_async, tungstenite::client::IntoClientRequest, tungstenite::Message, MaybeTlsStream,
    WebSocketStream,
};

use super::openai_responses::{OpenAIResponsesOptions, ResponsesClient};
use super::openai_responses_shared::ResponsesEventStream;
use crate::types::{Model, ProviderResponse};

pub(crate) fn observe_event(options: &crate::types::StreamOptions, event: &Value) {
    let Some(observer) = &options.on_stream_observation else {
        return;
    };
    let kind = event["type"].as_str().unwrap_or("");
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        observer("raw_event");
        match kind {
            "response.output_text.delta" | "response.refusal.delta" => observer("text"),
            "response.reasoning_summary_text.delta"
            | "response.reasoning_text.delta"
            | "response.reasoning_summary_part.done" => observer("thinking"),
            "response.function_call_arguments.delta" | "response.function_call_arguments.done" => {
                observer("tool")
            }
            "response.output_item.added" if event["item"]["type"] == "function_call" => {
                observer("tool")
            }
            "response.completed"
            | "response.done"
            | "response.failed"
            | "response.incomplete"
            | "error" => observer("terminal"),
            _ => {}
        }
    }));
}

type Socket = WebSocketStream<MaybeTlsStream<TcpStream>>;
#[derive(Debug)]
pub(crate) struct TransportError {
    pub message: String,
    pub sent: bool,
}
impl From<&str> for TransportError {
    fn from(message: &str) -> Self {
        Self {
            message: message.into(),
            sent: false,
        }
    }
}
impl From<String> for TransportError {
    fn from(message: String) -> Self {
        Self {
            message,
            sent: false,
        }
    }
}
impl TransportError {
    fn interrupted(message: &str) -> Self {
        Self {
            message: message.into(),
            sent: true,
        }
    }
}
const MAX_CONNECTIONS: usize = 64;
// A split compaction and the parent request may coexist. A socket still carries
// exactly one request; the bounded pool provides separate leases, not multiplexing.
const MAX_SESSION_CONNECTIONS: usize = 4;
const MAX_AGE: Duration = Duration::from_secs(55 * 60);
const IDLE_AGE: Duration = Duration::from_secs(5 * 60);

#[derive(Default)]
struct Connection {
    socket: Option<Socket>,
    born: Option<Instant>,
}
struct PoolEntry {
    connection: Arc<Mutex<Connection>>,
    used: Instant,
}
struct Capability {
    models: Vec<String>,
    expires: Instant,
}
static POOL: OnceLock<std::sync::Mutex<HashMap<String, PoolEntry>>> = OnceLock::new();
static CAPABILITIES: OnceLock<Mutex<HashMap<String, Capability>>> = OnceLock::new();

pub(crate) fn opted_in(provider: &str) -> bool {
    matches!(provider, "azure-openai-managed" | "github-copilot")
}

/// No account credentials/default headers are stored on a shared HTTP client.
pub(crate) fn http_client(provider: &str) -> reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    if opted_in(provider) {
        CLIENT
            .get_or_init(|| {
                reqwest::Client::builder()
                    .redirect(reqwest::redirect::Policy::none())
                    .pool_idle_timeout(Duration::from_secs(90))
                    .build()
                    .expect("Responses HTTP client")
            })
            .clone()
    } else {
        reqwest::Client::new()
    }
}

pub(crate) fn request_headers(client: &ResponsesClient) -> Result<HeaderMap, String> {
    let mut headers = HeaderMap::new();
    for (key, value) in &client.default_headers {
        if let Some(value) = value {
            headers.insert(
                HeaderName::from_bytes(key.as_bytes()).map_err(|_| "Invalid header name")?,
                HeaderValue::from_str(value).map_err(|_| "Invalid header value")?,
            );
        }
    }
    if !headers.contains_key(AUTHORIZATION) {
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", client.api_key))
                .map_err(|_| "Invalid authorization header")?,
        );
    }
    Ok(headers)
}

fn identity(model: &Model, client: &ResponsesClient, headers: &HeaderMap) -> String {
    let mut digest = Sha256::new();
    for value in [&model.provider, &client.base_url, &client.api_key] {
        digest.update((value.len() as u64).to_le_bytes());
        digest.update(value.as_bytes());
    }
    let mut headers = headers
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_bytes()))
        .collect::<Vec<_>>();
    headers.sort_by(|a, b| a.0.cmp(b.0));
    for (key, value) in headers {
        digest.update(key);
        digest.update([0]);
        digest.update(value);
        digest.update([0]);
    }
    format!("{:x}", digest.finalize())
}

fn supported_models(provider: &str, body: &Value) -> Vec<String> {
    if provider == "azure-openai-managed" {
        let cap = &body["responsesWebSocket"];
        if cap["enabled"] != true
            || cap["store"] != false
            || cap["version"] != 1
            || cap["path"] != "/azure-openai/v1/responses"
            || cap["maxInFlightPerConnection"] != 1
        {
            return vec![];
        }
        return cap["models"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect();
    }
    body["data"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|item| {
            item["policy"]["state"]
                .as_str()
                .map(|s| s == "enabled")
                .unwrap_or(true)
                && item["supported_endpoints"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .any(|v| v == "ws:/responses")
        })
        .filter_map(|item| item["id"].as_str())
        .map(str::to_owned)
        .collect()
}

async fn capable(model: &Model, client: &ResponsesClient, headers: &HeaderMap, key: &str) -> bool {
    let cache = CAPABILITIES.get_or_init(|| Mutex::new(HashMap::new()));
    {
        let guard = cache.lock().await;
        if let Some(entry) = guard.get(key).filter(|e| e.expires > Instant::now()) {
            return entry.models.contains(&model.id);
        }
    }
    let Ok(mut base) = url::Url::parse(&client.base_url) else {
        return false;
    };
    if model.provider == "azure-openai-managed" {
        if base.path().trim_end_matches('/') != "/azure-openai/v1" {
            return false;
        }
        base.set_path("/health");
        base.set_query(None);
    } else {
        base.set_path(&format!("{}/models", base.path().trim_end_matches('/')));
    }
    let result = http_client(&model.provider)
        .get(base)
        .headers(headers.clone())
        .timeout(Duration::from_secs(3))
        .send()
        .await;
    let mut models = Vec::new();
    if let Ok(response) = result {
        if response.status().is_success() && response.content_length().unwrap_or(0) <= 2_000_000 {
            // Bound chunked bodies too. Catalogue failure is an HTTP fallback, not a login failure.
            let mut bytes = Vec::new();
            let mut chunks = response.bytes_stream();
            while let Some(Ok(chunk)) = chunks.next().await {
                if bytes.len() + chunk.len() > 2_000_000 {
                    bytes.clear();
                    break;
                }
                bytes.extend_from_slice(&chunk);
            }
            if let Ok(body) = serde_json::from_slice::<Value>(&bytes) {
                models = supported_models(&model.provider, &body);
            }
        }
    }
    let enabled = models.contains(&model.id);
    let mut guard = cache.lock().await;
    guard.retain(|_, e| e.expires > Instant::now());
    if guard.len() >= MAX_CONNECTIONS {
        guard.clear();
    }
    guard.insert(
        key.to_owned(),
        Capability {
            models,
            expires: Instant::now() + Duration::from_secs(if enabled { 300 } else { 30 }),
        },
    );
    enabled
}

fn connection(key: &str) -> Option<OwnedMutexGuard<Connection>> {
    let mut pool = POOL
        .get_or_init(|| std::sync::Mutex::new(HashMap::new()))
        .lock()
        .ok()?;
    let now = Instant::now();
    pool.retain(|_, e| {
        now.duration_since(e.used) < IDLE_AGE || Arc::strong_count(&e.connection) > 1
    });
    for slot in 0..MAX_SESSION_CONNECTIONS {
        let slot_key = format!("{key}:slot:{slot}");
        if !pool.contains_key(&slot_key) && pool.len() >= MAX_CONNECTIONS {
            let oldest = pool
                .iter()
                .filter(|(_, e)| Arc::strong_count(&e.connection) == 1)
                .min_by_key(|(_, e)| e.used)
                .map(|(k, _)| k.clone());
            if let Some(oldest) = oldest {
                pool.remove(&oldest);
            } else {
                return None;
            }
        }
        let entry = pool.entry(slot_key).or_insert_with(|| PoolEntry {
            connection: Arc::new(Mutex::new(Connection::default())),
            used: now,
        });
        if let Ok(lease) = entry.connection.clone().try_lock_owned() {
            entry.used = now;
            return Some(lease);
        }
    }
    // No request has been sent: an occupied pool can safely use HTTP/SSE rather
    // than producing a retryable 10-second "active request" failure.
    None
}

fn create_payload(params: &Map<String, Value>) -> Map<String, Value> {
    let mut payload = params.clone();
    payload.remove("stream");
    payload.remove("background");
    // Full context every turn: safe across compaction, rewinds and cross-provider histories.
    // Do not infer a previous_response_id merely from a transcript ID.
    payload.remove("previous_response_id");
    payload.insert("type".into(), Value::String("response.create".into()));
    payload.insert("store".into(), Value::Bool(false));
    payload
}

struct Lease {
    socket: Socket,
    owner: OwnedMutexGuard<Connection>,
    signal: Option<tokio_util::sync::CancellationToken>,
    deadline: tokio::time::Instant,
}

/// None permits SSE only before response.create was sent. Errors after sending are final.
pub(crate) async fn try_websocket(
    model: &Model,
    client: &ResponsesClient,
    params: &Map<String, Value>,
    options: &OpenAIResponsesOptions,
) -> Result<Option<(ResponsesEventStream, ProviderResponse)>, TransportError> {
    if !opted_in(&model.provider)
        || matches!(options.stream.transport.as_deref(), Some("sse" | "http"))
    {
        return Ok(None);
    }
    let Some(session) = options
        .stream
        .session_id
        .as_deref()
        .filter(|s| !s.is_empty())
    else {
        return Ok(None);
    };
    let headers = request_headers(client)?;
    let account_key = identity(model, client, &headers);
    let capability = tokio::time::timeout(
        Duration::from_secs(4),
        capable(model, client, &headers, &account_key),
    );
    let supported = if let Some(signal) = &options.stream.signal {
        tokio::select! { biased; _ = signal.cancelled() => return Err("Request was aborted".into()), result = capability => result.unwrap_or(false) }
    } else {
        capability.await.unwrap_or(false)
    };
    if options
        .stream
        .signal
        .as_ref()
        .map(|s| s.is_cancelled())
        .unwrap_or(false)
    {
        return Err("Request was aborted".into());
    }
    if !supported {
        return Ok(None);
    }
    let key = format!("{}:{}:{}", account_key, model.id, session);
    let Some(mut owner) = connection(&key) else {
        return Ok(None);
    };
    if owner.born.map(|t| t.elapsed() >= MAX_AGE).unwrap_or(false) {
        owner.socket = None;
    }
    let socket = if let Some(socket) = owner.socket.take() {
        socket
    } else {
        let mut url = url::Url::parse(&format!(
            "{}/responses",
            client.base_url.trim_end_matches('/')
        ))
        .map_err(|_| "Invalid Responses URL")?;
        let scheme = if url.scheme() == "https" { "wss" } else { "ws" };
        url.set_scheme(scheme)
            .map_err(|_| "Invalid Responses scheme")?;
        let mut request = url
            .as_str()
            .into_client_request()
            .map_err(|_| "Invalid WebSocket request")?;
        for (name, value) in &headers {
            if !matches!(
                name.as_str(),
                "accept" | "content-type" | "content-length" | "connection" | "upgrade" | "host"
            ) {
                request.headers_mut().insert(name.clone(), value.clone());
            }
        }
        let connect = tokio::time::timeout(Duration::from_secs(5), connect_async(request));
        let result = if let Some(signal) = &options.stream.signal {
            tokio::select! { _ = signal.cancelled() => return Err("Request was aborted".into()), result = connect => result }
        } else {
            connect.await
        };
        let Ok(Ok((socket, _response))) = result else {
            // An unsuccessful HTTP upgrade cannot have submitted model work.
            CAPABILITIES.get().unwrap().lock().await.insert(
                account_key,
                Capability {
                    models: vec![],
                    expires: Instant::now() + Duration::from_secs(30),
                },
            );
            return Ok(None);
        };
        owner.born = Some(Instant::now());
        socket
    };
    let mut lease = Lease {
        socket,
        owner,
        signal: options.stream.signal.clone(),
        deadline: tokio::time::Instant::now()
            + Duration::from_millis(
                options
                    .stream
                    .timeout_ms
                    .unwrap_or(600_000.0)
                    .clamp(1.0, 3_600_000.0) as u64,
            ),
    };
    let payload = serde_json::to_string(&create_payload(params))
        .map_err(|_| "Could not encode Responses request")?;
    // If this fails we cannot know how many bytes reached the server: never retry/fallback.
    let send = tokio::time::timeout_at(
        lease.deadline,
        lease.socket.send(Message::Text(payload.into())),
    );
    let sent = if let Some(signal) = &lease.signal {
        tokio::select! { biased; _ = signal.cancelled() => return Err("Request was aborted".into()), result = send => result }
    } else {
        send.await
    };
    sent.map_err(|_| {
        TransportError::interrupted("WebSocket request send timed out; not replayed")
    })?
    .map_err(|_| TransportError::interrupted("WebSocket request send failed; not replayed"))?;
    // The socket accepted our bytes; no server acknowledgement has arrived yet. The
    // edge marker keeps that distinct from a real HTTP response header edge so the
    // host records this as `transport_open_ack_ms` and leaves the header stage null.
    let response = ProviderResponse {
        status: 101,
        headers: IndexMap::from([
            ("x-optimus-transport".into(), "websocket".into()),
            ("x-optimus-response-edge".into(), "transport_send_ack".into()),
        ]),
    };
    let events = futures::stream::unfold(Some(lease), |state| async move {
        let mut lease = state?;
        loop {
            let next = tokio::time::timeout_at(lease.deadline, lease.socket.next());
            let next = if let Some(signal) = &lease.signal {
                tokio::select! { biased; _ = signal.cancelled() => return Some((transport_error("Request was aborted"), None)), result = next => result }
            } else {
                next.await
            };
            // B7: never fold the deadline into a generic socket close. A hit deadline and
            // a socket that stopped producing frames are different failures, and the
            // operator must be able to tell them apart. Neither is replayed.
            let mut event = match next {
                Ok(Some(Ok(Message::Text(text)))) => match serde_json::from_str::<Value>(&text) {
                    Ok(event) => event,
                    Err(_) => {
                        return Some((
                            transport_error("Invalid WebSocket JSON; request not replayed"),
                            None,
                        ))
                    }
                },
                Ok(Some(Ok(Message::Ping(bytes)))) => {
                    let pong = tokio::time::timeout_at(
                        lease.deadline,
                        lease.socket.send(Message::Pong(bytes)),
                    );
                    let pong = if let Some(signal) = &lease.signal {
                        tokio::select! { biased; _ = signal.cancelled() => return Some((transport_error("Request was aborted"), None)), result = pong => result }
                    } else {
                        pong.await
                    };
                    if !matches!(pong, Ok(Ok(()))) {
                        return Some((transport_error("WebSocket pong failed"), None));
                    }
                    continue;
                }
                Ok(Some(Ok(Message::Pong(_)))) => continue,
                Ok(Some(Err(error))) => {
                    return Some((
                        transport_error(&format!(
                            "WebSocket transport error ({error}); request not replayed"
                        )),
                        None,
                    ))
                }
                Ok(None) => {
                    return Some((
                        transport_error(
                            "WebSocket closed before response completion; request not replayed",
                        ),
                        None,
                    ))
                }
                Ok(Some(Ok(Message::Close(_)))) => {
                    return Some((
                        transport_error(
                            "WebSocket closed before response completion; request not replayed",
                        ),
                        None,
                    ))
                }
                Ok(Some(Ok(_))) => {
                    return Some((
                        transport_error(
                            "WebSocket frame stream ended before response completion; request not replayed",
                        ),
                        None,
                    ))
                }
                Err(_) => {
                    return Some((
                        transport_error(
                            "WebSocket response deadline expired before response completion; request not replayed",
                        ),
                        None,
                    ))
                }
            };
            normalize_event(&mut event);
            let kind = event["type"].as_str().unwrap_or("");
            if kind == "response.completed" {
                if event["response"]["status"] == "completed" {
                    lease.owner.socket = Some(lease.socket);
                }
                return Some((event, None));
            }
            if matches!(kind, "error" | "response.failed" | "response.incomplete") {
                return Some((event, None));
            }
            return Some((event, Some(lease)));
        }
    });
    Ok(Some((Box::pin(events), response)))
}

fn transport_error(message: &str) -> Value {
    json!({"type":"error","code":"responses_request_interrupted","message":message})
}

fn normalize_event(event: &mut Value) {
    if !event.is_object() {
        *event = transport_error("Invalid WebSocket event envelope");
        return;
    }
    if matches!(
        event["type"].as_str(),
        Some("response.completed" | "response.incomplete" | "response.done")
    ) && !event["response"].is_object()
    {
        *event = transport_error("Invalid WebSocket terminal envelope");
        return;
    }
    match event["type"].as_str() {
        Some("error") => {
            if event.get("code").is_none() {
                event["code"] = event["error"]["code"].clone();
            }
            if event.get("message").is_none() {
                event["message"] = event["error"]["message"].clone();
            }
        }
        Some("response.incomplete") => {
            event["type"] = json!("response.completed");
            event["response"]["status"] = json!("incomplete");
        }
        Some("response.done") => {
            event["type"] = json!("response.completed");
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    #[test]
    fn transport_errors_and_incomplete_responses_keep_shared_parser_contract() {
        let mut error =
            json!({"type":"error","error":{"code":"rate_limit_error","message":"wait"}});
        normalize_event(&mut error);
        assert_eq!(error["message"], "wait");
        assert_eq!(error["code"], "rate_limit_error");
        let mut incomplete =
            json!({"type":"response.incomplete","response":{"id":"r","usage":{"output_tokens":3}}});
        normalize_event(&mut incomplete);
        assert_eq!(incomplete["response"]["status"], "incomplete");
        assert_eq!(incomplete["type"], "response.completed");
        assert_eq!(incomplete["response"]["usage"]["output_tokens"], 3);
    }

    #[tokio::test]
    async fn pooled_http_reuses_connection_without_reusing_authorization() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let requests = [
            ("azure-openai-managed", "fake-azure-first"),
            ("azure-openai-managed", "fake-azure-rotated"),
            ("github-copilot", "fake-github-first"),
            ("github-copilot", "fake-github-rotated"),
        ];
        let server = tokio::spawn(async move {
            // All four requests must arrive on this one accepted connection.
            let (mut socket, _) = listener.accept().await.unwrap();
            for (index, (_, expected_key)) in requests.iter().enumerate() {
                let mut request = Vec::new();
                let mut buffer = [0_u8; 2048];
                while !request.windows(4).any(|bytes| bytes == b"\r\n\r\n") {
                    let count =
                        tokio::time::timeout(Duration::from_secs(5), socket.read(&mut buffer))
                            .await
                            .expect("request did not reuse the pooled connection")
                            .unwrap();
                    assert!(count > 0, "pooled connection closed before request {index}");
                    request.extend_from_slice(&buffer[..count]);
                    assert!(request.len() < 8192);
                }
                let request = String::from_utf8(request).unwrap();
                assert!(request.starts_with(&format!("GET /probe/{index} HTTP/1.1\r\n")));
                let authorization = request
                    .lines()
                    .filter_map(|line| line.split_once(':'))
                    .filter(|(name, _)| name.eq_ignore_ascii_case("authorization"))
                    .map(|(_, value)| value.trim())
                    .collect::<Vec<_>>();
                assert_eq!(authorization, vec![format!("Bearer {expected_key}")]);
                socket
                    .write_all(
                        b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: keep-alive\r\n\r\nok",
                    )
                    .await
                    .unwrap();
            }
        });
        for (index, (provider, api_key)) in requests.iter().enumerate() {
            let client = ResponsesClient {
                api_key: (*api_key).into(),
                base_url: format!("http://{address}"),
                default_headers: IndexMap::new(),
            };
            let response = http_client(provider)
                .get(format!("{}/probe/{index}", client.base_url))
                .headers(request_headers(&client).unwrap())
                .timeout(Duration::from_secs(5))
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), reqwest::StatusCode::OK);
            // Fully consume the response so reuse is observable on the next call.
            assert_eq!(response.text().await.unwrap(), "ok");
        }
        server.await.unwrap();
    }

    async fn fake_endpoint(
        fail_partial: bool,
    ) -> (String, Arc<AtomicUsize>, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let connections = Arc::new(AtomicUsize::new(0));
        let counter = connections.clone();
        let server = tokio::spawn(async move {
            let mut children = tokio::task::JoinSet::new();
            while let Ok((mut tcp, _)) = listener.accept().await {
                let counter = counter.clone();
                children.spawn(async move {
                    let mut buf = [0;4096];
                    let count = tcp.peek(&mut buf).await.unwrap();
                    if String::from_utf8_lossy(&buf[..count]).starts_with("GET /models ") {
                        let _ = tcp.read(&mut buf).await.unwrap();
                        let body = r#"{"data":[{"id":"model","policy":{"state":"enabled"},"supported_endpoints":["ws:/responses"]}]}"#;
                        tcp.write_all(format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",body.len(),body).as_bytes()).await.unwrap();
                    } else {
                        counter.fetch_add(1, Ordering::SeqCst);
                        let mut ws = tokio_tungstenite::accept_async(tcp).await.unwrap();
                        while let Some(Ok(Message::Text(text))) = ws.next().await {
                            let request: Value = serde_json::from_str(&text).unwrap();
                            assert_eq!(request["type"], "response.create");
                            assert_eq!(request["store"], false);
                            assert!(request.get("stream").is_none());
                            for event in [json!({"type":"response.created","response":{"id":"response1"}}),json!({"type":"response.output_text.delta","delta":"héllo"})] {
                                ws.send(Message::Text(event.to_string().into())).await.unwrap();
                            }
                            if fail_partial { ws.close(None).await.unwrap(); break; }
                            ws.send(Message::Text(json!({"type":"response.completed","response":{"id":"response1","status":"completed"}}).to_string().into())).await.unwrap();
                        }
                    }
                });
            }
        });
        (format!("http://{address}"), connections, server)
    }

    fn fake_request(
        base: &str,
    ) -> (
        Model,
        ResponsesClient,
        OpenAIResponsesOptions,
        Map<String, Value>,
    ) {
        let model = Model::new("model", "Model", "openai-responses", "github-copilot", base);
        let client = ResponsesClient {
            api_key: "test-not-a-real-key".into(),
            base_url: base.into(),
            default_headers: IndexMap::new(),
        };
        let options = OpenAIResponsesOptions {
            stream: crate::types::StreamOptions {
                session_id: Some("fake-chat".into()),
                timeout_ms: Some(1000.0),
                ..Default::default()
            },
            ..Default::default()
        };
        let params =
            json!({"model":"model","input":[{"role":"user","content":"hello"}],"stream":true})
                .as_object()
                .unwrap()
                .clone();
        (model, client, options, params)
    }

    #[tokio::test]
    async fn websocket_reuses_one_session_socket_and_resets_full_context_each_turn() {
        let (base, connections, server) = fake_endpoint(false).await;
        let (model, client, options, mut params) = fake_request(&base);
        for _ in 0..2 {
            let (events, metadata) = try_websocket(&model, &client, &params, &options)
                .await
                .unwrap()
                .unwrap();
            assert_eq!(metadata.status, 101);
            let events = events.collect::<Vec<_>>().await;
            assert_eq!(events.len(), 3);
            assert_eq!(events[1]["delta"], "héllo");
            params.insert(
                "input".into(),
                json!([{"type":"compaction","encrypted_content":"checkpoint"}]),
            );
        }
        assert_eq!(connections.load(Ordering::SeqCst), 1);
        server.abort();
    }

    #[tokio::test]
    async fn websocket_parallel_summaries_use_distinct_sockets_without_waiting_for_each_other() {
        let (base, connections, server) = fake_endpoint(false).await;
        let (model, client, options, params) = fake_request(&base);
        let (history, _) = try_websocket(&model, &client, &params, &options)
            .await.unwrap().unwrap();
        // Keep the first stream leased: split compaction starts its prefix summary
        // before the history summary finishes, with the same account/session ID.
        let (prefix, _) = tokio::time::timeout(
            Duration::from_secs(2), try_websocket(&model, &client, &params, &options),
        ).await.expect("parallel summary must not wait on the other stream")
            .unwrap().expect("parallel summary should retain WebSockets");
        let (history, prefix) = tokio::join!(history.collect::<Vec<_>>(), prefix.collect::<Vec<_>>());
        assert_eq!(history.last().unwrap()["type"], "response.completed");
        assert_eq!(prefix.last().unwrap()["type"], "response.completed");
        assert_eq!(connections.load(Ordering::SeqCst), 2);
        // Sequential work still reuses a warm socket after both summaries finish.
        let (events, _) = try_websocket(&model, &client, &params, &options).await.unwrap().unwrap();
        assert_eq!(events.collect::<Vec<_>>().await.len(), 3);
        assert_eq!(connections.load(Ordering::SeqCst), 2);
        server.abort();
    }

    #[tokio::test]
    async fn websocket_session_capacity_falls_back_only_before_send() {
        let (base, connections, server) = fake_endpoint(false).await;
        let (model, client, options, params) = fake_request(&base);
        let mut active = Vec::new();
        for _ in 0..MAX_SESSION_CONNECTIONS {
            active.push(try_websocket(&model, &client, &params, &options).await.unwrap().unwrap().0);
        }
        assert!(tokio::time::timeout(Duration::from_secs(2),
            try_websocket(&model, &client, &params, &options))
            .await.unwrap().unwrap().is_none());
        assert_eq!(connections.load(Ordering::SeqCst), MAX_SESSION_CONNECTIONS);
        for stream in active {
            assert_eq!(stream.collect::<Vec<_>>().await.last().unwrap()["type"], "response.completed");
        }
        server.abort();
    }

    #[tokio::test]
    async fn websocket_partial_output_is_not_replayed_or_successfully_completed() {
        let (base, connections, server) = fake_endpoint(true).await;
        let (model, client, options, params) = fake_request(&base);
        let (events, _) = try_websocket(&model, &client, &params, &options)
            .await
            .unwrap()
            .unwrap();
        let events = events.collect::<Vec<_>>().await;
        assert_eq!(events.len(), 3);
        assert_eq!(events[2]["type"], "error");
        // B7: a socket that goes away is reported as a close, not as a deadline, and it
        // is still never replayed.
        assert_eq!(events[2]["code"], "responses_request_interrupted");
        let message = events[2]["message"].as_str().unwrap();
        assert!(message.contains("closed before response completion"), "{message}");
        assert!(message.contains("request not replayed"), "{message}");
        assert!(
            !message.contains("deadline"),
            "a close must stay distinguishable from a deadline: {message}"
        );
        assert_eq!(connections.load(Ordering::SeqCst), 1);
        server.abort();
    }

    /// B7: a response deadline is reported as a deadline, still without replaying the
    /// already-sent request.
    #[tokio::test]
    async fn websocket_deadline_is_reported_as_a_deadline_and_never_replayed() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let accepted = Arc::new(AtomicUsize::new(0));
        let counter = accepted.clone();
        let server = tokio::spawn(async move {
            let mut children = tokio::task::JoinSet::new();
            while let Ok((mut tcp, _)) = listener.accept().await {
                let counter = counter.clone();
                children.spawn(async move {
                    let mut buf = [0; 4096];
                    let count = tcp.peek(&mut buf).await.unwrap();
                    if String::from_utf8_lossy(&buf[..count]).starts_with("GET /models ") {
                        let _ = tcp.read(&mut buf).await.unwrap();
                        let body = r#"{"data":[{"id":"model","policy":{"state":"enabled"},"supported_endpoints":["ws:/responses"]}]}"#;
                        tcp.write_all(format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body).as_bytes()).await.unwrap();
                    } else {
                        counter.fetch_add(1, Ordering::SeqCst);
                        let mut ws = tokio_tungstenite::accept_async(tcp).await.unwrap();
                        while let Some(Ok(Message::Text(_))) = ws.next().await {
                            // Accept the request, stream one frame, then stall. The client
                            // deadline (1000 ms in this fixture) must expire.
                            ws.send(Message::Text(
                                json!({"type":"response.created","response":{"id":"resp_deadline"}}).to_string().into(),
                            )).await.unwrap();
                            futures::future::pending::<()>().await;
                        }
                    }
                });
            }
        });
        let base = format!("http://{address}");
        let (model, client, options, params) = fake_request(&base);
        let (events, metadata) = try_websocket(&model, &client, &params, &options)
            .await
            .unwrap()
            .expect("the capability probe must keep WebSockets");
        assert_eq!(metadata.status, 101);
        assert_eq!(
            metadata.headers.get("x-optimus-response-edge").map(String::as_str),
            Some("transport_send_ack"),
            "a local send acknowledgement must be labelled as such"
        );
        let events = tokio::time::timeout(Duration::from_secs(5), events.collect::<Vec<_>>())
            .await
            .expect("the deadline must fire instead of hanging");
        let terminal = events.last().expect("a terminal event");
        assert_eq!(terminal["type"], "error");
        let message = terminal["message"].as_str().unwrap();
        assert!(message.contains("deadline expired"), "{message}");
        assert!(message.contains("request not replayed"), "{message}");
        assert!(
            !message.contains("closed before"),
            "a deadline must stay distinguishable from a socket close: {message}"
        );
        assert_eq!(accepted.load(Ordering::SeqCst), 1, "the payload is sent once and never replayed");
        server.abort();
    }

    #[tokio::test]
    async fn websocket_cancellation_drops_lease_and_never_reuses_unfinished_socket() {
        let (base, connections, server) = fake_endpoint(false).await;
        let (model, client, mut options, params) = fake_request(&base);
        let cancel = tokio_util::sync::CancellationToken::new();
        options.stream.signal = Some(cancel.clone());
        let (events, _) = try_websocket(&model, &client, &params, &options)
            .await
            .unwrap()
            .unwrap();
        cancel.cancel();
        drop(events);
        options.stream.signal = None;
        let (events, _) = try_websocket(&model, &client, &params, &options)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            events.collect::<Vec<_>>().await.last().unwrap()["type"],
            "response.completed"
        );
        assert_eq!(connections.load(Ordering::SeqCst), 2);
        server.abort();
    }

    #[test]
    fn observer_emits_only_whitelisted_phases_and_contains_panics() {
        let phases = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let capture = phases.clone();
        let mut options = crate::types::StreamOptions::default();
        options.on_stream_observation = Some(Arc::new(move |phase| {
            capture.lock().unwrap().push(phase.into())
        }));
        observe_event(
            &options,
            &json!({"type":"response.output_text.delta","delta":"PRIVATE"}),
        );
        assert_eq!(*phases.lock().unwrap(), vec!["raw_event", "text"]);
        assert!(serde_json::to_string(&options)
            .unwrap()
            .find("observation")
            .is_none());
        options.on_stream_observation = Some(Arc::new(|_| panic!("observer")));
        observe_event(&options, &json!({"type":"response.completed"}));
    }
    #[test]
    fn websocket_capability_is_provider_and_model_gated() {
        assert!(!opted_in("ollama-cloud"));
        assert!(!opted_in("openai-codex"));
        assert!(opted_in("github-copilot"));
        assert!(opted_in("azure-openai-managed"));
        let body = json!({"data":[{"id":"yes","supported_endpoints":["ws:/responses"]},{"id":"no","supported_endpoints":["/responses"]},{"id":"disabled","policy":{"state":"disabled"},"supported_endpoints":["ws:/responses"]}]});
        assert_eq!(supported_models("github-copilot", &body), vec!["yes"]);
        assert!(supported_models("azure-openai-managed", &body).is_empty());
    }
    #[test]
    fn azure_capability_requires_explicit_enable_and_no_storage() {
        let body = json!({"responsesWebSocket":{"enabled":true,"store":false,"version":1,
            "path":"/azure-openai/v1/responses","maxInFlightPerConnection":1,"models":["yes"]}});
        assert_eq!(supported_models("azure-openai-managed", &body), vec!["yes"]);
        for (field, value) in [
            ("enabled", json!(false)),
            ("enabled", Value::Null),
            ("store", json!(true)),
            ("store", Value::Null),
        ] {
            let mut rejected = body.clone();
            rejected["responsesWebSocket"][field] = value;
            assert!(supported_models("azure-openai-managed", &rejected).is_empty());
        }
    }
    #[test]
    fn websocket_full_context_preserves_tools_and_compaction_without_stale_continuation() {
        let params = json!({"stream":true,"background":false,"store":true,"previous_response_id":"stale","input":[{"type":"compaction","encrypted_content":"abc"},{"type":"function_call_output","call_id":"c","output":"ok"}],"tools":[{"name":"python"}],"reasoning":{"effort":"xhigh"}}).as_object().unwrap().clone();
        let payload = create_payload(&params);
        assert_eq!(payload["input"], params["input"]);
        assert_eq!(payload["tools"], params["tools"]);
        assert_eq!(payload["reasoning"], params["reasoning"]);
        assert_eq!(payload["store"], false);
        assert_eq!(payload["type"], "response.create");
        assert!(!payload.contains_key("previous_response_id"));
        assert!(!payload.contains_key("stream"));
        assert!(!payload.contains_key("background"));
    }
    #[test]
    fn pool_identity_changes_on_credentials_route_and_headers() {
        let model = Model::new(
            "a",
            "a",
            "openai-responses",
            "github-copilot",
            "http://localhost/v1",
        );
        let mut client = ResponsesClient {
            api_key: "one".into(),
            base_url: model.base_url.clone(),
            default_headers: IndexMap::new(),
        };
        let first = identity(&model, &client, &request_headers(&client).unwrap());
        client.api_key = "two".into();
        assert_ne!(
            first,
            identity(&model, &client, &request_headers(&client).unwrap())
        );
        assert!(!first.contains("one"));
    }
}
