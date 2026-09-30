//! Offline wire fixtures; synthetic credentials and a loopback server only.
use pi_ai::providers::google_vertex::stream_simple_google_vertex;
use pi_ai::types::{Context, Model, SimpleStreamOptions, StreamOptions};
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[tokio::test]
async fn gemma4_simple_stream_uses_supported_thinking_levels_on_the_wire() {
    for id in ["gemma-4-26b-a4b-it", "publishers/google/models/gemma4-31b-it"] {
        for (effort, expected) in [
            (None, json!({"thinkingLevel":"MINIMAL"})),
            (Some("off"), json!({"thinkingLevel":"MINIMAL"})),
            (Some("minimal"), json!({"includeThoughts":true,"thinkingLevel":"MINIMAL"})),
            (Some("low"), json!({"includeThoughts":true,"thinkingLevel":"MINIMAL"})),
            (Some("medium"), json!({"includeThoughts":true,"thinkingLevel":"HIGH"})),
            (Some("high"), json!({"includeThoughts":true,"thinkingLevel":"HIGH"})),
            (Some("xhigh"), json!({"includeThoughts":true,"thinkingLevel":"HIGH"})),
            (Some("max"), json!({"includeThoughts":true,"thinkingLevel":"HIGH"})),
        ] {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            let server = tokio::spawn(async move {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut bytes = Vec::new();
                let body = loop {
                    let mut chunk = [0; 4096];
                    let count = socket.read(&mut chunk).await.unwrap();
                    assert_ne!(count, 0);
                    bytes.extend_from_slice(&chunk[..count]);
                    if let Some(end) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
                        let headers = std::str::from_utf8(&bytes[..end]).unwrap();
                        assert!(headers.contains("streamGenerateContent?alt=sse"));
                        let length = headers.lines().filter_map(|line| line.split_once(':'))
                            .find(|(key, _)| key.eq_ignore_ascii_case("content-length"))
                            .unwrap().1.trim().parse::<usize>().unwrap();
                        if bytes.len() >= end + 4 + length {
                            break serde_json::from_slice::<Value>(&bytes[end + 4..end + 4 + length]).unwrap();
                        }
                    }
                };
                let reply = "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"fixture-ok\"}]},\"finishReason\":\"STOP\"}]}\n\n";
                socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{reply}", reply.len()).as_bytes()).await.unwrap();
                body
            });
            let mut model = Model::new(id, id, "google-vertex", "google-vertex", &format!("http://{address}"));
            model.reasoning = true;
            model.max_tokens = 8192.0;
            let stream = stream_simple_google_vertex(&model, &Context::default(), Some(SimpleStreamOptions {
                reasoning: effort.map(str::to_string),
                stream: StreamOptions { api_key: Some("synthetic-vertex-key".into()), ..Default::default() },
                ..Default::default()
            }));
            let output = tokio::time::timeout(std::time::Duration::from_secs(5), stream.result()).await.unwrap();
            assert_eq!(output.stop_reason, "stop", "{id} {effort:?}: {:?}", output.error_message);
            let body = tokio::time::timeout(std::time::Duration::from_secs(5), server).await.unwrap().unwrap();
            assert_eq!(body["generationConfig"]["thinkingConfig"], expected, "{id} {effort:?}");
        }
    }
}
