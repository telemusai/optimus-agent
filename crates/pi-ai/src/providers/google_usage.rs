//! Gemini and Vertex report cached input inside prompt tokens and thoughts
//! separately from candidate output. Keep the normalized component totals consistent.
use crate::types::{Usage, UsageCost};
use serde_json::Value;

pub(super) fn normalize(metadata: &Value) -> Usage {
    let count = |key| metadata.get(key).and_then(Value::as_f64)
        .filter(|value| value.is_finite()).unwrap_or(0.0).max(0.0);
    let cache_read = count("cachedContentTokenCount");
    let input = (count("promptTokenCount") - cache_read).max(0.0);
    let output = count("candidatesTokenCount") + count("thoughtsTokenCount");
    Usage { input, output, cache_read, cache_write: 0.0,
        total_tokens: input + output + cache_read, cost: UsageCost::zero() }
}

#[cfg(test)]
mod tests {
    use crate::providers::{google, google_vertex};
    use crate::types::{Context, Model, ModelCost, StreamOptions};
    use serde_json::json;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    #[tokio::test]
    async fn both_google_transports_normalize_missing_totals_and_overreported_cache() {
        for vertex in [false, true] {
            for (prompt, cache, total) in [(100, 80, 115.0), (10, 20, 35.0)] {
                let listener = tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).await.unwrap();
                let address = listener.local_addr().unwrap();
                let body = format!("data: {}\n\n", json!({"candidates":[{"finishReason":"STOP"}],"usageMetadata":{
                    "promptTokenCount":prompt,"cachedContentTokenCount":cache,
                    "candidatesTokenCount":12,"thoughtsTokenCount":3
                }}));
                let server = tokio::spawn(async move {
                    let (mut socket, _) = listener.accept().await.unwrap();
                    let mut request = Vec::new();
                    loop {
                        let mut buffer = [0; 4096];
                        let count = socket.read(&mut buffer).await.unwrap(); assert_ne!(count, 0);
                        request.extend_from_slice(&buffer[..count]);
                        if let Some(end) = request.windows(4).position(|bytes| bytes == b"\r\n\r\n") {
                            let headers = std::str::from_utf8(&request[..end]).unwrap();
                            let length: usize = headers.lines().filter_map(|line| line.split_once(':'))
                                .find(|(key,_)| key.eq_ignore_ascii_case("content-length")).unwrap().1.trim().parse().unwrap();
                            if request.len() >= end + 4 + length { break; }
                        }
                    }
                    let response = format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
                    socket.write_all(response.as_bytes()).await.unwrap();
                });
                let mut model = Model::new("fixture", "fixture", if vertex {"google-vertex"} else {"google-generative-ai"}, "fixture", &format!("http://{address}/v1"));
                model.cost = ModelCost {input:4.0,output:12.0,cache_read:0.4,cache_write:0.0};
                let base = StreamOptions { api_key:Some("synthetic-fixture-key".into()), ..Default::default() };
                let context = Context::new(None, vec![], None);
                let stream = if vertex {
                    google_vertex::stream_google_vertex(&model, &context, Some(google_vertex::GoogleVertexOptions::from_base(&base)))
                } else {
                    google::stream_google(&model, &context, Some(google::GoogleOptions::from_base(&base)))
                };
                let message = tokio::time::timeout(std::time::Duration::from_secs(5), stream.result()).await.unwrap();
                assert_eq!(message.stop_reason, "stop", "{:?}", message.error_message);
                assert_eq!(message.usage.input, if prompt == 100 {20.0} else {0.0});
                assert_eq!(message.usage.output, 15.0);
                assert_eq!(message.usage.total_tokens, total);
                assert!((message.usage.cost.total - (message.usage.input*4.0 + 15.0*12.0 + cache as f64*0.4)/1_000_000.0).abs() < 1e-12);
                server.await.unwrap();
            }
        }
    }
}
