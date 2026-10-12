//! Isolated model configuration and local HTTP fixtures; no live credentials or providers.
use std::time::Duration;

use indexmap::IndexMap;
use pi_ai::providers::{
    anthropic::stream_simple_anthropic, openai_responses::stream_simple_openai_responses,
    simple_options::build_base_options,
};
use pi_ai::types::{
    Context, Message, SimpleStreamOptions, StreamOptions, UserContent, UserMessage,
};
use pi_coding_agent::core::auth_storage::{AuthStorage, AuthStorageOptions};
use pi_coding_agent::core::model_registry::{ModelDefinition, ModelRegistry, ProviderConfigInput};
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

fn auth() -> AuthStorage {
    AuthStorage::in_memory(
        IndexMap::new(),
        Some(AuthStorageOptions {
            prime_cli_config_path: None,
            use_prime_cli_config: false,
        }),
    )
}

#[test]
fn disk_overrides_custom_models_and_refresh_preserve_explicit_limits() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("models.json");
    let mut config = json!({"providers": {
        "anthropic": {"modelOverrides": {"claude-opus-5-5": {"maxTokens": 128000}}},
        "fixture": {"baseUrl": "http://127.0.0.1:1", "api": "openai-responses", "apiKey": "synthetic-key",
            "models": [{"id": "explicit", "maxTokens": 128000}, {"id": "default"}]}
    }});
    std::fs::write(&path, config.to_string()).unwrap();
    let mut registry = ModelRegistry::create(auth(), Some(path.to_string_lossy().into_owned()));
    assert_eq!(registry.get_error(), None);
    for (provider, id) in [("anthropic", "claude-opus-5-5"), ("fixture", "explicit")] {
        let model = registry.find(provider, id).unwrap();
        assert!(model.max_tokens_explicit);
        assert_eq!(
            build_base_options(&model, None, None).max_tokens,
            Some(128000.0)
        );
    }
    let default = registry.find("fixture", "default").unwrap();
    assert!(!default.max_tokens_explicit);
    assert_eq!(
        build_base_options(&default, None, None).max_tokens,
        Some(16384.0)
    );
    config["providers"]["anthropic"]["modelOverrides"]["claude-opus-5-5"] =
        json!({"name": "Renamed"});
    config["providers"]["fixture"]["models"][0] = json!({"id": "explicit"});
    std::fs::write(&path, config.to_string()).unwrap();
    registry.refresh();
    assert_eq!(registry.get_error(), None);
    assert!(
        !registry
            .find("anthropic", "claude-opus-5-5")
            .unwrap()
            .max_tokens_explicit
    );
    assert!(
        !registry
            .find("fixture", "explicit")
            .unwrap()
            .max_tokens_explicit
    );
}

#[test]
fn registered_provider_limits_survive_refresh_and_request_options_take_precedence() {
    let mut registry = ModelRegistry::in_memory(auth());
    registry
        .register_provider(
            "fixture",
            ProviderConfigInput {
                base_url: Some("http://127.0.0.1:1".into()),
                api: Some("openai-responses".into()),
                api_key: Some("synthetic-key".into()),
                models: Some(vec![
                    ModelDefinition {
                        id: "explicit".into(),
                        max_tokens: Some(128000.0),
                        ..Default::default()
                    },
                    ModelDefinition {
                        id: "default".into(),
                        ..Default::default()
                    },
                ]),
                ..Default::default()
            },
        )
        .unwrap();
    registry.refresh();
    let model = registry.find("fixture", "explicit").unwrap();
    assert!(model.max_tokens_explicit);
    assert_eq!(
        build_base_options(&model, None, None).max_tokens,
        Some(128000.0)
    );
    let options = SimpleStreamOptions {
        stream: StreamOptions {
            max_tokens: Some(64000.0),
            ..Default::default()
        },
        ..Default::default()
    };
    assert_eq!(
        build_base_options(&model, Some(&options), None).max_tokens,
        Some(64000.0)
    );
    assert!(
        !registry
            .find("fixture", "default")
            .unwrap()
            .max_tokens_explicit
    );
}

async fn serve(listener: TcpListener, anthropic: bool, expected: u64) {
    let (mut socket, _) = listener.accept().await.unwrap();
    let mut received = Vec::new();
    let (body_start, body_len) = loop {
        let mut buffer = [0; 4096];
        let count = socket.read(&mut buffer).await.unwrap();
        assert_ne!(count, 0);
        received.extend_from_slice(&buffer[..count]);
        assert!(received.len() < 128 * 1024);
        if let Some(end) = received.windows(4).position(|bytes| bytes == b"\r\n\r\n") {
            let headers = std::str::from_utf8(&received[..end]).unwrap();
            assert!(headers.starts_with(if anthropic {
                "POST /v1/messages "
            } else {
                "POST /responses "
            }));
            let length = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().unwrap())
                })
                .unwrap();
            assert!(length < 128 * 1024);
            break (end + 4, length);
        }
    };
    while received.len() < body_start + body_len {
        let mut buffer = [0; 4096];
        let count = socket.read(&mut buffer).await.unwrap();
        assert_ne!(count, 0);
        received.extend_from_slice(&buffer[..count]);
    }
    let request: Value =
        serde_json::from_slice(&received[body_start..body_start + body_len]).unwrap();
    assert_eq!(
        request[if anthropic {
            "max_tokens"
        } else {
            "max_output_tokens"
        }]
        .as_u64(),
        Some(expected)
    );
    let events = if anthropic {
        vec![
            json!({"type":"message_start","message":{"id":"fixture","usage":{"input_tokens":1,"output_tokens":0}}}),
            json!({"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":0}}),
            json!({"type":"message_stop"}),
        ]
    } else {
        vec![
            json!({"type":"response.created","response":{"id":"fixture"}}),
            json!({"type":"response.completed","response":{"id":"fixture","status":"completed","usage":{"input_tokens":1,"output_tokens":0,"total_tokens":1}}}),
        ]
    };
    let body: String = events
        .iter()
        .map(|event| {
            format!(
                "event: {}\ndata: {event}\n\n",
                event["type"].as_str().unwrap()
            )
        })
        .collect();
    socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).as_bytes()).await.unwrap();
}

#[tokio::test]
async fn configured_limits_reach_anthropic_and_responses_http_requests() {
    for api in ["anthropic-messages", "openai-responses"] {
        for (requested, expected) in [(None, 128000), (Some(64000.0), 64000)] {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let url = format!("http://{}", listener.local_addr().unwrap());
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("models.json");
            std::fs::write(
                &path,
                json!({"providers":{"fixture":{
                    "baseUrl":url,"api":api,"apiKey":"synthetic-key",
                    "models":[{"id":"fixture","contextWindow":200000,"maxTokens":128000}]
                }}})
                .to_string(),
            )
            .unwrap();
            let registry = ModelRegistry::create(auth(), Some(path.to_string_lossy().into_owned()));
            assert_eq!(registry.get_error(), None);
            let model = registry.find("fixture", "fixture").unwrap();
            let anthropic = api == "anthropic-messages";
            let server = tokio::spawn(async move {
                tokio::time::timeout(
                    Duration::from_secs(10),
                    serve(listener, anthropic, expected),
                )
                .await
                .unwrap();
            });
            let options = SimpleStreamOptions {
                stream: StreamOptions {
                    api_key: Some("synthetic-key".into()),
                    max_tokens: requested,
                    timeout_ms: Some(5000.0),
                    ..Default::default()
                },
                ..Default::default()
            };
            let context = Context::new(
                None,
                vec![Message::user(UserMessage::new(
                    UserContent::Text("Synthetic fixture".into()),
                    0,
                ))],
                None,
            );
            let stream = if anthropic {
                stream_simple_anthropic(&model, &context, Some(options))
            } else {
                stream_simple_openai_responses(&model, &context, Some(options))
            };
            let result = tokio::time::timeout(Duration::from_secs(10), stream.result()).await;
            if result.is_err() {
                server.abort();
            }
            let result = result.unwrap();
            server.await.unwrap();
            assert_eq!(
                result.stop_reason,
                "stop",
                "{}",
                result.error_message.unwrap_or_default()
            );
        }
    }
}
