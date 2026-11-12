//! Native compaction transport failures must not become "unsupported" text fallback.
//! All requests use a loopback fixture and synthetic credentials.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use pi_agent_core::types::AgentMessage;
use pi_ai::compaction::CompactionOptions;
use pi_ai::types::{Context, Message, Model, UserContent, UserMessage};
use pi_coding_agent::core::compaction::compaction::{
    compact, default_compaction_settings, CompactionPreparation, ProviderRetryPolicy,
    SummaryCallRunner,
};
use pi_coding_agent::core::compaction::utils::create_file_ops;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

async fn read_request(socket: &mut tokio::net::TcpStream) -> Vec<u8> {
    let mut request = Vec::new();
    let mut buffer = [0_u8; 4096];
    loop {
        let read = socket.read(&mut buffer).await.unwrap();
        assert!(read > 0, "client closed before sending a complete request");
        request.extend_from_slice(&buffer[..read]);
        assert!(request.len() < 64 * 1024);
        if let Some(header_end) = request.windows(4).position(|bytes| bytes == b"\r\n\r\n") {
            let headers = String::from_utf8_lossy(&request[..header_end]);
            let content_length = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().unwrap())
                })
                .unwrap();
            if request.len() >= header_end + 4 + content_length {
                return request;
            }
        }
    }
}

fn preparation() -> CompactionPreparation {
    CompactionPreparation {
        first_kept_entry_id: "retained-turn".to_string(),
        messages_to_summarize: vec![AgentMessage::Message(Message::User(UserMessage::new(
            UserContent::Text("Preserve this task through native compaction.".to_string()),
            0,
        )))],
        turn_prefix_messages: Vec::new(),
        is_split_turn: false,
        tokens_before: 250_000.0,
        retained_state_anchor: None,
        previous_summary: None,
        file_ops: create_file_ops(),
        settings: default_compaction_settings(),
        task_state: None,
        typed_state: None,
    }
}

#[tokio::test]
async fn native_http_429_is_not_unsupported_and_never_calls_text_summary() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let request = read_request(&mut socket).await;
        assert!(String::from_utf8_lossy(&request).starts_with("POST /responses/compact "));
        socket
            .write_all(
                b"HTTP/1.1 429 Too Many Requests\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}",
            )
            .await
            .unwrap();
    });

    let mut model = Model::new(
        "gpt-6-astra",
        "Synthetic native compaction",
        "openai-responses",
        "openai",
        format!("http://{address}"),
    );
    model.context_window = 1_000_000.0;
    model.max_tokens = 32_000.0;
    let summary_calls = Arc::new(AtomicUsize::new(0));
    let observed_summary_calls = summary_calls.clone();
    let runner: SummaryCallRunner = Arc::new(move |_| {
        observed_summary_calls.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { Err("text fallback invoked".to_string()) })
    });
    let provider_context = Context::default();
    let options = CompactionOptions::default();
    let no_retry = ProviderRetryPolicy {
        enabled: false,
        max_retries: 0,
        base_delay_ms: 0.0,
        max_retry_delay_ms: 0.0,
    };

    let error = compact(
        &preparation(),
        &model,
        "synthetic-test-key",
        None,
        None,
        None,
        runner,
        Some(&no_retry),
        Some((&provider_context, Some(&options))),
    )
    .await
    .expect_err("429 must fail native compaction rather than report it as unsupported");
    server.await.unwrap();

    let summary_call_count = summary_calls.load(Ordering::SeqCst);
    eprintln!("case=http_429 text_summary_calls={summary_call_count} error={error}");
    assert!(error.contains("HTTP 429"), "unexpected error: {error}");
    assert_eq!(summary_call_count, 0);
}

#[tokio::test]
async fn native_body_transport_and_malformed_json_never_call_text_summary() {
    for response in [
        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 1000\r\nConnection: close\r\n\r\n{\"output\":[]".as_slice(),
        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 8\r\nConnection: close\r\n\r\nnot-json".as_slice(),
    ] {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let response = response.to_vec();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            read_request(&mut socket).await;
            socket.write_all(&response).await.unwrap();
        });
        let mut model = Model::new(
            "gpt-6-astra",
            "Synthetic native compaction",
            "openai-responses",
            "openai",
            format!("http://{address}"),
        );
        model.context_window = 1_000_000.0;
        model.max_tokens = 32_000.0;
        let summary_calls = Arc::new(AtomicUsize::new(0));
        let observed_summary_calls = summary_calls.clone();
        let runner: SummaryCallRunner = Arc::new(move |_| {
            observed_summary_calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { Err("text fallback invoked".to_string()) })
        });
        let provider_context = Context::default();
        let options = CompactionOptions::default();
        let no_retry = ProviderRetryPolicy {
            enabled: false,
            max_retries: 0,
            base_delay_ms: 0.0,
            max_retry_delay_ms: 0.0,
        };
        let error = compact(
            &preparation(),
            &model,
            "synthetic-test-key",
            None,
            None,
            None,
            runner,
            Some(&no_retry),
            Some((&provider_context, Some(&options))),
        )
        .await
        .expect_err("body failures must remain native errors");
        server.await.unwrap();
        let summary_call_count = summary_calls.load(Ordering::SeqCst);
        eprintln!("case=body_failure text_summary_calls={summary_call_count} error={error}");
        assert_ne!(error, "text fallback invoked");
        assert_eq!(summary_call_count, 0);
    }
}

// Stage 2-only helpers. They do not alter the stage 1 timing or persistence evidence.
fn native_model(base_url: String) -> Model {
    let mut model = Model::new(
        "gpt-6-astra",
        "Synthetic native compaction",
        "openai-responses",
        "openai",
        base_url,
    );
    model.context_window = 1_000_000.0;
    model.max_tokens = 32_000.0;
    model
}

fn zero_delay_retry(max_retries: u32) -> ProviderRetryPolicy {
    ProviderRetryPolicy {
        enabled: true,
        max_retries,
        base_delay_ms: 0.0,
        max_retry_delay_ms: 0.0,
    }
}

async fn compact_error_with_options(
    model: &Model,
    retry: &ProviderRetryPolicy,
    options: CompactionOptions,
    signal: Option<tokio_util::sync::CancellationToken>,
) -> (String, usize) {
    let summary_calls = Arc::new(AtomicUsize::new(0));
    let observed_summary_calls = summary_calls.clone();
    let runner: SummaryCallRunner = Arc::new(move |_| {
        observed_summary_calls.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { Err("text fallback invoked".to_string()) })
    });
    let provider_context = Context::default();
    let result = compact(
        &preparation(),
        model,
        "synthetic-test-key",
        None,
        signal.as_ref(),
        None,
        runner,
        Some(retry),
        Some((&provider_context, Some(&options))),
    )
    .await;
    let error =
        result.expect_err("a native error/cancellation must not return a compaction result");
    (error, summary_calls.load(Ordering::SeqCst))
}

async fn compact_error_only(
    model: &Model,
    retry: &ProviderRetryPolicy,
    timeout_ms: Option<f64>,
    signal: Option<tokio_util::sync::CancellationToken>,
) -> (String, usize) {
    let mut options = CompactionOptions::default();
    options.simple.stream.timeout_ms = timeout_ms;
    compact_error_with_options(model, retry, options, signal).await
}

#[tokio::test]
async fn native_status_retry_contract_has_exact_counts_and_zero_text_calls() {
    for (status, expected_requests) in [(401_u16, 1_usize), (408, 3), (429, 3), (500, 3)] {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let requests = Arc::new(AtomicUsize::new(0));
        let observed_requests = requests.clone();
        let server = tokio::spawn(async move {
            loop {
                let (mut socket, _) = listener.accept().await.unwrap();
                read_request(&mut socket).await;
                observed_requests.fetch_add(1, Ordering::SeqCst);
                socket
                    .write_all(
                        format!(
                            "HTTP/1.1 {status} Fixture\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{{}}"
                        )
                        .as_bytes(),
                    )
                    .await
                    .unwrap();
            }
        });
        let model = native_model(format!("http://{address}"));
        let retry = zero_delay_retry(2);
        let (error, text_calls) = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            compact_error_only(&model, &retry, None, None),
        )
        .await
        .expect("status fixture must not hang");
        server.abort();
        let _ = server.await;

        assert!(
            error.contains(&format!("HTTP {status}")),
            "unexpected status error: {error}"
        );
        assert_eq!(
            requests.load(Ordering::SeqCst),
            expected_requests,
            "status {status}"
        );
        assert_eq!(text_calls, 0, "status {status}");
        eprintln!("case=http_{status} native_requests={expected_requests} text_summary_calls=0 error={error}");
    }
}

#[tokio::test]
async fn native_delayed_body_timeout_is_native_error_without_text_summary() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        read_request(&mut socket).await;
        socket
            .write_all(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 1000\r\nConnection: close\r\n\r\n",
            )
            .await
            .unwrap();
        // Hold the body open without sleeping. The client's 25 ms request deadline
        // closes the socket and is the only delay that ends this read.
        let mut buffer = [0_u8; 1];
        let closed = socket.read(&mut buffer).await;
        assert!(
            matches!(closed, Ok(0) | Err(_)),
            "client must close the timed-out body"
        );
    });
    let model = native_model(format!("http://{address}"));
    let no_retry = ProviderRetryPolicy {
        enabled: false,
        max_retries: 0,
        base_delay_ms: 0.0,
        max_retry_delay_ms: 0.0,
    };
    let (error, text_calls) = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        compact_error_only(&model, &no_retry, Some(25.0), None),
    )
    .await
    .expect("body timeout escaped its external bound");
    server.await.unwrap();

    // reqwest 0.12.28 exposes the timeout through Response::bytes as this top-level
    // decode error; the pending body and 25 ms option prove the transport cause.
    assert_eq!(error, "error decoding response body");
    assert_eq!(text_calls, 0);
    eprintln!("case=body_timeout native_requests=1 text_summary_calls=0 error={error}");
}

#[tokio::test]
async fn native_pre_cancel_is_aborted_without_native_or_text_call() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let model = native_model(format!("http://{address}"));
    let cancel = tokio_util::sync::CancellationToken::new();
    cancel.cancel();
    let retry = zero_delay_retry(2);
    let (error, text_calls) = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        compact_error_only(&model, &retry, None, Some(cancel)),
    )
    .await
    .expect("pre-cancel must complete immediately");

    assert_eq!(error, "Aborted");
    assert_eq!(text_calls, 0);
    let listener = listener.into_std().unwrap();
    listener.set_nonblocking(true).unwrap();
    let connection = listener
        .accept()
        .expect_err("pre-cancel opened a native connection");
    assert_eq!(connection.kind(), std::io::ErrorKind::WouldBlock);
    eprintln!("case=pre_cancel native_requests=0 text_summary_calls=0 error={error}");
}

#[tokio::test]
async fn native_inflight_header_and_body_cancellation_is_aborted_without_text_call() {
    for phase in ["headers", "body"] {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let cancel = tokio_util::sync::CancellationToken::new();
        let server_cancel = cancel.clone();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            read_request(&mut socket).await;
            if phase == "body" {
                socket
                    .write_all(
                        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 1000\r\nConnection: close\r\n\r\n",
                    )
                    .await
                    .unwrap();
            }
            server_cancel.cancel();
        });
        let model = native_model(format!("http://{address}"));
        let retry = zero_delay_retry(2);
        let (error, text_calls) = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            compact_error_only(&model, &retry, None, Some(cancel)),
        )
        .await
        .unwrap_or_else(|_| panic!("in-flight {phase} cancellation hung"));
        server.await.unwrap();

        assert_eq!(error, "Aborted", "phase {phase}");
        assert_eq!(text_calls, 0, "phase {phase}");
        eprintln!("case=inflight_{phase} native_requests=1 text_summary_calls=0 error={error}");
    }
}

#[tokio::test]
async fn native_direct_unsupported_context_length_and_managed_contract() {
    let direct_cases = [
        (404_u16, "{}", true),
        (405, "{}", true),
        (501, "{}", true),
        (400, r#"{"error":{"code":"context_length_exceeded"}}"#, true),
        (413, "{}", true),
        (400, "{}", false),
    ];
    for (status, response_body, unsupported) in direct_cases {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let response_body = response_body.to_string();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            read_request(&mut socket).await;
            socket
                .write_all(
                    format!(
                        "HTTP/1.1 {status} Fixture\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{response_body}",
                        response_body.len(),
                    )
                    .as_bytes(),
                )
                .await
                .unwrap();
        });
        let model = native_model(format!("http://{address}"));
        let mut options = CompactionOptions::default();
        options.simple.stream.api_key = Some("synthetic-test-key".to_string());
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            pi_ai::stream::try_compact_simple(&model, &Context::default(), Some(&options)),
        )
        .await
        .expect("direct status fixture must not hang");
        server.await.unwrap();
        if unsupported {
            assert!(
                matches!(result, Ok(None)),
                "status {status} must remain native-unsupported"
            );
        } else {
            let error = result.expect_err("direct non-context-length 400 must stay an error");
            assert_eq!(error.message, "Server compaction failed (HTTP 400)");
        }
        eprintln!("case=direct_status_{status} unsupported={unsupported} text_layer_not_invoked");
    }

    let managed_cases = [
        (404_u16, "{}", true),
        (405, "{}", true),
        (501, "{}", true),
        (
            400,
            r#"{"error":{"code":"context_length_exceeded"}}"#,
            false,
        ),
        (413, "{}", false),
    ];
    for (status, body, unsupported) in managed_cases {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let body = body.to_string();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            read_request(&mut socket).await;
            socket
                .write_all(
                    format!(
                        "HTTP/1.1 {status} Fixture\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len(),
                    )
                    .as_bytes(),
                )
                .await
                .unwrap();
        });
        let base_url = format!("http://{address}/azure-openai/v1");
        let mut model = native_model(base_url.clone());
        model.provider = "azure-openai-managed".to_string();
        model.native_compaction = Some(pi_ai::types::NativeCompactionCapability {
            protocol: "openai-responses-compact-v1".to_string(),
            provider: "azure-openai-managed".to_string(),
            model: "gpt-6-astra".to_string(),
            endpoint: format!("{base_url}/responses/compact"),
            api_version: "v1".to_string(),
            enabled: true,
            validation: pi_ai::types::NativeCompactionValidation::LiveVerified,
        });
        let mut options = CompactionOptions::default();
        options.simple.stream.api_key = Some("synthetic-test-key".to_string());
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            pi_ai::stream::try_compact_simple(&model, &Context::default(), Some(&options)),
        )
        .await
        .expect("managed status fixture must not hang");
        server.await.unwrap();
        if unsupported {
            assert!(
                matches!(result, Ok(None)),
                "managed unsupported endpoint status {status}"
            );
            eprintln!("case=managed_status_{status} unsupported=true");
        } else {
            let error = result.expect_err(
                "validated managed context-length responses must not become unsupported",
            );
            assert_eq!(
                error.message,
                format!("Server compaction failed (HTTP {status})")
            );
            eprintln!(
                "case=managed_context_length_{status} native_error={}",
                error.message
            );
        }
    }
}

#[tokio::test]
async fn public_openai_compaction_wrappers_preserve_option_and_string_errors() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let requests = Arc::new(AtomicUsize::new(0));
    let observed_requests = requests.clone();
    let server = tokio::spawn(async move {
        loop {
            let (mut socket, _) = listener.accept().await.unwrap();
            read_request(&mut socket).await;
            observed_requests.fetch_add(1, Ordering::SeqCst);
            socket
                .write_all(
                    b"HTTP/1.1 429 Fixture\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}",
                )
                .await
                .unwrap();
        }
    });
    let model = native_model(format!("http://{address}"));
    let context = Context::default();
    let mut options = CompactionOptions::default();
    options.simple.stream.api_key = Some("synthetic-test-key".to_string());
    let legacy = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        pi_ai::providers::openai_responses::compact_openai_responses(
            &model,
            &context,
            Some(&options),
        ),
    )
    .await
    .expect("legacy compact wrapper must not hang");
    assert!(
        legacy.is_none(),
        "public Option wrapper must keep erasing Err to None"
    );
    let string_error = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        pi_ai::providers::openai_responses::try_compact_openai_responses(
            &model,
            &context,
            Some(&options),
        ),
    )
    .await
    .expect("legacy try wrapper must not hang")
    .expect_err("public try wrapper must retain its String error");
    server.abort();
    let _ = server.await;

    assert_eq!(string_error, "Server compaction failed (HTTP 429)");
    assert_eq!(requests.load(Ordering::SeqCst), 2);

    let mut codex_model = native_model("https://unused.invalid".to_string());
    codex_model.api = "openai-codex-responses".to_string();
    codex_model.provider = "openai-codex".to_string();
    options.simple.stream.api_key = Some("synthetic-malformed-jwt".to_string());
    let codex_legacy = pi_ai::providers::openai_codex_responses::compact_openai_codex_responses(
        &codex_model,
        &context,
        Some(&options),
    )
    .await;
    assert!(
        codex_legacy.is_none(),
        "Codex public Option wrapper must retain legacy behavior"
    );
    let codex_string_error =
        pi_ai::providers::openai_codex_responses::try_compact_openai_codex_responses(
            &codex_model,
            &context,
            Some(&options),
        )
        .await
        .expect_err("Codex malformed JWT must remain a public String error before network");
    assert_eq!(codex_string_error, "Failed to extract accountId from token");
}

#[tokio::test(flavor = "current_thread")]
async fn native_retry_delay_cancellation_is_aborted_without_second_request_or_text() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let requests = Arc::new(AtomicUsize::new(0));
    let observed_requests = requests.clone();
    let server = tokio::spawn(async move {
        loop {
            let (mut socket, _) = listener.accept().await.unwrap();
            read_request(&mut socket).await;
            observed_requests.fetch_add(1, Ordering::SeqCst);
            socket
                .write_all(
                    b"HTTP/1.1 429 Fixture\r\nContent-Type: application/json\r\nRetry-After-Ms: 60000\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}",
                )
                .await
                .unwrap();
        }
    });
    let response_seen = Arc::new(tokio::sync::Notify::new());
    let observed_response = response_seen.clone();
    let mut options = CompactionOptions::default();
    options.simple.stream.on_response = Some(Arc::new(move |response, _model| {
        let observed_response = observed_response.clone();
        Box::pin(async move {
            assert_eq!(response.status, 429);
            observed_response.notify_one();
        })
    }));
    let model = native_model(format!("http://{address}"));
    let retry = zero_delay_retry(1);
    let cancel = tokio_util::sync::CancellationToken::new();
    let mut attempt = Box::pin(compact_error_with_options(
        &model,
        &retry,
        options,
        Some(cancel.clone()),
    ));
    // A 429 does not read its response body. after on_response returns, production computes
    // Retry-After-Ms=60000 and polls the real delay. Biased polling drives that future first;
    // once its notification branch wins, the nonzero delay is pending, not the network or text.
    tokio::select! {
        biased;
        result = &mut attempt => panic!("native retry delay completed before cancellation: {result:?}"),
        _ = response_seen.notified() => {}
    }
    cancel.cancel();
    let (error, text_calls) = tokio::time::timeout(std::time::Duration::from_secs(2), attempt)
        .await
        .expect("retry-delay cancellation must abort promptly");
    server.abort();
    let _ = server.await;

    assert_eq!(error, "Aborted");
    assert_eq!(text_calls, 0);
    assert_eq!(
        requests.load(Ordering::SeqCst),
        1,
        "retry delay opened a second native request"
    );
    eprintln!("case=retry_delay_cancel native_requests=1 text_summary_calls=0 error={error}");
}
