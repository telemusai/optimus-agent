//! OCC-3 retained content-only capture lifecycle — red-first regression suite.
//!
//! Real Agent / isolated native family controller / `Occ3NativeSession` /
//! `SglangOcc3Connector` / OpenAI-completions provider path against a
//! synthetic loopback HTTP fixture (same style as `occ3_source_handoff.rs`).
//! Deterministic; no GPU, daemon, credential or network beyond 127.0.0.1.
//! No live-semantics claim.
//!
//! Covered invariants (PR #215 retained mode):
//! 1. A retained (`content_only_orchestrator_v1`) capture accepts a plain
//!    content turn with zero tool bindings and publishes exactly once through
//!    the authorized family controller with an `occ3.publish.v2` orchestrator
//!    body that carries no tool binding.
//! 2. Any tool call inside the retained captured turn is refused by the
//!    completed hook (`occ3_completed_mode_tool_forbidden`), nothing is
//!    published, and the session tombstones (no re-capture on a later prompt).
//! 3. An earlier unarmed content turn can never be claimed by a later armed
//!    tool-only turn.
//! 4. Legacy (`same_turn_tool_v1`) arming still requires exactly one
//!    `occ3_handoff` binding and its wire carries no `capture_mode` key.
//! 5. The retained publication claim is one-shot.
//! 6. Publication against an unauthorized recipient is refused before any
//!    publish request.
//! 7. The modes cannot cross: a legacy capture cannot be published through
//!    `publish_retained`.

use pi_agent_core::agent::{Agent, AgentOptions, PromptInput};
use pi_agent_core::types::{AgentState, ThinkingLevel};
use pi_ai::source_capture::{canonical_json_bytes, sha256, SourceCaptureProfile};
use pi_ai::types::{Model, API_OPENAI_COMPLETIONS};
use pi_coding_agent::core::agent_messages::*;
use pi_coding_agent::core::agent_session::occ3_state::{NativeSourceBudget, Occ3NativeSession};
use pi_coding_agent::core::occ::occ3::connector::Occ3Connector;
use pi_coding_agent::core::occ::occ3::types::Occ3ModelIdentity;
use pi_coding_agent::core::occ3_connectors::sglang::{
    NativeExperimentalPermit, SglangOcc3Config, SglangOcc3Connector,
};
use pi_coding_agent::modes::daemon::daemon_mode::occ3_native::{
    IsolatedNativeFamily, IsolatedNativeMember,
};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

const METHOD: &str = "statebridge.glm53.final-prenorm-predictor.continuous-prefix.v1";
const FACT: &str = "retained-7c31ab-source-only-nonce-orchid-quartz";
const HANDLE: &str = "private-source-handle";

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

fn identity() -> Occ3ModelIdentity {
    Occ3ModelIdentity {
        model_id: "fixture-model".into(),
        architecture: "fixture-arch".into(),
        checkpoint_revision: "fixture-revision".into(),
        weight_fingerprint: "metadata-fixture-not-weights".into(),
        tokenizer_fingerprint: "fixture-tokenizer".into(),
        chat_template_fingerprint: "fixture-template".into(),
        quantisation: "fixture-bf16".into(),
        tensor_geometry: "fixture-d8".into(),
        kv_layout: "fixture-tp8".into(),
        latent_method: METHOD.into(),
        runtime: "synthetic-local-fixture".into(),
    }
}

/// Which arm the scenario takes before the captured source turn.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Arm {
    /// `arm_source_retained`: v3 intent, content-only no-tool turn expected.
    Retained,
    /// `arm_source`: legacy v2 default intent, one `occ3_handoff` binding
    /// expected in the same captured turn.
    Legacy,
}

/// The turn shape the fixture serves for the captured (armed) request.
#[derive(Clone, Copy, PartialEq, Eq)]
enum CapturedTurn {
    /// Reasoning + content delta, `finish_reason: "stop"`, no tool calls.
    Content,
    /// Reasoning + one `occ3_handoff` tool call, `finish_reason: "tool_calls"`.
    Tool,
    /// Tool call only, no content run.
    ToolOnly,
}

/// One scenario for the shared offline runner.
#[derive(Clone)]
struct Scenario {
    arm: Arm,
    /// Serve an unarmed plain content turn before arming.
    turn1_unarmed: bool,
    captured_turn: CapturedTurn,
    /// `publish_retained(relationship, selector)` after the captured turn.
    publish: Option<(&'static str, &'static str)>,
    /// A second `publish_retained` on the same session.
    republish: Option<(&'static str, &'static str)>,
    /// A follow-up source prompt after the captured turn (tombstone probe).
    follow_up_prompt: bool,
    /// Parent the receiver catalog under an unrelated session.
    unrelated_receiver: bool,
}

impl Default for Scenario {
    fn default() -> Self {
        Self {
            arm: Arm::Retained,
            turn1_unarmed: false,
            captured_turn: CapturedTurn::Content,
            publish: Some(("child", "receiver")),
            republish: None,
            follow_up_prompt: false,
            unrelated_receiver: false,
        }
    }
}

/// Content-only SSE: reasoning + content, finish_reason "stop", NO tool calls.
fn sse_content(id: &str) -> String {
    let chunks = vec![
        json!({"id":id,"choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"retained native reasoning"},"finish_reason":null}]}),
        json!({"id":id,"choices":[{"index":0,"delta":{"content":"synthetic retained fixture completion"},"finish_reason":null}]}),
        json!({"id":id,"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}),
    ];
    let mut body = chunks
        .into_iter()
        .map(|v| format!("data: {v}\n\n"))
        .collect::<String>();
    body.push_str("data: [DONE]\n\n");
    body
}

/// Same-turn legacy handoff SSE: reasoning + one occ3_handoff tool call.
fn sse_tool(id: &str, content_run: bool) -> String {
    let mut chunks = vec![json!({"id":id,"choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"retained native reasoning"},"finish_reason":null}]})];
    if content_run {
        chunks.push(json!({"id":id,"choices":[{"index":0,"delta":{"content":"synthetic fixture completion"},"finish_reason":null}]}));
    }
    chunks.push(json!({"id":id,"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"actual-tool-1","type":"function","function":{"name":"occ3_handoff","arguments":"{\"receiver_role\":\"child\",\"receiver_name\":\"receiver\"}"}}]},"finish_reason":null}]}));
    chunks.push(json!({"id":id,"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}));
    let mut body = chunks
        .into_iter()
        .map(|v| format!("data: {v}\n\n"))
        .collect::<String>();
    body.push_str("data: [DONE]\n\n");
    body
}

async fn read_request(socket: &mut tokio::net::TcpStream) -> (String, Value) {
    let mut bytes = Vec::new();
    let end;
    loop {
        let mut part = [0u8; 4096];
        let n = socket.read(&mut part).await.unwrap();
        assert!(n > 0);
        bytes.extend_from_slice(&part[..n]);
        if let Some(i) = bytes.windows(4).position(|v| v == b"\r\n\r\n") {
            end = i + 4;
            break;
        }
    }
    let headers = String::from_utf8(bytes[..end].to_vec()).unwrap();
    assert!(headers
        .to_lowercase()
        .contains("authorization: bearer synthetic-fixture-key"));
    let size = headers
        .lines()
        .find_map(|l| {
            l.to_ascii_lowercase()
                .strip_prefix("content-length: ")
                .map(|v| v.trim().parse::<usize>().unwrap())
        })
        .unwrap();
    while bytes.len() < end + size {
        let mut p = [0u8; 4096];
        let n = socket.read(&mut p).await.unwrap();
        assert!(n > 0);
        bytes.extend_from_slice(&p[..n]);
    }
    (
        headers
            .lines()
            .next()
            .unwrap()
            .split_whitespace()
            .nth(1)
            .unwrap()
            .into(),
        serde_json::from_slice(&bytes[end..end + size]).unwrap(),
    )
}

async fn send(socket: &mut tokio::net::TcpStream, body: String, extra: Vec<(&str, String)>) {
    let mut headers = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n",
        body.len()
    );
    for (k, v) in extra {
        headers.push_str(&format!("{k}: {v}\r\n"));
    }
    headers.push_str("\r\n");
    socket.write_all(headers.as_bytes()).await.unwrap();
    socket.write_all(body.as_bytes()).await.unwrap();
    socket.shutdown().await.unwrap();
}

#[derive(Default)]
struct FixtureRecords {
    requests: Vec<(String, Value)>,
    /// Every `occ3_capture` intent body the fixture received, in order.
    source_intents: Vec<Value>,
    /// The single authenticated receipt the fixture served for the capture.
    source_receipt: Option<Value>,
    /// Every `/occ3/v1/native/publish` body, in order.
    publish_bodies: Vec<Value>,
    /// The last accepted publication reply.
    publication: Option<Value>,
}

struct FixtureHandles {
    records: Arc<Mutex<FixtureRecords>>,
    endpoint: String,
    server: tokio::task::JoinHandle<()>,
}

async fn serve(scenario: Scenario) -> FixtureHandles {
    let records = Arc::new(Mutex::new(FixtureRecords::default()));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let records_outer = records.clone();
    let task = tokio::spawn(async move {
        let records = records_outer.clone();
        loop {
            let (mut socket, _) = listener.accept().await.unwrap();
            let records = records.clone();
            let scenario = scenario.clone();
            tokio::spawn(async move {
                let (path, body) = read_request(&mut socket).await;
                {
                    let mut records = records.lock().unwrap();
                    records.requests.push((path.clone(), body.clone()));
                }
                let mut headers = Vec::new();
                let result = match path.as_str() {
                    "/occ3/v1/native/sessions/enroll" => json!({
                        "schema":"occ3.sessions.enrolled.v1","operation_id":body["operation_id"],
                        "session_id":body["session_id"],"principal_digest":"a".repeat(64),
                        "runtime_id":body["runtime_id"],"runtime_generation":body["runtime_generation"],
                        "model_fingerprint":body["model_fingerprint"],"expires_at_epoch":body["expires_at_epoch"],
                        "readiness":{"runtime_epoch":body["runtime_id"],"runtime_generation":body["runtime_generation"],
                            "model_fingerprint":body["model_fingerprint"],"method_id":METHOD,
                            "expected_ranks":8,"ready_ranks":8,"tp_layout_digest":"b".repeat(64),
                            "hidden_dim":8,"layer_index":3,"dtype":"bfloat16","profile_digest":"c".repeat(64),
                            "source_capture":true,"continuous_prefix":true,"terminal_journal":true}
                    }),
                    "/occ3/v1/native/publish" => {
                        let source = {
                            let records = records.lock().unwrap();
                            records.source_intents.last().cloned().unwrap()
                        };
                        assert_eq!(body["source_session_id"], source["source_session_id"]);
                        assert_eq!(body["source_turn_nonce"], source["turn_nonce"]);
                        if body.get("tool_call_id").is_some() {
                            // Legacy same-turn tool dispatch.
                            assert_eq!(body["schema"], "occ3.publish.v1");
                            assert_eq!(body["tool_call_id"], "actual-tool-1");
                        } else {
                            // Retained orchestrator dispatch: v2 body, no tool
                            // binding, the dispatch mode names the authority.
                            assert_eq!(body["schema"], "occ3.publish.v2");
                            assert_eq!(body["dispatch_mode"], "orchestrator_v1");
                        }
                        let mut binding = body.clone();
                        binding.as_object_mut().unwrap().remove("schema");
                        binding["principal_digest"] = json!("a".repeat(64));
                        let reply = json!({
                            "schema":"occ3.publication.v1","publication_id":body["publication_id"],
                            "binding_digest":sha256(&canonical_json_bytes(&binding).unwrap()),
                            "state":"published","source_handle":body["source_handle"],
                            "expires_at_epoch":body["expires_at_epoch"]
                        });
                        {
                            let mut records = records.lock().unwrap();
                            records.publication = Some(reply.clone());
                            records.publish_bodies.push(body.clone());
                        }
                        reply
                    },
                    "/occ3/v1/native/publication/status" => {
                        records.lock().unwrap().publication.clone().unwrap_or_else(|| json!({
                            "schema":"occ3.publication.v1","publication_id":body["publication_id"],"state":"absent"
                        }))
                    },
                    "/occ3/v1/native/retire" => json!({
                        "schema":"occ3.retired.v1","operation_id":body["operation_id"],
                        "source_handle":body["source_handle"],"revoked":true,"gpu_freed":false,"state":"retired"
                    }),
                    "/v1/chat/completions" => {
                        assert_eq!(
                            body.get("max_tokens").or_else(|| body.get("max_completion_tokens")),
                            Some(&json!(131072)),
                            "the real native default budget is preserved"
                        );
                        if let Some(intent) = body.get("occ3_capture") {
                            // The captured (armed) source turn. The receipt
                            // shape follows the intent the client serialized:
                            // v1 intent -> bare v1 receipt, v2/v3 intent ->
                            // occ3.source.v2 with the full span block bound to
                            // the intent's own span policy.
                            let output = match scenario.captured_turn {
                                CapturedTurn::Content => sse_content("native-source-rid"),
                                CapturedTurn::Tool => sse_tool("native-source-rid", true),
                                CapturedTurn::ToolOnly => sse_tool("native-source-rid", false),
                            };
                            let v1 = intent["schema"] == "occ3.capture.v1";
                            let mut receipt = json!({
                                "schema": if v1 {"occ3.source.v1"} else {"occ3.source.v2"},
                                "intent_digest": sha256(&canonical_json_bytes(intent).unwrap()),
                                "source_handle":HANDLE,
                                "native_rid":"native-source-rid","response_id":"native-source-rid",
                                "capture_commit_id":"synthetic-retained-capture-commit",
                                "accepted_span_start":0,"accepted_span_end":8,"retained_count":8,
                                "predictor_position_offset":-1,"representation":"final_layer_prenorm_predictor",
                                "layer_index":3,"hidden_dim":8,"dtype":"bfloat16",
                                "layout_digest":"b".repeat(64),"ready_ranks":8,"expected_ranks":8,
                                "accepted_sequence_digest":"d".repeat(64),
                                "output_sha256":sha256(output.as_bytes()),
                                "expires_at_epoch":intent["expires_at_epoch"],"byte_count":128
                            });
                            if !v1 {
                                // Counts always sum to the window width
                                // (span_contiguous), retained_count <= sum.
                                // The no-tool policy must attest zero tool
                                // rows regardless of what the turn emitted:
                                // a tool-bearing turn under a no-tool intent
                                // is the client-side refusal under test.
                                let no_tool = intent["span_policy"] == "content_tail_no_tool_v2";
                                let (thinking, content, tool_calls, tool_args) =
                                    if no_tool {
                                        (2, 6, 0, 0)
                                    } else {
                                        (2, 2, 1, 3)
                                    };
                                let span = json!({
                                    "span_schema":"occ3.span.v1","span_policy":intent["span_policy"],
                                    "span_marker_profile":"glm53","span_marker_schema":"occ3.span.markers.v1",
                                    "span_reasoning_terminator":"thinking_end","span_turn_end_attested":true,
                                    "span_thinking_count":thinking,"span_content_count":content,
                                    "span_tool_call_count":tool_calls,"span_tool_arguments_count":tool_args,
                                    "span_control_count":0,"span_unknown_count":0,
                                    "span_window_start_position":0,"span_window_end_position":8,
                                    "span_contiguous":true,"span_contains_final_content":true,
                                    "span_contains_tool_call":!no_tool,"span_message_truncated":false
                                });
                                for (k, v) in span.as_object().unwrap() {
                                    receipt[k.clone()] = v.clone();
                                }
                            }
                            {
                                let mut records = records.lock().unwrap();
                                records.source_intents.push(intent.clone());
                                records.source_receipt = Some(receipt.clone());
                            }
                            headers.push(("x-optimus-occ3-source", receipt.to_string()));
                            send(&mut socket, output, headers).await;
                            return;
                        }
                        if let Some(attachment) = body.get("occ3") {
                            // The receiver's actual attached request: the
                            // private fact never reaches the wire and the
                            // served receipt proves consumption of the exact
                            // attached state.
                            assert!(!body["messages"].to_string().contains(FACT));
                            assert_eq!(attachment["method"], METHOD);
                            headers.push(("x-optimus-occ3-receipt", json!({
                                "attachment":attachment,"consumed":true,"injected_state":true,"fallback_used":false
                            }).to_string()));
                        }
                        send(&mut socket, sse_content("native-completion-rid"), headers).await;
                        return;
                    },
                    _ => panic!("Unexpected fixture route: {path}"),
                };
                send(&mut socket, result.to_string(), headers).await;
            });
        }
    });
    FixtureHandles {
        records,
        endpoint,
        server: task,
    }
}

fn native_options(model: Model) -> AgentOptions {
    AgentOptions {
        initial_state: Some(AgentState {
            model,
            thinking_level: ThinkingLevel::Max,
            system_prompt: "Retained fixture. Plain content turn.".into(),
            ..Default::default()
        }),
        get_api_key: Some(Arc::new(|_| {
            Box::pin(async { Some("synthetic-fixture-key".into()) })
        })),
        ..Default::default()
    }
}

/// Everything the tests observe from one scenario run.
struct RetainedResult {
    requests: Vec<(String, Value)>,
    source_intents: Vec<Value>,
    source_receipt: Option<Value>,
    publish_bodies: Vec<Value>,
    /// Source agent error state after the captured turn.
    sender_error: Option<String>,
    /// Source agent error state after the optional follow-up prompt.
    follow_up_error: Option<String>,
    receiver_context: String,
    audit_before_publish: Value,
    source_audit: Value,
    receiver_audit: Value,
    admitted: usize,
    publish_outcome: Option<Result<AgentSessionMessageReceipt, String>>,
    republish_outcome: Option<Result<AgentSessionMessageReceipt, String>>,
}

impl RetainedResult {
    fn model_requests(&self) -> usize {
        self.requests
            .iter()
            .filter(|(p, _)| p == "/v1/chat/completions")
            .count()
    }
    fn capture_requests(&self) -> usize {
        self.requests
            .iter()
            .filter(|(_, b)| b.get("occ3_capture").is_some())
            .count()
    }
    fn receiver_requests(&self) -> usize {
        self.requests
            .iter()
            .filter(|(_, b)| b.get("occ3").is_some())
            .count()
    }
}

async fn run_retained_fixture(scenario: Scenario) -> Result<RetainedResult, String> {
    let handles = serve(scenario.clone()).await;
    let mut model = Model::new(
        "fixture-model",
        "fixture",
        API_OPENAI_COMPLETIONS,
        "native-fixture",
        format!("{}/v1", handles.endpoint),
    );
    model.max_tokens = 262144.;
    model.default_max_tokens = Some(131072.);
    model.context_window = 1048576.;
    model.reasoning = true;
    let permit = NativeExperimentalPermit {
        attempt_id: "offline-attempt".into(),
        identity: identity(),
        target: pi_ai::inference_attachment::InferenceStateTarget {
            provider: model.provider.clone(),
            model_id: model.id.clone(),
            base_url: model.base_url.clone(),
        },
        runtime_id: "runtime-epoch-1".into(),
        runtime_generation: 1,
        access_domain: "isolated-family".into(),
        expires_at_epoch: now() + 240,
        source_profile: SourceCaptureProfile {
            layer_index: 3,
            hidden_dim: 8,
            dtype: "bfloat16".into(),
            layout_digest: "b".repeat(64),
            expected_ranks: 8,
        },
        profile_digest: "c".repeat(64),
        max_capture_bytes: 1048576,
    };
    let connector = Arc::new(
        SglangOcc3Connector::new_experimental(
            SglangOcc3Config {
                endpoint: url::Url::parse(&handles.endpoint).unwrap(),
                api_key: Some("synthetic-fixture-key".into()),
                control_headers: None,
            },
            permit,
        )
        .map_err(|_| "Connector invalid")?,
    );
    let source = Occ3NativeSession::enroll("source-session", &model, connector.clone()).await?;
    let receiver = Occ3NativeSession::enroll("receiver-session", &model, connector.clone()).await?;
    let source_agent = Agent::new(native_options(model.clone()));
    let receiver_agent = Agent::new(native_options(model));
    let family = IsolatedNativeFamily::new(vec![
        IsolatedNativeMember {
            catalog: AgentFamilyCatalogEntry {
                id: "source-session".into(),
                name: Some("source".into()),
                depth: 0.,
                ..Default::default()
            },
            endpoint: AgentSessionMessageEndpoint {
                active_session_id: "source-active".into(),
                session_id: "source-session".into(),
                session_name: Some("source".into()),
                ..Default::default()
            },
            agent: source_agent.clone(),
            native: source.clone(),
        },
        IsolatedNativeMember {
            catalog: AgentFamilyCatalogEntry {
                id: "receiver-session".into(),
                name: Some("receiver".into()),
                depth: 1.,
                parent_session_id: Some(
                    if scenario.unrelated_receiver {
                        "other-parent"
                    } else {
                        "source-session"
                    }
                    .into(),
                ),
                ..Default::default()
            },
            endpoint: AgentSessionMessageEndpoint {
                active_session_id: "receiver-active".into(),
                session_id: "receiver-session".into(),
                session_name: Some("receiver".into()),
                ..Default::default()
            },
            agent: receiver_agent.clone(),
            native: receiver.clone(),
        },
    ])?;
    family.install()?;

    if scenario.turn1_unarmed {
        source_agent
            .prompt(PromptInput::Text {
                input: format!("Private source facts: {FACT}. Reply with one short sentence."),
                images: Vec::new(),
            })
            .await
            .map_err(|_| "Turn 1 prompt failed")?;
    }
    let budget = NativeSourceBudget {
        retained_count_limit: 128,
        max_capture_bytes: 1048576,
        deadline: Instant::now() + Duration::from_secs(20),
        max_response_bytes: 65536,
    };
    match scenario.arm {
        Arm::Retained => source.arm_source_retained(budget)?,
        Arm::Legacy => source.arm_source(budget)?,
    }
    source_agent
        .prompt(PromptInput::Text {
            input: format!("Private source facts: {FACT}. Complete the source turn."),
            images: Vec::new(),
        })
        .await
        .map_err(|_| "Captured turn prompt failed")?;
    let sender_error = source_agent.state().error_message;
    let audit_before_publish = source.audit();

    let controller = family.controller("source-session")?;
    let publish_outcome = match scenario.publish {
        Some((relationship, selector)) => {
            Some(source.publish_retained(controller.clone(), relationship, selector).await)
        }
        None => None,
    };
    let republish_outcome = match scenario.republish {
        Some((relationship, selector)) => {
            Some(source.publish_retained(controller, relationship, selector).await)
        }
        None => None,
    };
    let follow_up_error = if scenario.follow_up_prompt {
        source_agent
            .prompt(PromptInput::Text {
                input: "Retry the same failed source turn without a new grant.".into(),
                images: Vec::new(),
            })
            .await
            .map_err(|_| "Follow-up prompt failed")?;
        source_agent.state().error_message
    } else {
        None
    };

    let receiver_context = serde_json::to_string(&receiver_agent.state().messages).unwrap();
    assert!(!receiver_context.contains(FACT));
    assert!(!receiver_context.contains(HANDLE));
    let source_audit = source.audit();
    let receiver_audit = receiver.audit();
    let admitted = family.receipts().len();
    let _ = family.close().await;
    let _ = connector.shutdown().await;
    handles.server.abort();
    let records = handles.records.lock().unwrap();
    Ok(RetainedResult {
        requests: records.requests.clone(),
        source_intents: records.source_intents.clone(),
        source_receipt: records.source_receipt.clone(),
        publish_bodies: records.publish_bodies.clone(),
        sender_error,
        follow_up_error,
        receiver_context,
        audit_before_publish,
        source_audit,
        receiver_audit,
        admitted,
        publish_outcome,
        republish_outcome,
    })
}

fn budget_error(result: &RetainedResult) -> &str {
    result
        .sender_error
        .as_deref()
        .unwrap_or("NO OBSERVED SOURCE ERROR")
}

#[tokio::test]
async fn retained_capture_accepts_content_only_turn() {
    let result = run_retained_fixture(Scenario::default()).await.unwrap();
    // The captured turn completed without error and captured zero tool
    // bindings: the completed hook refuses any binding in retained mode, so
    // a clean turn is itself the zero-binding observation.
    assert!(result.sender_error.is_none(), "{}", budget_error(&result));
    let before = &result.audit_before_publish["source"];
    assert_eq!(before["capture_mode"], "content_only_orchestrator_v1");
    assert_eq!(before["state"], "Captured");
    assert_eq!(before["retained"], true);
    assert_eq!(before["tool_claimed"], false, "no tool binding was claimed");
    assert_eq!(before["publication_claimed"], false);
    // The armed intent the fixture received negotiated the retained mode.
    assert_eq!(result.capture_requests(), 1);
    let intent = &result.source_intents[0];
    assert_eq!(intent["schema"], "occ3.capture.v3");
    assert_eq!(intent["capture_mode"], "content_only_orchestrator_v1");
    assert_eq!(intent["span_policy"], "content_tail_no_tool_v2");
    // Authorized orchestrator publication: one v2 publish, dispatch mode
    // named, no tool binding on the wire.
    let receipt = result
        .publish_outcome
        .as_ref()
        .expect("publish_retained succeeded")
        .as_ref()
        .expect("publish_retained result");
    assert_eq!(receipt.target.session_id, "receiver-session");
    assert!(matches!(receipt.delivery_status.as_str(), "delivered" | "queued"));
    assert_eq!(result.publish_bodies.len(), 1);
    let publish = &result.publish_bodies[0];
    // The connector accepts an Orchestrator dispatch only when the lease
    // holds ZERO tool bindings, so this accepted v2 publish body is the
    // observed proof of the empty binding set next to tool_claimed=false.
    assert_eq!(publish["schema"], "occ3.publish.v2");
    assert_eq!(publish["dispatch_mode"], "orchestrator_v1");
    assert!(publish.get("tool_call_id").is_none(), "no tool binding key");
    assert_eq!(publish["source_session_id"], "source-session");
    assert_eq!(publish["receiver_session_id"], "receiver-session");
    assert_eq!(publish["source_turn_nonce"], intent["turn_nonce"]);
    // The receiver consumed the attached state: one attached request, a
    // delivered receipt, a Consumed lease.
    assert_eq!(result.receiver_requests(), 1);
    assert_eq!(result.admitted, 1);
    assert_eq!(result.receiver_audit["receiver"][0]["lease_state"], "Consumed");
    assert_eq!(
        result.source_audit["source"]["publication_claimed"],
        true,
        "one-shot publication claim recorded"
    );
    assert_eq!(result.source_audit["source"]["handoff_admitted"], true);
    // The receiver plaintext/context carries neither the private fact nor
    // the source handle.
    assert!(!result.receiver_context.contains(FACT));
    assert!(!result.receiver_context.contains(HANDLE));
    // Exactly one source request: the retained content turn ends the run.
    assert_eq!(result.model_requests(), 2, "one source turn + one receiver turn");
    // The receipt the fixture served is the v2 wire with the no-tool span
    // block: span counts sum to the contiguous window width, zero tool rows.
    let served = result.source_receipt.as_ref().expect("fixture served one receipt");
    assert_eq!(served["schema"], "occ3.source.v2");
    assert_eq!(served["span_schema"], "occ3.span.v1");
    assert_eq!(served["span_policy"], "content_tail_no_tool_v2");
    assert_eq!(served["span_marker_schema"], "occ3.span.markers.v1");
    assert_eq!(served["span_tool_call_count"], 0);
    assert_eq!(served["span_tool_arguments_count"], 0);
    assert_eq!(served["span_thinking_count"], 2);
    assert_eq!(served["span_content_count"], 6);
    assert_eq!(
        served["span_thinking_count"].as_u64().unwrap()
            + served["span_content_count"].as_u64().unwrap()
            + served["span_tool_call_count"].as_u64().unwrap()
            + served["span_tool_arguments_count"].as_u64().unwrap()
            + served["span_control_count"].as_u64().unwrap()
            + served["span_unknown_count"].as_u64().unwrap(),
        served["span_window_end_position"].as_u64().unwrap()
            - served["span_window_start_position"].as_u64().unwrap(),
        "counts sum to the contiguous window width"
    );
    assert_eq!(served["span_contains_tool_call"], false);
    assert_eq!(served["retained_count"], 8);
    assert_eq!(served["intent_digest"], sha256(&canonical_json_bytes(&result.source_intents[0]).unwrap()));
}

#[tokio::test]
async fn retained_capture_rejects_unexpected_tool_output() {
    let result = run_retained_fixture(Scenario {
        captured_turn: CapturedTurn::Tool,
        publish: Some(("child", "receiver")),
        follow_up_prompt: true,
        ..Default::default()
    })
    .await
    .unwrap();
    // The turn ends in a sanitized failure naming the forbidden mode check.
    assert!(result.sender_error.is_some());
    assert!(
        result.sender_error.as_deref().unwrap().contains("occ3_completed_mode_tool_forbidden"),
        "observed: {:?}",
        result.sender_error
    );
    // No publication request was sent and the receiver got nothing.
    assert!(result.publish_bodies.is_empty(), "no publish body hit the fixture");
    assert!(result.publish_outcome.is_some());
    assert!(result.publish_outcome.as_ref().unwrap().is_err());
    assert_eq!(result.receiver_requests(), 0);
    assert_eq!(result.admitted, 0);
    assert_eq!(result.receiver_audit["receiver"].as_array().map(Vec::len), Some(0));
    // The session tombstones: a follow-up source prompt produces no new
    // occ3_capture request and no model request at all.
    assert!(result.follow_up_error.is_some(), "the retry prompt was refused");
    assert_eq!(result.capture_requests(), 1);
    assert_eq!(result.model_requests(), 1, "no post-failure continuation or retry");
}

#[tokio::test]
async fn unarmed_content_then_armed_tool_only_cannot_claim_earlier_content() {
    let result = run_retained_fixture(Scenario {
        arm: Arm::Retained,
        turn1_unarmed: true,
        captured_turn: CapturedTurn::ToolOnly,
        publish: Some(("child", "receiver")),
        ..Default::default()
    })
    .await
    .unwrap();
    // Turn 1 was plain content with no capture; only the armed turn 2 asked
    // for a capture, and it was a tool-only turn that the retained mode
    // refuses.
    assert!(result.sender_error.is_some());
    assert!(
        result
            .sender_error
            .as_deref()
            .unwrap()
            .contains("occ3_completed_mode_tool_forbidden"),
        "observed: {:?}",
        result.sender_error
    );
    assert_eq!(result.capture_requests(), 1, "only the armed turn carried occ3_capture");
    let intent = &result.source_intents[0];
    assert_eq!(intent["schema"], "occ3.capture.v3");
    assert!(result.publish_bodies.is_empty());
    assert!(result.publish_outcome.as_ref().unwrap().is_err());
    assert_eq!(result.receiver_requests(), 0);
    assert_eq!(result.admitted, 0);
    // Turn 1's content never became a captured lease: no CapturedRetained
    // lease exists, nothing was published or admitted, and the receiver
    // context stayed clean.
    assert_ne!(result.source_audit["source"]["state"], "Captured");
    assert_eq!(result.source_audit["source"]["publication_claimed"], false);
    assert_eq!(result.source_audit["source"]["handoff_admitted"], false);
    assert!(!result.receiver_context.contains(FACT));
    assert!(!result.receiver_context.contains(HANDLE));
    // Turn 1 (unarmed content) + turn 2 (armed tool-only) and nothing else.
    assert_eq!(result.model_requests(), 2);
}

#[tokio::test]
async fn legacy_mode_still_requires_exactly_one_binding() {
    // Negative: a legacy arm with a zero-tool captured turn fails the
    // completed hook and never publishes.
    let refused = run_retained_fixture(Scenario {
        arm: Arm::Legacy,
        captured_turn: CapturedTurn::Content,
        publish: Some(("child", "receiver")),
        ..Default::default()
    })
    .await
    .unwrap();
    assert!(refused.sender_error.is_some());
    assert!(
        refused
            .sender_error
            .as_deref()
            .unwrap()
            .contains("occ3_completed_binding_count"),
        "observed: {:?}",
        refused.sender_error
    );
    assert!(refused.publish_bodies.is_empty());
    assert!(refused.publish_outcome.as_ref().unwrap().is_err());
    assert_eq!(refused.receiver_requests(), 0);
    assert_eq!(refused.admitted, 0);
    // The legacy intent wire carries no capture_mode key.
    assert_eq!(refused.capture_requests(), 1);
    let intent = &refused.source_intents[0];
    // Legacy arms negotiate the v1/v2 family (the operator span-policy knob
    // decides which) but never the retained v3 mode.
    assert!(
        intent["schema"] == "occ3.capture.v2" || intent["schema"] == "occ3.capture.v1",
        "observed legacy schema: {}",
        intent["schema"]
    );
    assert!(intent.get("capture_mode").is_none(), "legacy wire has no capture_mode");

    // Positive: the existing legacy same-turn tool path is unchanged — one
    // occ3_handoff binding, model-tool publication, receiver consumption.
    let positive = run_retained_fixture(Scenario {
        arm: Arm::Legacy,
        captured_turn: CapturedTurn::Tool,
        publish: None,
        ..Default::default()
    })
    .await
    .unwrap();
    assert!(positive.sender_error.is_none(), "{:?}", positive.sender_error);
    assert_eq!(positive.publish_bodies.len(), 1);
    assert_eq!(positive.publish_bodies[0]["schema"], "occ3.publish.v1");
    assert_eq!(positive.publish_bodies[0]["tool_call_id"], "actual-tool-1");
    assert_eq!(positive.receiver_requests(), 1);
    assert_eq!(positive.admitted, 1);
    assert_eq!(positive.receiver_audit["receiver"][0]["lease_state"], "Consumed");
    assert_eq!(positive.audit_before_publish["source"]["capture_mode"], "same_turn_tool_v1");
    assert!(positive.source_intents[0].get("capture_mode").is_none());
    assert!(!positive.receiver_context.contains(FACT));
    assert!(!positive.receiver_context.contains(HANDLE));
}

#[tokio::test]
async fn earlier_capture_cannot_be_republished() {
    let result = run_retained_fixture(Scenario {
        republish: Some(("child", "receiver")),
        ..Default::default()
    })
    .await
    .unwrap();
    // The first publication succeeded; the second is refused fast by the
    // one-shot claim and exactly one v2 publish body reached the fixture.
    assert!(result.publish_outcome.as_ref().unwrap().is_ok());
    let second = result
        .republish_outcome
        .as_ref()
        .expect("second publish attempted")
        .as_ref();
    assert!(second.is_err());
    assert!(
        second
            .as_ref()
            .unwrap_err()
            .contains("occ3_publish_already_claimed"),
        "observed: {second:?}"
    );
    assert_eq!(result.publish_bodies.len(), 1, "exactly one publish body");
    assert_eq!(result.publish_bodies[0]["schema"], "occ3.publish.v2");
    // The successful first capture still negotiated the retained mode.
    assert_eq!(result.capture_requests(), 1);
    assert_eq!(result.source_intents[0]["schema"], "occ3.capture.v3");
    assert_eq!(result.source_intents[0]["capture_mode"], "content_only_orchestrator_v1");
    assert_eq!(result.source_intents[0]["span_policy"], "content_tail_no_tool_v2");
    assert!(result.sender_error.is_none(), "{:?}", result.sender_error);
    assert_eq!(result.receiver_requests(), 1, "the receiver consumed exactly once");
    assert_eq!(result.admitted, 1);
    assert_eq!(result.source_audit["source"]["publication_claimed"], true);
}

#[tokio::test]
async fn publish_retained_wrong_recipient_refused() {
    // A selector that is not in the family catalog at all.
    let nonexistent = run_retained_fixture(Scenario {
        publish: Some(("child", "nonexistent-receiver")),
        ..Default::default()
    })
    .await
    .unwrap();
    let outcome = nonexistent.publish_outcome.as_ref().unwrap();
    assert!(outcome.is_err());
    assert!(
        outcome
            .as_ref()
            .unwrap_err()
            .contains("occ3_publish_authorize"),
        "observed: {outcome:?}"
    );
    assert!(
        nonexistent.publish_bodies.is_empty(),
        "no publish request was sent"
    );
    assert_eq!(nonexistent.receiver_requests(), 0);
    assert_eq!(nonexistent.admitted, 0);
    // The capture itself succeeded under the retained mode before the
    // recipient check refused the publication.
    assert!(nonexistent.sender_error.is_none(), "{:?}", nonexistent.sender_error);
    assert_eq!(nonexistent.source_intents[0]["schema"], "occ3.capture.v3");
    assert_eq!(
        nonexistent.source_intents[0]["capture_mode"],
        "content_only_orchestrator_v1"
    );
    assert_eq!(nonexistent.source_intents[0]["span_policy"], "content_tail_no_tool_v2");

    // A real session that is not reachable under the claimed relationship.
    let unrelated = run_retained_fixture(Scenario {
        publish: Some(("child", "receiver")),
        unrelated_receiver: true,
        ..Default::default()
    })
    .await
    .unwrap();
    let outcome = unrelated.publish_outcome.as_ref().unwrap();
    assert!(outcome.is_err());
    assert!(
        outcome
            .as_ref()
            .unwrap_err()
            .contains("occ3_publish_authorize"),
        "observed: {outcome:?}"
    );
    assert!(unrelated.publish_bodies.is_empty());
    assert_eq!(unrelated.receiver_requests(), 0);
    assert_eq!(unrelated.admitted, 0);
}

#[tokio::test]
async fn mode_crossing_refused() {
    // A captured legacy turn (one occ3_handoff binding, published through the
    // model tool by the existing path) can never be re-published through the
    // retained orchestrator dispatch.
    let result = run_retained_fixture(Scenario {
        arm: Arm::Legacy,
        captured_turn: CapturedTurn::Tool,
        publish: Some(("child", "receiver")),
        ..Default::default()
    })
    .await
    .unwrap();
    let outcome = result.publish_outcome.as_ref().unwrap();
    assert!(outcome.is_err());
    assert!(
        outcome
            .as_ref()
            .unwrap_err()
            .contains("occ3_publish_mode_not_retained"),
        "observed: {outcome:?}"
    );
    // The only publish body the fixture saw is the legacy v1 tool dispatch.
    assert_eq!(result.publish_bodies.len(), 1);
    assert_eq!(result.publish_bodies[0]["schema"], "occ3.publish.v1");
    assert_eq!(result.publish_bodies[0]["tool_call_id"], "actual-tool-1");
    // The captured legacy intent stayed on the legacy wire: no capture_mode
    // key, no v3 negotiation.
    assert!(
        result.source_intents[0]["schema"] == "occ3.capture.v2"
            || result.source_intents[0]["schema"] == "occ3.capture.v1",
        "observed legacy schema: {}",
        result.source_intents[0]["schema"]
    );
    assert!(result.source_intents[0].get("capture_mode").is_none());
    assert!(
        result
            .publish_bodies
            .iter()
            .all(|body| body.get("dispatch_mode").is_none()),
        "no orchestrator v2 body was ever sent"
    );
}
