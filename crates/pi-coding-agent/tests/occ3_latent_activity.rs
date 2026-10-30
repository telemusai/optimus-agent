//! OCC-3 latent activity: one authenticated delivery path from a completed native
//! receiver consumption to the existing content-free communication projection.
//!
//! The fixture is local and synthetic: a TCP server answers the sglang connector
//! control plane and the two model requests, so the receiver lease reaches
//! `Consumed` only through a verified `x-optimus-occ3-receipt` header. No GPU, no
//! provider account, no semantic evidence.
use indexmap::IndexMap;
use pi_agent_core::agent::{Agent, AgentOptions, PromptInput};
use pi_agent_core::types::{AgentMessage, AgentState, ThinkingLevel};
use pi_ai::source_capture::{canonical_json_bytes, sha256, SourceCaptureProfile};
use pi_ai::types::{Model, API_OPENAI_COMPLETIONS};
use pi_coding_agent::core::agent_messages::{
    create_agent_session_message_delivery, stamp_occ3_delivery_record, AgentSessionMessageEndpoint,
    AgentSessionMessagePayload, AgentSessionMessageSender, AgentFamilyCatalogEntry, Occ2StrikeLedger,
    Occ3DeliveryDetails, AGENT_MESSAGE_SOURCE, OCC3_DELIVERY_LATENT_NATIVE, OCC3_DELIVERY_PENDING,
};
use pi_coding_agent::core::agent_session::occ3_state::{NativeSourceBudget, Occ3NativeSession};
use pi_coding_agent::core::occ::activity::{from_admitted_messages, CommunicationKind};
use pi_coding_agent::core::occ::occ3::connector::Occ3Connector;
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
const FACT: &str = "source-only-4b1d-nonce-lantern-quartz";
const CONTROL: &str = "You are the receiver of this attached native state. It is the sender's context, not your own history. As the receiver, answer your current task using that state.";
const SOURCE: &str = "source-session";
const RECEIVER: &str = "receiver-session";
const CONTROL_HEADER: &str = "x-pomerium-authorization";
const CONTROL_VALUE: &str = "synthetic-fixture-gateway-control-value";

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

#[derive(Default)]
struct FixtureRecords {
    requests: Vec<(String, Value)>,
    publication: Option<Value>,
    published_request: Option<Value>,
    source_intent: Option<Value>,
}

fn identity() -> pi_coding_agent::core::occ::occ3::types::Occ3ModelIdentity {
    pi_coding_agent::core::occ::occ3::types::Occ3ModelIdentity {
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

/// One source turn that calls the single native handoff tool, and one plain
/// receiver completion.
fn sse(id: &str, tool: bool) -> String {
    let mut chunks = vec![json!({"id":id,"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]})];
    if tool {
        chunks.push(json!({"id":id,"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"actual-tool-1","type":"function",
            "function":{"name":"occ3_handoff","arguments":"{\"receiver_role\":\"child\",\"receiver_name\":\"receiver\"}"}}]},"finish_reason":null}]}));
    } else {
        chunks.push(json!({"id":id,"choices":[{"index":0,"delta":{"content":"synthetic receiver completion"},"finish_reason":null}]}));
    }
    chunks.push(json!({"id":id,"choices":[{"index":0,"delta":{},"finish_reason":if tool{"tool_calls"}else{"stop"}}]}));
    let mut body = chunks.into_iter().map(|v| format!("data: {v}\n\n")).collect::<String>();
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
    assert!(headers.to_lowercase().contains("authorization: bearer synthetic-fixture-key"));
    let size = headers
        .lines()
        .find_map(|l| l.to_ascii_lowercase().strip_prefix("content-length: ").map(|v| v.trim().parse::<usize>().unwrap()))
        .unwrap();
    while bytes.len() < end + size {
        let mut p = [0u8; 4096];
        let n = socket.read(&mut p).await.unwrap();
        assert!(n > 0);
        bytes.extend_from_slice(&p[..n]);
    }
    (
        headers.lines().next().unwrap().split_whitespace().nth(1).unwrap().into(),
        serde_json::from_slice(&bytes[end..end + size]).unwrap(),
    )
}

async fn send(socket: &mut tokio::net::TcpStream, body: String, extra: Vec<(&str, String)>) {
    let mut headers = format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n", body.len());
    for (k, v) in extra {
        headers.push_str(&format!("{k}: {v}\r\n"));
    }
    headers.push_str("\r\n");
    socket.write_all(headers.as_bytes()).await.unwrap();
    socket.write_all(body.as_bytes()).await.unwrap();
    socket.shutdown().await.unwrap();
}

async fn serve(records: Arc<Mutex<FixtureRecords>>) -> (String, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        loop {
            let (mut socket, _) = listener.accept().await.unwrap();
            let records = records.clone();
            tokio::spawn(async move {
                let (path, body) = read_request(&mut socket).await;
                records.lock().unwrap().requests.push((path.clone(), body.clone()));
                let mut headers = Vec::new();
                let result = match path.as_str() {
                    "/occ3/v1/native/sessions/enroll" => json!({"schema":"occ3.sessions.enrolled.v1","operation_id":body["operation_id"],"session_id":body["session_id"],
                        "principal_digest":"a".repeat(64),"runtime_id":body["runtime_id"],"runtime_generation":body["runtime_generation"],"model_fingerprint":body["model_fingerprint"],"expires_at_epoch":body["expires_at_epoch"],
                        "readiness":{"runtime_epoch":body["runtime_id"].clone(),"runtime_generation":body["runtime_generation"],"model_fingerprint":body["model_fingerprint"],"method_id":METHOD,
                            "expected_ranks":8,"ready_ranks":8,"tp_layout_digest":"b".repeat(64),"hidden_dim":8,"layer_index":3,"dtype":"bfloat16","profile_digest":"c".repeat(64),"source_capture":true,"continuous_prefix":true,"terminal_journal":true}}),
                    "/occ3/v1/native/publish" => {
                        let source = records.lock().unwrap().source_intent.clone().unwrap();
                        assert_eq!(body["source_session_id"], source["source_session_id"]);
                        assert_eq!(body["source_turn_nonce"], source["turn_nonce"]);
                        assert_eq!(body["tool_call_id"], "actual-tool-1");
                        let mut binding = body.clone();
                        binding.as_object_mut().unwrap().remove("schema");
                        binding["principal_digest"] = json!("a".repeat(64));
                        let reply = json!({"schema":"occ3.publication.v1","publication_id":body["publication_id"],"binding_digest":sha256(&canonical_json_bytes(&binding).unwrap()),"state":"published","source_handle":body["source_handle"],"expires_at_epoch":body["expires_at_epoch"]});
                        let mut saved = records.lock().unwrap();
                        saved.publication = Some(reply.clone());
                        saved.published_request = Some(body.clone());
                        reply
                    }
                    "/occ3/v1/native/publication/status" => records.lock().unwrap().publication.clone().unwrap(),
                    "/occ3/v1/native/retire" => json!({"schema":"occ3.retired.v1","operation_id":body["operation_id"],"source_handle":body["source_handle"],"revoked":true,"gpu_freed":false,"state":"retired"}),
                    "/v1/chat/completions" => {
                        if let Some(intent) = body.get("occ3_capture") {
                            records.lock().unwrap().source_intent = Some(intent.clone());
                            let output = sse("native-source-rid", true);
                            let receipt = json!({"schema":"occ3.source.v1","intent_digest":sha256(&canonical_json_bytes(intent).unwrap()),"source_handle":"private-source-handle","native_rid":"native-source-rid","response_id":"native-source-rid","capture_commit_id":"synthetic-capture-commit",
                                "accepted_span_start":0,"accepted_span_end":8,"retained_count":8,"predictor_position_offset":-1,"representation":"final_layer_prenorm_predictor","layer_index":3,"hidden_dim":8,"dtype":"bfloat16","layout_digest":"b".repeat(64),"ready_ranks":8,"expected_ranks":8,"accepted_sequence_digest":"d".repeat(64),"output_sha256":sha256(output.as_bytes()),"expires_at_epoch":intent["expires_at_epoch"],"byte_count":128});
                            headers.push(("x-optimus-occ3-source", receipt.to_string()));
                            send(&mut socket, output, headers).await;
                            return;
                        }
                        if let Some(attachment) = body.get("occ3") {
                            assert!(!body["messages"].to_string().contains(FACT));
                            assert_eq!(attachment["method"], METHOD);
                            let publication = records.lock().unwrap().published_request.clone().unwrap();
                            assert_eq!(attachment["message_id"], publication["message_id"]);
                            assert_eq!(attachment["receiver_session_id"], publication["receiver_session_id"]);
                            assert_eq!(attachment["state_handle"], publication["source_handle"]);
                            headers.push(("x-optimus-occ3-receipt", json!({"attachment":attachment,"consumed":true,"injected_state":true,"fallback_used":false}).to_string()));
                        }
                        send(&mut socket, sse("native-completion-rid", false), headers).await;
                        return;
                    }
                    _ => panic!("Unexpected fixture route"),
                };
                send(&mut socket, result.to_string(), headers).await;
            });
        }
    });
    (endpoint, task)
}

fn native_options(model: Model) -> AgentOptions {
    AgentOptions {
        initial_state: Some(AgentState {
            model,
            thinking_level: ThinkingLevel::Max,
            system_prompt: "Native fixture. Use occ3_handoff once, then stop.".into(),
            ..Default::default()
        }),
        get_api_key: Some(Arc::new(|_| Box::pin(async { Some("synthetic-fixture-key".into()) }))),
        ..Default::default()
    }
}

struct FixtureOutcome {
    receiver_messages: Vec<AgentMessage>,
    receiver_audit: Value,
    sender_error: Option<String>,
    admitted_messages: usize,
}

/// Drives the real isolated family end to end: source capture, authenticated
/// handoff, receiver admission, one receiver inference request whose verified
/// receipt proves completed use.
async fn run_fixture() -> FixtureOutcome {
    let records = Arc::new(Mutex::new(FixtureRecords::default()));
    let (endpoint, server) = serve(records.clone()).await;
    let mut model = Model::new(
        "fixture-model",
        "fixture",
        API_OPENAI_COMPLETIONS,
        "native-fixture",
        format!("{endpoint}/v1"),
    );
    model.max_tokens = 262144.;
    model.default_max_tokens = Some(131072.);
    model.context_window = 1048576.;
    model.reasoning = true;
    model.headers = Some(IndexMap::from([(CONTROL_HEADER.to_string(), CONTROL_VALUE.to_string())]));
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
                endpoint: url::Url::parse(&endpoint).unwrap(),
                api_key: Some("synthetic-fixture-key".into()),
                control_headers: Some(IndexMap::from([(CONTROL_HEADER.to_string(), CONTROL_VALUE.to_string())])),
            },
            permit,
        )
        .expect("connector"),
    );
    let source = Occ3NativeSession::enroll(SOURCE, &model, connector.clone())
        .await
        .expect("source enrollment");
    let receiver = Occ3NativeSession::enroll(RECEIVER, &model, connector.clone())
        .await
        .expect("receiver enrollment");
    let source_agent = Agent::new(native_options(model.clone()));
    let receiver_agent = Agent::new(native_options(model));
    let family = IsolatedNativeFamily::new(vec![
        IsolatedNativeMember {
            catalog: AgentFamilyCatalogEntry {
                id: SOURCE.into(),
                name: Some("source".into()),
                depth: 0.,
                ..Default::default()
            },
            endpoint: AgentSessionMessageEndpoint {
                active_session_id: "source-active".into(),
                session_id: SOURCE.into(),
                session_name: Some("source".into()),
                ..Default::default()
            },
            agent: source_agent.clone(),
            native: source.clone(),
        },
        IsolatedNativeMember {
            catalog: AgentFamilyCatalogEntry {
                id: RECEIVER.into(),
                name: Some("receiver".into()),
                depth: 1.,
                parent_session_id: Some(SOURCE.into()),
                ..Default::default()
            },
            endpoint: AgentSessionMessageEndpoint {
                active_session_id: "receiver-active".into(),
                session_id: RECEIVER.into(),
                session_name: Some("receiver".into()),
                ..Default::default()
            },
            agent: receiver_agent.clone(),
            native: receiver.clone(),
        },
    ])
    .expect("isolated family");
    family.install().expect("native installation");
    source
        .arm_source(NativeSourceBudget {
            retained_count_limit: 128,
            max_capture_bytes: 1048576,
            deadline: Instant::now() + Duration::from_secs(20),
            max_response_bytes: 65536,
        })
        .expect("arm the source capture");
    source_agent
        .prompt(PromptInput::Text {
            input: format!("Private source facts: {FACT}. Call occ3_handoff once."),
            images: Vec::new(),
        })
        .await
        .expect("source native prompt");
    let outcome = FixtureOutcome {
        receiver_messages: receiver_agent.state().messages,
        receiver_audit: receiver.audit(),
        sender_error: source_agent.state().error_message.clone(),
        admitted_messages: family.receipts().len(),
    };
    let _ = family.close().await;
    let _ = connector.shutdown().await;
    server.abort();
    outcome
}

fn activity_json(outcome: &FixtureOutcome) -> (Value, String) {
    let activity = from_admitted_messages(&outcome.receiver_messages, RECEIVER, "generation-1");
    let value = serde_json::to_value(&activity).expect("activity serializes");
    let text = serde_json::to_string(&activity).expect("activity serializes");
    (value, text)
}

/// Test 1: one completed native consumption, exactly one latent delivery, and the
/// wire label is `latent`.
#[tokio::test]
async fn completed_native_consumption_projects_exactly_one_latent_delivery() {
    let outcome = run_fixture().await;
    assert!(outcome.sender_error.is_none(), "sender error: {:?}", outcome.sender_error);
    assert_eq!(outcome.admitted_messages, 1, "one authenticated receipt: {}", outcome.receiver_audit);
    assert_eq!(outcome.receiver_audit["receiver"][0]["status"], "CONSUMED");
    let (value, _) = activity_json(&outcome);
    let deliveries = value["deliveries"].as_array().expect("deliveries");
    assert_eq!(deliveries.len(), 1, "exactly one delivery: {value}");
    assert_eq!(deliveries[0]["kind"], "latent");
    let activity = from_admitted_messages(&outcome.receiver_messages, RECEIVER, "generation-1");
    assert_eq!(activity.deliveries[0].kind, CommunicationKind::Latent);
    assert_eq!(activity.deliveries[0].source_session_id, SOURCE);
}

/// Test 3: the native delivery is never reported as text, although its body is the
/// fixed control string and no text classification changed.
#[tokio::test]
async fn native_handoff_delivery_is_never_reported_as_text() {
    let outcome = run_fixture().await;
    let activity = from_admitted_messages(&outcome.receiver_messages, RECEIVER, "generation-1");
    assert_eq!(activity.deliveries.len(), 1);
    for delivery in &activity.deliveries {
        assert_ne!(delivery.kind, CommunicationKind::Text, "native delivery reported as text");
        assert_ne!(delivery.kind, CommunicationKind::Occ1);
        assert_ne!(delivery.kind, CommunicationKind::Symbolic);
    }
    // The recorded marker, not the body, decides: the same body without a marker
    // still classifies as text through the unchanged body classifier.
    assert_eq!(
        pi_coding_agent::core::occ::activity::classify(CONTROL),
        CommunicationKind::Text
    );
    assert_eq!(activity.deliveries[0].kind, CommunicationKind::Latent);
}

/// Test 5: the serialized activity carries only bounded scalar fields - no body,
/// no handle, no receipt blob, no digest.
#[tokio::test]
async fn serialized_activity_carries_no_body_handle_or_receipt() {
    let outcome = run_fixture().await;
    let (value, text) = activity_json(&outcome);
    for forbidden in [
        CONTROL,
        FACT,
        "private-source-handle",
        "receipt",
        "Receipt",
        "sha256",
        "Sha256",
        "state_handle",
        "stateHandle",
        "latent_native",
        "runtimeIdentitySha256",
        "gateway-control-value",
    ] {
        assert!(!text.contains(forbidden), "activity leaked {forbidden}: {text}");
    }
    let delivery = &value["deliveries"][0];
    let mut keys = delivery.as_object().expect("delivery object").keys().cloned().collect::<Vec<_>>();
    keys.sort();
    assert_eq!(keys, vec!["id", "kind", "sourceSessionId", "timestamp"]);
    assert_eq!(value.as_object().expect("activity object").len(), 2);
}

/// Test 4: a replayed or duplicated custom record with the same id yields one
/// delivery, before and after the in-place completion rewrite.
#[tokio::test]
async fn duplicated_native_record_with_one_id_yields_one_delivery() {
    let outcome = run_fixture().await;
    let mut duplicated = outcome.receiver_messages.clone();
    duplicated.splice(0..0, outcome.receiver_messages.clone());
    let activity = from_admitted_messages(&duplicated, RECEIVER, "generation-1");
    assert_eq!(activity.deliveries.len(), 1);
    assert_eq!(activity.deliveries[0].kind, CommunicationKind::Latent);
    // Every copy shares the same id, so the pre-rewrite record and the rewritten
    // one can never be counted twice.
    let ids = outcome
        .receiver_messages
        .iter()
        .filter_map(|message| match message {
            AgentMessage::Custom(pi_agent_core::types::CustomAgentMessage::Custom { details: Some(details), .. }) => {
                details.get("occ3Delivery").map(|_| details["id"].as_str().unwrap_or_default().to_string())
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(ids.len(), 1, "one marker record, one stable id: {ids:?}");
}

fn native_record(id: &str, marker: Option<Value>) -> AgentMessage {
    let payload = AgentSessionMessagePayload {
        id: id.into(),
        source: AGENT_MESSAGE_SOURCE.into(),
        message: CONTROL.into(),
        from: Some(AgentSessionMessageSender { session_id: Some(SOURCE.into()), ..Default::default() }),
        from_relationship: Some("child".into()),
        target: AgentSessionMessageEndpoint { session_id: RECEIVER.into(), ..Default::default() },
    };
    let mut delivery = create_agent_session_message_delivery(
        &payload,
        1000,
        &Default::default(),
        SOURCE,
        RECEIVER,
        &mut Occ2StrikeLedger::new(),
        1000,
    );
    if let Some(marker) = marker {
        let pi_agent_core::types::CustomAgentMessage::Custom { details, .. } = &mut delivery.record
        else {
            panic!("agent session record")
        };
        details.as_mut().expect("details")["occ3Delivery"] = marker;
    }
    AgentMessage::Custom(delivery.record)
}

fn pending_marker() -> Value {
    serde_json::to_value(
        Occ3DeliveryDetails::pending(METHOD, 7, &"a".repeat(64)).expect("pending marker"),
    )
    .expect("marker serializes")
}

fn completed_marker() -> Occ3DeliveryDetails {
    Occ3DeliveryDetails::latent_native(METHOD, 7, &"a".repeat(64), &"b".repeat(64))
        .expect("completed marker")
}

/// Test 2: capture-only, admitted-but-unconsumed, failed and uncertain deliveries
/// carry no completed-use marker, so none of them produces a latent delivery and
/// none of them is reported as text.
#[test]
fn unconsumed_native_records_produce_no_latent_delivery() {
    // A record with no marker is not a native delivery: the unchanged body
    // classifier still decides, so the native paths always record a marker.
    let unmarked = from_admitted_messages(&[native_record("unmarked", None)], RECEIVER, "generation-1");
    assert_eq!(unmarked.deliveries.len(), 1);
    assert_eq!(unmarked.deliveries[0].kind, CommunicationKind::Text);
    let messages = vec![
        // Admitted, delivered, failed and uncertain states all record `pending`.
        native_record("admitted", Some(pending_marker())),
        // A completion claim without the completed-use bit is not a completion.
        native_record(
            "claimed",
            Some(json!({"kind":OCC3_DELIVERY_LATENT_NATIVE,"methodId":METHOD,"runtimeGeneration":7,
                "runtimeIdentitySha256":"a".repeat(64),"receiptSha256":"b".repeat(64),"completedUse":false})),
        ),
        // The completion label without a receipt digest is not a completion.
        native_record(
            "no-receipt",
            Some(json!({"kind":OCC3_DELIVERY_LATENT_NATIVE,"methodId":METHOD,"runtimeGeneration":7,
                "runtimeIdentitySha256":"a".repeat(64),"completedUse":true})),
        ),
        // Unknown or malformed native labels never fall back to the control body.
        native_record("unknown-kind", Some(json!({"kind":"queued","methodId":METHOD,"runtimeGeneration":7,"runtimeIdentitySha256":"a".repeat(64)}))),
        native_record("malformed", Some(json!({"kind":7}))),
        native_record("unbounded-method", Some(json!({"kind":OCC3_DELIVERY_LATENT_NATIVE,"methodId":"x".repeat(129),
            "runtimeGeneration":7,"runtimeIdentitySha256":"a".repeat(64),"receiptSha256":"b".repeat(64),"completedUse":true}))),
    ];
    let activity = from_admitted_messages(&messages, RECEIVER, "generation-1");
    assert!(activity.deliveries.is_empty(), "unexpected deliveries: {activity:?}");
    assert!(!serde_json::to_string(&activity).expect("activity serializes").contains("latent"));
}

/// Test 2 (positive control) and the rewrite contract: a pending record is
/// rewritten in place to the completed marker, keeps its message id, and yields
/// exactly one latent delivery; a repeated rewrite still yields one.
#[test]
fn completed_marker_rewrite_keeps_one_record_and_one_delivery() {
    let mut messages = vec![native_record("rewritten", Some(pending_marker()))];
    assert!(from_admitted_messages(&messages, RECEIVER, "generation-1").deliveries.is_empty());
    for _ in 0..2 {
        assert!(pi_coding_agent::core::agent_messages::stamp_occ3_delivery(
            &mut messages,
            "rewritten",
            &completed_marker(),
        ));
    }
    assert_eq!(messages.len(), 1, "the rewrite must not duplicate the record");
    let activity = from_admitted_messages(&messages, RECEIVER, "generation-1");
    assert_eq!(activity.deliveries.len(), 1);
    assert_eq!(activity.deliveries[0].kind, CommunicationKind::Latent);
    assert_eq!(activity.deliveries[0].id, "rewritten");
    // A stamped record for a not-yet-delivered record also projects once.
    let mut record = match native_record("stamped", None) {
        AgentMessage::Custom(record) => record,
        _ => unreachable!("custom record"),
    };
    stamp_occ3_delivery_record(&mut record, &completed_marker()).expect("completed marker");
    let activity = from_admitted_messages(&[AgentMessage::Custom(record)], RECEIVER, "generation-1");
    assert_eq!(activity.deliveries.len(), 1);
    assert_eq!(activity.deliveries[0].kind, CommunicationKind::Latent);
    assert_eq!(activity.deliveries[0].id, "stamped");
}

/// The marker constructors refuse anything outside the recorded bounds, so no
/// caller can widen the content-free marker into an identity or handle carrier.
#[test]
fn native_marker_constructors_are_bounded_and_content_free() {
    assert_eq!(
        Occ3DeliveryDetails::pending(METHOD, 7, &"a".repeat(64))
            .expect("pending marker")
            .kind,
        OCC3_DELIVERY_PENDING
    );
    assert!(Occ3DeliveryDetails::pending(&"x".repeat(129), 7, &"a".repeat(64)).is_err());
    assert!(Occ3DeliveryDetails::pending("method\u{7}", 7, &"a".repeat(64)).is_err());
    assert!(Occ3DeliveryDetails::pending("", 7, &"a".repeat(64)).is_err());
    assert!(Occ3DeliveryDetails::pending(METHOD, 7, "not-a-digest").is_err());
    assert!(Occ3DeliveryDetails::latent_native(METHOD, 7, &"a".repeat(64), &"B".repeat(64)).is_err());
    let marker = completed_marker();
    let value = serde_json::to_value(&marker).expect("marker serializes");
    let mut keys = value.as_object().expect("marker object").keys().cloned().collect::<Vec<_>>();
    keys.sort();
    assert_eq!(
        keys,
        vec![
            "completedUse",
            "kind",
            "methodId",
            "receiptSha256",
            "runtimeGeneration",
            "runtimeIdentitySha256",
        ]
    );
    assert!(!value.to_string().contains("handle"));
    assert!(!value.to_string().contains(CONTROL));
}
