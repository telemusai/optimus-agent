//! OCC-3 method-identity seams (spec 14/18): the coarse `Occ3Method` enum is
//! only the negotiation family; the full dotted method id is the identity that
//! the string-equality seams compare. Two different full method ids must each
//! bind through their own permit, and a method mismatch must decline by name.
//!
//! Offline: one loopback control plane serves the enrollment, publication and
//! completion routes, parameterized by the method id the runtime serves. No
//! GPU, no provider account, no semantic evidence.

use indexmap::IndexMap;
use pi_agent_core::agent::{Agent, AgentOptions, PromptInput};
use pi_agent_core::types::{AgentState, ThinkingLevel};
use pi_ai::inference_attachment::InferenceStateTarget;
use pi_ai::source_capture::{canonical_json_bytes, sha256, SourceCaptureProfile};
use pi_ai::types::{Model, API_OPENAI_COMPLETIONS};
use pi_coding_agent::core::agent_messages::{
    AgentFamilyCatalogEntry, AgentSessionMessageEndpoint,
};
use pi_coding_agent::core::agent_session::occ3_state::{NativeSourceBudget, Occ3NativeSession};
use pi_coding_agent::core::occ::occ3::connector::Occ3Connector;
use pi_coding_agent::core::occ::occ3::types::{Occ3Error, Occ3Method, Occ3ModelIdentity};
use pi_coding_agent::core::occ::occ3::inference::{Occ3InferenceInbox, PrepareInferenceRequest};
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

/// Two full dotted method ids of one negotiation family.
const METHOD_A: &str = "statebridge.glm53.final-prenorm-predictor.continuous-prefix.v1";
const METHOD_B: &str = "statebridge.glm53.calibrated-lowrank-procrustes.v1";
/// A full dotted method id of the second negotiation family.
const METHOD_KV: &str = "kv-handoff.glm53.final-kv.v1";

const FACT: &str = "method-identity-seam-fixture-fact-4b1d";
const CONTROL: &str = "You are the receiver of this attached native state. It is the sender's context, not your own history. As the receiver, answer your current task using that state.";
const SOURCE: &str = "source-session";
const RECEIVER: &str = "receiver-session";

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

fn identity(method: &str) -> Occ3ModelIdentity {
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
        latent_method: method.into(),
        runtime: "synthetic-local-fixture".into(),
    }
}

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

#[derive(Default)]
struct FixtureRecords {
    requests: Vec<(String, Value)>,
    publication: Option<Value>,
    published_request: Option<Value>,
    source_intent: Option<Value>,
    /// The full dotted method id the receiver's attachment carried on the wire.
    attached_method: Mutex<Option<String>>,
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

/// One loopback control plane serving exactly one method id.
async fn serve(method: &'static str) -> (String, Arc<Mutex<FixtureRecords>>, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let records = Arc::new(Mutex::new(FixtureRecords::default()));
    let records_task = Arc::clone(&records);
    let task = tokio::spawn(async move {
        loop {
            let (mut socket, _) = listener.accept().await.unwrap();
            let records = Arc::clone(&records_task);
            tokio::spawn(async move {
                let (path, body) = read_request(&mut socket).await;
                records.lock().unwrap().requests.push((path.clone(), body.clone()));
                let mut headers = Vec::new();
                let result = match path.as_str() {
                    "/occ3/v1/native/sessions/enroll" => json!({"schema":"occ3.sessions.enrolled.v1","operation_id":body["operation_id"],"session_id":body["session_id"],
                        "principal_digest":"a".repeat(64),"runtime_id":body["runtime_id"],"runtime_generation":body["runtime_generation"],"model_fingerprint":body["model_fingerprint"],"expires_at_epoch":body["expires_at_epoch"],
                        "readiness":{"runtime_epoch":body["runtime_id"].clone(),"runtime_generation":body["runtime_generation"],"model_fingerprint":body["model_fingerprint"],"method_id":method,
                            "expected_ranks":8,"ready_ranks":8,"tp_layout_digest":"b".repeat(64),"hidden_dim":8,"layer_index":3,"dtype":"bfloat16","profile_digest":"c".repeat(64),"source_capture":true,"continuous_prefix":true,"terminal_journal":true}}),
                    "/occ3/v1/native/publish" => {
                        let source = records.lock().unwrap().source_intent.clone().unwrap();
                        assert_eq!(body["source_session_id"], source["source_session_id"]);
                        assert_eq!(body["source_turn_nonce"], source["turn_nonce"]);
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
                            assert_eq!(attachment["method"], method, "the wire attachment must carry the full method id, not its family");
                            let publication = records.lock().unwrap().published_request.clone().unwrap();
                            assert_eq!(attachment["message_id"], publication["message_id"]);
                            records.lock().unwrap().attached_method = Mutex::new(Some(attachment["method"].as_str().unwrap().to_string()));
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
    (endpoint, records, task)
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

#[derive(Clone)]
struct MethodOutcome {
    sender_error: Option<String>,
    consumed: bool,
    /// The full dotted method id the receiver's attachment carried.
    attached_method: Option<String>,
    publication_message_id: String,
}

/// Drives the real isolated family end to end under one full method id:
/// source capture, authenticated handoff, receiver admission, and one receiver
/// inference request whose verified receipt proves completed use.
async fn run_method(method: &'static str) -> (MethodOutcome, Arc<SglangMethodFixture>) {
    let (endpoint, records, _server) = serve(method).await;
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
    let permit = NativeExperimentalPermit {
        attempt_id: "method-identity-attempt".into(),
        identity: identity(method),
        target: InferenceStateTarget {
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
                control_headers: Some(IndexMap::new()),
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
    let receiver_agent = Agent::new(native_options(model.clone()));
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
    let audit = receiver.audit();
    let receiver_records = audit["receiver"].as_array().cloned().unwrap_or_default();
    let consumed = receiver_records
        .iter()
        .any(|record| record["status"] == json!("CONSUMED"));
    let attached_method = records
        .lock()
        .unwrap()
        .attached_method
        .lock()
        .unwrap()
        .clone();
    let publication_message_id = records
        .lock()
        .unwrap()
        .published_request
        .as_ref()
        .and_then(|published| published["message_id"].as_str())
        .unwrap_or_default()
        .to_string();
    let outcome = MethodOutcome {
        sender_error: source_agent.state().error_message.clone(),
        consumed,
        attached_method,
        publication_message_id,
    };
    let fixture = Arc::new(SglangMethodFixture {
        _connector: connector,
        _receiver_agent: receiver_agent.clone(),
        _receiver_session: receiver.clone(),
    });
    let _ = family.close().await;
    (outcome, fixture)
}

pub struct SglangMethodFixture {
    _connector: Arc<SglangOcc3Connector>,
    _receiver_agent: Arc<pi_agent_core::agent::Agent>,
    _receiver_session: Arc<Occ3NativeSession>,
}

/// Two different full method ids of one family each bind through their own
/// permit: the publication, the wire attachment and the completed-use receipt
/// all carry the exact full dotted method id, never just the family.
#[tokio::test]
async fn two_full_method_ids_of_one_family_each_bind_end_to_end() {
    let (outcome_a, _) = run_method(METHOD_A).await;
    assert!(outcome_a.sender_error.is_none(), "sender error: {:?}", outcome_a.sender_error);
    assert!(outcome_a.consumed, "method A must complete a native consumption");
    assert_eq!(outcome_a.attached_method.as_deref(), Some(METHOD_A));

    let (outcome_b, _) = run_method(METHOD_B).await;
    assert!(outcome_b.sender_error.is_none(), "sender error: {:?}", outcome_b.sender_error);
    assert!(outcome_b.consumed, "method B must complete a native consumption");
    assert_eq!(outcome_b.attached_method.as_deref(), Some(METHOD_B));
    assert_ne!(outcome_a.publication_message_id, outcome_b.publication_message_id);
}

/// A full method id of the second family binds end to end too: the derived
/// family follows the permit's method id everywhere, never a hardcoded one.
#[tokio::test]
async fn a_kv_handoff_full_method_id_binds_end_to_end_with_its_own_family() {
    let (outcome, _) = run_method(METHOD_KV).await;
    assert!(outcome.sender_error.is_none(), "sender error: {:?}", outcome.sender_error);
    assert!(outcome.consumed, "the kv-handoff method must complete a native consumption");
    assert_eq!(outcome.attached_method.as_deref(), Some(METHOD_KV));
    assert_eq!(
        pi_coding_agent::core::occ3_connectors::method_family(METHOD_KV),
        Some(Occ3Method::KvHandoff)
    );
}

/// The receiver-side string-equality seam: an enrollment identity whose full
/// method id differs must not bind the delivery. Same family, different id:
/// declined.
#[tokio::test]
async fn bind_lease_declines_a_same_family_method_id_swap() {
    // Two complete identities of one family, differing only in the full
    // dotted method id.
    let identity_a = identity(METHOD_A);
    let identity_b = identity(METHOD_B);
    let envelope = fixture_envelope(&identity_a);
    let mut inbox = Occ3InferenceInbox::new(RECEIVER.to_string()).unwrap();
    assert!(inbox
        .admit(envelope.clone(), &envelope.message_id, SOURCE, now())
        .unwrap());
    assert!(inbox
        .claim(&[envelope.message_id.clone()], now())
        .unwrap()
        .is_some());
    // The attachment carries the full method id of identity A.
    let attachment = pi_ai::inference_attachment::InferenceAttachment {
        protocol_version: 1,
        request_id: "consume-1".into(),
        message_id: envelope.message_id.clone(),
        connector_id: envelope.connector_id.clone(),
        receiver_session_id: RECEIVER.into(),
        model_fingerprint: envelope.model_fingerprint.clone(),
        runtime_id: envelope.runtime_id.clone(),
        runtime_generation: envelope.state_handle.runtime_generation,
        state_handle: envelope.state_handle.handle_id.clone(),
        expires_at_epoch: envelope.state_handle.expires_at_epoch,
        method: identity_a.latent_method.clone(),
    };
    let target = InferenceStateTarget {
        provider: "native-fixture".into(),
        model_id: "fixture-model".into(),
        base_url: "http://127.0.0.1:1/v1".into(),
    };
    let lease = pi_ai::inference_attachment::InferenceStateLease::new(target, attachment).unwrap();
    // The same family is not the same method: identity B must refuse.
    assert!(matches!(
        inbox.bind_lease(&lease, &identity_b),
        Err(Occ3Error::Unauthorized)
    ));
    // The exact full method id binds.
    assert!(inbox.bind_lease(&lease, &identity_a).is_ok());
}

/// The connector-side string-equality seam: prepare_inference with a receiver
/// identity whose full method id differs declines with the named
/// "native control binding invalid" transfer failure.
#[tokio::test]
async fn prepare_inference_declines_a_swapped_full_method_id() {
    let identity_a = identity(METHOD_A);
    let identity_b = identity(METHOD_B);
    let envelope = fixture_envelope(&identity_a);
    let target = InferenceStateTarget {
        provider: "native-fixture".into(),
        model_id: "fixture-model".into(),
        base_url: "http://127.0.0.1:1/v1".into(),
    };
    let request = PrepareInferenceRequest {
        envelope,
        request_id: "consume-2".into(),
        receiver_session_id: RECEIVER.into(),
        receiver_identity: identity_b,
        now_epoch: now(),
        target,
    };
    let connector = offline_connector(&identity_a);
    let error = connector.prepare_inference(request).await.unwrap_err();
    assert!(
        matches!(error, Occ3Error::TransferFailed { ref reason } if reason == "native control binding invalid"),
        "a swapped full method id must decline by name: {error:?}"
    );
}

fn fixture_envelope(identity: &Occ3ModelIdentity) -> pi_coding_agent::core::occ::occ3::types::Occ3Envelope {
    pi_coding_agent::core::occ::occ3::types::Occ3Envelope {
        schema: "optimus.occ3.v1".into(),
        protocol_version: 1,
        message_id: "method-message-1".into(),
        sender_session_id: SOURCE.into(),
        receiver_session_id: RECEIVER.into(),
        model_fingerprint: identity.fingerprint(),
        runtime_id: "runtime-epoch-1".into(),
        connector_id: "sglang-occ3".into(),
        latent_method: pi_coding_agent::core::occ3_connectors::method_family(&identity.latent_method)
            .expect("family"),
        state_handle: pi_coding_agent::core::occ::occ3::types::Occ3HandleRef {
            handle_id: "handle-1".into(),
            connector_id: "sglang-occ3".into(),
            model_fingerprint: identity.fingerprint(),
            runtime_generation: 1,
            expires_at_epoch: now() + 600,
            state_size_bytes: 128,
        },
        control_summary: CONTROL.into(),
        fallback_reference: "text".into(),
    }
}

/// An offline permit-bearing connector with no reachable endpoint: identity
/// checks decline before any network.
fn offline_connector(identity: &Occ3ModelIdentity) -> Arc<SglangOcc3Connector> {
    Arc::new(
        SglangOcc3Connector::new_experimental(
            SglangOcc3Config {
                endpoint: url::Url::parse("http://127.0.0.1:9").unwrap(),
                api_key: None,
                control_headers: None,
            },
            NativeExperimentalPermit {
                attempt_id: "offline-attempt".into(),
                identity: identity.clone(),
                target: InferenceStateTarget {
                    provider: "native-fixture".into(),
                    model_id: "fixture-model".into(),
                    base_url: "http://127.0.0.1:9/v1".into(),
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
            },
        )
        .expect("offline connector"),
    )
}
