//! Real native Agent/controller/HTTP request integration. The local backend is
//! a synthetic protocol fixture, not GPU or semantic evidence.
use indexmap::IndexMap;
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
const FACT: &str = "source-only-7e568c-nonce-planet-saffron";
// Synthetic gateway control header: the fixture rides the same resolved value
// on the connector control plane and the inference model headers, mirroring the
// existing-credential propagation the connector implements. Never a real value.
const CONTROL_HEADER: &str = "x-pomerium-authorization";
const CONTROL_VALUE: &str = "synthetic-fixture-gateway-control-value";
fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}
#[derive(Clone, Default)]
pub struct FixtureMode {
    pub lose_publish: bool,
    pub lose_first_retire: bool,
    pub absent_publish: bool,
    pub wrong_epoch: bool,
    pub bad_source_hash: bool,
    pub multiple_tools: bool,
    pub unrelated: bool,
    pub retry_source_request: bool,
    pub queue_receiver: bool,
    /// Control plane replies 401 to publish and the one same-key status query:
    /// the attempt fails closed with no refresh, replay or regeneration.
    pub unauthorized_publish: bool,
}
#[derive(Default)]
struct FixtureRecords {
    requests: Vec<(String, Value)>,
    publication: Option<Value>,
    published_request: Option<Value>,
    source_intent: Option<Value>,
    // The single authenticated receipt the fixture served for that intent.
    source_receipt: Option<Value>,
    // In-memory only: the observed control header value per request, in order.
    // Never copied into the report or any evidence file.
    control_header_seen: Vec<(String, Option<String>)>,
}
pub struct FixtureResult {
    pub report: Value,
    pub requests: Vec<(String, Value)>,
    pub source_receipt: Option<Value>,
    pub sender_error: Option<String>,
    pub control_header_seen: Vec<(String, Option<String>)>,
    /// The sender agent's serialized transcript (messages only, never the
    /// report or any evidence file). Stable recorded evidence: the first
    /// turn's failure and the ordinary recovery turn are asserted from
    /// here, not from the volatile agent error field, which a later
    /// successful turn may legitimately clear.
    pub source_transcript: String,
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
fn sse(id: &str, tool: bool, multiple: bool) -> String {
    let mut chunks = vec![
        json!({"id":id,"choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"retained native reasoning"},"finish_reason":null}]}),
    ];
    if tool {
        let mut calls = vec![
            json!({"index":0,"id":"actual-tool-1","type":"function","function":{"name":"occ3_handoff","arguments":"{\"receiver_role\":\"child\",\"receiver_name\":\"receiver\"}"}}),
        ];
        if multiple {
            calls.push(json!({"index":1,"id":"actual-tool-2","type":"function","function":{"name":"occ3_handoff","arguments":"{\"receiver_role\":\"child\",\"receiver_name\":\"receiver\"}"}}));
        }
        chunks.push(json!({"id":id,"choices":[{"index":0,"delta":{"tool_calls":calls},"finish_reason":null}]}));
    } else {
        chunks.push(json!({"id":id,"choices":[{"index":0,"delta":{"content":"synthetic fixture completion"},"finish_reason":null}]}));
    }
    chunks.push(json!({"id":id,"choices":[{"index":0,"delta":{},"finish_reason":if tool {"tool_calls"}else{"stop"}}]}));
    let mut body = chunks
        .into_iter()
        .map(|v| format!("data: {v}\n\n"))
        .collect::<String>();
    body.push_str("data: [DONE]\n\n");
    body
}
async fn read_request(socket: &mut tokio::net::TcpStream) -> (String, Value, Option<String>) {
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
    let control_prefix = format!("{CONTROL_HEADER}: ");
    let control = headers.lines().find_map(|l| {
        l.to_ascii_lowercase()
            .strip_prefix(&control_prefix)
            .map(str::to_string)
    });
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
        control,
    )
}
async fn send(socket: &mut tokio::net::TcpStream, body: String, extra: Vec<(&str, String)>) {
    let mut headers=format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n",body.len());
    for (k, v) in extra {
        headers.push_str(&format!("{k}: {v}\r\n"));
    }
    headers.push_str("\r\n");
    socket.write_all(headers.as_bytes()).await.unwrap();
    socket.write_all(body.as_bytes()).await.unwrap();
    socket.shutdown().await.unwrap();
}
async fn serve(
    mode: FixtureMode,
    records: Arc<Mutex<FixtureRecords>>,
) -> (String, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        loop {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mode = mode.clone();
            let records = records.clone();
            tokio::spawn(async move {
                let (path, body, control) = read_request(&mut socket).await;
                {
                    let mut records = records.lock().unwrap();
                    records.requests.push((path.clone(), body.clone()));
                    records.control_header_seen.push((path.clone(), control));
                }
                if mode.unauthorized_publish
                    && matches!(
                        path.as_str(),
                        "/occ3/v1/native/publish" | "/occ3/v1/native/publication/status"
                    )
                {
                    // Mid-attempt 401: the attempt fails closed. No refresh,
                    // replay, regeneration or inference request may follow.
                    let body = json!({"detail": "unauthorized"}).to_string();
                    let response = format!(
                        "HTTP/1.1 401 Unauthorized\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    );
                    socket.write_all(response.as_bytes()).await.unwrap();
                    socket.shutdown().await.unwrap();
                    return;
                }
                let mut headers = Vec::new();
                let result=match path.as_str() {
                "/occ3/v1/native/sessions/enroll"=>json!({"schema":"occ3.sessions.enrolled.v1","operation_id":body["operation_id"],"session_id":body["session_id"],
                    "principal_digest":"a".repeat(64),"runtime_id":body["runtime_id"],"runtime_generation":body["runtime_generation"],"model_fingerprint":body["model_fingerprint"],"expires_at_epoch":body["expires_at_epoch"],
                    "readiness":{"runtime_epoch":if mode.wrong_epoch {"wrong-epoch".into()}else{body["runtime_id"].clone()},"runtime_generation":body["runtime_generation"],"model_fingerprint":body["model_fingerprint"],"method_id":METHOD,
                        "expected_ranks":8,"ready_ranks":8,"tp_layout_digest":"b".repeat(64),"hidden_dim":8,"layer_index":3,"dtype":"bfloat16","profile_digest":"c".repeat(64),"source_capture":true,"continuous_prefix":true,"terminal_journal":true}}),
                "/occ3/v1/native/publish"=>{
                    let source=records.lock().unwrap().source_intent.clone().unwrap();
                    assert_eq!(body["source_session_id"],source["source_session_id"]);assert_eq!(body["source_turn_nonce"],source["turn_nonce"]);assert_eq!(body["tool_call_id"],"actual-tool-1");
                    let mut binding=body.clone();binding.as_object_mut().unwrap().remove("schema");binding["principal_digest"]=json!("a".repeat(64));
                    let reply=json!({"schema":"occ3.publication.v1","publication_id":body["publication_id"],"binding_digest":sha256(&canonical_json_bytes(&binding).unwrap()),"state":"published","source_handle":body["source_handle"],"expires_at_epoch":body["expires_at_epoch"]});
                    if !mode.absent_publish {let mut saved=records.lock().unwrap();saved.publication=Some(reply.clone());saved.published_request=Some(body.clone());}
                    if mode.lose_publish || mode.absent_publish {return} reply
                },
                "/occ3/v1/native/publication/status"=>records.lock().unwrap().publication.clone().unwrap_or_else(||json!({"schema":"occ3.publication.v1","publication_id":body["publication_id"],"state":"absent"})),
                "/occ3/v1/native/retire"=>{
                    let retire_count=records.lock().unwrap().requests.iter().filter(|(p,_)|p.ends_with("/retire")).count();
                    if mode.lose_first_retire && retire_count==1 {return}
                    json!({"schema":"occ3.retired.v1","operation_id":body["operation_id"],"source_handle":body["source_handle"],"revoked":true,"gpu_freed":false,"state":"retired"})
                },
                "/v1/chat/completions"=>{
                    assert_eq!(body.get("max_tokens").or_else(||body.get("max_completion_tokens")), Some(&json!(131072))); // real native default is preserved, fixture output is short.
                    assert!(body.get("reasoning_effort").and_then(Value::as_str).is_some_and(|v|matches!(v,"high"|"xhigh"|"max")), "MAX goes through the existing model-specific conversion");
                    if let Some(intent)=body.get("occ3_capture") {
                        records.lock().unwrap().source_intent=Some(intent.clone());
                        let output=sse("native-source-rid",true,mode.multiple_tools);
                        // The default arming is occ3.capture.v2, so the
                        // intent is answered on the v2 wire with the
                        // occ3.span.v1 block: the reasoning region, the
                        // thinking_end terminator marker row and the
                        // role-token stop row (both CONTROL), the single
                        // occ3_handoff tool-call row and its argument rows.
                        // The turn emits no content delta, so the window
                        // carries no CONTENT row.
                        let mut receipt=json!({"schema":"occ3.source.v2","intent_digest":sha256(&canonical_json_bytes(intent).unwrap()),"source_handle":"private-source-handle","native_rid":"native-source-rid","response_id":"native-source-rid","capture_commit_id":"synthetic-capture-commit",
                            "accepted_span_start":0,"accepted_span_end":8,"retained_count":8,"predictor_position_offset":-1,"representation":"final_layer_prenorm_predictor","layer_index":3,"hidden_dim":8,"dtype":"bfloat16","layout_digest":"b".repeat(64),"ready_ranks":8,"expected_ranks":8,"accepted_sequence_digest":"d".repeat(64),"output_sha256":if mode.bad_source_hash{"0".repeat(64)}else{sha256(output.as_bytes())},"expires_at_epoch":intent["expires_at_epoch"],"byte_count":128});
                        let span=json!({"span_schema":"occ3.span.v1","span_policy":intent["span_policy"],
                            "span_marker_profile":"glm45_glm47_xml_kv.v1","span_marker_schema":"occ3.span.markers.v1",
                            "span_reasoning_terminator":"thinking_end","span_turn_end_attested":true,
                            "span_thinking_count":2,"span_content_count":0,"span_tool_call_count":1,
                            "span_tool_arguments_count":3,"span_control_count":2,"span_unknown_count":0,
                            "span_window_start_position":0,"span_window_end_position":8,
                            "span_contiguous":true,"span_contains_final_content":true,
                            "span_contains_tool_call":true,"span_message_truncated":false});
                        for (key,value) in span.as_object().unwrap() {
                            receipt[key.clone()]=value.clone();
                        }
                        records.lock().unwrap().source_receipt=Some(receipt.clone());
                        headers.push(("x-optimus-occ3-source",receipt.to_string()));send(&mut socket,output,headers).await;return;
                    }
                    if let Some(attachment)=body.get("occ3") {
                        assert!(!body["messages"].to_string().contains(FACT));
                        assert_eq!(attachment["method"],METHOD);
                        let publication=records.lock().unwrap().published_request.clone().unwrap();
                        assert_eq!(attachment["message_id"],publication["message_id"]);assert_eq!(attachment["receiver_session_id"],publication["receiver_session_id"]);
                        assert_eq!(attachment["state_handle"],publication["source_handle"]);assert_eq!(attachment["runtime_id"],"runtime-epoch-1");assert_eq!(attachment["runtime_generation"],1);
                        headers.push(("x-optimus-occ3-receipt",json!({"attachment":attachment,"consumed":true,"injected_state":true,"fallback_used":false}).to_string()));
                    }
                    send(&mut socket,sse("native-completion-rid",false,false),headers).await;return;
                },
                _=>panic!("Unexpected fixture route"),
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
        get_api_key: Some(Arc::new(|_| {
            Box::pin(async { Some("synthetic-fixture-key".into()) })
        })),
        ..Default::default()
    }
}
// Test-only asynchronous queue boundary. Authorize through the actual sealed
// family controller and use actual receiver sidecar admission. Deliberately do
// not run receiver inference; a queued acknowledgment must not imply Consumed.
struct QueuedAdmissionFixture {
    inner: Arc<dyn AgentSessionMessageController>,
    receiver: Arc<Occ3NativeSession>,
}
impl AgentSessionMessageController for QueuedAdmissionFixture {
    fn roster(&self) -> pi_ai::types::BoxFuture<Result<AgentFamilyRosterResult, String>> {
        self.inner.roster()
    }
    fn await_pending_child_publication(
        &self,
        selector: String,
    ) -> pi_ai::types::BoxFuture<Result<Option<String>, String>> {
        self.inner.await_pending_child_publication(selector)
    }
    fn send_agent_message(
        &self,
        input: AgentSessionMessageSendInput,
    ) -> pi_ai::types::BoxFuture<Result<AgentSessionMessageReceipt, String>> {
        self.inner.send_agent_message(input)
    }
    fn authorize_native_handoff(
        &self,
        relationship: String,
        selector: String,
    ) -> pi_ai::types::BoxFuture<Result<AuthorizedNativeHandoff, String>> {
        self.inner.authorize_native_handoff(relationship, selector)
    }
    fn send_prepared_native_handoff(
        &self,
        prepared: PreparedNativeHandoff,
    ) -> pi_ai::types::BoxFuture<Result<AgentSessionMessageReceipt, String>> {
        let receiver = self.receiver.clone();
        Box::pin(async move {
            if !receiver.admit(&prepared)? {
                return Err("duplicate queued fixture admission".into());
            }
            let payload = prepared.control_payload();
            Ok(create_agent_session_message_receipt(
                &payload,
                &DELIVERY_STATUS_QUEUED.into(),
                &chrono::Utc::now().to_rfc3339(),
            ))
        })
    }
}
pub async fn run_offline_fixture(mode: FixtureMode) -> Result<FixtureResult, String> {
    let records = Arc::new(Mutex::new(FixtureRecords::default()));
    let (endpoint, server) = serve(mode.clone(), records.clone()).await;
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
    // Same resolved control header context for inference (model headers) and
    // the native control plane (connector config), exactly like the live path.
    model.headers = Some(IndexMap::from([(
        CONTROL_HEADER.to_string(),
        CONTROL_VALUE.to_string(),
    )]));
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
                control_headers: Some(IndexMap::from([(
                    CONTROL_HEADER.to_string(),
                    CONTROL_VALUE.to_string(),
                )])),
            },
            permit,
        )
        .map_err(|_| "Connector invalid")?,
    );
    let source = match Occ3NativeSession::enroll("source-session", &model, connector.clone()).await
    {
        Ok(v) => v,
        Err(e) => {
            server.abort();
            return Err(e);
        }
    };
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
                    if mode.unrelated {
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
    if mode.queue_receiver {
        let mut state = source_agent.state();
        state.tools = Some(vec![source.handoff_tool(Arc::new(
            QueuedAdmissionFixture {
                inner: family.controller("source-session")?,
                receiver: receiver.clone(),
            },
        ))]);
        source_agent.set_state(state);
    }
    source.arm_source(NativeSourceBudget {
        retained_count_limit: 128,
        max_capture_bytes: 1048576,
        deadline: Instant::now() + Duration::from_secs(20),
        max_response_bytes: 65536,
    })?;
    source_agent.prompt(PromptInput::Text {input:format!("Private source facts: {FACT}. Call occ3_handoff to child receiver. Arguments contain only role and name."),images:Vec::new()}).await.map_err(|_|"Source native prompt failed")?;
    if mode.retry_source_request {
        source_agent
            .prompt(PromptInput::Text {
                input: "Retry the same failed source turn without a new grant".into(),
                images: Vec::new(),
            })
            .await
            .map_err(|_| "Source retry future failed")?;
    }
    let receipts = family.receipts();
    let sender_error = source_agent.state().error_message;
    let source_transcript = serde_json::to_string(&source_agent.state().messages).unwrap();
    let receiver_context = serde_json::to_string(&receiver_agent.state().messages).unwrap();
    assert!(!receiver_context.contains(FACT));
    assert!(!receiver_context.contains("private-source-handle"));
    let source_phase = source.audit();
    let receiver_phase = receiver.audit();
    let cleanup = family.close().await;
    let shutdown = connector.shutdown().await;
    server.abort();
    let requests = records.lock().unwrap().requests.clone();
    let source_receipt = records.lock().unwrap().source_receipt.clone();
    let control_header_seen = records.lock().unwrap().control_header_seen.clone();
    let report = json!({"mode":"offline-synthetic-native-controller","source_requests":requests.iter().filter(|(_,v)|v.get("occ3_capture").is_some()).count(),
        "receiver_requests":requests.iter().filter(|(_,v)|v.get("occ3").is_some()).count(),"total_model_requests":requests.iter().filter(|(p,_)|p=="/v1/chat/completions").count(),"admitted_messages":receipts.len(),"receiver_source_facts_absent":true,
        "source_phase":source_phase,"receiver_phase":receiver_phase,"queue_fixture_only":mode.queue_receiver,
        "cleanup_logical_only":cleanup.is_ok(),"shutdown":shutdown.is_ok(),"gpu_proof":false,"semantic_proof":false});
    Ok(FixtureResult {
        report,
        requests,
        source_receipt,
        sender_error,
        control_header_seen,
        source_transcript,
    })
}
#[tokio::test]
async fn actual_native_source_to_authorized_controller_to_actual_receiver_preserves_budget_and_hides_facts(
) {
    let result = run_offline_fixture(Default::default()).await.unwrap();
    assert_eq!(result.report["admitted_messages"], 1);
    assert_eq!(result.report["source_requests"], 1);
    assert_eq!(result.report["receiver_requests"], 1);
    assert!(result.sender_error.is_none());
    let control: Vec<_> = result
        .requests
        .iter()
        .filter(|(path, _)| path.starts_with("/occ3/"))
        .collect();
    assert!(control.iter().all(|(_, v)| !v.to_string().contains(FACT)));
    assert_eq!(
        control
            .iter()
            .filter(|(p, _)| p.ends_with("/publish"))
            .count(),
        1
    );
    assert_eq!(result.report["cleanup_logical_only"], true);
    // The audit projection must expose the two authenticated span integers next
    // to retained_count: retained_count alone cannot prove the accepted total
    // or rule out an early suffix drop. Values come from the served receipt.
    let served = result
        .source_receipt
        .as_ref()
        .expect("fixture served one authenticated source receipt");
    let audit_receipt = &result.report["source_phase"]["source"]["receipt"];
    assert_eq!(
        audit_receipt["accepted_span_start"], served["accepted_span_start"],
        "audit must expose the authenticated accepted span start"
    );
    assert_eq!(
        audit_receipt["accepted_span_end"], served["accepted_span_end"],
        "audit must expose the authenticated accepted span end"
    );
}
#[tokio::test]
async fn lost_publication_reply_queries_same_key_and_does_not_regenerate_source() {
    let result = run_offline_fixture(FixtureMode {
        lose_publish: true,
        ..Default::default()
    })
    .await
    .unwrap();
    assert_eq!(result.report["admitted_messages"], 1);
    let publishes: Vec<_> = result
        .requests
        .iter()
        .filter(|(p, _)| p.ends_with("/publish"))
        .collect();
    let statuses: Vec<_> = result
        .requests
        .iter()
        .filter(|(p, _)| p.ends_with("/publication/status"))
        .collect();
    assert_eq!(publishes.len(), 1);
    assert_eq!(statuses.len(), 1);
    assert_eq!(
        publishes[0].1["publication_id"],
        statuses[0].1["publication_id"]
    );
    assert_eq!(result.report["source_requests"], 1);
}
#[tokio::test]
async fn unknown_absent_publication_is_not_text_fallback_or_regeneration() {
    let result = run_offline_fixture(FixtureMode {
        absent_publish: true,
        ..Default::default()
    })
    .await
    .unwrap();
    assert_eq!(result.report["admitted_messages"], 0);
    assert_eq!(result.report["receiver_requests"], 0);
    assert_eq!(result.report["source_requests"], 1);
    assert_eq!(
        result
            .requests
            .iter()
            .filter(|(p, _)| p.ends_with("/publish"))
            .count(),
        1
    );
}
#[tokio::test]
async fn genuine_family_authorization_rejects_unrelated_target_before_publish() {
    let result = run_offline_fixture(FixtureMode {
        unrelated: true,
        ..Default::default()
    })
    .await
    .unwrap();
    assert_eq!(result.report["admitted_messages"], 0);
    assert_eq!(result.report["receiver_requests"], 0);
    assert!(!result.requests.iter().any(|(p, _)| p.ends_with("/publish")));
}
#[tokio::test]
async fn invalid_source_digest_and_ambiguous_tools_never_publish() {
    for mode in [
        FixtureMode {
            bad_source_hash: true,
            ..Default::default()
        },
        FixtureMode {
            multiple_tools: true,
            ..Default::default()
        },
    ] {
        let result = run_offline_fixture(mode).await.unwrap();
        assert_eq!(result.report["admitted_messages"], 0);
        assert_eq!(result.report["receiver_requests"], 0);
        assert!(!result.requests.iter().any(|(p, _)| p.ends_with("/publish")));
    }
}
#[tokio::test]
async fn runtime_epoch_mismatch_fails_before_any_model_request() {
    assert!(run_offline_fixture(FixtureMode {
        wrong_epoch: true,
        ..Default::default()
    })
    .await
    .is_err());
}

#[tokio::test]
async fn close_and_shutdown_reuse_retirement_key_even_after_unknown_reply() {
    for lose_first_retire in [false, true] {
        let result = run_offline_fixture(FixtureMode {
            lose_first_retire,
            ..Default::default()
        })
        .await
        .unwrap();
        let retirements: Vec<_> = result
            .requests
            .iter()
            .filter(|(path, _)| path.ends_with("/retire"))
            .collect();
        assert_eq!(retirements.len(), 2);
        assert_eq!(
            retirements[0].1, retirements[1].1,
            "same operation and exact bound body"
        );
        assert_eq!(result.report["cleanup_logical_only"], !lose_first_retire);
        assert_eq!(result.report["shutdown"], true);
        assert_eq!(result.report["source_requests"], 1);
        assert_eq!(result.report["receiver_requests"], 1);
    }
}

/// The transcript's messages with their serialized roles (`"user"`,
/// `"assistant"`, `"toolResult"`; pi-ai types.rs ROLE_* constants). The role
/// field is the documented shape of every serialized AgentMessage; an entry
/// without one fails loudly instead of silently skipping, so a transcript
/// shape change can never fake a passing assertion.
fn transcript_roles(transcript: &str) -> Vec<(String, Value)> {
    let messages: Vec<Value> =
        serde_json::from_str(transcript).expect("serialized agent messages");
    messages
        .into_iter()
        .enumerate()
        .map(|(index, message)| {
            let role = message
                .get("role")
                .and_then(Value::as_str)
                .map(str::to_string)
                .unwrap_or_else(|| {
                    panic!("transcript entry {index} lacks the documented role field: {message}")
                });
            (role, message)
        })
        .collect()
}

/// Index of the explicit retry prompt: the user turn carrying the fixture's
/// fixed retry text (`run_offline_fixture` issues it verbatim).
fn explicit_retry_index(roles: &[(String, Value)]) -> Option<usize> {
    roles.iter().position(|(role, message)| {
        role == "user" && message.to_string().contains("Retry the same failed source turn")
    })
}

/// Index of the first assistant message whose recorded diagnostics carry an
/// occ3 provider failure code in the structured `details` field (a provider
/// diagnostic, not transcript text): the capture-level failure evidence.
fn assistant_occ3_failure_index(roles: &[(String, Value)]) -> Option<usize> {
    roles.iter().position(|(role, message)| {
        role == "assistant"
            && message["diagnostics"].as_array().is_some_and(|diagnostics| {
                diagnostics.iter().any(|diagnostic| {
                    diagnostic["details"]["providerErrorType"]
                        .as_str()
                        .is_some_and(|code| code.starts_with("occ3_"))
                })
            })
    })
}

/// Index of the first error tool result (`"isError": true`) whose text
/// content carries the failure-mode bracket code: the refused handoff
/// call's recorded outcome.
fn tool_error_code_index(roles: &[(String, Value)], code: &str) -> Option<usize> {
    roles.iter().position(|(role, message)| {
        role == "toolResult"
            && message["isError"].as_bool() == Some(true)
            && message["content"].as_array().is_some_and(|blocks| {
                blocks.iter().any(|block| {
                    block["type"] == "text"
                        && block["text"].as_str().is_some_and(|text| text.contains(code))
                })
            })
    })
}

/// Index of the LAST ordinary assistant completion. The captured turn emits
/// no content delta (fixture sse for a capture is tool-only), so every
/// occurrence of the completion text is an ordinary uncaptured turn's
/// output - never the captured turn's.
fn ordinary_completion_index(roles: &[(String, Value)]) -> Option<usize> {
    roles.iter().rposition(|(role, message)| {
        role == "assistant"
            && message["content"].as_array().is_some_and(|blocks| {
                blocks.iter().any(|block| {
                    block["type"] == "text"
                        && block["text"]
                            .as_str()
                            .is_some_and(|text| text.contains("synthetic fixture completion"))
                })
            })
    })
}

/// PR235 contract (upstream #235): a failed armed capture or publication
/// tombstones the slot and counts the failure; an explicitly requested
/// LATER turn then runs ORDINARY and uncaptured instead of bricking the
/// session (the pre-#235 behavior this test used to pin). What stays
/// forbidden - and is asserted per failure mode below - is any automatic
/// REPLAY of the failed attempt: the capture intent never repeats, no
/// second publish or same-key status query follows the failed one, and
/// nothing ever reaches the receiver.
#[tokio::test]
async fn failed_source_capture_or_publication_recovers_as_explicit_ordinary_turn_and_never_replays() {
    for mode in [
        FixtureMode {
            bad_source_hash: true,
            retry_source_request: true,
            ..Default::default()
        },
        FixtureMode {
            multiple_tools: true,
            retry_source_request: true,
            ..Default::default()
        },
        FixtureMode {
            absent_publish: true,
            retry_source_request: true,
            ..Default::default()
        },
        FixtureMode {
            unauthorized_publish: true,
            retry_source_request: true,
            ..Default::default()
        },
        FixtureMode {
            unrelated: true,
            retry_source_request: true,
            ..Default::default()
        },
    ] {
        // Two distinct ordinary-recovery shapes, derived from the failure
        // point (FixtureMode is Clone, not Copy - read the flags before the
        // fixture consumes the mode):
        // - Capture-level failures (bad hash / ambiguous tools) abort the
        //   turn terminally: no tool runs, the loop answers the streamed
        //   calls and stops (agent_loop terminal-failure branch). Exactly
        //   TWO model requests: the capture request plus the explicit
        //   retry turn. No publish, no status query.
        // - Tool-path failures (unrelated target / absent or refused
        //   publish) leave the tool error on the SAME turn, and the loop's
        //   ordinary same-turn continuation follows (one more uncaptured
        //   request - ordinary continuation, NOT native replay), then the
        //   explicit retry turn adds its own. Exactly THREE model requests.
        //   Publish-failure modes carry exactly one publish and the one
        //   same-key status query of the failed attempt; an unrelated
        //   target is refused before any publish.
        let capture_level_failure = mode.bad_source_hash || mode.multiple_tools;
        let unrelated_target = mode.unrelated;
        let (want_total, want_publishes, want_statuses): (u64, usize, usize) =
            if capture_level_failure {
                (2, 0, 0)
            } else if unrelated_target {
                (3, 0, 0)
            } else {
                (3, 1, 1)
            };
        let result = run_offline_fixture(mode).await.unwrap();
        let roles = transcript_roles(&result.source_transcript);
        // The failed attempt is durably recorded BEFORE the explicit retry
        // prompt exists: capture-level failures record an assistant
        // provider-failure diagnostic; tool-path failures record the refused
        // handoff call's error tool result (isError with the bracket code).
        // These are recorded events, not loose transcript substrings.
        let retry_index = explicit_retry_index(&roles)
            .expect("the fixture issues the explicit retry prompt");
        let failure_index = assistant_occ3_failure_index(&roles)
            .or_else(|| {
                let code = if unrelated_target {
                    "[occ3_tool_authorize"
                } else {
                    "[occ3_tool_publish"
                };
                tool_error_code_index(&roles, code)
            })
            .expect("the first turn's failure is durably recorded on the transcript");
        assert!(
            failure_index < retry_index,
            "the failed attempt predates the explicit retry prompt (indexes {failure_index} < {retry_index})"
        );
        // Exactly one capture-intent request EVER, and it is the first
        // model request: every later request - the same-turn ordinary
        // continuation after a failed tool call and the explicit retry
        // turn - runs uncaptured. The tombstone never re-arms the capture.
        let capture_positions: Vec<usize> = result
            .requests
            .iter()
            .filter(|(path, _)| path == "/v1/chat/completions")
            .enumerate()
            .filter(|(_, (_, body))| body.get("occ3_capture").is_some())
            .map(|(index, _)| index)
            .collect();
        assert_eq!(
            capture_positions,
            vec![0usize],
            "only the first model request carries the capture intent; every later request is ordinary and uncaptured"
        );
        assert_eq!(result.report["source_requests"], 1);
        // The explicitly requested retry turn completed ordinarily AFTER
        // the retry prompt: the captured turn emits no content delta, so
        // the last recorded completion can only be an ordinary turn's.
        let completion_index = ordinary_completion_index(&roles)
            .expect("an ordinary uncaptured completion is recorded");
        assert!(
            retry_index < completion_index,
            "the explicit retry turn completed ordinarily (indexes {retry_index} < {completion_index})"
        );
        // The exact per-mode bound: no unbounded ordinary retries. The
        // fixture's plain response carries no tools and no capture, so the
        // loop cannot continue past the retry turn's completion.
        assert_eq!(
            result.report["total_model_requests"], want_total,
            "exactly the capture request plus the derived ordinary turns"
        );
        // No receiver-side consumption of anything.
        assert_eq!(result.report["receiver_requests"], 0);
        assert_eq!(result.report["admitted_messages"], 0);
        // No publish or status query beyond the failed attempt itself: the
        // same-key status query of a lost/refused publish fires exactly once
        // (the connector's publication recovery), and nothing follows the
        // failure.
        let publishes = result
            .requests
            .iter()
            .filter(|(p, _)| p.ends_with("/publish"))
            .count();
        let statuses = result
            .requests
            .iter()
            .filter(|(p, _)| p.ends_with("/publication/status"))
            .count();
        assert_eq!(publishes, want_publishes, "publish count is exactly the failed attempt's");
        assert_eq!(
            statuses, want_statuses,
            "status queries are exactly the same-key query of the failed publish"
        );
    }
}

#[tokio::test]
async fn gateway_control_header_rides_control_and_inference_with_the_same_value() {
    let result = run_offline_fixture(Default::default()).await.unwrap();
    assert!(!result.control_header_seen.is_empty());
    let control: Vec<_> = result
        .control_header_seen
        .iter()
        .filter(|(path, _)| path.starts_with("/occ3/"))
        .collect();
    let inference: Vec<_> = result
        .control_header_seen
        .iter()
        .filter(|(path, _)| path == "/v1/chat/completions")
        .collect();
    assert!(!control.is_empty(), "control-plane requests were exercised");
    assert!(!inference.is_empty(), "inference requests were exercised");
    for (_, value) in &result.control_header_seen {
        assert_eq!(value.as_deref(), Some(CONTROL_VALUE));
    }
    // The value is never serialized into the report or any evidence.
    assert!(!result.report.to_string().contains(CONTROL_VALUE));
    assert!(!result
        .requests
        .iter()
        .any(|(_, body)| body.to_string().contains(CONTROL_VALUE)));
    // Budget stays bound: default 131072 requested, ceiling and context unchanged.
    assert_eq!(result.report["receiver_requests"], 1);
}

#[tokio::test]
async fn mid_attempt_401_is_failed_not_negative_proof_and_never_replays() {
    let result = run_offline_fixture(FixtureMode {
        unauthorized_publish: true,
        ..Default::default()
    })
    .await
    .unwrap();
    // Exactly one publish and one same-key status query; no second publish,
    // no credential refresh, no source regeneration, no receiver inference.
    let publishes: Vec<_> = result
        .requests
        .iter()
        .filter(|(p, _)| p.ends_with("/publish"))
        .collect();
    let statuses: Vec<_> = result
        .requests
        .iter()
        .filter(|(p, _)| p.ends_with("/publication/status"))
        .collect();
    assert_eq!(publishes.len(), 1);
    assert_eq!(statuses.len(), 1);
    assert_eq!(
        publishes[0].1["publication_id"],
        statuses[0].1["publication_id"]
    );
    assert_eq!(result.report["admitted_messages"], 0);
    assert_eq!(result.report["receiver_requests"], 0);
    assert_eq!(result.report["source_requests"], 1);
    // The 401 attempt fails closed INSIDE the first turn; the agent loop's
    // ordinary continuation after the failed tool call is one more
    // UNCAPPED model request (PR235 recovery, in-turn). That continuation
    // is not a replay: the capture intent never repeats, no second publish
    // or status query follows, and nothing reaches the receiver.
    assert_eq!(
        result.report["total_model_requests"], 2,
        "the capture turn plus exactly one ordinary uncaptured continuation"
    );
    let capture_positions: Vec<usize> = result
        .requests
        .iter()
        .filter(|(path, _)| path == "/v1/chat/completions")
        .enumerate()
        .filter(|(_, (_, body))| body.get("occ3_capture").is_some())
        .map(|(index, _)| index)
        .collect();
    assert_eq!(
        capture_positions,
        vec![0usize],
        "only the first model request carries the capture intent"
    );
    // No explicit retry prompt exists in this mode (retry_source_request
    // is false): the second request is the SAME-TURN ordinary continuation
    // after the failed tool call, not an explicit retry turn and not a
    // native replay.
    let roles = transcript_roles(&result.source_transcript);
    assert!(
        explicit_retry_index(&roles).is_none(),
        "this mode issues no retry prompt; the extra request is the same-turn continuation"
    );
    // The 401 failure is durably recorded as the failed handoff call's
    // error tool result (isError with the bracket code), and the ordinary
    // same-turn continuation completes after it.
    let failure_index = tool_error_code_index(&roles, "[occ3_tool_publish")
        .expect("the failed publish is the recorded error tool result");
    let completion_index = ordinary_completion_index(&roles)
        .expect("the ordinary same-turn continuation completed");
    assert!(
        failure_index < completion_index,
        "the same-turn continuation completed after the failed publish (indexes {failure_index} < {completion_index})"
    );
    // The gateway header value never leaks into the report.
    assert!(!result.report.to_string().contains(CONTROL_VALUE));
    // The refused control requests still carried the resolved header.
    for (path, value) in &result.control_header_seen {
        if path.ends_with("/publish") || path.ends_with("/publication/status") {
            assert_eq!(value.as_deref(), Some(CONTROL_VALUE));
        }
    }
}

#[tokio::test]
async fn queued_admission_allows_source_continuation_but_is_not_receiver_consumption() {
    let queued = run_offline_fixture(FixtureMode {
        queue_receiver: true,
        ..Default::default()
    })
    .await
    .unwrap();
    assert_eq!(queued.report["source_phase"]["source"]["state"], "Captured");
    assert_eq!(
        queued.report["source_phase"]["source"]["handoff_admitted"],
        true
    );
    assert_eq!(
        queued.report["receiver_phase"]["receiver"][0]["status"],
        "ADMITTED"
    );
    assert!(queued.report["receiver_phase"]["receiver"][0]["lease_state"].is_null());
    assert_eq!(queued.report["receiver_requests"], 0);
    assert_eq!(queued.report["source_requests"], 1);
    assert_eq!(
        queued.report["total_model_requests"], 2,
        "only ordinary source continuation after known queue admission"
    );
    assert_eq!(
        queued.report["admitted_messages"], 0,
        "isolated consumed-receipt journal is not fabricated"
    );
    assert!(queued.sender_error.is_none());
    let consumed = run_offline_fixture(Default::default()).await.unwrap();
    assert_eq!(
        consumed.report["receiver_phase"]["receiver"][0]["lease_state"],
        "Consumed"
    );
    assert_eq!(consumed.report["receiver_requests"], 1);
    assert_eq!(consumed.report["admitted_messages"], 1);
}
