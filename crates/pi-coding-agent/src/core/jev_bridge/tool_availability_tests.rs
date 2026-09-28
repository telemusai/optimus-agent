// Included in the bridge's environment-serialized test module.
struct AvailabilityEnv(Option<std::ffi::OsString>);
impl Drop for AvailabilityEnv {
    fn drop(&mut self) {
        match self.0.take() {
            Some(value) => std::env::set_var(crate::config::env_agent_dir(), value),
            None => std::env::remove_var(crate::config::env_agent_dir()),
        }
        invalidate_settings_cache();
    }
}

struct AvailabilitySession;
impl crate::core::extensions::types::ReadonlySessionManager for AvailabilitySession {
    fn get_session_id(&self) -> String { "jev-availability-fixture".into() }
    fn get_session_file(&self) -> Option<String> { None }
    fn get_session_dir(&self) -> String { String::new() }
    fn get_branch(&self) -> Vec<crate::core::extensions::types::SessionEntry> { Vec::new() }
}
impl crate::core::extensions::types::SessionManager for AvailabilitySession {}

fn availability_fixture(dir: &Path) -> (
    AvailabilityEnv, Arc<JevBridgeCore>, Arc<crate::core::extensions::runner::ExtensionRunner>,
    Arc<pi_jev::mock::MockJevTransport>, SharedExtension,
) {
    use crate::core::extensions::runner::{ExtensionRunner, NullModelRegistry};
    use crate::core::extensions::types::ExtensionRuntime;
    use pi_jev::mock::{MockJevTransport, MockStep};
    crate::modes::interactive::theme::theme::init_theme(Some("neon"), false);
    let guard = AvailabilityEnv(std::env::var_os(crate::config::env_agent_dir()));
    std::env::set_var(crate::config::env_agent_dir(), dir);
    let mut settings = JevSettings::default();
    settings.global_default = Some(JevMode::Active);
    settings.transport = Some("mock".into());
    settings.features.complexity = false;
    JevSettingsStore::new(dir).save(&settings).unwrap();
    invalidate_settings_cache();
    let settings = JevSettingsStore::new(dir).load();
    let transport = Arc::new(MockJevTransport::scripted(vec![MockStep::Body(json!({
        "model":"synthetic-jev", "answers":{"tool_requirement.0":{
            "type":"choice","choice":"none","confidence":0.83,
            "probabilities":{"none":0.83,"read":0.0,"search":0.0,"shell":0.0,"python":0.0,"delegate":0.0,"multiple":0.17}
        }}
    }).to_string())]));
    let client = Arc::new(pi_jev::JevSystemOne::new(JevMode::Active,
        pi_jev::SecretString::new("synthetic-fixture-only"), transport.clone(),
        pi_jev::JevLimits { max_retries: 0, ..Default::default() }, Arc::new(pi_jev::JevStats::default())).unwrap());
    let store_dir = dir.to_path_buf();
    let observer = JevObserver::new(pi_jev::hooks::JevObserverConfig {
        enabled_categories: ["tool_requirement".to_string()].into_iter().collect(),
        authoritative_gate: Some(Arc::new(move |session, _| {
            let current = JevSettingsStore::new(&store_dir).load();
            let session = session.unwrap_or("");
            pi_jev::hooks::JevRequestGate {
                mode: current.effective_mode(session),
                requested_model: current.requested_model_or_default().into(),
                policy_generation: decision_policy_generation(&current, session),
                allowed: current.effective_mode(session).is_enabled(),
            }
        })),
        ..Default::default()
    }, client, dir.join("records.jsonl"));
    let core = Arc::new(JevBridgeCore::new(settings.clone()));
    core.own_session("jev-availability-fixture");
    *core.observer.lock().unwrap() = Some(ObserverBuild { cheap_stamp: cheap_credential_stamp(&settings), observer });
    live_bridges().lock().unwrap().push(Arc::downgrade(&core));
    let extension = build_observer_extension(core.clone());
    core.attach_extension(&extension);
    core.set_active_handler(&core, true);
    let runner = Arc::new(ExtensionRunner::new(vec![extension.clone()], ExtensionRuntime::new(Default::default()),
        dir.to_string_lossy().into_owned(), Arc::new(AvailabilitySession), Arc::new(NullModelRegistry)));
    (guard, core, runner, transport, extension)
}

#[tokio::test]
async fn advisory_none_real_observer_handler_runner_retains_modes_and_short_followups() {
    use crate::core::execution_mode::ExecutionMode;
    let _lock = env_lock().lock().unwrap_or_else(|p| p.into_inner());
    let dir = tempfile::tempdir().unwrap();
    let (_guard, core, runner, transport, _extension) = availability_fixture(dir.path());
    let mut turn = 0;
    for mode in [ExecutionMode::Ipython, ExecutionMode::Direct, ExecutionMode::Node, ExecutionMode::Clang] {
        for dynamic in [false, true] {
            let mut names = mode.tools(&[]);
            if dynamic { names.push("jev_decide".into()); }
            let tools: Vec<_> = names.iter().map(|name| json!({"type":"function","name":name,"description":"keep","parameters":{"type":"object"}})).collect();
            for text in ["ok", "continue", "status"] {
                turn += 1;
                core.note_turn("jev-availability-fixture", turn);
                core.remember_task_text("jev-availability-fixture", text);
                let body = json!({"tools":tools,"tool_choice":"auto","input":[{"role":"user","content":text}]});
                let outgoing = runner.emit_before_provider_request(body.clone()).await;
                assert_eq!(serde_json::to_vec(&outgoing).unwrap(),serde_json::to_vec(&body).unwrap());
                let diagnostic = core.tool_diagnostics("jev-availability-fixture");
                assert_eq!(diagnostic["lastAdvertisedTools"],json!(names));
                assert_eq!(diagnostic["advertisedAtTurn"],turn);
                assert_eq!(diagnostic["toolStateScope"],"local_post_hook_catalog_not_provider_receipt");
            }
        }
    }
    assert_eq!(transport.call_count(),24,"real accepted decisions, not a disabled hook");
    let rows: Vec<Value> = std::fs::read_to_string(dir.path().join("records.jsonl")).unwrap().lines()
        .map(|line|serde_json::from_str::<Value>(line).unwrap())
        .filter(|row| row["category"] == "tool_requirement").collect();
    assert_eq!(rows.len(),24);
    for row in rows {
        assert_eq!(row["selected_value"],"none");
        assert_eq!(row["confidence"],0.83);
        assert_eq!(row["acceptance"],"accepted");
        assert_eq!(row["outcome"],"accepted_no_effect");
        assert_eq!(row["applied"],false);
        assert_eq!(row["baseline_action"],row["actual_action"]);
        assert_eq!(row["actual_action"]["tool_state_scope"],"local_jev_hook_not_provider_receipt");
    }
    for mode in [JevMode::Off,JevMode::Compare,JevMode::CompareAndActive] {
        let store = JevSettingsStore::new(dir.path());
        let mut settings = store.load(); settings.global_default = Some(mode); store.save(&settings).unwrap();
        invalidate_settings_cache();
        let before = transport.call_count();
        let body = json!({"tools":[{"type":"function","name":"ipython"},{"type":"function","name":"jev_decide"}],"tool_choice":"none"});
        assert_eq!(runner.emit_before_provider_request(body.clone()).await,body);
        assert_eq!(transport.call_count()-before,usize::from(mode.allows_active()));
    }
}

#[tokio::test]
async fn final_advertisement_observes_later_hook_not_jev_baseline() {
    let _lock = env_lock().lock().unwrap_or_else(|p| p.into_inner());
    let dir = tempfile::tempdir().unwrap();
    let (_guard, core, runner, transport, extension) = availability_fixture(dir.path());
    core.note_turn("jev-availability-fixture",7);
    core.remember_task_text("jev-availability-fixture","status");
    extension.lock().unwrap().handlers.get_mut(JEV_ACTIVE_EVENT).unwrap().push(Arc::new(|event, _| Box::pin(async move {
        let ExtensionEvent::BeforeProviderRequest(mut payload) = event else { return None; };
        payload.payload["tools"] = json!([{"type":"function","name":"jev_decide"}]);
        payload.payload["tool_choice"] = json!("none");
        Some(payload.payload)
    })));
    let outgoing = runner.emit_before_provider_request(json!({"tools":[{"type":"function","name":"ipython"},{"type":"function","name":"jev_decide"}],"tool_choice":"auto"})).await;
    assert_eq!(transport.call_count(),1);
    assert_eq!(outgoing["tool_choice"],"none","final observer must not recreate executors or weaken bans");
    let diagnostic = core.tool_diagnostics("jev-availability-fixture");
    assert_eq!(diagnostic["lastAdvertisedTools"],json!(["jev_decide"]));
    assert_eq!(diagnostic["recentRequestsRetainingTools"],"1/1","one final observation per request");
    let row = std::fs::read_to_string(dir.path().join("records.jsonl")).unwrap().lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .find(|row| row["category"] == "tool_requirement").unwrap();
    assert_eq!(row["baseline_action"]["tools"],"count:2");
    assert_eq!(row["actual_action"]["tools"],"count:2","Jev ledger is scoped before the later hook");
}

#[tokio::test]
async fn advisory_none_codex_sse_wire_preserves_actual_executor_catalog() {
    use base64::Engine;
    use tokio::io::{AsyncReadExt,AsyncWriteExt};
    use pi_ai::providers::openai_codex_responses::{stream_openai_codex_responses,OpenAICodexResponsesOptions};
    use pi_ai::types::{Context,Model,StreamOptions,Tool,Message,UserMessage,UserContent};
    use crate::core::execution_mode::ExecutionMode;
    let _lock = env_lock().lock().unwrap_or_else(|p| p.into_inner());
    let dir = tempfile::tempdir().unwrap();
    let (_guard, core, runner, transport, _extension) = availability_fixture(dir.path());
    for mode in [ExecutionMode::Ipython,ExecutionMode::Direct] {
        let listener = tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST,0)).await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            tokio::time::timeout(Duration::from_secs(10),async move {
                let (mut socket,_) = listener.accept().await.unwrap();
                let mut bytes = Vec::new(); let mut buffer = [0u8;4096];
                let (offset,length) = loop {
                    let count = socket.read(&mut buffer).await.unwrap(); assert!(count>0);
                    bytes.extend_from_slice(&buffer[..count]); assert!(bytes.len()<65536);
                    if let Some(end) = bytes.windows(4).position(|part|part==b"\r\n\r\n") {
                        let headers = String::from_utf8_lossy(&bytes[..end]).to_ascii_lowercase();
                        let length: usize = headers.lines().find_map(|line|line.strip_prefix("content-length:")).unwrap().trim().parse().unwrap();
                        break (end+4,length);
                    }
                };
                while bytes.len()<offset+length {
                    let count=socket.read(&mut buffer).await.unwrap(); assert!(count>0);
                    bytes.extend_from_slice(&buffer[..count]); assert!(bytes.len()<65536);
                }
                let body: Value = serde_json::from_slice(&bytes[offset..offset+length]).unwrap();
                let data = format!("data: {}\n\ndata: {}\n\n",
                    json!({"type":"response.created","response":{"id":"fixture"}}),
                    json!({"type":"response.completed","response":{"id":"fixture","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":0,"total_tokens":1}}}));
                socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",data.len(),data).as_bytes()).await.unwrap();
                body
            }).await.expect("bounded loopback fixture")
        });
        core.remember_task_text("jev-availability-fixture","continue");
        let mut names = mode.tools(&[]); names.push("jev_decide".into());
        let context = Context::new(Some("unchanged system".into()),vec![Message::user(UserMessage::new(UserContent::Text("continue".into()),1))],
            Some(names.iter().map(|name|Tool { name:name.clone(),description:"unchanged description".into(),parameters:json!({"type":"object","properties":{}}) }).collect()));
        let model = Model::new("synthetic-codex","Fixture","openai-codex-responses","openai-codex",format!("http://{address}"));
        let token = format!("aaa.{}.bbb",base64::engine::general_purpose::STANDARD.encode(br#"{"https://api.openai.com/auth":{"chatgpt_account_id":"synthetic-only"}}"#));
        let expected = Arc::new(Mutex::new(None));
        let options = OpenAICodexResponsesOptions { stream: StreamOptions {
            api_key:Some(token),transport:Some("sse".into()),timeout_ms:Some(5000.0),
            on_payload:Some(Arc::new({let runner=runner.clone();let expected=expected.clone();move |body,_| {
                let runner=runner.clone();let expected=expected.clone();Box::pin(async move {
                    *expected.lock().unwrap()=Some(body.clone());
                    Some(runner.emit_before_provider_request(body).await)
                })
            }})),..Default::default()
        },..Default::default() };
        let result = tokio::time::timeout(Duration::from_secs(10),stream_openai_codex_responses(&model,&context,Some(options)).result()).await.unwrap();
        assert_ne!(result.stop_reason,"error","{:?}",result.error_message);
        let wire = server.await.unwrap();
        assert_eq!(wire,expected.lock().unwrap().clone().unwrap(),"actual serialized provider body must retain the full pre-Jev payload");
        assert_eq!(advertised_tool_names(&wire),names);
        assert_eq!(core.tool_diagnostics("jev-availability-fixture")["lastAdvertisedTools"],json!(names));
    }
    assert_eq!(transport.call_count(),2,"the actual provider called the accepted-none handler");
}
