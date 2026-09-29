use super::*;

fn response(body: &Value) -> (Value, &'static str) {
    let messages = body["messages"].as_array().unwrap();
    let last = messages.last().unwrap();
    if last["role"] == "user" {
        let text = last["content"]
            .as_str()
            .map(str::to_owned)
            .unwrap_or_else(|| {
                last["content"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(|block| block["text"].as_str())
                    .collect::<Vec<_>>()
                    .join("\n")
            });
        if let Some(args) = text.strip_prefix("HEARTBEAT_CALL ") {
            return (
                json!({"role":"assistant", "tool_calls":[{
                    "index":0,"id":format!("heartbeat-{}", messages.len()),"type":"function",
                    "function":{"name":"heartbeat","arguments":args}
                }]}),
                "tool_calls",
            );
        }
    }
    (json!({"role":"assistant", "content":REPLY}), "stop")
}

async fn request(client: &Arc<DaemonClient>, body: Value) -> Value {
    let result = client
        .request(
            body.as_object().unwrap().clone(),
            Some(30_000),
            Default::default(),
        )
        .await
        .unwrap();
    assert!(result.success, "{:?}", result.error);
    result.data.unwrap_or(Value::Null)
}

async fn create(client: &Arc<DaemonClient>, fixture: &PrivateCli, no_tools: bool) -> String {
    let value = request(
        client,
        json!({"type":"create", "config":{
            "cwd":fixture.workspace, "agentDir":fixture.agent_dir,
            "sessionDir":fixture.root.path().join("sessions"),
            "provider":"local-cli-fixture", "model":"fixture-model",
            "noSkills":true, "noExtensions":true, "noTools":no_tools
        }}),
    )
    .await;
    value
        .get("activeSessionId")
        .or_else(|| value.get("id"))
        .unwrap()
        .as_str()
        .unwrap()
        .into()
}

async fn call(client: &Arc<DaemonClient>, active: &str, args: Value, error: bool) -> Value {
    request(
        client,
        json!({"type":"prompt_and_wait", "activeSessionId":active, "streamingBehavior":"followUp",
        "message":format!("HEARTBEAT_CALL {args}")}),
    )
    .await;
    let state = request(
        client,
        json!({"type":"get_connection_state", "activeSessionId":active}),
    )
    .await;
    let entries: Vec<Value> = fs::read_to_string(state["sessionFile"].as_str().unwrap())
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let result = entries
        .iter()
        .rev()
        .map(|entry| &entry["message"])
        .find(|message| message["role"] == "toolResult" && message["toolName"] == "heartbeat")
        .unwrap_or_else(|| {
            panic!("native heartbeat result must be persisted for {args}: {entries:?}")
        });
    assert_eq!(result["isError"], error, "{result}");
    result["details"].clone()
}

#[test]
fn native_heartbeat_lifecycle_in_every_execution_mode() {
    let mut model = LocalModel::with_response(response);
    let fixture = PrivateCli::new(&format!("http://{}/v1", model.address));
    let mut daemon = OwnedDaemon {
        process: fixture.spawn("heartbeat-daemon", &["--mode", "daemon", "--offline"]),
        socket: fixture.socket.clone(),
    };
    wait_for_daemon(&mut daemon);
    tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(async {
        let client = DaemonClient::create(&fixture.socket);
        client.connect(1000).await.unwrap();
        client.wait_for_hello(1000).await.unwrap();
        let active = create(&client, &fixture, false).await;
        let other = create(&client, &fixture, false).await;
        // This is the daemon command used by the TUI's /heartbeat handler.
        request(&client, json!({"type":"heartbeat_set", "activeSessionId":active,
            "schedule":"every 30m", "deliveryMode":"follow_up", "prompt":"USER_HEARTBEAT_UNCHANGED"})).await;
        let jobs = request(&client, json!({"type":"cron_list", "activeSessionId":active})).await;
        let user_job = jobs["jobs"].as_array().unwrap().first().unwrap().clone();
        call(&client, &active, json!({"action":"delete", "id":user_job["id"]}), true).await;
        for mode in ["direct", "node", "clang", "ipython"] {
            request(&client, json!({"type":"prompt_and_wait", "activeSessionId":active,
                "message":format!("/mode {mode}")})).await;
            let state = request(&client, json!({"type":"get_connection_state", "activeSessionId":active})).await;
            assert!(state["activeToolNames"].as_array().unwrap().contains(&json!("heartbeat")));
            let created = call(&client, &active, json!({"action":"create", "instruction":"Inspect finished reviewers",
                "label":mode, "delivery_mode":"follow_up"}), false).await;
            let job = &created["heartbeat"];
            assert_eq!(job["status"], "active");
            assert_eq!(job["delivery_mode"], "follow_up");
            assert!(job["next_run_at"].is_string(), "{job}");
            let id = job["id"].as_str().unwrap();
            assert_eq!(call(&client, &active, json!({"action":"list"}), false).await["heartbeats"].as_array().unwrap().len(), 1);
            // Another session must neither see nor cancel this job.
            assert_eq!(call(&client, &other, json!({"action":"list"}), false).await["heartbeats"], json!([]));
            call(&client, &other, json!({"action":"delete", "id":id}), true).await;
            assert_eq!(call(&client, &active, json!({"action":"update", "id":id, "status":"pause"}), false).await["heartbeat"]["status"], "paused");
            assert_eq!(call(&client, &active, json!({"action":"update", "id":id, "status":"resume", "interval":"10m"}), false).await["heartbeat"]["status"], "active");
            assert_eq!(call(&client, &active, json!({"action":"delete", "id":id}), false).await["heartbeat"]["status"], "cancelled");
            assert_eq!(call(&client, &active, json!({"action":"list"}), false).await["heartbeats"], json!([]));
        }
        for args in [
            json!({"action":"create", "instruction":"   "}),
            json!({"action":"create", "instruction":"check", "interval":"1s"}),
            json!({"action":"create", "instruction":"check", "interval":"in 5m"}),
            json!({"action":"list", "instruction":"ignored field"}),
            json!({"action":"update", "id":"missing"}),
        ] { call(&client, &active, args, true).await; }
        // Invalid input must not kill the worker or create a hidden job.
        assert_eq!(call(&client, &active, json!({"action":"list"}), false).await["heartbeats"], json!([]));
        assert_eq!(call(&client, &active, json!({"action":"list", "include_inactive":true}), false).await["heartbeats"].as_array().unwrap().len(), 4);
        let jobs = request(&client, json!({"type":"cron_list", "activeSessionId":active})).await;
        assert_eq!(jobs["jobs"], json!([user_job]));
        let timer = call(&client, &active, json!({"action":"create", "interval":"10s",
            "instruction":"REAL_HEARTBEAT_TICK", "delivery_mode":"follow_up"}), false).await;
        request(&client, json!({"type":"prompt_and_wait", "activeSessionId":active,
            "message":"/mode node"})).await;
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            let delivered = model.requests.lock().unwrap().iter().any(|body| {
                let last = body["messages"].as_array().unwrap().last().unwrap();
                let content = last["content"].to_string();
                last["role"] == "user" && content.contains("REAL_HEARTBEAT_TICK")
                    && !content.contains("HEARTBEAT_CALL")
            });
            if delivered { break; }
            assert!(Instant::now() < deadline, "due heartbeat never reached the model");
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        let listed = call(&client, &active, json!({"action":"list"}), false).await;
        assert!(listed["heartbeats"][0]["run_count"].as_f64().unwrap() >= 1.0);
        call(&client, &active, json!({"action":"delete", "id":timer["heartbeat"]["id"]}), false).await;
        let no_tools = create(&client, &fixture, true).await;
        let state = request(&client, json!({"type":"get_connection_state", "activeSessionId":no_tools})).await;
        assert_eq!(state["activeToolNames"], json!([]));
        request(&client, json!({"type":"prompt_and_wait", "activeSessionId":no_tools,"message":"NO_TOOLS_CHECK"})).await;
        client.close().await;
    });
    let requests = model.requests.lock().unwrap();
    for request in requests.iter() {
        let enabled = request["tools"].as_array().is_some_and(|tools| {
            tools
                .iter()
                .any(|tool| tool["function"]["name"] == "heartbeat")
        });
        assert_eq!(
            request["messages"][0]["content"]
                .to_string()
                .contains("# Recurring Agent Heartbeats"),
            enabled
        );
    }
    drop(requests);
    drop(daemon);
    model.finish();
}
