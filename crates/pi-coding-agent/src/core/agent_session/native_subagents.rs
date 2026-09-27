//! Language-independent access to the session-owned RLM child lifecycle.
use super::*;

impl AgentSession {
    pub(in crate::core::agent_session) fn native_subagent_tool(
        self: &Arc<Self>,
    ) -> crate::core::extensions::types::ToolDefinition {
        let weak = Arc::downgrade(self);
        crate::core::tools::ToolDefinition::<Value> {
            name: "subagent".into(),
            label: "Subagent".into(),
            description: "Delegate to native child agents. spawn admits a child immediately and returns its id; children run concurrently and report completion to this chat. list shows direct children. collect returns status and short answer previews, optionally waiting up to 60 seconds. Children inherit this execution mode, working directory and tool restrictions. No Python workspace or CLI subprocess is needed.".into(),
            parameters: serde_json::json!({
                "type":"object", "additionalProperties":false,
                "properties":{
                    "action":{"type":"string","enum":["spawn","list","collect"]},
                    "prompt":{"type":"string","description":"Required for spawn: a self-contained task, scope and expected result."},
                    "name":{"type":"string","description":"Optional unique child name for spawn."},
                    "model":{"type":"string","description":"Optional provider/model selector for spawn; defaults to the parent model."},
                    "thinking":{"type":"string","description":"Optional thinking level for spawn."},
                    "targets":{"type":"array","items":{"type":"string"},"description":"Child ids or names for collect; omit to collect all direct children."},
                    "timeout_ms":{"type":"integer","minimum":0,"maximum":60000,"description":"collect wait, default 0; reaching it does not cancel children."}
                }, "required":["action"]
            }),
            execution_mode: Some(pi_agent_core::types::ToolExecutionMode::Sequential),
            execute: Arc::new(move |_, args, signal, _, _| {
                let weak = weak.clone();
                Box::pin(async move {
                    let session = weak.upgrade().ok_or_else(|| anyhow::anyhow!("Parent session disposed"))?;
                    let signal = signal.unwrap_or_default();
                    if signal.is_cancelled() { return Err(anyhow::anyhow!("Subagent call cancelled")); }
                    let action = args.get("action").and_then(Value::as_str).unwrap_or("");
                    let allowed: &[&str] = match action {
                        "spawn" => &["action", "prompt", "name", "model", "thinking"],
                        "list" => &["action"],
                        "collect" => &["action", "targets", "timeout_ms"],
                        _ => return Err(anyhow::anyhow!("action must be spawn, list or collect")),
                    };
                    if let Some(key) = args.as_object().and_then(|object| object.keys().find(|key| !allowed.contains(&key.as_str()))) {
                        return Err(anyhow::anyhow!("{key} is not supported for subagent {action}"));
                    }
                    let work = async {
                        match action {
                            "spawn" => {
                                let prompt = args.get("prompt").and_then(Value::as_str).filter(|s| !s.trim().is_empty())
                                    .ok_or_else(|| anyhow::anyhow!("spawn requires a non-empty prompt"))?;
                                let kwargs = ["name", "model", "thinking"].into_iter()
                                    .filter_map(|key| args.get(key).map(|value| (key.into(), value.clone())))
                                    .collect();
                                Ok(serde_json::to_value(session.start_rlm_child_run(prompt, &kwargs, None).await.map_err(anyhow::Error::msg)?)?)
                            }
                            "list" => Ok(serde_json::to_value(session.list_rlm_subagents().await.map_err(anyhow::Error::msg)?)?),
                            _ => {
                                let targets = match args.get("targets") {
                                    None => Vec::new(),
                                    Some(Value::Array(values)) => values.iter().map(|value| value.as_str().filter(|s| !s.trim().is_empty()).map(str::to_string)
                                        .ok_or_else(|| anyhow::anyhow!("targets must contain non-empty child ids or names"))).collect::<Result<Vec<_>, _>>()?,
                                    _ => return Err(anyhow::anyhow!("targets must be an array")),
                                };
                                let timeout = match args.get("timeout_ms") {
                                    None => 0,
                                    Some(value) => value.as_u64().filter(|n| *n <= 60_000)
                                        .ok_or_else(|| anyhow::anyhow!("timeout_ms must be an integer between 0 and 60000"))?,
                                };
                                Ok(serde_json::to_value(session.collect_rlm_children(&targets, timeout).await.map_err(anyhow::Error::msg)?)?)
                            }
                        }
                    };
                    let value: Value = tokio::select! {
                        biased;
                        _ = signal.cancelled() => return Err(anyhow::anyhow!("Subagent call cancelled")),
                        value = work => value?,
                    };
                    Ok(crate::core::tools::tool_definition_wrapper::text_tool_result(serde_json::to_string(&value)?, value))
                })
            }),
            ..Default::default()
        }.into()
    }
}
