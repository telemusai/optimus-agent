//! Execution-language-independent access to the existing session scheduler.
use super::*;
use crate::core::cron_jobs::{
    normalize_heartbeat_schedule, parse_agent_cron_schedule, SCHEDULE_ONCE,
};

impl AgentSession {
    pub(in crate::core::agent_session) fn native_heartbeat_tool(
        self: &Arc<Self>,
    ) -> crate::core::extensions::types::ToolDefinition {
        let weak = Arc::downgrade(self);
        crate::core::tools::ToolDefinition::<Value> {
            name: "heartbeat".into(),
            label: "Heartbeat".into(),
            description: "Manage real recurring agent-owned prompts in this session: create, list, update or delete. Use for requested periodic progress checks, including subagent supervision. Runs through the Optimus daemon scheduler in every execution mode; no Python kernel or shell timer is needed. Default interval is 5m; minimum 10s. Only this session's agent-owned heartbeats are accessible; the user's /heartbeat is separate. Confirm success from the returned id, status and next_run_at, never from an intention to check later.".into(),
            parameters: serde_json::json!({
                "type":"object", "additionalProperties":false,
                "properties":{
                    "action":{"type":"string","enum":["create","list","update","delete"]},
                    "instruction":{"type":"string","minLength":1,"description":"Required for create. Specific work to perform at each heartbeat."},
                    "interval":{"type":"string","description":"Recurring interval, e.g. 5m or 30s; defaults to 5m on create."},
                    "label":{"type":"string","description":"Short name for this heartbeat."},
                    "id":{"type":"string","minLength":1,"description":"Required for update/delete: id returned by create/list."},
                    "status":{"type":"string","enum":["pause","resume"],"description":"For update only."},
                    "delivery_mode":{"type":"string","enum":["steer","follow_up"],"description":"steer (default) interrupts the current turn; follow_up waits until the current turn finishes."},
                    "include_inactive":{"type":"boolean","description":"For list only; default false lists active and paused heartbeats."}
                }, "required":["action"]
            }),
            execution_mode: Some(pi_agent_core::types::ToolExecutionMode::Sequential),
            execute: Arc::new(move |_, args, signal, _, _| {
                let weak = weak.clone();
                Box::pin(async move {
                    let session = weak.upgrade().ok_or_else(|| anyhow::anyhow!("Session disposed"))?;
                    if signal.is_some_and(|signal| signal.is_cancelled()) {
                        return Err(anyhow::anyhow!("Heartbeat call cancelled"));
                    }
                    let action = validate(&args).map_err(anyhow::Error::msg)?;
                    // The legacy controller trait raises creation/storage errors as panics.
                    // Return them as tool failures rather than terminating the agent turn.
                    let value = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        session.handle_rlm_heartbeat_host_request(&format!("rlm_heartbeat.{action}"), Some(&args))
                    })).map_err(|error| anyhow::anyhow!("Heartbeat scheduler failed: {}",
                        error.downcast_ref::<String>().map(String::as_str)
                            .or_else(|| error.downcast_ref::<&str>().copied()).unwrap_or("unknown scheduler error")))?
                        .map_err(anyhow::Error::msg)?;
                    if action != "list" && value["heartbeat"].is_null() {
                        return Err(anyhow::anyhow!("No matching agent-owned heartbeat was changed in this session"));
                    }
                    Ok(crate::core::tools::tool_definition_wrapper::text_tool_result(
                        serde_json::to_string(&value)?, value))
                })
            }),
            ..Default::default()
        }.into()
    }
}

fn validate(args: &Value) -> Result<&str, String> {
    let action = args.get("action").and_then(Value::as_str).unwrap_or("");
    let allowed: &[&str] = match action {
        "create" => &[
            "action",
            "instruction",
            "interval",
            "label",
            "delivery_mode",
        ],
        "list" => &["action", "include_inactive"],
        "update" => &[
            "action",
            "id",
            "instruction",
            "interval",
            "label",
            "status",
            "delivery_mode",
        ],
        "delete" => &["action", "id"],
        _ => return Err("action must be create, list, update or delete".into()),
    };
    if let Some(key) = args
        .as_object()
        .and_then(|object| object.keys().find(|key| !allowed.contains(&key.as_str())))
    {
        return Err(format!("{key} is not supported for heartbeat {action}"));
    }
    for field in ["id", "instruction"] {
        let required = (field == "id" && matches!(action, "update" | "delete"))
            || (field == "instruction" && action == "create");
        if (required || args.get(field).is_some())
            && !args
                .get(field)
                .and_then(Value::as_str)
                .is_some_and(|s| !s.trim().is_empty())
        {
            return Err(format!("heartbeat {action} requires a non-empty {field}"));
        }
    }
    if let Some(interval) = args.get("interval") {
        let interval = interval.as_str().ok_or("interval must be a string")?;
        let schedule = normalize_heartbeat_schedule(Some(interval));
        let parsed = parse_agent_cron_schedule(&schedule, crate::core::cron_jobs::now_millis())?;
        if parsed.schedule.kind == SCHEDULE_ONCE {
            return Err("Heartbeat interval must be recurring".into());
        }
    }
    Ok(action)
}
