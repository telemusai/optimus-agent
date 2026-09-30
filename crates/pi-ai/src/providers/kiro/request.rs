use crate::types::*;
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

fn tool_use_id(id: &str) -> String {
    if (1..=64).contains(&id.len())
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"_.:-".contains(&b))
    {
        return id.to_owned();
    }
    // Responses IDs include a pipe and an item ID. Hash the full ID so calls and
    // results agree without truncation/sanitization collisions or session edits.
    format!(
        "optimus_{}",
        URL_SAFE_NO_PAD.encode(Sha256::digest(id.as_bytes()))
    )
}

fn append_array(target: &mut Value, key: &str, source: &Value) {
    if let Some(items) = source[key].as_array() {
        if !target[key].is_array() {
            target[key] = json!([]);
        }
        target[key]
            .as_array_mut()
            .unwrap()
            .extend(items.iter().cloned());
    }
}

fn user(content: &str, model: &Model, origin: &str) -> Value {
    json!({"content":content,"modelId":model.id,"origin":origin})
}

fn add_images(message: &mut Value, content: &[ImageOrTextContent]) -> Result<(), String> {
    for block in content {
        if let ImageOrTextContent::Image(image) = block {
            let format = match image.mime_type.as_str() {
                "image/png" => "png",
                "image/jpeg" => "jpeg",
                "image/gif" => "gif",
                "image/webp" => "webp",
                _ => return Err("Kiro supports PNG, JPEG, GIF and WebP images".into()),
            };
            if !message["images"].is_array() {
                message["images"] = json!([]);
            }
            message["images"]
                .as_array_mut()
                .unwrap()
                .push(json!({"format":format,"source":{"bytes":image.data}}));
        }
    }
    Ok(())
}

fn push_message(history: &mut Vec<Value>, key: &str, mut wire: Value) {
    if let Some(previous) = history.last_mut().and_then(|entry| entry.get_mut(key)) {
        previous["content"] = json!(format!(
            "{}\n\n{}",
            previous["content"].as_str().unwrap_or(""),
            wire["content"].as_str().unwrap_or("")
        ));
        append_array(previous, "images", &wire);
        append_array(previous, "toolUses", &wire);
        if wire["userInputMessageContext"]["toolResults"].is_array() {
            if !previous["userInputMessageContext"].is_object() {
                previous["userInputMessageContext"] = json!({});
            }
            append_array(
                &mut previous["userInputMessageContext"],
                "toolResults",
                &wire["userInputMessageContext"],
            );
        }
    } else {
        let mut entry = json!({});
        entry[key] = wire.take();
        history.push(entry);
    }
}

fn close_pending_calls(
    history: &mut Vec<Value>,
    calls: &mut Vec<String>,
    model: &Model,
    origin: &str,
) {
    if calls.is_empty() {
        return;
    }
    // /btw can snapshot a running tool. Repair only this request, never the session.
    let mut wire = user(
        "Tool results unavailable in this conversation snapshot.",
        model,
        origin,
    );
    wire["userInputMessageContext"] = json!({"toolResults":std::mem::take(calls)
        .into_iter().map(|id| json!({"toolUseId":tool_use_id(&id),"status":"error","content":[{
            "text":"Tool result unavailable in this conversation snapshot. The tool may still be running in the main conversation."
        }]})).collect::<Vec<_>>()});
    push_message(history, "userInputMessage", wire);
}

pub fn build(
    model: &Model,
    context: &Context,
    options: &SimpleStreamOptions,
    origin: &str,
    profile: Option<&str>,
) -> Result<Value, String> {
    let mut history: Vec<Value> = vec![];
    let mut calls = Vec::new();
    for message in &context.messages {
        let (key, wire) = match message {
            Message::User(message) => {
                close_pending_calls(&mut history, &mut calls, model, origin);
                let mut wire = user(&message.content.text(), model, origin);
                if let UserContent::Blocks(blocks) = &message.content {
                    add_images(&mut wire, blocks)?;
                }
                ("userInputMessage", wire)
            }
            Message::Assistant(message) => {
                if matches!(message.stop_reason.as_str(), "error" | "aborted") {
                    continue;
                }
                close_pending_calls(&mut history, &mut calls, model, origin);
                let mut text = String::new();
                let mut tools = vec![];
                for block in &message.content {
                    match block {
                        ContentBlock::Text(block) => text.push_str(&block.text),
                        ContentBlock::ToolCall(call) => {
                            calls.push(call.id.clone());
                            tools.push(json!({"name":call.name,"toolUseId":tool_use_id(&call.id),"input":call.arguments}));
                        }
                        // Reasoning signatures belong to their original provider, not Kiro's history.
                        ContentBlock::Thinking(_) => {}
                    }
                }
                if text.is_empty() && tools.is_empty() {
                    continue;
                }
                let mut wire = json!({"content":text});
                if !tools.is_empty() {
                    wire["toolUses"] = json!(tools);
                }
                ("assistantResponseMessage", wire)
            }
            Message::ToolResult(message) => {
                let text = message
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        ImageOrTextContent::Text(t) => Some(t.text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                let mut wire = user("Tool results provided.", model, origin);
                add_images(&mut wire, &message.content)?;
                if let Some(index) = calls.iter().position(|id| id == &message.tool_call_id) {
                    calls.remove(index);
                    wire["userInputMessageContext"] = json!({"toolResults":[{
                        "toolUseId":tool_use_id(&message.tool_call_id),"status":if message.is_error {"error"} else {"success"},
                        "content":[{"text":text}]
                    }]});
                } else {
                    // Preserve orphan, duplicate and late output without an unmatched result block.
                    wire["content"] = json!(format!(
                        "Previous tool output ({}):\n{text}",
                        message.tool_name
                    ));
                }
                ("userInputMessage", wire)
            }
        };
        push_message(&mut history, key, wire);
    }
    close_pending_calls(&mut history, &mut calls, model, origin);
    if history
        .first()
        .is_some_and(|entry| entry["assistantResponseMessage"].is_object())
    {
        history.insert(
            0,
            json!({"userInputMessage":user("Continue the conversation.", model, origin)}),
        );
    }
    if !history
        .last()
        .is_some_and(|entry| entry["userInputMessage"].is_object())
    {
        history.push(json!({"userInputMessage":user("Please continue.", model, origin)}));
    }
    let mut system = context.system_prompt.clone().unwrap_or_default();
    if model.reasoning {
        if let Some(reasoning) = options.reasoning.as_deref() {
            let budget = match reasoning {
                "off" => 0,
                "minimal" => 1024,
                "low" => 4096,
                "medium" => 8192,
                "high" => 16384,
                "xhigh" | "max" => 32768,
                _ => 8192,
            };
            let directive = if budget == 0 {
                "<thinking_mode>disabled</thinking_mode>".into()
            } else {
                format!("<thinking_mode>enabled</thinking_mode><max_thinking_length>{budget}</max_thinking_length>")
            };
            system = format!("{directive}\n{system}");
        }
    }
    if !system.is_empty() {
        let first = &mut history[0]["userInputMessage"];
        first["content"] = json!(format!(
            "{system}\n\n{}",
            first["content"].as_str().unwrap_or("")
        ));
    }
    let mut current = history.pop().unwrap();
    if let Some(tools) = context.tools.as_ref().filter(|tools| !tools.is_empty()) {
        let user = &mut current["userInputMessage"];
        if !user["userInputMessageContext"].is_object() {
            user["userInputMessageContext"] = json!({});
        }
        user["userInputMessageContext"]["tools"] =
            json!(tools.iter().map(|tool| json!({"toolSpecification":{
            "name":tool.name,"description":tool.description,"inputSchema":{"json":tool.parameters}
        }})).collect::<Vec<_>>());
    }
    let id = options
        .stream
        .session_id
        .as_deref()
        .and_then(|id| uuid::Uuid::parse_str(id).ok())
        .unwrap_or_else(uuid::Uuid::new_v4);
    let mut body = json!({"conversationState":{"chatTriggerType":"MANUAL","agentTaskType":"vibe",
        "conversationId":id.to_string(),"currentMessage":current},"agentMode":"vibe"});
    if !history.is_empty() {
        body["conversationState"]["history"] = json!(history);
    }
    if let Some(profile) = profile {
        body["profileArn"] = json!(profile);
    }
    Ok(body)
}
