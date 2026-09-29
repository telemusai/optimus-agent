//! Explicit live diagnostic. Never invoked by regression tests.
//! cargo run --locked -p pi-ai --example kiro_probe -- --catalog
//! cargo run --locked -p pi-ai --example kiro_probe -- --model claude-haiku-4.5
use pi_ai::{providers::kiro, types::*};

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args != ["--catalog"] && !(args.len() == 2 && args[0] == "--model") {
        return Err(
            "Use --catalog for direct discovery, or --model <id> for a live tool round trip".into(),
        );
    }
    let key = if let Ok(key) = std::env::var("KIRO_API_KEY") {
        key
    } else {
        let path = kiro::auth::default_cli_path().ok_or("No Kiro CLI login or KIRO_API_KEY")?;
        kiro::auth::resolve_cli(&path).await?.credential.encode()
    };
    let catalog = kiro::catalog::discover(&key, None).await?;
    if args == ["--catalog"] {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({"providers":{"kiro":{
                "api":"kiro-api","baseUrl":kiro::DEFAULT_ENDPOINT,"models":catalog
            }}}))
            .map_err(|_| "Cannot serialize Kiro catalog")?
        );
        return Ok(());
    }
    let model = catalog
        .into_iter()
        .find(|model| model.id == args[1])
        .ok_or("Requested model is not in the authorized Kiro catalog")?;
    let mut context = Context::new(Some("Call the supplied echo tool once with value OPTIMUS_KIRO_OK. After receiving its result reply with that value only.".into()),
        vec![Message::User(UserMessage::new(UserContent::Text("Run the validation.".into()),0))],
        Some(vec![Tool { name:"echo".into(),description:"Echo a validation string without side effects".into(),parameters:serde_json::json!({"type":"object","properties":{"value":{"type":"string"}},"required":["value"]}) }]));
    let options = SimpleStreamOptions {
        stream: StreamOptions {
            api_key: Some(key),
            ..Default::default()
        },
        reasoning: Some("off".into()),
        ..Default::default()
    };
    let first = kiro::stream_simple_kiro(&model, &context, Some(&options))
        .result()
        .await;
    if first.stop_reason != "toolUse" {
        return Err(first
            .error_message
            .unwrap_or_else(|| "Kiro did not return the validation tool call".into()));
    }
    let call = first
        .content
        .iter()
        .find_map(ContentBlock::as_tool_call)
        .cloned()
        .ok_or("Kiro returned no tool call")?;
    if call.name != "echo"
        || call
            .arguments
            .get("value")
            .and_then(serde_json::Value::as_str)
            != Some("OPTIMUS_KIRO_OK")
    {
        return Err("Kiro returned unexpected validation arguments; no tool was executed".into());
    }
    context.messages.push(Message::Assistant(first));
    context
        .messages
        .push(Message::ToolResult(ToolResultMessage::new(
            call.id,
            "echo",
            vec![TextContent::new("OPTIMUS_KIRO_OK").into()],
            false,
            0,
        )));
    let second = kiro::stream_simple_kiro(&model, &context, Some(&options))
        .result()
        .await;
    if second.stop_reason != "stop" {
        return Err(second
            .error_message
            .unwrap_or_else(|| "Kiro did not complete the second turn".into()));
    }
    let text = second
        .content
        .iter()
        .filter_map(ContentBlock::as_text)
        .map(|t| t.text.as_str())
        .collect::<String>();
    if !text.contains("OPTIMUS_KIRO_OK") {
        return Err("Kiro did not return the expected validation result".into());
    }
    println!("PASS: {} native Kiro tool round trip", model.id);
    Ok(())
}
