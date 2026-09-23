//! Port of packages/coding-agent/src/core/tools/acp-mcp.ts

use std::sync::{Arc, OnceLock};

use regex::Regex;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use super::ipython::IpythonKernelProvisioner;
use super::{ExtensionContext, ToolDefinition, ToolExecuteFn};
use crate::core::kernel::shared::{ExecuteResult, ExecuteStatus};

// 48 keeps `mcp_list_tools_<name>` within providers' 64-char tool-name limits.
fn acp_mcp_server_name_pattern() -> &'static Regex {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    PATTERN.get_or_init(|| Regex::new(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,48}$").expect("valid server name pattern"))
}

/// TypeScript `interface AcpMcpServerConfig` (only the members this file reads).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AcpMcpServerConfig {
    pub name: String,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

pub fn acp_mcp_tool_names(servers: &[AcpMcpServerConfig]) -> Result<Vec<String>, String> {
    let mut names: Vec<String> = Vec::new();
    let mut seen_servers: std::collections::HashSet<String> = std::collections::HashSet::new();
    for server in servers {
        if !acp_mcp_server_name_pattern().is_match(&server.name) {
            return Err(format!("Invalid ACP MCP server name: {}", server.name));
        }
        if seen_servers.contains(&server.name) {
            return Err(format!("Duplicate ACP MCP server: {}", server.name));
        }
        seen_servers.insert(server.name.clone());
        names.push(format!("mcp_list_tools_{}", server.name));
        names.push(format!("mcp_call_{}", server.name));
    }
    Ok(names)
}

fn execution_result(result: ExecuteResult) -> Result<(String, Value), String> {
    let mut text = result.stdout.clone();
    if !result.stderr.is_empty() {
        text += &format!("{}{}", if text.is_empty() { "" } else { "\n" }, result.stderr);
    }
    if let Some(value) = result.result.clone() {
        if !value.is_empty() {
            text += &format!("{}{}", if text.is_empty() { "" } else { "\n" }, value);
        }
    }
    if let Some(error) = result.error.as_ref() {
        text += &format!(
            "{}{}",
            if text.is_empty() { "" } else { "\n" },
            error.traceback.join("\n")
        );
    }
    let status = result.status;
    if status != ExecuteStatus::Ok {
        let message = if text.is_empty() {
            format!("MCP kernel execution {}", execute_status_str(status))
        } else {
            text
        };
        return Err(message);
    }
    Ok((
        if text.is_empty() { "(empty)".to_string() } else { text },
        serde_json::json!({
            "durationMs": result.duration_ms,
            "status": execute_status_str(status),
            "stdout": result.stdout,
            "stderr": result.stderr,
            "result": result.result,
        }),
    ))
}

fn execute_status_str(status: ExecuteStatus) -> &'static str {
    match status {
        ExecuteStatus::Ok => "ok",
        ExecuteStatus::Error => "error",
        ExecuteStatus::Aborted => "aborted",
    }
}

async fn execute_mcp_code(
    provisioner: &Arc<IpythonKernelProvisioner>,
    code: &str,
    signal: Option<CancellationToken>,
) -> Result<(String, Value), String> {
    // `ensure` takes the shared provisioner, a progress handler and the kernel
    // bootstrap `AbortSignal`; the tool signal is a `CancellationToken`.
    let abort_signal = signal.clone().map(super::ipython::abort_signal_from_token);
    let manager = provisioner
        .ensure(None, abort_signal.clone())
        .await
        .map_err(|error| error.to_string())?;
    let result = manager
        .execute(code, abort_signal, None)
        .await
        .map_err(|error| error.to_string())?;
    execution_result(result)
}

pub const ACP_MCP_SERVER_NAME_PATTERN_SOURCE: &str = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,48}$";

pub fn create_acp_mcp_tool_definitions(
    servers: &[AcpMcpServerConfig],
    provisioner: Arc<IpythonKernelProvisioner>,
) -> Result<Vec<ToolDefinition<Value>>, String> {
    let names = acp_mcp_tool_names(servers)?;
    let mut definitions: Vec<ToolDefinition<Value>> = Vec::new();
    for (index, server) in servers.iter().enumerate() {
        let list_tool_name = names[index * 2].clone();
        let call_tool_name = names[index * 2 + 1].clone();
        let server_name = serde_json::to_string(&server.name).unwrap_or_else(|_| "\"\"".to_string());

        let list_provisioner = provisioner.clone();
        let list_server_name = server_name.clone();
        let list_execute: ToolExecuteFn<Value> = Arc::new(
            move |_tool_call_id: String,
                  _params: Value,
                  signal: Option<CancellationToken>,
                  _on_update: Option<pi_agent_core::types::AgentToolUpdateCallback>,
                  _ctx: ExtensionContext| {
                let provisioner = list_provisioner.clone();
                let server_name = list_server_name.clone();
                Box::pin(async move {
                    let code = format!(
                        "print(__import__(\"json\").dumps(await mcp.list_tools({server_name}), default=str))"
                    );
                    let (text, details) = execute_mcp_code(&provisioner, &code, signal)
                        .await
                        .map_err(anyhow::Error::msg)?;
                    Ok(pi_agent_core::types::AgentToolResult::new(
                        vec![pi_agent_core::types::ContentBlock::text(text)],
                        details,
                    ))
                })
            },
        );

        definitions.push(ToolDefinition {
            name: list_tool_name.clone(),
            label: format!("list tools from {}", server.name),
            description: format!(
                "List every tool the \"{}\" MCP server exposes. Call this first, then use {} to invoke a specific tool.",
                server.name, call_tool_name
            ),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": false
            }),
            execute: list_execute,
            ..ToolDefinition::default()
        });

        let call_provisioner = provisioner.clone();
        let call_server_name = server_name.clone();
        let call_list_tool_name = list_tool_name.clone();
        let call_execute: ToolExecuteFn<Value> = Arc::new(
            move |_tool_call_id: String,
                  params: Value,
                  signal: Option<CancellationToken>,
                  _on_update: Option<pi_agent_core::types::AgentToolUpdateCallback>,
                  _ctx: ExtensionContext| {
                let provisioner = call_provisioner.clone();
                let server_name = call_server_name.clone();
                Box::pin(async move {
                    let tool = params
                        .get("tool")
                        .and_then(|tool| tool.as_str())
                        .unwrap_or_default()
                        .to_string();
                    let args = params.get("arguments").cloned().unwrap_or(Value::Null);
                    let args_json = serde_json::to_string(&args).unwrap_or_else(|_| "{}".to_string());
                    let code = format!(
                        "print(__import__(\"json\").dumps(await mcp.call_tool({server_name}, {}, __import__(\"json\").loads({})), default=str))",
                        serde_json::to_string(&tool).unwrap_or_else(|_| "\"\"".to_string()),
                        serde_json::to_string(&args_json).unwrap_or_else(|_| "\"\"".to_string())
                    );
                    let (text, details) = execute_mcp_code(&provisioner, &code, signal)
                        .await
                        .map_err(anyhow::Error::msg)?;
                    Ok(pi_agent_core::types::AgentToolResult::new(
                        vec![pi_agent_core::types::ContentBlock::text(text)],
                        details,
                    ))
                })
            },
        );

        definitions.push(ToolDefinition {
            name: call_tool_name.clone(),
            label: format!("call tool on {}", server.name),
            description: format!(
                "Call a tool on the \"{}\" MCP server. Use {} first to discover available tool names and argument schemas.",
                server.name, call_list_tool_name
            ),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "tool": { "type": "string", "description": format!("Tool name on \"{}\".", server.name) },
                    "arguments": { "type": "object", "description": "JSON arguments for the tool.", "additionalProperties": true }
                },
                "required": ["tool", "arguments"],
                "additionalProperties": false
            }),
            execute: call_execute,
            ..ToolDefinition::default()
        });
    }
    Ok(definitions)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn server(name: &str) -> AcpMcpServerConfig {
        AcpMcpServerConfig {
            name: name.to_string(),
            extra: serde_json::Map::new(),
        }
    }

    fn result(
        status: ExecuteStatus,
        stdout: &str,
        stderr: &str,
        value: Option<&str>,
        error: Option<crate::core::kernel::shared::ExecError>,
    ) -> ExecuteResult {
        ExecuteResult {
            stdout: stdout.to_string(),
            stderr: stderr.to_string(),
            result: value.map(str::to_string),
            diffs: None,
            attachments: None,
            sent_agent_messages: None,
            background_output: None,
            status,
            error,
            execution_reports: None,
            duration_ms: 12.0,
        }
    }

    #[test]
    fn tool_names_are_prefixed_pairs() {
        let names = acp_mcp_tool_names(&[server("files"), server("git")]).expect("names");
        assert_eq!(
            names,
            vec![
                "mcp_list_tools_files",
                "mcp_call_files",
                "mcp_list_tools_git",
                "mcp_call_git"
            ]
        );
    }

    #[test]
    fn tool_names_reject_invalid_and_duplicate_servers() {
        let invalid = acp_mcp_tool_names(&[server("bad name")]).expect_err("invalid");
        assert_eq!(invalid, "Invalid ACP MCP server name: bad name");

        let duplicate = acp_mcp_tool_names(&[server("files"), server("files")]).expect_err("duplicate");
        assert_eq!(duplicate, "Duplicate ACP MCP server: files");

        let too_long = acp_mcp_tool_names(&[server(&"a".repeat(50))]).expect_err("too long");
        assert!(too_long.starts_with("Invalid ACP MCP server name:"));
        let longest = acp_mcp_tool_names(&[server(&"a".repeat(49))]).expect("64-character tool name");
        assert_eq!(longest[0].len(), 64);
    }

    #[test]
    fn execution_result_joins_sections_in_order() {
        let (text, details) =
            execution_result(result(ExecuteStatus::Ok, "out", "err", Some("res"), None)).expect("ok");
        assert_eq!(text, "out\nerr\nres");
        assert_eq!(details["durationMs"], serde_json::json!(12.0));
        assert_eq!(details["status"], serde_json::json!("ok"));
    }

    #[test]
    fn execution_result_falls_back_to_empty_marker() {
        let (text, _) = execution_result(result(ExecuteStatus::Ok, "", "", None, None)).expect("ok");
        assert_eq!(text, "(empty)");
    }

    #[test]
    fn execution_result_throws_status_message_for_failures() {
        let error = execution_result(result(
            ExecuteStatus::Error,
            "",
            "",
            None,
            Some(crate::core::kernel::shared::ExecError {
                ename: "ValueError".to_string(),
                evalue: "bad".to_string(),
                traceback: vec!["line 1".to_string(), "line 2".to_string()],
            }),
        ))
        .expect_err("must fail");
        assert_eq!(error, "line 1\nline 2");

        let empty_error = execution_result(result(ExecuteStatus::Aborted, "", "", None, None)).expect_err("must fail");
        assert_eq!(empty_error, "MCP kernel execution aborted");
    }
}
