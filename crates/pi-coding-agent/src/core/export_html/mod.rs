//! Port of packages/coding-agent/src/core/export-html/index.ts

pub mod ansi_to_html;
pub mod tool_renderer;

use std::collections::HashMap;
use std::path::Path;

use base64::Engine;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::core::extensions::types::ToolDefinition;
use crate::core::export_html::tool_renderer::ToolHtmlRenderer;
use crate::core::session_manager::{SessionEntry, SessionManager};
use crate::modes::interactive::theme::theme::{
    get_resolved_theme_colors, get_theme_export_colors,
};

/// `APP_NAME` from config.ts (`pkg.piConfig.name`).
/// blocked_on: needs crate::config::APP_NAME
const APP_NAME: &str = "prime-agent";

/// Pre-rendered HTML for a custom tool call and result.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RenderedToolHtml {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub call_html: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result_html_collapsed: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result_html_expanded: Option<String>,
}

pub struct ExportOptions {
    pub output_path: Option<String>,
    pub theme_name: Option<String>,
    /// Optional tool renderer for custom tools.
    pub tool_renderer: Option<Box<dyn ToolHtmlRenderer>>,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            output_path: None,
            theme_name: None,
            tool_renderer: None,
        }
    }
}

/// Parse a color string to RGB values. Supports hex (#RRGGBB) and rgb(r,g,b) formats.
fn parse_color(color: &str) -> Option<(f64, f64, f64)> {
    let hex = regex_hex();
    if let Some(captures) = hex.captures(color) {
        return Some((
            i64::from_str_radix(&captures[1], 16).ok()? as f64,
            i64::from_str_radix(&captures[2], 16).ok()? as f64,
            i64::from_str_radix(&captures[3], 16).ok()? as f64,
        ));
    }
    let rgb = regex_rgb();
    if let Some(captures) = rgb.captures(color) {
        return Some((
            captures[1].parse::<f64>().ok()?,
            captures[2].parse::<f64>().ok()?,
            captures[3].parse::<f64>().ok()?,
        ));
    }
    None
}

fn regex_hex() -> &'static regex::Regex {
    static REGEX: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    REGEX.get_or_init(|| regex::Regex::new("^#([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})$").expect("hex regex"))
}

fn regex_rgb() -> &'static regex::Regex {
    static REGEX: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    REGEX.get_or_init(|| {
        regex::Regex::new("^rgb\\s*\\(\\s*(\\d+)\\s*,\\s*(\\d+)\\s*,\\s*(\\d+)\\s*\\)$").expect("rgb regex")
    })
}

/// Calculate relative luminance of a color (0-1, higher = lighter).
fn get_luminance(r: f64, g: f64, b: f64) -> f64 {
    let to_linear = |c: f64| {
        let s = c / 255.0;
        if s <= 0.03928 {
            s / 12.92
        } else {
            ((s + 0.055) / 1.055).powf(2.4)
        }
    };
    0.2126 * to_linear(r) + 0.7152 * to_linear(g) + 0.0722 * to_linear(b)
}

/// Adjust color brightness. Factor > 1 lightens, < 1 darkens.
fn adjust_brightness(color: &str, factor: f64) -> String {
    let Some((r, g, b)) = parse_color(color) else {
        return color.to_string();
    };
    let adjust = |c: f64| (c * factor).round().min(255.0).max(0.0);
    format!(
        "rgb({}, {}, {})",
        adjust(r) as i64,
        adjust(g) as i64,
        adjust(b) as i64
    )
}

/// Derive export background colors from a base color (e.g., userMessageBg).
fn derive_export_colors(base_color: &str) -> ExportColors {
    let Some((r, g, b)) = parse_color(base_color) else {
        return ExportColors {
            page_bg: "rgb(24, 24, 30)".to_string(),
            card_bg: "rgb(30, 30, 36)".to_string(),
            info_bg: "rgb(60, 55, 40)".to_string(),
        };
    };

    let luminance = get_luminance(r, g, b);
    let is_light = luminance > 0.5;

    if is_light {
        return ExportColors {
            page_bg: adjust_brightness(base_color, 0.96),
            card_bg: base_color.to_string(),
            info_bg: format!(
                "rgb({}, {}, {})",
                (r + 10.0).min(255.0) as i64,
                (g + 5.0).min(255.0) as i64,
                (b - 20.0).max(0.0) as i64
            ),
        };
    }
    ExportColors {
        page_bg: adjust_brightness(base_color, 0.7),
        card_bg: adjust_brightness(base_color, 0.85),
        info_bg: format!(
            "rgb({}, {}, {})",
            (r + 20.0).min(255.0) as i64,
            (g + 15.0).min(255.0) as i64,
            b as i64
        ),
    }
}

#[derive(Debug, Clone, PartialEq)]
struct ExportColors {
    page_bg: String,
    card_bg: String,
    info_bg: String,
}

/// Generate CSS custom property declarations from theme colors.
fn generate_theme_vars(theme_name: Option<&str>) -> Result<String, String> {
    let colors = get_resolved_theme_colors(theme_name)?;
    let mut lines: Vec<String> = Vec::new();
    for (key, value) in colors.iter() {
        lines.push(format!("--{key}: {value};"));
    }
    let theme_export = get_theme_export_colors(theme_name);
    let user_message_bg = colors
        .get("userMessageBg")
        .cloned()
        .unwrap_or_else(|| "#343541".to_string());
    let derived_colors = derive_export_colors(&user_message_bg);

    lines.push(format!(
        "--exportPageBg: {};",
        theme_export.page_bg.clone().unwrap_or_else(|| derived_colors.page_bg.clone())
    ));
    lines.push(format!(
        "--exportCardBg: {};",
        theme_export.card_bg.clone().unwrap_or_else(|| derived_colors.card_bg.clone())
    ));
    lines.push(format!(
        "--exportInfoBg: {};",
        theme_export.info_bg.clone().unwrap_or_else(|| derived_colors.info_bg.clone())
    ));

    Ok(lines.join("\n      "))
}

/// `SessionData` passed to the template.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionData {
    pub header: Option<SessionEntry>,
    pub entries: Vec<SessionEntry>,
    #[serde(rename = "leafId")]
    pub leaf_id: Option<String>,
    #[serde(rename = "systemPrompt", skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ExportToolSummary>>,
    /// Pre-rendered HTML for custom tool calls/results, keyed by tool call ID.
    #[serde(rename = "renderedTools", skip_serializing_if = "Option::is_none")]
    pub rendered_tools: Option<Map<String, Value>>,
}

/// `Pick<ToolDefinition, "name" | "description" | "parameters">`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportToolSummary {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

/// Core HTML generation logic shared by both export functions.
fn generate_html(session_data: &SessionData, theme_name: Option<&str>) -> Result<String, String> {
    let template_dir = crate::config::get_export_template_dir();
    let template = read_template(&template_dir, "template.html")?;
    let template_css = read_template(&template_dir, "template.css")?;
    let template_js = read_template(&template_dir, "template.js")?;
    let marked_js = read_template(&Path::new(&template_dir).join("vendor").to_string_lossy(), "marked.min.js")?;
    let hljs_js = read_template(&Path::new(&template_dir).join("vendor").to_string_lossy(), "highlight.min.js")?;

    let theme_vars = generate_theme_vars(theme_name)?;
    let colors = get_resolved_theme_colors(theme_name)?;
    let theme_export = get_theme_export_colors(theme_name);
    let user_message_bg = colors
        .get("userMessageBg")
        .cloned()
        .unwrap_or_else(|| "#343541".to_string());
    let derived_export_colors = derive_export_colors(&user_message_bg);
    let body_bg = theme_export.page_bg.unwrap_or(derived_export_colors.page_bg);
    let container_bg = theme_export.card_bg.unwrap_or(derived_export_colors.card_bg);
    let info_bg = theme_export.info_bg.unwrap_or(derived_export_colors.info_bg);
    let session_data_base64 = base64::engine::general_purpose::STANDARD
        .encode(serde_json::to_string(session_data).map_err(|error| error.to_string())?);
    let css = template_css
        .replace("{{THEME_VARS}}", &theme_vars)
        .replace("{{BODY_BG}}", &body_bg)
        .replace("{{CONTAINER_BG}}", &container_bg)
        .replace("{{INFO_BG}}", &info_bg);

    Ok(template
        .replace("{{CSS}}", &css)
        .replace("{{JS}}", &template_js)
        .replace("{{SESSION_DATA}}", &session_data_base64)
        .replace("{{MARKED_JS}}", &marked_js)
        .replace("{{HIGHLIGHT_JS}}", &hljs_js))
}

fn read_template(dir: &str, name: &str) -> Result<String, String> {
    let path = Path::new(dir).join(name);
    std::fs::read_to_string(&path).map_err(|error| format!("{}: {error}", path.to_string_lossy()))
}

/// Tools rendered directly by the HTML template (not pre-rendered via TUI->ANSI->HTML pipeline)
fn is_template_rendered_tool(name: &str) -> bool {
    name == "bash" || name == "edit"
}

/// Pre-render custom tools to HTML using their TUI renderers.
///
/// Session entries are plain JSON objects (`SessionEntry = Map<String, Value>`),
/// exactly like the TypeScript union is read structurally here.
fn pre_render_custom_tools(
    entries: &[SessionEntry],
    tool_renderer: &dyn ToolHtmlRenderer,
) -> HashMap<String, RenderedToolHtml> {
    let mut rendered_tools: HashMap<String, RenderedToolHtml> = HashMap::new();

    for entry in entries {
        if entry.get("type").and_then(Value::as_str) != Some("message") {
            continue;
        }
        let Some(message) = entry.get("message") else {
            continue;
        };
        let role = message.get("role").and_then(Value::as_str).unwrap_or("");

        if role == "assistant" {
            if let Some(content) = message.get("content").and_then(Value::as_array) {
                for block in content {
                    if block.get("type").and_then(Value::as_str) != Some("toolCall") {
                        continue;
                    }
                    let name = block.get("name").and_then(Value::as_str).unwrap_or("");
                    if is_template_rendered_tool(name) {
                        continue;
                    }
                    let id = block.get("id").and_then(Value::as_str).unwrap_or("");
                    let arguments = block.get("arguments").cloned().unwrap_or(Value::Null);
                    if let Some(call_html) = tool_renderer.render_call(id, name, arguments) {
                        rendered_tools.insert(
                            id.to_string(),
                            RenderedToolHtml {
                                call_html: Some(call_html),
                                ..Default::default()
                            },
                        );
                    }
                }
            }
        }

        if role == "toolResult" {
            let Some(tool_call_id) = message.get("toolCallId").and_then(Value::as_str) else {
                continue;
            };
            let tool_name = message.get("toolName").and_then(Value::as_str).unwrap_or("");
            let existing = rendered_tools.get(tool_call_id).cloned();
            if existing.is_some() || !is_template_rendered_tool(tool_name) {
                let result = message
                    .get("content")
                    .and_then(Value::as_array)
                    .map(|content| {
                        content
                            .iter()
                            .map(|part| crate::core::export_html::tool_renderer::ToolResultContentPart {
                                content_type: part
                                    .get("type")
                                    .and_then(Value::as_str)
                                    .unwrap_or("")
                                    .to_string(),
                                text: part.get("text").and_then(Value::as_str).map(str::to_string),
                                data: part.get("data").and_then(Value::as_str).map(str::to_string),
                                mime_type: part
                                    .get("mimeType")
                                    .and_then(Value::as_str)
                                    .map(str::to_string),
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                let details = message.get("details").cloned().unwrap_or(Value::Null);
                let is_error = message.get("isError").and_then(Value::as_bool).unwrap_or(false);
                if let Some(rendered) =
                    tool_renderer.render_result(tool_call_id, tool_name, result, details, is_error)
                {
                    let mut merged = existing.unwrap_or_default();
                    merged.result_html_collapsed = rendered.collapsed;
                    merged.result_html_expanded = rendered.expanded;
                    rendered_tools.insert(tool_call_id.to_string(), merged);
                }
            }
        }
    }

    rendered_tools
}

/// Export session to HTML using SessionManager and AgentState.
/// Used by TUI's /export command.
pub fn export_session_to_html(
    session_manager: &mut SessionManager,
    state: Option<&ExportState>,
    options: Option<ExportOptions>,
) -> Result<String, String> {
    let options = options.unwrap_or_default();

    let Some(session_file) = session_manager.get_session_file() else {
        return Err("Cannot export in-memory session to HTML".to_string());
    };
    if !Path::new(&session_file).exists() {
        return Err("Nothing to export yet - start a conversation first".to_string());
    }

    let entries = session_manager.get_entries();
    let mut rendered_tools: Option<Map<String, Value>> = None;
    if let Some(tool_renderer) = options.tool_renderer.as_deref() {
        let rendered = pre_render_custom_tools(&entries, tool_renderer);
        if !rendered.is_empty() {
            let mut map = Map::new();
            for (key, value) in rendered {
                map.insert(key, serde_json::to_value(value).map_err(|error| error.to_string())?);
            }
            rendered_tools = Some(map);
        }
    }

    let session_data = SessionData {
        header: session_manager.get_header(),
        entries,
        leaf_id: session_manager.get_leaf_id(),
        system_prompt: state.map(|state| state.system_prompt.clone()),
        tools: state.map(|state| {
            state
                .tools
                .iter()
                .map(|tool| ExportToolSummary {
                    name: tool.name.clone(),
                    description: tool.description.clone(),
                    parameters: tool.parameters.clone(),
                })
                .collect()
        }),
        rendered_tools,
    };

    let html = generate_html(&session_data, options.theme_name.as_deref())?;

    let output_path = match &options.output_path {
        Some(path) => path.clone(),
        None => {
            let session_basename = basename_without_jsonl(&session_file);
            format!("{APP_NAME}-session-{session_basename}.html")
        }
    };

    std::fs::write(&output_path, html).map_err(|error| error.to_string())?;
    Ok(output_path)
}

/// Export session file to HTML (standalone, without AgentState).
/// Used by CLI for exporting arbitrary session files.
pub fn export_from_file(input_path: &str, options: Option<ExportOptions>) -> Result<String, String> {
    let options = options.unwrap_or_default();

    if !Path::new(input_path).exists() {
        return Err(format!("File not found: {input_path}"));
    }

    let mut session_manager = SessionManager::in_memory(None, None)
        .map_err(|error| error.to_string())?;
    session_manager
        .set_session_file(input_path, None, None)
        .map_err(|error| error.to_string())?;

    let session_data = SessionData {
        header: session_manager.get_header(),
        entries: session_manager.get_entries(),
        leaf_id: session_manager.get_leaf_id(),
        system_prompt: None,
        tools: None,
        rendered_tools: None,
    };

    let html = generate_html(&session_data, options.theme_name.as_deref())?;

    let output_path = match &options.output_path {
        Some(path) => path.clone(),
        None => {
            let input_basename = basename_without_jsonl(input_path);
            format!("{APP_NAME}-session-{input_basename}.html")
        }
    };

    std::fs::write(&output_path, html).map_err(|error| error.to_string())?;
    Ok(output_path)
}

fn basename_without_jsonl(path: &str) -> String {
    let base = Path::new(path)
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_default();
    base.strip_suffix(".jsonl").map(|value| value.to_string()).unwrap_or(base)
}

/// `AgentState` subset used by the export (`systemPrompt` and `tools`).
#[derive(Debug, Clone, Default)]
pub struct ExportState {
    pub system_prompt: String,
    pub tools: Vec<ToolDefinition>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_hex_and_rgb_colors() {
        assert_eq!(parse_color("#343541"), Some((52.0, 53.0, 65.0)));
        assert_eq!(parse_color("rgb(1, 2, 3)"), Some((1.0, 2.0, 3.0)));
        assert_eq!(parse_color("rgb( 1 , 2 , 3 )"), Some((1.0, 2.0, 3.0)));
        assert_eq!(parse_color("not-a-color"), None);
    }

    #[test]
    fn light_and_dark_derivations_match_the_typescript() {
        let light = derive_export_colors("#ffffff");
        assert_eq!(light.card_bg, "#ffffff");
        assert_eq!(light.page_bg, "rgb(245, 245, 245)");
        assert_eq!(light.info_bg, "rgb(255, 255, 235)");

        let dark = derive_export_colors("#343541");
        assert_eq!(dark.page_bg, "rgb(36, 37, 46)");
        assert_eq!(dark.card_bg, "rgb(44, 45, 55)");
        assert_eq!(dark.info_bg, "rgb(72, 68, 65)");
    }

    #[test]
    fn invalid_base_color_uses_the_fallback_palette() {
        let fallback = derive_export_colors("bogus");
        assert_eq!(fallback.page_bg, "rgb(24, 24, 30)");
        assert_eq!(fallback.card_bg, "rgb(30, 30, 36)");
        assert_eq!(fallback.info_bg, "rgb(60, 55, 40)");
    }

    #[test]
    fn template_rendered_tools_match_the_typescript_set() {
        assert!(is_template_rendered_tool("bash"));
        assert!(is_template_rendered_tool("edit"));
        assert!(!is_template_rendered_tool("ipython"));
    }

    #[test]
    fn basename_strips_only_a_jsonl_suffix() {
        assert_eq!(basename_without_jsonl("/a/b/session.jsonl"), "session");
        assert_eq!(basename_without_jsonl("/a/b/session.txt"), "session.txt");
    }
}
