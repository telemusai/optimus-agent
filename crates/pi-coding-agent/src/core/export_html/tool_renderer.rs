//! Port of packages/coding-agent/src/core/export-html/tool-renderer.ts
//!
//! Tool HTML renderer for custom tools in HTML export.
//!
//! Renders custom tool calls and results to HTML by invoking their TUI renderers
//! and converting the ANSI output to HTML.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use regex::Regex;
use serde_json::{Map, Value};

use crate::core::extensions::types::{
    Component, Theme, ToolDefinition, ToolRenderContext, ToolRenderResultOptions,
};
use crate::core::export_html::ansi_to_html::ansi_lines_to_html;
use pi_agent_core::types::AgentToolResult;

/// TypeScript `Array<{ type: string; text?: string; data?: string; mimeType?: string }>`.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ToolResultContentPart {
    #[serde(rename = "type")]
    pub content_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<String>,
    #[serde(rename = "mimeType", skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
}

pub struct ToolHtmlRendererDeps {
    /// Function to look up tool definition by name.
    pub get_tool_definition: Arc<dyn Fn(&str) -> Option<ToolDefinition> + Send + Sync>,
    /// Theme for styling.
    pub theme: Theme,
    /// Working directory for render context.
    pub cwd: String,
    /// Terminal width for rendering (default: 100).
    pub width: Option<usize>,
}

/// Port of the `ToolHtmlRenderer` interface.
pub trait ToolHtmlRenderer: Send + Sync {
    /// Render a tool call to HTML. Returns `None` if tool has no custom renderer.
    fn render_call(&self, tool_call_id: &str, tool_name: &str, args: Value) -> Option<String>;
    /// Render a tool result to collapsed/expanded HTML.
    fn render_result(
        &self,
        tool_call_id: &str,
        tool_name: &str,
        result: Vec<ToolResultContentPart>,
        details: Value,
        is_error: bool,
    ) -> Option<RenderedToolResultHtml>;
}

/// `{ collapsed?: string; expanded?: string }`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RenderedToolResultHtml {
    pub collapsed: Option<String>,
    pub expanded: Option<String>,
}

fn ansi_escape_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| Regex::new("\\x1b\\[[\\d;]*m").expect("ansi escape regex"))
}

fn is_blank_rendered_line(line: &str) -> bool {
    ansi_escape_regex().replace_all(line, "").trim().is_empty()
}

fn trim_rendered_result_lines(lines: Vec<String>) -> Vec<String> {
    let mut start = 0usize;
    let mut end = lines.len();
    while start < end && is_blank_rendered_line(&lines[start]) {
        start += 1;
    }
    while end > start && is_blank_rendered_line(&lines[end - 1]) {
        end -= 1;
    }
    lines[start..end].to_vec()
}

struct DefaultToolHtmlRenderer {
    get_tool_definition: Arc<dyn Fn(&str) -> Option<ToolDefinition> + Send + Sync>,
    theme: Theme,
    cwd: String,
    width: usize,
    rendered_call_components: Mutex<HashMap<String, Arc<dyn Component>>>,
    rendered_result_components: Mutex<HashMap<String, Arc<dyn Component>>>,
    rendered_states: Mutex<HashMap<String, Value>>,
    rendered_args: Mutex<HashMap<String, Value>>,
}

impl DefaultToolHtmlRenderer {
    fn get_state(&self, tool_call_id: &str) -> Value {
        let mut states = self.rendered_states.lock().expect("rendered states");
        states
            .entry(tool_call_id.to_string())
            .or_insert_with(|| Value::Object(Map::new()))
            .clone()
    }

    fn create_render_context(
        &self,
        tool_call_id: &str,
        last_component: Option<Arc<dyn Component>>,
        expanded: bool,
        is_partial: bool,
        is_error: bool,
    ) -> ToolRenderContext {
        ToolRenderContext {
            args: self
                .rendered_args
                .lock()
                .expect("rendered args")
                .get(tool_call_id)
                .cloned()
                .unwrap_or(Value::Null),
            tool_call_id: tool_call_id.to_string(),
            invalidate: Arc::new(|| {}),
            last_component,
            state: self.get_state(tool_call_id),
            cwd: self.cwd.clone(),
            execution_started: true,
            args_complete: true,
            is_partial,
            expanded,
            show_expand_hint: None,
            show_images: false,
            include_image_dimensions: true,
            is_error,
        }
    }
}

impl ToolHtmlRenderer for DefaultToolHtmlRenderer {
    fn render_call(&self, tool_call_id: &str, tool_name: &str, args: Value) -> Option<String> {
        self.rendered_args
            .lock()
            .expect("rendered args")
            .insert(tool_call_id.to_string(), args.clone());
        let tool_def = (self.get_tool_definition)(tool_name)?;
        let render_call = tool_def.render_call.as_ref()?;

        let last_component = self
            .rendered_call_components
            .lock()
            .expect("rendered call components")
            .get(tool_call_id)
            .cloned();
        let context = self.create_render_context(tool_call_id, last_component, false, true, false);
        let component = render_call(args, self.theme.clone(), context);
        self.rendered_call_components
            .lock()
            .expect("rendered call components")
            .insert(tool_call_id.to_string(), component.clone());
        let lines = component.render(self.width);
        Some(ansi_lines_to_html(&lines))
    }

    fn render_result(
        &self,
        tool_call_id: &str,
        tool_name: &str,
        result: Vec<ToolResultContentPart>,
        details: Value,
        is_error: bool,
    ) -> Option<RenderedToolResultHtml> {
        let tool_def = (self.get_tool_definition)(tool_name)?;
        let render_result = tool_def.render_result.as_ref()?;

        let agent_tool_result = AgentToolResult {
            content: result
                .iter()
                .map(|part| match part.content_type.as_str() {
                    "image" => pi_agent_core::types::ContentBlock::Image(pi_ai::types::ImageContent {
                        type_: "image".to_string(),
                        data: part.data.clone().unwrap_or_default(),
                        mime_type: part.mime_type.clone().unwrap_or_default(),
                    }),
                    _ => pi_agent_core::types::ContentBlock::text(part.text.clone().unwrap_or_default()),
                })
                .collect(),
            details: details.clone(),
            is_error: Some(is_error),
            terminate: None,
        };

        let last_component = self
            .rendered_result_components
            .lock()
            .expect("rendered result components")
            .get(tool_call_id)
            .cloned();
        let collapsed_component = render_result(
            agent_tool_result.clone(),
            ToolRenderResultOptions {
                expanded: false,
                is_partial: false,
            },
            self.theme.clone(),
            self.create_render_context(tool_call_id, last_component, false, false, is_error),
        );
        self.rendered_result_components
            .lock()
            .expect("rendered result components")
            .insert(tool_call_id.to_string(), collapsed_component.clone());
        let collapsed = ansi_lines_to_html(&trim_rendered_result_lines(collapsed_component.render(self.width)));

        let last_component = self
            .rendered_result_components
            .lock()
            .expect("rendered result components")
            .get(tool_call_id)
            .cloned();
        let expanded_component = render_result(
            agent_tool_result,
            ToolRenderResultOptions {
                expanded: true,
                is_partial: false,
            },
            self.theme.clone(),
            self.create_render_context(tool_call_id, last_component, true, false, is_error),
        );
        self.rendered_result_components
            .lock()
            .expect("rendered result components")
            .insert(tool_call_id.to_string(), expanded_component.clone());
        let expanded = ansi_lines_to_html(&trim_rendered_result_lines(expanded_component.render(self.width)));

        Some(RenderedToolResultHtml {
            collapsed: if !collapsed.is_empty() && collapsed != expanded {
                Some(collapsed)
            } else {
                None
            },
            expanded: Some(expanded),
        })
    }
}

/// Create a tool HTML renderer.
///
/// The renderer looks up tool definitions and invokes their renderCall/renderResult
/// methods, converting the resulting TUI Component output (ANSI) to HTML.
pub fn create_tool_html_renderer(deps: ToolHtmlRendererDeps) -> Box<dyn ToolHtmlRenderer> {
    Box::new(DefaultToolHtmlRenderer {
        get_tool_definition: deps.get_tool_definition,
        theme: deps.theme,
        cwd: deps.cwd,
        width: deps.width.unwrap_or(100),
        rendered_call_components: Mutex::new(HashMap::new()),
        rendered_result_components: Mutex::new(HashMap::new()),
        rendered_states: Mutex::new(HashMap::new()),
        rendered_args: Mutex::new(HashMap::new()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blank_line_handling() {
        assert!(is_blank_rendered_line("   "));
        assert!(is_blank_rendered_line("\u{1b}[31m   \u{1b}[0m"));
        assert!(!is_blank_rendered_line("x"));
    }

    #[test]
    fn trims_leading_and_trailing_blank_rendered_lines() {
        blank_line_handling();
        let lines = vec![
            "".to_string(),
            "  ".to_string(),
            "a".to_string(),
            "\u{1b}[0m ".to_string(),
        ];
        assert_eq!(trim_rendered_result_lines(lines), vec!["a".to_string()]);
    }

    #[test]
    fn trims_all_blank_lines_to_nothing() {
        let lines = vec![" ".to_string(), "".to_string()];
        assert!(trim_rendered_result_lines(lines).is_empty());
    }

    #[test]
    fn renderer_returns_none_without_a_custom_renderer() {
        let renderer = create_tool_html_renderer(ToolHtmlRendererDeps {
            get_tool_definition: Arc::new(|_name: &str| None),
            theme: Theme::default(),
            cwd: "/tmp".to_string(),
            width: None,
        });
        assert_eq!(renderer.render_call("id", "bash", Value::Null), None);
        assert_eq!(
            renderer.render_result("id", "bash", Vec::new(), Value::Null, false),
            None
        );
    }
}
