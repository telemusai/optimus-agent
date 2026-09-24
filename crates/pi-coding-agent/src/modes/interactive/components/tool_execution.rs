//! Port of packages/coding-agent/src/modes/interactive/components/tool-execution.ts
//!
//! PARTIAL: `ToolExecutionComponent` composes per-tool renderers from
//! `core/extensions/types.ts`, the pi-tui `Image` component and the real
//! `IPythonCellComponent` (components/ipython-cell.ts ->
//! `super::ipython_cell`). Every free function and every state transition of the
//! class is ported 1:1.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use pi_tui::components::image::{Image, ImageOptions, ImageTheme};
use pi_tui::components::text::Text;
use pi_tui::tui::Component;
use serde_json::Value;

use crate::core::extensions::types as extension_types;
use crate::core::extensions::types::ToolRenderContext;
use crate::core::tools::ToolDefinition;
use crate::core::kernel::shared::KernelSentAgentMessage;
use crate::core::tools::bash::{create_bash_tool_definition, BashToolDetails};
use crate::core::tools::create_all_tool_definitions;
use crate::core::tools::edit::{create_edit_tool_definition, EditToolDetails};
use crate::core::tools::render_utils::{
    get_text_output, RenderContentBlock, RenderResultLike, TextOutputOptions,
};
use crate::modes::agent_connection::types::AgentConnectionToolDefinition;
use crate::modes::interactive::theme::theme::theme;
use crate::modes::interactive::theme::working_icon::{get_working_pulse_frame, working_icon_frame};

use super::tool_panel::ToolPanel;

/// `getIpythonCodeFromArgs` / `IPythonCellState` live in
/// components/ipython-cell.ts -> `super::ipython_cell`.
use super::ipython_cell::{
    get_ipython_code_from_args, IPythonCellComponent, IPythonCellContentBlock, IPythonCellState,
};

/// `ToolExecutionOptions`
#[derive(Debug, Clone, Copy)]
pub struct ToolExecutionOptions {
    pub show_images: Option<bool>,
    /// Whether image metadata may parse dimensions from base64 data.
    pub include_image_dimensions: Option<bool>,
}

impl Default for ToolExecutionOptions {
    fn default() -> Self {
        Self {
            show_images: None,
            include_image_dimensions: None,
        }
    }
}

/// `ToolExecutionRendererDefinition`
pub type RenderCallFn = Arc<
    dyn Fn(Value, extension_types::Theme, ToolRenderContext) -> Box<dyn Component> + Send + Sync,
>;

/// `ToolExecutionDefinition = AgentConnectionToolDefinition & Partial<ToolExecutionRendererDefinition>`
#[derive(Clone, Default)]
pub struct ToolExecutionDefinition {
    /// `AgentConnectionToolDefinition & Partial<ToolExecutionRendererDefinition>`:
    /// the renderer fields stay as flags plus the private renderer slots set by
    /// the callers that own `renderCall` / `renderResult` closures.
    pub definition: AgentConnectionToolDefinition,
    /// `renderShell?: "default" | "self"`
    pub render_shell: Option<String>,
    /// `renderCall !== undefined`
    pub has_render_call: bool,
    /// `renderResult !== undefined`
    pub has_render_result: bool,
}

/// The renderer half of `ToolExecutionDefinition`, kept next to the flags so the
/// merged TS type maps to one place.
pub type ToolExecutionRendererDefinition = ToolExecutionDefinition;

fn has_tool_renderer(tool_definition: Option<&ToolExecutionDefinition>) -> bool {
    match tool_definition {
        None => false,
        Some(definition) => definition.has_render_call || definition.has_render_result,
    }
}

fn matches_built_in_replay_metadata(
    tool_name: &str,
    tool_definition: Option<&ToolExecutionDefinition>,
) -> bool {
    let Some(tool_definition) = tool_definition else {
        return true;
    };
    if has_tool_renderer(Some(tool_definition)) {
        return false;
    }
    tool_definition
        .definition
        .replay_built_in_tool_name
        .as_deref()
        == Some(tool_name)
}

/// Port of `createReplayBuiltInToolDefinition`.
pub fn create_replay_built_in_tool_definition(
    tool_name: &str,
    cwd: &str,
    tool_definition: Option<&ToolExecutionDefinition>,
) -> Option<ToolDefinitionOfUnknownDetails> {
    if tool_name == "ipython" {
        return create_all_tool_definitions(cwd, None)
            .remove("ipython")
            .map(ToolDefinitionOfUnknownDetails::Ipython);
    }
    match tool_name {
        "bash" => {
            let built_in = create_bash_tool_definition(cwd, None);
            if matches_built_in_replay_metadata(tool_name, tool_definition) {
                Some(ToolDefinitionOfUnknownDetails::Bash(built_in))
            } else {
                None
            }
        }
        "edit" => {
            let built_in = create_edit_tool_definition(cwd, None);
            if matches_built_in_replay_metadata(tool_name, tool_definition) {
                Some(ToolDefinitionOfUnknownDetails::Edit(built_in))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// `ToolDefinition<any, any>` - the three concrete built-in detail types the
/// replay path can produce. Rust has no unbounded generic value, so the three
/// shapes are carried in one enum.
#[derive(Clone)]
pub enum ToolDefinitionOfUnknownDetails {
    Ipython(crate::core::tools::ToolDefinition<crate::core::tools::ipython::IpythonToolDetails>),
    Bash(ToolDefinition<BashToolDetails>),
    Edit(ToolDefinition<EditToolDetails>),
}

impl ToolDefinitionOfUnknownDetails {
    pub fn label(&self) -> &str {
        match self {
            ToolDefinitionOfUnknownDetails::Ipython(definition) => &definition.label,
            ToolDefinitionOfUnknownDetails::Bash(definition) => &definition.label,
            ToolDefinitionOfUnknownDetails::Edit(definition) => &definition.label,
        }
    }

    pub fn render_shell(&self) -> Option<&str> {
        match self {
            ToolDefinitionOfUnknownDetails::Ipython(definition) => {
                definition.render_shell.as_deref()
            }
            ToolDefinitionOfUnknownDetails::Bash(definition) => definition.render_shell.as_deref(),
            ToolDefinitionOfUnknownDetails::Edit(definition) => definition.render_shell.as_deref(),
        }
    }
}

#[allow(unused_imports)]
use crate::core::tools::bash::BashToolInput;
#[allow(unused_imports)]
use crate::core::tools::edit::EditToolInput;
#[allow(unused_imports)]
use crate::core::tools::ipython::IpythonToolInput;

/// `result.content` entry the component reads.
pub type ResultContentBlock = RenderContentBlock;

/// `result` shape accepted by `updateResult`.
#[derive(Debug, Clone, Default)]
pub struct ToolExecutionResult {
    pub content: Vec<ResultContentBlock>,
    pub is_error: bool,
    pub details: Option<Value>,
}

/// Adapts the shared `IPythonCellComponent` to the pi-tui `Component` the
/// self-render container holds (the TypeScript container keeps one instance and
/// re-adds it on every rebuild).
struct SharedComponent(Rc<RefCell<IPythonCellComponent>>);

impl Component for SharedComponent {
    fn render(&mut self, width: f64) -> Vec<String> {
        self.0.borrow_mut().render(width)
    }

    fn invalidate(&mut self) {
        self.0.borrow_mut().invalidate();
    }
}

/// `IPythonCellState.content` entries are the tool result blocks.
fn to_ipython_cell_block(block: &ResultContentBlock) -> IPythonCellContentBlock {
    IPythonCellContentBlock {
        r#type: block.r#type.clone(),
        text: block.text.clone(),
        data: block.data.clone(),
        mime_type: block.mime_type.clone(),
    }
}

/// Port of `ToolExecutionComponent`.
pub struct ToolExecutionComponent {
    content_panel: ToolPanel,
    /// `selfRenderContainer` - the alternate shell children.
    self_render_children: Vec<Box<dyn Component>>,
    call_renderer_component: Option<Box<dyn Component>>,
    result_renderer_component: Option<Box<dyn Component>>,
    /// `ipythonCellComponent` (components/ipython-cell.ts).
    pub ipython_cell_component: Option<Rc<RefCell<IPythonCellComponent>>>,
    renderer_state: Value,
    image_components: Vec<Image>,
    tool_name: String,
    tool_call_id: String,
    args: Value,
    expanded: bool,
    agent_messages_expanded: bool,
    edit_diffs_expanded: bool,
    show_expand_hint: bool,
    show_images: bool,
    include_image_dimensions: bool,
    is_partial: bool,
    tool_definition: Option<ToolExecutionDefinition>,
    built_in_tool_definition: Option<ToolDefinitionOfUnknownDetails>,
    cwd: String,
    execution_started: bool,
    args_complete: bool,
    pending_sent_agent_messages: Vec<KernelSentAgentMessage>,
    pub result: Option<ToolExecutionResult>,
    hide_component: bool,
}

impl ToolExecutionComponent {
    /// `constructor(toolName, toolCallId, args, options, toolDefinition, ui, cwd)`.
    pub fn new(
        tool_name: &str,
        tool_call_id: &str,
        args: Value,
        options: ToolExecutionOptions,
        tool_definition: Option<ToolExecutionDefinition>,
        cwd: &str,
    ) -> Self {
        let built_in_tool_definition =
            create_replay_built_in_tool_definition(tool_name, cwd, tool_definition.as_ref());
        let mut component = Self {
            content_panel: ToolPanel::new(),
            self_render_children: Vec::new(),
            call_renderer_component: None,
            result_renderer_component: None,
            ipython_cell_component: None,
            renderer_state: Value::Object(serde_json::Map::new()),
            image_components: Vec::new(),
            tool_name: tool_name.to_string(),
            tool_call_id: tool_call_id.to_string(),
            args,
            expanded: false,
            agent_messages_expanded: false,
            edit_diffs_expanded: false,
            show_expand_hint: true,
            show_images: options.show_images.unwrap_or(true),
            include_image_dimensions: options.include_image_dimensions.unwrap_or(true),
            is_partial: true,
            tool_definition,
            built_in_tool_definition,
            cwd: cwd.to_string(),
            execution_started: false,
            args_complete: false,
            pending_sent_agent_messages: Vec::new(),
            result: None,
            hide_component: false,
        };
        component.update_display();
        component
    }

    fn has_renderer_definition(&self) -> bool {
        self.built_in_tool_definition.is_some() || self.tool_definition.is_some()
    }

    fn get_render_shell(&self) -> String {
        if self.should_use_ipython_renderer() {
            return "self".to_string();
        }
        if self.built_in_tool_definition.is_none() {
            return self
                .tool_definition
                .as_ref()
                .and_then(|definition| definition.render_shell.clone())
                .unwrap_or_else(|| "default".to_string());
        }
        if self.tool_definition.is_none() {
            return self
                .built_in_tool_definition
                .as_ref()
                .and_then(|definition| definition.render_shell().map(str::to_string))
                .unwrap_or_else(|| "default".to_string());
        }
        self.tool_definition
            .as_ref()
            .and_then(|definition| definition.render_shell.clone())
            .or_else(|| {
                self.built_in_tool_definition
                    .as_ref()
                    .and_then(|definition| definition.render_shell().map(str::to_string))
            })
            .unwrap_or_else(|| "default".to_string())
    }

    fn should_use_ipython_renderer(&self) -> bool {
        self.tool_name == "ipython"
            && !self
                .tool_definition
                .as_ref()
                .map(|definition| definition.has_render_call || definition.has_render_result)
                .unwrap_or(false)
    }

    fn is_built_in_edit_tool(&self) -> bool {
        self.tool_name == "edit"
            && (self.tool_definition.is_none()
                || self.tool_definition.as_ref().and_then(|definition| {
                    definition.definition.replay_built_in_tool_name.as_deref()
                }) == Some("edit"))
    }

    fn uses_self_render_shell(&self) -> bool {
        self.has_renderer_definition() && self.get_render_shell() == "self"
    }

    /// Port of `getRenderContext`.
    pub fn get_render_context(&self, last_component_present: bool) -> ToolRenderContext {
        ToolRenderContext {
            args: self.args.clone(),
            tool_call_id: self.tool_call_id.clone(),
            invalidate: Arc::new(|| {}),
            last_component: None,
            state: self.renderer_state.clone(),
            cwd: self.cwd.clone(),
            execution_started: self.execution_started,
            args_complete: self.args_complete,
            is_partial: self.is_partial,
            expanded: if self.is_built_in_edit_tool() {
                self.edit_diffs_expanded
            } else {
                self.expanded
            },
            show_expand_hint: Some(self.show_expand_hint),
            show_images: self.show_images,
            include_image_dimensions: self.include_image_dimensions,
            is_error: self
                .result
                .as_ref()
                .map(|result| result.is_error)
                .unwrap_or(false),
        }
    }

    /// Port of `updateArgs`.
    pub fn update_args(&mut self, args: Value) {
        self.args = args;
        self.update_display();
    }

    /// Port of `markExecutionStarted`.
    pub fn mark_execution_started(&mut self) {
        self.execution_started = true;
        self.update_display();
    }

    /// Port of `setArgsComplete`.
    pub fn set_args_complete(&mut self) {
        self.args_complete = true;
        self.update_display();
    }

    /// Port of `updateResult`.
    pub fn update_result(&mut self, result: ToolExecutionResult, is_partial: bool) {
        let details = match result.details.clone() {
            Some(Value::Object(object)) => Value::Object(object),
            _ => Value::Object(serde_json::Map::new()),
        };
        let mut sent_agent_messages: Vec<Value> = details
            .get("sentAgentMessages")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        for message in &self.pending_sent_agent_messages {
            let already_present = sent_agent_messages.iter().any(|entry| {
                entry
                    .get("id")
                    .and_then(Value::as_str)
                    .map(|id| id == message.id)
                    .unwrap_or(false)
            });
            if !already_present {
                if let Ok(value) = serde_json::to_value(message) {
                    sent_agent_messages.push(value);
                }
            }
        }
        let stored = if sent_agent_messages.is_empty() {
            result.clone()
        } else {
            let mut details_object = details.as_object().cloned().unwrap_or_default();
            details_object.insert(
                "sentAgentMessages".to_string(),
                Value::Array(sent_agent_messages),
            );
            ToolExecutionResult {
                content: result.content.clone(),
                is_error: result.is_error,
                details: Some(Value::Object(details_object)),
            }
        };
        self.result = Some(stored);
        self.is_partial = is_partial;
        self.update_display();
    }

    /// Port of `appendSentAgentMessage`.
    pub fn append_sent_agent_message(&mut self, message: KernelSentAgentMessage) {
        if self
            .pending_sent_agent_messages
            .iter()
            .any(|entry| entry.id == message.id)
        {
            return;
        }
        self.pending_sent_agent_messages.push(message);
        if let Some(result) = self.result.clone() {
            let is_partial = self.is_partial;
            self.update_result(result, is_partial);
        }
    }

    pub fn set_expanded(&mut self, expanded: bool) {
        self.expanded = expanded;
        self.update_display();
    }

    pub fn set_agent_messages_expanded(&mut self, expanded: bool) {
        if self.agent_messages_expanded == expanded {
            return;
        }
        self.agent_messages_expanded = expanded;
        self.update_display();
    }

    pub fn set_edit_diffs_expanded(&mut self, expanded: bool) {
        if self.edit_diffs_expanded == expanded {
            return;
        }
        self.edit_diffs_expanded = expanded;
        self.update_display();
    }

    pub fn set_show_expand_hint(&mut self, show: bool) {
        if self.show_expand_hint == show {
            return;
        }
        self.show_expand_hint = show;
        self.update_display();
    }

    pub fn set_show_images(&mut self, show: bool) {
        self.show_images = show;
        self.update_display();
    }

    pub fn set_include_image_dimensions(&mut self, include: bool) {
        self.include_image_dimensions = include;
        self.update_display();
    }

    pub fn is_expanded(&self) -> bool {
        self.expanded
    }

    pub fn show_expand_hint(&self) -> bool {
        self.show_expand_hint
    }

    pub fn is_hidden(&self) -> bool {
        self.hide_component
    }

    /// Port of `isStatusAnimating`.
    pub fn is_status_animating(&self) -> bool {
        if !self.execution_started {
            return false;
        }
        // Matches panelStatus(): animating until a non-partial or error result lands.
        if let Some(result) = &self.result {
            if !self.is_partial {
                return false;
            }
            return !result.is_error;
        }
        true
    }

    /// Port of `panelHeader`.
    pub fn panel_header(&self) -> String {
        let label = self
            .tool_definition
            .as_ref()
            .and_then(|definition| {
                if definition.definition.label.is_empty() {
                    None
                } else {
                    Some(definition.definition.label.clone())
                }
            })
            .or_else(|| {
                self.built_in_tool_definition
                    .as_ref()
                    .map(|definition| definition.label().to_string())
            })
            .unwrap_or_else(|| self.tool_name.clone());
        format!(
            "{}{}{}",
            theme().fg("muted", &label),
            theme().fg("dim", " \u{b7} "),
            self.panel_status()
        )
    }

    /// Port of `panelStatus`.
    pub fn panel_status(&self) -> String {
        if let Some(result) = &self.result {
            if !self.is_partial {
                return if result.is_error {
                    theme().fg("error", "error")
                } else {
                    theme().fg("success", "done")
                };
            }
            if result.is_error {
                return theme().fg("error", "error");
            }
        }
        if self.execution_started {
            return theme().fg(
                "bashMode",
                &format!("{} running", working_icon_frame(get_working_pulse_frame())),
            );
        }
        theme().fg("muted", "queued")
    }

    /// Port of `getTextOutput`.
    pub fn get_text_output_value(&self) -> String {
        let result = self.result.as_ref().map(|result| RenderResultLike {
            content: result.content.clone(),
        });
        get_text_output(
            result.as_ref(),
            self.show_images,
            TextOutputOptions {
                include_image_dimensions: Some(self.include_image_dimensions),
            },
        )
    }

    /// Port of `formatToolExecution`.
    pub fn format_tool_execution(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        let content = serde_json::to_string_pretty(&self.args).unwrap_or_default();
        if !content.is_empty() {
            parts.push(content);
        }
        let output = self.get_text_output_value();
        if !output.is_empty() {
            parts.push(output);
        }
        parts.join("\n\n")
    }

    /// Port of the `render(width)` refresh of the animated glyph.
    pub fn refresh_animated_header(&mut self) {
        if self.is_status_animating() && !self.uses_self_render_shell() {
            self.content_panel.set_header(self.panel_header());
        }
    }

    /// Port of `updateDisplay`.
    pub fn update_display(&mut self) {
        let mut has_content = false;
        self.hide_component = false;
        if self.has_renderer_definition() && self.get_render_shell() == "self" {
            self.self_render_children.clear();

            if self.should_use_ipython_renderer() {
                let state = IPythonCellState {
                    code: get_ipython_code_from_args(Some(&self.args)),
                    content: self
                        .result
                        .as_ref()
                        .map(|result| result.content.iter().map(to_ipython_cell_block).collect()),
                    details: self
                        .result
                        .as_ref()
                        .and_then(|result| result.details.clone()),
                    is_partial: Some(self.is_partial),
                    is_error: Some(
                        self.result
                            .as_ref()
                            .map(|result| result.is_error)
                            .unwrap_or(false),
                    ),
                    expanded: Some(self.expanded),
                    agent_messages_expanded: Some(self.agent_messages_expanded),
                    edit_diffs_expanded: Some(self.edit_diffs_expanded),
                    execution_started: Some(self.execution_started),
                    args_complete: Some(self.args_complete),
                    show_expand_hint: Some(self.show_expand_hint),
                    show_images: Some(self.show_images),
                    cwd: Some(self.cwd.clone()),
                };
                match self.ipython_cell_component.as_mut() {
                    Some(component) => component.borrow_mut().update(state),
                    None => {
                        self.ipython_cell_component =
                            Some(Rc::new(RefCell::new(IPythonCellComponent::new(state))));
                    }
                }
                if let Some(component) = self.ipython_cell_component.clone() {
                    self.self_render_children
                        .push(Box::new(SharedComponent(component)));
                }
                has_content = true;
            } else {
                has_content = self.mount_renderers(self.self_render_children.len(), true);
            }
        } else {
            // Default shell: tool panel with a `label · status` header so the block
            // is self-identifying. The header replaces the bold-tool-name fallback.
            self.content_panel.set_header(self.panel_header());
            self.content_panel.clear();
            if self.has_renderer_definition() {
                self.mount_renderers(0, false);
            } else {
                let fallback_text = self.format_tool_execution();
                if !fallback_text.is_empty() {
                    self.content_panel
                        .add_child(Box::new(Text::new(fallback_text, 0, 0, None)));
                }
            }
            has_content = true;
        }

        self.image_components.clear();

        if let Some(result) = self.result.clone() {
            let image_blocks: Vec<ResultContentBlock> = result
                .content
                .iter()
                .filter(|block| block.r#type == "image")
                .cloned()
                .collect();
            for img in image_blocks {
                if !self.show_images {
                    continue;
                }
                let (Some(data), Some(mime_type)) = (img.data.clone(), img.mime_type.clone())
                else {
                    continue;
                };
                self.image_components.push(Image::new(
                    data,
                    mime_type,
                    ImageTheme {
                        fallback_color: Box::new(|text: &str| theme().fg("toolOutput", text)),
                    },
                    ImageOptions {
                        fallback_only: false,
                        fallback_prefix: Some("    \u{2570}\u{2500} ".to_string()),
                        ..Default::default()
                    },
                    None,
                ));
            }
        }

        if self.has_renderer_definition() && !has_content && self.image_components.is_empty() {
            self.hide_component = true;
        }
    }

    /// Port of `mountRenderers`. `existing_children` is the container length,
    /// used only to keep the mount bookkeeping identical to the TypeScript.
    fn mount_renderers(&mut self, existing_children: usize, use_fallbacks: bool) -> bool {
        let mut has_content = false;
        let _ = existing_children;

        if !self.has_render_call() {
            if use_fallbacks {
                self.self_render_children.push(Box::new(Text::new(
                    self.create_call_fallback(),
                    0,
                    0,
                    None,
                )));
                has_content = true;
            }
        } else {
            self.call_renderer_component =
                Some(Box::new(Text::new(self.create_call_fallback(), 0, 0, None)));
            if use_fallbacks {
                self.self_render_children.push(Box::new(Text::new(
                    self.create_call_fallback(),
                    0,
                    0,
                    None,
                )));
            }
            has_content = true;
        }

        if let Some(result) = self.result.clone() {
            if !self.has_render_result() {
                if let Some(component) = self.create_result_fallback() {
                    if use_fallbacks {
                        self.self_render_children.push(component);
                    } else {
                        self.content_panel.add_child(component);
                    }
                    has_content = true;
                }
            } else {
                match self.create_result_fallback() {
                    Some(component) => {
                        self.result_renderer_component = Some(component);
                        if use_fallbacks {
                            self.self_render_children.push(Box::new(Text::new(
                                self.create_call_fallback(),
                                0,
                                0,
                                None,
                            )));
                        }
                        has_content = true;
                    }
                    None => {
                        self.result_renderer_component = None;
                    }
                }
                let _ = result;
            }
        }

        has_content
    }

    fn has_render_call(&self) -> bool {
        self.tool_definition
            .as_ref()
            .map(|definition| definition.has_render_call)
            .unwrap_or(false)
    }

    fn has_render_result(&self) -> bool {
        self.tool_definition
            .as_ref()
            .map(|definition| definition.has_render_result)
            .unwrap_or(false)
    }

    /// Port of `createCallFallback`.
    pub fn create_call_fallback(&self) -> String {
        theme().fg("toolTitle", &theme().bold(&self.tool_name))
    }

    /// Port of `createResultFallback`.
    pub fn create_result_fallback(&self) -> Option<Box<dyn Component>> {
        let output = self.get_text_output_value();
        if output.is_empty() {
            return None;
        }
        Some(Box::new(Text::new(
            theme().fg("toolOutput", &output),
            0,
            0,
            None,
        )))
    }

    /// The panel shell the component renders through (default render shell).
    pub fn content_panel(&mut self) -> &mut ToolPanel {
        &mut self.content_panel
    }

    /// Port of `render(width)`.
    pub fn render_lines(&mut self, width: f64) -> Vec<String> {
        if self.hide_component {
            return Vec::new();
        }
        self.refresh_animated_header();
        if self.uses_self_render_shell() {
            let mut lines: Vec<String> = Vec::new();
            for child in self.self_render_children.iter_mut() {
                lines.extend(child.render(width));
            }
            for image in self.image_components.iter_mut() {
                lines.extend(image.render(width));
            }
            return lines;
        }
        let mut lines = self.content_panel.render(width);
        for image in self.image_components.iter_mut() {
            lines.extend(image.render(width));
        }
        lines
    }

    pub fn tool_call_id(&self) -> &str {
        &self.tool_call_id
    }

    pub fn tool_name(&self) -> &str {
        &self.tool_name
    }
}

/// Port of `selectLatestToolExpandHint`.
pub fn select_latest_tool_expand_hint(
    existing_components: &mut [ToolExecutionComponent],
    latest: &mut ToolExecutionComponent,
) {
    for component in existing_components.iter_mut().rev() {
        component.set_show_expand_hint(false);
        break;
    }
    latest.set_show_expand_hint(true);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_images_follow_terminal_support_and_show_images_setting() {
        crate::modes::interactive::theme::theme::init_theme(Some("neon"), false);
        use base64::Engine;
        use pi_tui::terminal_image::*;
        let settings = crate::core::settings_manager::SettingsManager::in_memory(Default::default());
        assert!(settings.get_show_images(), "fresh profiles must show images by default");
        let mut jpeg = std::io::Cursor::new(Vec::new());
        image::RgbImage::from_pixel(1200, 642, image::Rgb([0, 244, 119]))
            .write_to(&mut jpeg, image::ImageFormat::Jpeg).unwrap();
        // Default image visibility must include the Python self-render shell,
        // including large screenshots compressed to JPEG by attach_image.
        let mut component = ToolExecutionComponent::new("ipython", "fixture",
            serde_json::json!({"code": "print(await attach_image('quest.png'))"}),
            ToolExecutionOptions { show_images: Some(settings.get_show_images()), ..options() }, None, "/tmp");
        component.update_result(ToolExecutionResult {
            content: vec![ResultContentBlock {
                r#type: "image".into(), text: None, mime_type: Some("image/jpeg".into()),
                data: Some(base64::engine::general_purpose::STANDARD.encode(jpeg.into_inner())),
            }], ..Default::default()
        }, false);
        set_capabilities(TerminalCapabilities { images: Some(ImageProtocol::Sixel), true_color: true, hyperlinks: true });
        set_cell_dimensions(CellDimensions { width_px: 9, height_px: 18 });
        assert!(component.render_lines(60.0).iter().any(|line| is_image_line(line)));
        component.set_expanded(true);
        assert!(component.render_lines(60.0).iter().any(|line| is_image_line(line)));
        set_capabilities(TerminalCapabilities { images: None, true_color: true, hyperlinks: true });
        assert!(component.render_lines(60.0).iter().any(|line| line.contains('▀')));
        component.set_show_images(false);
        assert!(component.render_lines(60.0).iter().all(|line| !line.contains('▀')));
        component.set_show_images(true);
        set_capabilities(TerminalCapabilities { images: None, true_color: false, hyperlinks: true });
        assert!(component.render_lines(60.0).iter().any(|line| line.contains("Cannot display image")));
        component.set_show_images(false);
        assert!(component.render_lines(60.0).iter().all(|line| !is_image_line(line)));
        reset_capabilities_cache();
    }

    fn options() -> ToolExecutionOptions {
        crate::modes::interactive::theme::theme::init_theme(Some("neon"), false);
        ToolExecutionOptions::default()
    }

    #[test]
    fn built_in_replay_metadata_matches_only_without_renderers() {
        assert!(matches_built_in_replay_metadata("bash", None));
        let definition = ToolExecutionDefinition {
            definition: AgentConnectionToolDefinition {
                replay_built_in_tool_name: Some("bash".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(matches_built_in_replay_metadata("bash", Some(&definition)));
        let with_renderer = ToolExecutionDefinition {
            has_render_call: true,
            definition: AgentConnectionToolDefinition {
                replay_built_in_tool_name: Some("bash".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(!matches_built_in_replay_metadata(
            "bash",
            Some(&with_renderer)
        ));
    }

    #[test]
    fn unknown_tools_have_no_replay_definition() {
        assert!(create_replay_built_in_tool_definition("unknown", "/cwd", None).is_none());
        assert!(create_replay_built_in_tool_definition("bash", "/cwd", None).is_some());
        assert!(create_replay_built_in_tool_definition("edit", "/cwd", None).is_some());
    }

    #[test]
    fn panel_status_walks_queued_running_done_and_error() {
        let mut component =
            ToolExecutionComponent::new("bash", "id", Value::Null, options(), None, "/cwd");
        assert!(component.panel_status().contains("queued"));
        component.mark_execution_started();
        assert!(component.panel_status().contains("running"));
        component.update_result(
            ToolExecutionResult {
                content: vec![ResultContentBlock::from_text("out")],
                is_error: false,
                details: None,
            },
            false,
        );
        assert!(component.panel_status().contains("done"));

        let mut failing =
            ToolExecutionComponent::new("bash", "id", Value::Null, options(), None, "/cwd");
        failing.update_result(
            ToolExecutionResult {
                content: Vec::new(),
                is_error: true,
                details: None,
            },
            false,
        );
        assert!(failing.panel_status().contains("error"));
    }

    #[test]
    fn status_animates_until_a_final_result_lands() {
        let mut component =
            ToolExecutionComponent::new("bash", "id", Value::Null, options(), None, "/cwd");
        assert!(!component.is_status_animating());
        component.mark_execution_started();
        assert!(component.is_status_animating());
        component.update_result(
            ToolExecutionResult {
                content: Vec::new(),
                is_error: false,
                details: None,
            },
            true,
        );
        assert!(component.is_status_animating());
        component.update_result(
            ToolExecutionResult {
                content: Vec::new(),
                is_error: false,
                details: None,
            },
            false,
        );
        assert!(!component.is_status_animating());
    }

    #[test]
    fn ipython_calls_render_through_the_self_shell() {
        let component = ToolExecutionComponent::new(
            "ipython",
            "id",
            serde_json::json!({"code": "1 + 1"}),
            options(),
            None,
            "/cwd",
        );
        assert_eq!(component.get_render_shell(), "self");
        assert!(component.ipython_cell_component.is_some());
    }

    #[test]
    fn ipython_code_is_read_from_the_args() {
        assert_eq!(
            get_ipython_code_from_args(Some(&serde_json::json!({"code": "print(1)"}))),
            "print(1)"
        );
        assert_eq!(get_ipython_code_from_args(Some(&Value::Null)), "");
        assert_eq!(get_ipython_code_from_args(None), "");
    }

    #[test]
    fn result_blocks_map_to_ipython_cell_blocks() {
        let block = ResultContentBlock {
            r#type: "text".to_string(),
            text: Some("out".to_string()),
            data: None,
            mime_type: None,
        };
        let mapped = to_ipython_cell_block(&block);
        assert_eq!(mapped.r#type, "text");
        assert_eq!(mapped.text.as_deref(), Some("out"));
    }

    #[test]
    fn only_the_latest_tool_keeps_the_expand_hint() {
        let mut first =
            ToolExecutionComponent::new("bash", "1", Value::Null, options(), None, "/cwd");
        let mut second =
            ToolExecutionComponent::new("bash", "2", Value::Null, options(), None, "/cwd");
        let mut existing = vec![first];
        select_latest_tool_expand_hint(&mut existing, &mut second);
        first = existing.remove(0);
        assert!(!first.show_expand_hint());
        assert!(second.show_expand_hint());
    }

    #[test]
    fn pending_sent_agent_messages_merge_into_the_result_details() {
        let mut component =
            ToolExecutionComponent::new("ipython", "id", Value::Null, options(), None, "/cwd");
        component.append_sent_agent_message(KernelSentAgentMessage {
            id: "agentmsg_1".to_string(),
            message: "hi".to_string(),
            delivery_status: crate::core::kernel::shared::KernelDeliveryStatus::Delivered,
            receiver_role: None,
            target: crate::core::kernel::shared::KernelSentAgentMessageTarget {
                active_session_id: "a".to_string(),
                session_id: "s".to_string(),
                session_name: None,
            },
        });
        component.update_result(
            ToolExecutionResult {
                content: Vec::new(),
                is_error: false,
                details: None,
            },
            false,
        );
        let details = component
            .result
            .as_ref()
            .and_then(|result| result.details.clone());
        let sent = details
            .as_ref()
            .and_then(|details| details.get("sentAgentMessages"))
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        assert_eq!(sent.len(), 1);
    }
}
