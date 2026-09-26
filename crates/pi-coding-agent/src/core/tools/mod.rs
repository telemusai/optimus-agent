//! Port of packages/coding-agent/src/core/tools/index.ts
//!
//! Built-in definitions retain typed details locally and convert into the
//! canonical extension definition when registered with an AgentSession.

pub mod acp_mcp;
pub mod bash;
pub mod code_preview;
pub mod edit;
pub mod edit_diff;
pub mod file_mutation_queue;
pub mod ipython;
pub mod node;
pub mod ipython_cell_code;
pub mod output_accumulator;
pub mod path_utils;
pub mod render_utils;
pub mod tool_definition_wrapper;
pub mod truncate;

use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::sync::Arc;

use futures::future::BoxFuture;
use pi_agent_core::types::{AgentToolResult, AgentToolUpdateCallback, ToolExecutionMode};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

pub use acp_mcp::{acp_mcp_tool_names, create_acp_mcp_tool_definitions, AcpMcpServerConfig};
pub use bash::{
    create_bash_tool, create_bash_tool_definition, create_local_bash_operations, BashOperations, BashSpawnContext,
    BashSpawnHook, BashToolDetails, BashToolInput, BashToolOptions, LocalBashOperationsOptions,
};
pub use edit::{
    create_edit_tool, create_edit_tool_definition, EditOperations, EditToolDetails, EditToolInput, EditToolOptions,
};
pub use file_mutation_queue::with_file_mutation_queue;
pub use ipython::{
    create_ipython_tool, create_ipython_tool_definition, IpythonKernelProvisioner, IpythonToolDetails,
    IpythonToolInput, IpythonToolOptions,
};
pub use truncate::{
    format_size, truncate_head, truncate_line, truncate_tail, TruncationOptions, TruncationResult, DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
};

use ipython::create_ipython_tool_definition as create_ipython_tool_definition_inner;

/// TypeScript `ToolName = "ipython"`.
pub type ToolName = &'static str;

/// TypeScript `interface ToolsOptions`.
#[derive(Clone, Default)]
pub struct ToolsOptions {
    pub ipython: Option<IpythonToolOptions>,
}

/// TypeScript `export type ReplayBuiltInToolName = "bash" | "edit"`.
pub const REPLAY_BUILT_IN_TOOL_NAMES: [&str; 2] = ["bash", "edit"];

/// TypeScript `interface ToolRenderResultOptions`.
#[derive(Debug, Clone, Copy, Default)]
pub struct ToolRenderResultOptions {
    /// Whether the result view is expanded
    pub expanded: bool,
    /// Whether this is a partial/streaming result
    pub is_partial: bool,
}

/// TypeScript `interface ExtensionUIDialogOptions`.
#[derive(Clone, Default)]
pub struct ExtensionUiDialogOptions {
    /// AbortSignal to programmatically dismiss the dialog.
    pub signal: Option<CancellationToken>,
    /// Timeout in milliseconds. Dialog auto-dismisses with live countdown display.
    pub timeout: Option<f64>,
}

/// The `ui` member of `ExtensionContext` this package calls.
#[derive(Clone, Default)]
pub struct ExtensionUiContext {
    /// `ctx.ui.setWorkingMessage(message)`.
    pub set_working_message: Option<Arc<dyn Fn(Option<String>) + Send + Sync>>,
    /// `ctx.ui.select(title, options, opts)`.
    pub select: Option<
        Arc<
            dyn Fn(String, Vec<String>, ExtensionUiDialogOptions) -> BoxFuture<'static, Option<String>>
                + Send
                + Sync,
        >,
    >,
}

impl ExtensionUiContext {
    pub fn set_working_message(&self, message: Option<&str>) {
        if let Some(setter) = self.set_working_message.as_ref() {
            setter(message.map(str::to_string));
        }
    }

    pub async fn select(
        &self,
        title: String,
        options: Vec<String>,
        opts: ExtensionUiDialogOptions,
    ) -> Option<String> {
        match self.select.as_ref() {
            Some(select) => select(title, options, opts).await,
            None => None,
        }
    }
}

/// TypeScript `interface ExtensionContext` (only the members this package uses).
#[derive(Clone, Default)]
pub struct ExtensionContext {
    /// Whether an interactive UI is attached (`ctx.hasUI`).
    pub has_ui: bool,
    pub cwd: String,
    pub ui: ExtensionUiContext,
}

/// TypeScript `interface ToolRenderContext<TState, TArgs>` (members this package uses).
pub struct ToolRenderContext<TState, TArgs> {
    pub args: TArgs,
    pub tool_call_id: String,
    pub invalidate: Arc<dyn Fn() + Send + Sync>,
    pub state: TState,
    pub cwd: String,
    pub execution_started: bool,
    pub args_complete: bool,
    pub is_partial: bool,
    pub expanded: bool,
    pub show_expand_hint: Option<bool>,
    pub show_images: bool,
    pub include_image_dimensions: bool,
    pub is_error: bool,
}

/// Plumbing carrier for [`ToolExecuteFn`]: Rust rejects a type alias whose
/// parameter is unused (E0091), so the details parameter is projected through
/// this trait instead of being written directly in the alias body.
///
/// The associated type is the same `dyn Fn` for every details type, which
/// matches the TypeScript signature: `execute` always returns an
/// `AgentToolResult`, and `AgentToolResult` carries details as JSON, so the
/// details type never reaches the callback.
pub trait ToolExecuteFnDetails {
    type Execute: ?Sized;
}

impl<TDetails> ToolExecuteFnDetails for TDetails {
    type Execute = dyn Fn(
            String,
            Value,
            Option<CancellationToken>,
            Option<AgentToolUpdateCallback>,
            ExtensionContext,
        ) -> BoxFuture<'static, Result<AgentToolResult, anyhow::Error>>
        + Send
        + Sync;
}

/// TypeScript `ToolDefinition.execute` signature.
pub type ToolExecuteFn<TDetails> = Arc<<TDetails as ToolExecuteFnDetails>::Execute>;

/// TypeScript `interface ToolDefinition<TParams, TDetails, TState>`.
///
/// `TParams` is a TypeBox schema in TypeScript; the JSON schema value is kept
/// as `Value`. `TDetails` is a phantom parameter because the shared
/// `AgentToolResult` stores details as JSON.
pub struct ToolDefinition<TDetails = Value> {
    /// Tool name (used in LLM tool calls)
    pub name: String,
    /// Human-readable label for UI
    pub label: String,
    /// Description for LLM
    pub description: String,
    /// Optional short text extensions may use when composing custom prompts.
    pub prompt_snippet: Option<String>,
    /// Optional guideline bullets appended to the default system prompt when this tool is active.
    pub prompt_guidelines: Vec<String>,
    /// Parameter schema (TypeBox)
    pub parameters: Value,
    /// Controls whether the tool renders the standard colored shell or its own framing.
    pub render_shell: Option<String>,
    /// Replay renderer to use for removed built-ins in saved transcripts.
    pub replay_built_in_tool_name: Option<String>,
    /// Optional compatibility shim to prepare raw tool call arguments before schema validation.
    pub prepare_arguments: Option<Arc<dyn Fn(Value) -> Value + Send + Sync>>,
    /// Per-tool execution mode override.
    pub execution_mode: Option<ToolExecutionMode>,
    /// Execute the tool.
    pub execute: ToolExecuteFn<TDetails>,
    pub details_type: PhantomData<fn() -> TDetails>,
}

impl<TDetails> Clone for ToolDefinition<TDetails> {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            label: self.label.clone(),
            description: self.description.clone(),
            prompt_snippet: self.prompt_snippet.clone(),
            prompt_guidelines: self.prompt_guidelines.clone(),
            parameters: self.parameters.clone(),
            render_shell: self.render_shell.clone(),
            replay_built_in_tool_name: self.replay_built_in_tool_name.clone(),
            prepare_arguments: self.prepare_arguments.clone(),
            execution_mode: self.execution_mode,
            execute: self.execute.clone(),
            details_type: PhantomData,
        }
    }
}

impl<TDetails> Default for ToolDefinition<TDetails> {
    fn default() -> Self {
        let execute: ToolExecuteFn<TDetails> = Arc::new(
            |_tool_call_id: String,
             _params: Value,
             _signal: Option<CancellationToken>,
             _on_update: Option<AgentToolUpdateCallback>,
             _ctx: ExtensionContext| {
                Box::pin(async move { Err(anyhow::anyhow!("tool has no execute handler")) })
            },
        );
        Self {
            name: String::new(),
            label: String::new(),
            description: String::new(),
            prompt_snippet: None,
            prompt_guidelines: Vec::new(),
            parameters: Value::Null,
            render_shell: None,
            replay_built_in_tool_name: None,
            prepare_arguments: None,
            execution_mode: None,
            execute,
            details_type: PhantomData,
        }
    }
}

impl<TDetails: 'static> From<ToolDefinition<TDetails>> for crate::core::extensions::types::ToolDefinition {
    fn from(definition: ToolDefinition<TDetails>) -> Self {
        let execute = definition.execute;
        Self {
            name: definition.name,
            label: definition.label,
            description: definition.description,
            prompt_snippet: definition.prompt_snippet,
            prompt_guidelines: if definition.prompt_guidelines.is_empty() {
                None
            } else {
                Some(definition.prompt_guidelines)
            },
            parameters: definition.parameters,
            render_shell: definition.render_shell,
            replay_built_in_tool_name: definition.replay_built_in_tool_name,
            prepare_arguments: definition.prepare_arguments,
            execution_mode: definition.execution_mode,
            execute: Arc::new(move |id, params, signal, update, context| {
                let execute = execute.clone();
                let working_ui = context.ui();
                let select_ui = context.ui();
                let context = ExtensionContext {
                    has_ui: context.has_ui(),
                    cwd: context.cwd(),
                    ui: ExtensionUiContext {
                        set_working_message: Some(Arc::new(move |message| working_ui.set_working_message(message))),
                        select: Some(Arc::new(move |title, choices, options| {
                            select_ui.select(title, choices, Some(crate::core::extensions::types::ExtensionUIDialogOptions {
                                signal: options.signal,
                                timeout: options.timeout,
                            }))
                        })),
                    },
                };
                Box::pin(async move {
                    execute(id, params, signal, update, context).await.map_err(|error| error.to_string())
                })
            }),
            render_call: None,
            render_result: None,
        }
    }
}

/// TypeScript `createAllToolDefinitions`.
pub fn create_all_tool_definitions(
    cwd: &str,
    options: Option<&ToolsOptions>,
) -> BTreeMap<ToolName, ToolDefinition<IpythonToolDetails>> {
    let mut definitions = BTreeMap::new();
    definitions.insert(
        "ipython",
        create_ipython_tool_definition_inner(cwd, options.and_then(|options| options.ipython.clone())),
    );
    definitions
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::core::extensions::runner::{ExtensionRunner, NullModelRegistry, NullSessionManager};
    use crate::core::extensions::types::{ExtensionRuntime, ExtensionRuntimeState};

    #[tokio::test]
    async fn builtin_conversion_preserves_metadata_and_live_extension_context() {
        let builtin = ToolDefinition::<Value> {
            name: "fixture".to_string(),
            label: "Fixture".to_string(),
            description: "Fixture tool".to_string(),
            prompt_snippet: Some("fixture prompt".to_string()),
            prompt_guidelines: vec!["fixture guideline".to_string()],
            parameters: serde_json::json!({"type": "object"}),
            execute: Arc::new(|_, _, _, _, context| Box::pin(async move {
                Ok(tool_definition_wrapper::text_tool_result(context.cwd, Value::Null))
            })),
            ..ToolDefinition::default()
        };
        let definition: crate::core::extensions::types::ToolDefinition = builtin.into();
        assert_eq!(definition.prompt_snippet.as_deref(), Some("fixture prompt"));
        assert_eq!(definition.prompt_guidelines, Some(vec!["fixture guideline".to_string()]));
        assert_eq!(definition.parameters, serde_json::json!({"type": "object"}));
        let runner = Arc::new(ExtensionRunner::new(
            Vec::new(),
            ExtensionRuntime::new(ExtensionRuntimeState::default()),
            "fixture-directory".to_string(),
            Arc::new(NullSessionManager),
            Arc::new(NullModelRegistry),
        ));
        let result = (definition.execute)("call".to_string(), Value::Null, None, None, runner.create_context()).await.unwrap();
        assert_eq!(result.content[0].as_text(), Some("fixture-directory"));
    }
}
