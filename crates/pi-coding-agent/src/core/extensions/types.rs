//! Port of packages/coding-agent/src/core/extensions/types.ts
//!
//! Extension system types: lifecycle events, tools, commands, UI context.
//!
//! Cross-crate types that do not exist yet are declared here as minimal local
//! definitions (see `blocked_on` in evidence/status/ca-extensions.json).

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokio_util::sync::CancellationToken;

use crate::core::diagnostics::ResourceDiagnostic;
use crate::core::refinement::refinement::{HarnessState, RefinementProposal, RefinementResult};
use crate::core::slash_commands::SlashCommandInfo;
use crate::core::source_info::SourceInfo;

// ---------------------------------------------------------------------------
// Minimal local stand-ins for types owned by other slices.
// blocked_on: needs modes::interactive::theme::theme::Theme
// ---------------------------------------------------------------------------

/// blocked_on: needs modes::interactive::theme::theme::Theme
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Theme {
    pub name: Option<String>,
    #[serde(rename = "sourcePath", skip_serializing_if = "Option::is_none")]
    pub source_path: Option<String>,
    #[serde(rename = "sourceInfo", skip_serializing_if = "Option::is_none")]
    pub source_info: Option<SourceInfo>,
    /// Raw theme document (colors etc.); the theme slice owns the typed shape.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// blocked_on: needs pi-tui::components::Component
pub trait Component: Send + Sync {
    fn render(&self, width: usize) -> Vec<String>;
    fn handle_input(&self, _data: &str) {}
    fn set_focused(&self, _focused: bool) {}
    fn invalidate(&self) {}
    fn dispose(&self) {}
}

/// blocked_on: needs pi-tui::autocomplete::AutocompleteItem
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AutocompleteItem {
    pub value: String,
    pub label: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(rename = "argumentHint", skip_serializing_if = "Option::is_none")]
    pub argument_hint: Option<String>,
    #[serde(rename = "sourceTag", skip_serializing_if = "Option::is_none")]
    pub source_tag: Option<String>,
    #[serde(rename = "takesArgument", skip_serializing_if = "Option::is_none")]
    pub takes_argument: Option<bool>,
}

/// blocked_on: needs pi-tui::autocomplete::AutocompleteProvider
pub trait AutocompleteProvider: Send + Sync {
    fn get_suggestions(
        &self,
        lines: Vec<String>,
        cursor_line: usize,
        cursor_col: usize,
        signal: CancellationToken,
        force: Option<bool>,
    ) -> Pin<Box<dyn std::future::Future<Output = Option<Value>> + Send>>;
}

/// blocked_on: needs pi-tui::components::EditorComponent
pub trait EditorComponent: Component {}

/// blocked_on: needs pi-tui::tui::TUI
pub trait Tui: Send + Sync {
    fn request_render(&self);
}

/// blocked_on: needs pi-tui::components::OverlayOptions
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct OverlayOptions {
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// blocked_on: needs pi-tui::components::OverlayHandle
pub trait OverlayHandle: Send + Sync {
    fn hide(&self);
    fn set_hidden(&self, hidden: bool);
}

/// `KeyId` from @earendil-works/pi-tui.
pub type KeyId = String;

/// `AbortSignal`.
pub type AbortSignal = CancellationToken;

/// `EventBus` from `core/event-bus.ts`.
///
/// The concrete bus now lives in `core::event_bus` (another slice), so the
/// extension layer re-exports it instead of keeping a local stand-in.
pub use crate::core::event_bus::{create_event_bus as create_event_bus_impl, EventBus, EventBusImpl};

/// `createEventBus()`.
pub(crate) fn create_event_bus() -> Arc<dyn EventBus> {
    Arc::new(EventBusImpl::new())
}

/// `Pick<CustomMessage, "customType" | "content" | "display" | "details">`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CustomMessagePayload {
    #[serde(rename = "customType")]
    pub custom_type: String,
    pub content: Value,
    pub display: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
}

/// blocked_on: needs core::session_manager::{SessionEntry, SessionManager}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionEntry {
    #[serde(rename = "type")]
    pub entry_type: String,
    pub id: String,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// blocked_on: needs core::session_manager::CompactionEntry
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompactionEntry {
    #[serde(rename = "type")]
    pub entry_type: String,
    pub summary: String,
    #[serde(rename = "firstKeptEntryId")]
    pub first_kept_entry_id: String,
    #[serde(rename = "tokensBefore")]
    pub tokens_before: f64,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// blocked_on: needs core::session_manager::BranchSummaryEntry
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BranchSummaryEntry {
    #[serde(rename = "type")]
    pub entry_type: String,
    #[serde(rename = "fromId")]
    pub from_id: String,
    pub summary: String,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// blocked_on: needs core::compaction::CompactionPreparation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactionPreparation {
    pub first_kept_entry_id: String,
    pub messages_to_summarize: Vec<Value>,
    pub turn_prefix_messages: Vec<Value>,
    pub is_split_turn: bool,
    pub tokens_before: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_summary: Option<String>,
}

/// blocked_on: needs core::compaction::CompactionResult
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactionResult {
    pub summary: String,
    pub first_kept_entry_id: String,
    pub tokens_before: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Value>,
}

/// blocked_on: needs core::bash_executor::BashResult
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BashResult {
    pub output: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i64>,
    pub cancelled: bool,
    pub truncated: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub full_output_path: Option<String>,
}

/// blocked_on: needs core::tools::bash::BashOperations
pub trait BashOperations: Send + Sync {
    fn exec(
        &self,
        command: String,
        cwd: String,
        on_data: Arc<dyn Fn(Vec<u8>) + Send + Sync>,
        signal: Option<CancellationToken>,
        timeout: Option<f64>,
        env: Option<Map<String, Value>>,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<Option<i64>, String>> + Send>>;
}

/// blocked_on: needs core::exec::{ExecOptions, ExecResult}
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExecOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub env: Option<Map<String, Value>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecResult {
    pub stdout: String,
    pub stderr: String,
    pub code: f64,
    pub killed: bool,
}

/// blocked_on: needs core::tools::{BashToolInput, EditToolInput, IpythonToolInput}
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BashToolInput {
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EditToolInput {
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IpythonToolInput {
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BashToolDetails {
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EditToolDetails {
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IpythonToolDetails {
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// blocked_on: needs core::model_registry::ModelRegistry
pub trait ModelRegistry: Send + Sync {
    fn register_provider(&self, name: &str, config: &ProviderConfig);
    fn unregister_provider(&self, name: &str);
    fn get_api_key_and_headers(
        &self,
        model: &pi_ai::types::Model,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<Value, String>> + Send>>;
}

/// blocked_on: needs core::session_manager::ReadonlySessionManager
pub trait ReadonlySessionManager: Send + Sync {
    fn get_session_id(&self) -> String;
    fn get_session_file(&self) -> Option<String>;
    fn get_session_dir(&self) -> String;
    fn get_branch(&self) -> Vec<SessionEntry>;
    /// Optional no-copy entry count. Unavailable implementations must not materialize
    /// a transcript merely to report this monitoring field.
    fn get_entry_count(&self) -> Option<usize> { None }
}

/// blocked_on: needs core::session_manager::SessionManager
pub trait SessionManager: ReadonlySessionManager {}

/// blocked_on: needs core::keybindings::KeybindingsManager
pub trait KeybindingsManager: Send + Sync {
    fn get_keys(&self, keybinding: &str) -> Vec<KeyId>;
}

/// blocked_on: needs core::keybindings::KeybindingsConfig
pub type KeybindingsConfig = Map<String, Value>;

pub use crate::core::footer_data_provider::ReadonlyFooterDataProvider;

/// blocked_on: needs core::system_prompt::BuildSystemPromptOptions
pub use crate::core::system_prompt::BuildSystemPromptOptions;

/// blocked_on: needs pi-agent-core::types::{AgentToolResult, AgentToolUpdateCallback, ThinkingLevel, ToolExecutionMode}
pub use pi_agent_core::types::{AgentToolResult, AgentToolUpdateCallback, CustomAgentMessage, ThinkingLevel, ToolExecutionMode};

/// blocked_on: needs pi-ai::types::{AssistantMessageEvent, AssistantMessageEventStream, Context, ImageContent, Model, TextContent, ToolResultMessage}
pub use pi_ai::types::{
    AssistantMessageEvent, Context, ImageContent, Model, SimpleStreamOptions, TextContent, ToolResultMessage,
};
pub use pi_ai::utils::event_stream::AssistantMessageEventStream;
pub use pi_ai::utils::oauth::types::{OAuthCredentials, OAuthLoginCallbacks};

// ---------------------------------------------------------------------------
// UI context
// ---------------------------------------------------------------------------

/// Options for extension UI dialogs.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExtensionUIDialogOptions {
    /// AbortSignal to programmatically dismiss the dialog.
    #[serde(skip)]
    pub signal: Option<AbortSignal>,
    /// Timeout in milliseconds. Dialog auto-dismisses with live countdown display.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout: Option<f64>,
}

/// Placement for extension widgets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WidgetPlacement {
    #[serde(rename = "aboveEditor")]
    AboveEditor,
    #[serde(rename = "belowEditor")]
    BelowEditor,
}

/// Options for extension widgets.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExtensionWidgetOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub placement: Option<WidgetPlacement>,
}

/// `{ consume?: boolean; data?: string }`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TerminalInputResult {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub consume: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<String>,
}

/// Raw terminal input listener for extensions.
pub type TerminalInputHandler =
    Arc<dyn Fn(String) -> Option<TerminalInputResult> + Send + Sync>;

/// Working indicator configuration for the interactive streaming loader.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkingIndicatorOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frames: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub interval_ms: Option<f64>,
}

pub type AutocompleteProviderFactory =
    Arc<dyn Fn(Arc<dyn AutocompleteProvider>) -> Arc<dyn AutocompleteProvider> + Send + Sync>;
pub type EditorFactory =
    Arc<dyn Fn(Arc<dyn Tui>, Value, Arc<dyn KeybindingsManager>) -> Arc<dyn EditorComponent> + Send + Sync>;

/// Theme entry returned by `getAllThemes()`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThemeInfo {
    pub name: String,
    pub path: Option<String>,
}

/// Result of `setTheme()`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SetThemeResult {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Widget component factory `(tui, theme) => Component`.
pub type WidgetFactory =
    Arc<dyn Fn(Arc<dyn Tui>, Theme) -> Arc<dyn Component> + Send + Sync>;
/// Footer component factory `(tui, theme, footerData) => Component`.
pub type FooterFactory = Arc<
    dyn Fn(Arc<dyn Tui>, Theme, Arc<dyn ReadonlyFooterDataProvider>) -> Arc<dyn Component> + Send + Sync,
>;
/// Header component factory `(tui, theme) => Component`.
pub type HeaderFactory = Arc<dyn Fn(Arc<dyn Tui>, Theme) -> Arc<dyn Component> + Send + Sync>;

/// Live factory and completion callback for `ui.custom(factory, options)`.
/// Functions cannot be serialized into JSON; daemon/RPC modes intentionally
/// decline these factories, while the interactive owner mounts the component.
pub type CustomComponentFactory = Arc<dyn Fn(
    Arc<dyn Tui>, Theme, Arc<dyn KeybindingsManager>, Arc<dyn Fn(Value) + Send + Sync>,
) -> Pin<Box<dyn std::future::Future<Output = Arc<dyn Component>> + Send>> + Send + Sync>;

/// The value passed to the custom factory's `done` callback.
pub type CustomComponentResult =
    Pin<Box<dyn std::future::Future<Output = Option<Value>> + Send>>;

/// Options for `ui.custom()`.
pub type CustomOptions = Value;

/// UI context for extensions to request interactive UI.
///
/// Every mode (interactive, RPC, print) provides its own implementation.
pub trait ExtensionUiContext: Send + Sync {
    /// Show a selector and return the user's choice.
    fn select(
        &self,
        title: String,
        options: Vec<String>,
        opts: Option<ExtensionUIDialogOptions>,
    ) -> Pin<Box<dyn std::future::Future<Output = Option<String>> + Send>>;

    /// Show a confirmation dialog.
    fn confirm(
        &self,
        title: String,
        message: String,
        opts: Option<ExtensionUIDialogOptions>,
    ) -> Pin<Box<dyn std::future::Future<Output = bool> + Send>>;

    /// Show a text input dialog.
    fn input(
        &self,
        title: String,
        placeholder: Option<String>,
        opts: Option<ExtensionUIDialogOptions>,
    ) -> Pin<Box<dyn std::future::Future<Output = Option<String>> + Send>>;

    /// Show a notification to the user.
    fn notify(&self, message: String, kind: Option<String>);

    /// Listen to raw terminal input (interactive mode only). Returns an unsubscribe function.
    fn on_terminal_input(&self, handler: TerminalInputHandler) -> Arc<dyn Fn() + Send + Sync>;

    /// Set status text in the footer/status bar. Pass undefined to clear.
    fn set_status(&self, key: String, text: Option<String>);

    /// Set the working/loading message shown during streaming.
    fn set_working_message(&self, message: Option<String>);

    /// Show or hide the built-in interactive working loader row during streaming.
    fn set_working_visible(&self, visible: bool);

    /// Configure the interactive working indicator shown during streaming.
    fn set_working_indicator(&self, options: Option<WorkingIndicatorOptions>);

    /// Set the label shown for hidden thinking blocks.
    fn set_hidden_thinking_label(&self, label: Option<String>);

    /// Set a widget to display above or below the editor (string-array form).
    fn set_widget_strings(
        &self,
        key: String,
        content: Option<Vec<String>>,
        options: Option<ExtensionWidgetOptions>,
    );

    /// Set a widget to display above or below the editor (component form).
    fn set_widget_factory(
        &self,
        key: String,
        content: Option<WidgetFactory>,
        options: Option<ExtensionWidgetOptions>,
    );

    /// Set a custom footer component, or undefined to restore the built-in footer.
    fn set_footer(&self, factory: Option<FooterFactory>);

    /// Set a custom header component, or undefined to restore the built-in header.
    fn set_header(&self, factory: Option<HeaderFactory>);

    /// Set the terminal window/tab title.
    fn set_title(&self, title: String);

    /// Show a custom component with keyboard focus.
    fn custom(&self, factory: CustomComponentFactory, options: Option<Value>) -> CustomComponentResult;

    /// Paste text into the editor, triggering paste handling.
    fn paste_to_editor(&self, text: String);

    /// Set the text in the core input editor.
    fn set_editor_text(&self, text: String);

    /// Get the current text from the core input editor.
    fn get_editor_text(&self) -> String;

    /// Show a multi-line editor for text editing.
    fn editor(
        &self,
        title: String,
        prefill: Option<String>,
    ) -> Pin<Box<dyn std::future::Future<Output = Option<String>> + Send>>;

    /// Stack additional autocomplete behavior on top of the built-in provider.
    fn add_autocomplete_provider(&self, factory: AutocompleteProviderFactory);

    /// Set a custom editor component via factory function.
    fn set_editor_component(&self, factory: Option<EditorFactory>);

    /// Get the currently configured custom editor factory.
    fn get_editor_component(&self) -> Option<EditorFactory>;

    /// Get the current theme for styling.
    fn theme(&self) -> Theme;

    /// Get all available themes with their names and file paths.
    fn get_all_themes(&self) -> Vec<ThemeInfo>;

    /// Load a theme by name without switching to it.
    fn get_theme(&self, name: String) -> Option<Theme>;

    /// Set the current theme by name or Theme object.
    fn set_theme(&self, theme: Value) -> SetThemeResult;

    /// Get current tool output expansion state.
    fn get_tools_expanded(&self) -> bool;

    /// Set tool output expansion state.
    fn set_tools_expanded(&self, expanded: bool);
}

/// `ContextUsage`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextUsage {
    /// Estimated context tokens, or null if unknown.
    pub tokens: Option<f64>,
    pub context_window: f64,
    /// Context usage as percentage of context window, or null if tokens is unknown.
    pub percent: Option<f64>,
}

/// `CompactOptions`.
#[derive(Clone)]
pub struct CompactOptions {
    pub custom_instructions: Option<String>,
    pub on_complete: Option<Arc<dyn Fn(CompactionResult) + Send + Sync>>,
    pub on_error: Option<Arc<dyn Fn(String) + Send + Sync>>,
}

impl std::fmt::Debug for CompactOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompactOptions")
            .field("custom_instructions", &self.custom_instructions)
            .finish_non_exhaustive()
    }
}

/// Context passed to extension event handlers.
pub trait ExtensionContext: Send + Sync {
    fn ui(&self) -> Arc<dyn ExtensionUiContext>;
    fn has_ui(&self) -> bool;
    fn cwd(&self) -> String;
    fn session_manager(&self) -> Arc<dyn ReadonlySessionManager>;
    fn model_registry(&self) -> Arc<dyn ModelRegistry>;
    fn model(&self) -> Option<Model>;
    fn is_idle(&self) -> bool;
    fn signal(&self) -> Option<AbortSignal>;
    fn abort(&self);
    fn has_pending_messages(&self) -> bool;
    fn shutdown(&self);
    fn get_context_usage(&self) -> Option<ContextUsage>;
    fn compact(&self, options: Option<CompactOptions>);
    fn get_system_prompt(&self) -> String;
}

/// Options for `ctx.newSession()`.
#[derive(Clone, Default)]
pub struct NewSessionOptions {
    pub parent_session: Option<String>,
    pub setup: Option<Arc<dyn Fn(Arc<dyn SessionManager>) -> Pin<Box<dyn std::future::Future<Output = ()> + Send>> + Send + Sync>>,
    pub with_session: Option<Arc<dyn Fn(Arc<dyn ReplacedSessionContext>) -> Pin<Box<dyn std::future::Future<Output = ()> + Send>> + Send + Sync>>,
}

impl std::fmt::Debug for NewSessionOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NewSessionOptions")
            .field("parent_session", &self.parent_session)
            .finish_non_exhaustive()
    }
}

/// Options for `ctx.fork()`.
#[derive(Clone, Default)]
pub struct ForkOptions {
    pub position: Option<String>,
    pub with_session: Option<Arc<dyn Fn(Arc<dyn ReplacedSessionContext>) -> Pin<Box<dyn std::future::Future<Output = ()> + Send>> + Send + Sync>>,
}

impl std::fmt::Debug for ForkOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ForkOptions").field("position", &self.position).finish()
    }
}

/// Options for `ctx.navigateTree()`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NavigateTreeOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summarize: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub custom_instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub replace_instructions: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

/// Options for `ctx.switchSession()`.
#[derive(Clone, Default)]
pub struct SwitchSessionOptions {
    pub with_session: Option<Arc<dyn Fn(Arc<dyn ReplacedSessionContext>) -> Pin<Box<dyn std::future::Future<Output = ()> + Send>> + Send + Sync>>,
}

impl std::fmt::Debug for SwitchSessionOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SwitchSessionOptions").finish()
    }
}

/// `{ cancelled: boolean }`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CancelledResult {
    pub cancelled: bool,
}

/// Extended context for command handlers.
pub trait ExtensionCommandContext: ExtensionContext {
    fn wait_for_idle(&self) -> Pin<Box<dyn std::future::Future<Output = ()> + Send>>;
    fn new_session(&self, options: Option<NewSessionOptions>) -> Pin<Box<dyn std::future::Future<Output = CancelledResult> + Send>>;
    fn fork(&self, entry_id: String, options: Option<ForkOptions>) -> Pin<Box<dyn std::future::Future<Output = CancelledResult> + Send>>;
    fn navigate_tree(&self, target_id: String, options: Option<NavigateTreeOptions>) -> Pin<Box<dyn std::future::Future<Output = CancelledResult> + Send>>;
    fn switch_session(&self, session_path: String, options: Option<SwitchSessionOptions>) -> Pin<Box<dyn std::future::Future<Output = CancelledResult> + Send>>;
    fn reload(&self) -> Pin<Box<dyn std::future::Future<Output = ()> + Send>>;
}

/// Fresh command-capable context bound to the replacement session.
pub trait ReplacedSessionContext: ExtensionCommandContext {
    fn send_message(&self, message: CustomMessagePayload, options: Option<SendMessageOptions>) -> Pin<Box<dyn std::future::Future<Output = ()> + Send>>;
    fn send_user_message(&self, content: Value, options: Option<SendUserMessageOptions>) -> Pin<Box<dyn std::future::Future<Output = ()> + Send>>;
}

/// `{ triggerTurn?: boolean; deliverAs?: "steer" | "followUp" | "nextTurn" }`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SendMessageOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trigger_turn: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deliver_as: Option<String>,
}

/// `{ deliverAs?: "steer" | "followUp" }`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SendUserMessageOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deliver_as: Option<String>,
}

// ---------------------------------------------------------------------------
// Tool rendering + tool definitions
// ---------------------------------------------------------------------------

/// Rendering options for tool results.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolRenderResultOptions {
    pub expanded: bool,
    pub is_partial: bool,
}

/// Context passed to tool renderers.
pub struct ToolRenderContext {
    /// Current tool call arguments.
    pub args: Value,
    /// Unique id for this tool execution.
    pub tool_call_id: String,
    /// Invalidate just this tool execution component for redraw.
    pub invalidate: Arc<dyn Fn() + Send + Sync>,
    /// Previously returned component for this render slot, if any.
    pub last_component: Option<Arc<dyn Component>>,
    /// Shared renderer state for this tool row.
    pub state: Value,
    /// Working directory for this tool execution.
    pub cwd: String,
    pub execution_started: bool,
    pub args_complete: bool,
    pub is_partial: bool,
    pub expanded: bool,
    /// Whether this row should show the global tool expansion shortcut.
    pub show_expand_hint: Option<bool>,
    pub show_images: bool,
    pub include_image_dimensions: bool,
    pub is_error: bool,
}

impl std::fmt::Debug for ToolRenderContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolRenderContext")
            .field("tool_call_id", &self.tool_call_id)
            .field("cwd", &self.cwd)
            .finish_non_exhaustive()
    }
}

pub type ReplayBuiltInToolName = String;

/// Tool definition for `registerTool()`.
pub struct ToolDefinition {
    /// Tool name (used in LLM tool calls).
    pub name: String,
    /// Human-readable label for UI.
    pub label: String,
    /// Description for LLM.
    pub description: String,
    /// Optional short text extensions may use when composing custom prompts.
    pub prompt_snippet: Option<String>,
    /// Optional guideline bullets appended to the default system prompt when this tool is active.
    pub prompt_guidelines: Option<Vec<String>>,
    /// Parameter schema (TypeBox/JSON schema).
    pub parameters: Value,
    /// Controls whether ToolExecutionComponent renders the standard colored shell.
    pub render_shell: Option<String>,
    /// Replay renderer to use for removed built-ins in saved transcripts.
    pub replay_built_in_tool_name: Option<ReplayBuiltInToolName>,
    /// Optional compatibility shim for raw tool call arguments before schema validation.
    pub prepare_arguments: Option<Arc<dyn Fn(Value) -> Value + Send + Sync>>,
    /// Per-tool execution mode override.
    pub execution_mode: Option<ToolExecutionMode>,
    /// Execute the tool.
    pub execute: Arc<
        dyn Fn(
                String,
                Value,
                Option<AbortSignal>,
                Option<AgentToolUpdateCallback>,
                Arc<dyn ExtensionContext>,
            ) -> Pin<Box<dyn std::future::Future<Output = Result<AgentToolResult, String>> + Send>>
            + Send
            + Sync,
    >,
    /// Custom rendering for tool call display.
    pub render_call: Option<Arc<dyn Fn(Value, Theme, ToolRenderContext) -> Arc<dyn Component> + Send + Sync>>,
    /// Custom rendering for tool result display.
    pub render_result: Option<
        Arc<
            dyn Fn(
                    AgentToolResult,
                    ToolRenderResultOptions,
                    Theme,
                    ToolRenderContext,
                ) -> Arc<dyn Component>
                + Send
                + Sync,
        >,
    >,
}

impl std::fmt::Debug for ToolDefinition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolDefinition")
            .field("name", &self.name)
            .field("label", &self.label)
            .finish_non_exhaustive()
    }
}

impl Clone for ToolDefinition {
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
            render_call: self.render_call.clone(),
            render_result: self.render_result.clone(),
        }
    }
}

/// `defineTool()` - identity helper preserving parameter inference.
pub fn define_tool(tool: ToolDefinition) -> ToolDefinition {
    tool
}

// ---------------------------------------------------------------------------
// Events
//
// Each TypeScript event interface carries a literal `type` discriminant. The
// payload fields live on the variant struct; the `type` string is added by the
// serde tag on the union enum, exactly as the TS discriminant behaves.
// ---------------------------------------------------------------------------

/// Payload of `resources_discover`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourcesDiscoverPayload {
    pub cwd: String,
    pub reason: String,
}

/// Result from a `resources_discover` handler.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResourcesDiscoverResult {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skill_paths: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_paths: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub theme_paths: Option<Vec<String>>,
}

/// Payload of `session_start`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionStartPayload {
    pub reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_session_file: Option<String>,
}

/// Payload of `session_before_switch`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionBeforeSwitchPayload {
    pub reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_session_file: Option<String>,
}

/// Payload of `session_before_fork`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionBeforeForkPayload {
    pub entry_id: String,
    pub position: String,
}

/// Payload of `session_before_compact`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionBeforeCompactPayload {
    pub preparation: CompactionPreparation,
    pub branch_entries: Vec<SessionEntry>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub custom_instructions: Option<String>,
}

/// Planning inputs for a refinement round.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RefinePreparation {
    pub trigger: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    pub scope: String,
    pub planning_state: HarnessState,
    pub history: Vec<RefinementResult>,
    pub conversation_text: String,
}

/// Payload of `session_before_refine`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionBeforeRefinePayload {
    pub preparation: RefinePreparation,
}

/// Typed, sanitized failure a `session_before_refine` hook reports when it
/// owned planning and the planning attempt failed. `message` must be a fixed
/// vocabulary string; `attempt_ms` carries per-attempt wall-clock durations.
/// Raw provider output, keys, and headers never belong here.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionBeforeRefineFailure {
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub category: Option<String>,
    #[serde(default)]
    pub attempts: u8,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attempt_ms: Option<Vec<u64>>,
}

/// `SessionBeforeRefineResult`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SessionBeforeRefineResult {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skip: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proposal: Option<RefinementProposal>,
    /// A hook that owned planning and failed reports the typed failure here so
    /// the core surfaces it instead of silently planning again (RF-001).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<SessionBeforeRefineFailure>,
}

/// Payload of `session_compact`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionCompactPayload {
    pub compaction_entry: CompactionEntry,
    pub from_extension: bool,
}

/// Payload of `session_shutdown`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionShutdownPayload {
    pub reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_session_file: Option<String>,
}

/// Preparation data for tree navigation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TreePreparation {
    pub target_id: String,
    pub old_leaf_id: Option<String>,
    pub common_ancestor_id: Option<String>,
    pub entries_to_summarize: Vec<SessionEntry>,
    pub user_wants_summary: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub custom_instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub replace_instructions: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

/// Payload of `session_before_tree`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionBeforeTreePayload {
    pub preparation: TreePreparation,
}

/// Payload of `session_tree`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionTreePayload {
    pub new_leaf_id: Option<String>,
    pub old_leaf_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary_entry: Option<BranchSummaryEntry>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from_extension: Option<bool>,
}

/// Payload of `context`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextPayload {
    pub messages: Vec<Value>,
}

/// Payload of `before_provider_request`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BeforeProviderRequestPayload {
    pub payload: Value,
}

/// Payload of `after_provider_response`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AfterProviderResponsePayload {
    pub status: f64,
    pub headers: Map<String, Value>,
}

/// Payload of `before_agent_start`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BeforeAgentStartPayload {
    pub prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<ImageContent>>,
    pub system_prompt: String,
    pub system_prompt_options: BuildSystemPromptOptions,
}

/// Payload of `agent_end`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentEndPayload {
    pub messages: Vec<Value>,
}

/// Payload of `refine_complete`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RefineCompletePayload {
    pub id: String,
    pub summary: String,
    pub applied_edits: f64,
    pub scope: String,
}

/// Payload of `turn_start`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TurnStartPayload {
    pub turn_index: f64,
    pub timestamp: f64,
}

/// Payload of `turn_end`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TurnEndPayload {
    pub turn_index: f64,
    pub message: Value,
    pub tool_results: Vec<Value>,
}

/// Payload of `message_start`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MessageStartPayload {
    pub message: Value,
}

/// Payload of `message_update`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MessageUpdatePayload {
    pub message: Value,
    pub assistant_message_event: AssistantMessageEvent,
}

/// Payload of `message_end`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MessageEndPayload {
    pub message: Value,
}

/// Payload of `tool_execution_start`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolExecutionStartPayload {
    pub tool_call_id: String,
    pub tool_name: String,
    pub args: Value,
}

/// Payload of `tool_execution_update`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolExecutionUpdatePayload {
    pub tool_call_id: String,
    pub tool_name: String,
    pub args: Value,
    pub partial_result: Value,
}

/// Payload of `tool_execution_end`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolExecutionEndPayload {
    pub tool_call_id: String,
    pub tool_name: String,
    pub result: Value,
    pub is_error: bool,
}

/// Payload of `model_select`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelSelectPayload {
    pub model: Model,
    pub previous_model: Option<Model>,
    pub source: String,
}

/// Payload of `thinking_level_select`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThinkingLevelSelectPayload {
    pub level: ThinkingLevel,
    pub previous_level: ThinkingLevel,
}

/// Payload of `user_bash`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UserBashPayload {
    pub command: String,
    pub exclude_from_context: bool,
    pub cwd: String,
}

/// `InputSource` identifies where user input enters the session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InputSource { Interactive, Rpc, Extension }

impl InputSource {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Interactive => "interactive",
            Self::Rpc => "rpc",
            Self::Extension => "extension",
        }
    }
}

/// Payload of `input`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InputPayload {
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<ImageContent>>,
    pub source: String,
}

/// Union of all extension events. `type` is the TypeScript discriminant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ExtensionEvent {
    #[serde(rename = "resources_discover")]
    ResourcesDiscover(ResourcesDiscoverPayload),
    #[serde(rename = "session_start")]
    SessionStart(SessionStartPayload),
    #[serde(rename = "session_before_switch")]
    SessionBeforeSwitch(SessionBeforeSwitchPayload),
    #[serde(rename = "session_before_fork")]
    SessionBeforeFork(SessionBeforeForkPayload),
    #[serde(rename = "session_before_compact")]
    SessionBeforeCompact(SessionBeforeCompactPayload),
    #[serde(rename = "session_compact")]
    SessionCompact(SessionCompactPayload),
    #[serde(rename = "session_before_refine")]
    SessionBeforeRefine(SessionBeforeRefinePayload),
    #[serde(rename = "session_shutdown")]
    SessionShutdown(SessionShutdownPayload),
    #[serde(rename = "session_before_tree")]
    SessionBeforeTree(SessionBeforeTreePayload),
    #[serde(rename = "session_tree")]
    SessionTree(SessionTreePayload),
    #[serde(rename = "context")]
    Context(ContextPayload),
    #[serde(rename = "before_provider_request")]
    BeforeProviderRequest(BeforeProviderRequestPayload),
    #[serde(rename = "after_provider_response")]
    AfterProviderResponse(AfterProviderResponsePayload),
    #[serde(rename = "before_agent_start")]
    BeforeAgentStart(BeforeAgentStartPayload),
    #[serde(rename = "agent_start")]
    AgentStart,
    #[serde(rename = "agent_end")]
    AgentEnd(AgentEndPayload),
    #[serde(rename = "turn_start")]
    TurnStart(TurnStartPayload),
    #[serde(rename = "turn_end")]
    TurnEnd(TurnEndPayload),
    #[serde(rename = "message_start")]
    MessageStart(MessageStartPayload),
    #[serde(rename = "message_update")]
    MessageUpdate(MessageUpdatePayload),
    #[serde(rename = "message_end")]
    MessageEnd(MessageEndPayload),
    #[serde(rename = "tool_execution_start")]
    ToolExecutionStart(ToolExecutionStartPayload),
    #[serde(rename = "tool_execution_update")]
    ToolExecutionUpdate(ToolExecutionUpdatePayload),
    #[serde(rename = "tool_execution_end")]
    ToolExecutionEnd(ToolExecutionEndPayload),
    #[serde(rename = "model_select")]
    ModelSelect(ModelSelectPayload),
    #[serde(rename = "thinking_level_select")]
    ThinkingLevelSelect(ThinkingLevelSelectPayload),
    #[serde(rename = "user_bash")]
    UserBash(UserBashPayload),
    #[serde(rename = "input")]
    Input(InputPayload),
    #[serde(rename = "tool_call")]
    ToolCall(ToolCallEvent),
    #[serde(rename = "tool_result")]
    ToolResult(ToolResultEvent),
    #[serde(rename = "refine_complete")]
    RefineComplete(RefineCompletePayload),
}

impl ExtensionEvent {
    /// The TypeScript `event.type` discriminant.
    pub fn event_type(&self) -> &'static str {
        match self {
            ExtensionEvent::ResourcesDiscover(_) => "resources_discover",
            ExtensionEvent::SessionStart(_) => "session_start",
            ExtensionEvent::SessionBeforeSwitch(_) => "session_before_switch",
            ExtensionEvent::SessionBeforeFork(_) => "session_before_fork",
            ExtensionEvent::SessionBeforeCompact(_) => "session_before_compact",
            ExtensionEvent::SessionCompact(_) => "session_compact",
            ExtensionEvent::SessionBeforeRefine(_) => "session_before_refine",
            ExtensionEvent::SessionShutdown(_) => "session_shutdown",
            ExtensionEvent::SessionBeforeTree(_) => "session_before_tree",
            ExtensionEvent::SessionTree(_) => "session_tree",
            ExtensionEvent::Context(_) => "context",
            ExtensionEvent::BeforeProviderRequest(_) => "before_provider_request",
            ExtensionEvent::AfterProviderResponse(_) => "after_provider_response",
            ExtensionEvent::BeforeAgentStart(_) => "before_agent_start",
            ExtensionEvent::AgentStart => "agent_start",
            ExtensionEvent::AgentEnd(_) => "agent_end",
            ExtensionEvent::TurnStart(_) => "turn_start",
            ExtensionEvent::TurnEnd(_) => "turn_end",
            ExtensionEvent::MessageStart(_) => "message_start",
            ExtensionEvent::MessageUpdate(_) => "message_update",
            ExtensionEvent::MessageEnd(_) => "message_end",
            ExtensionEvent::ToolExecutionStart(_) => "tool_execution_start",
            ExtensionEvent::ToolExecutionUpdate(_) => "tool_execution_update",
            ExtensionEvent::ToolExecutionEnd(_) => "tool_execution_end",
            ExtensionEvent::ModelSelect(_) => "model_select",
            ExtensionEvent::ThinkingLevelSelect(_) => "thinking_level_select",
            ExtensionEvent::UserBash(_) => "user_bash",
            ExtensionEvent::Input(_) => "input",
            ExtensionEvent::ToolCall(_) => "tool_call",
            ExtensionEvent::ToolResult(_) => "tool_result",
            ExtensionEvent::RefineComplete(_) => "refine_complete",
        }
    }
}

/// `ContextEventResult`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ContextEventResult {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub messages: Option<Vec<Value>>,
}

/// `BeforeProviderRequestEventResult = unknown`.
pub type BeforeProviderRequestEventResult = Value;

/// `ToolCallEventResult`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ToolCallEventResult {
    /// Block tool execution. To modify arguments, mutate `event.input` in place instead.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub block: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// `UserBashEventResult`.
#[derive(Clone, Default)]
pub struct UserBashEventResult {
    /// Custom operations to use for execution.
    pub operations: Option<Arc<dyn BashOperations>>,
    /// Full replacement: extension handled execution, use this result.
    pub result: Option<BashResult>,
}

impl std::fmt::Debug for UserBashEventResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UserBashEventResult").field("result", &self.result).finish_non_exhaustive()
    }
}

/// `ToolResultEventResult`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolResultEventResult {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Vec<Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
}

/// `MessageEndEventResult`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MessageEndEventResult {
    /// Replace the finalized message. Must keep the original message role.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<Value>,
}

/// `BeforeAgentStartEventResult`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BeforeAgentStartEventResult {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<CustomMessagePayload>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
}

/// `SessionBeforeSwitchResult`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SessionBeforeSwitchResult {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cancel: Option<bool>,
}

/// `SessionBeforeForkResult`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionBeforeForkResult {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cancel: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skip_conversation_restore: Option<bool>,
}

/// `SessionBeforeCompactResult`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SessionBeforeCompactResult {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cancel: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compaction: Option<CompactionResult>,
}

/// `SessionBeforeTreeResult.summary`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TreeSummaryResult {
    pub summary: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
}

/// `SessionBeforeTreeResult`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionBeforeTreeResult {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cancel: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<TreeSummaryResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub custom_instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub replace_instructions: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

/// `MessageRenderOptions`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MessageRenderOptions {
    pub expanded: bool,
}

/// `ExtensionRuntimeState.pendingProviderRegistrations` element.
///
/// The TypeScript declares this inline (`Array<{ name; config; extensionPath }>`),
/// so this module is its owner.
pub struct PendingProviderRegistration {
    pub name: String,
    pub config: ProviderConfig,
    pub extension_path: String,
}

/// `MessageRenderer<T>`.
///
/// The TypeScript parameter is `CustomMessage<T>` from `../messages.js`. In this
/// port the value handed to the renderer is the ported `CustomAgentMessage`
/// custom member, so the alias names that owner type.
pub type MessageRenderer =
    Arc<dyn Fn(CustomAgentMessage, MessageRenderOptions, Theme) -> Option<Arc<dyn Component>> + Send + Sync>;

// ---------------------------------------------------------------------------
// Tool call / tool result events
// ---------------------------------------------------------------------------

/// `ToolCallEvent` union. `toolName` is the discriminant; built-in names carry
/// their typed input, everything else uses `Record<string, unknown>`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "toolName")]
pub enum ToolCallEvent {
    #[serde(rename = "bash")]
    Bash {
        #[serde(rename = "toolCallId")]
        tool_call_id: String,
        input: BashToolInput,
    },
    #[serde(rename = "edit")]
    Edit {
        #[serde(rename = "toolCallId")]
        tool_call_id: String,
        input: EditToolInput,
    },
    #[serde(rename = "ipython")]
    Ipython {
        #[serde(rename = "toolCallId")]
        tool_call_id: String,
        input: IpythonToolInput,
    },
    #[serde(untagged)]
    Custom {
        #[serde(rename = "toolName")]
        tool_name: String,
        #[serde(rename = "toolCallId")]
        tool_call_id: String,
        input: Map<String, Value>,
    },
}

impl ToolCallEvent {
    pub fn tool_name(&self) -> &str {
        match self {
            ToolCallEvent::Bash { .. } => "bash",
            ToolCallEvent::Edit { .. } => "edit",
            ToolCallEvent::Ipython { .. } => "ipython",
            ToolCallEvent::Custom { tool_name, .. } => tool_name,
        }
    }

    pub fn tool_call_id(&self) -> &str {
        match self {
            ToolCallEvent::Bash { tool_call_id, .. } => tool_call_id,
            ToolCallEvent::Edit { tool_call_id, .. } => tool_call_id,
            ToolCallEvent::Ipython { tool_call_id, .. } => tool_call_id,
            ToolCallEvent::Custom { tool_call_id, .. } => tool_call_id,
        }
    }

    pub fn input(&self) -> Value {
        match self {
            ToolCallEvent::Bash { input, .. } => serde_json::to_value(input).unwrap_or(Value::Null),
            ToolCallEvent::Edit { input, .. } => serde_json::to_value(input).unwrap_or(Value::Null),
            ToolCallEvent::Ipython { input, .. } => serde_json::to_value(input).unwrap_or(Value::Null),
            ToolCallEvent::Custom { input, .. } => Value::Object(input.clone()),
        }
    }
}

/// `ToolResultEvent` union.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "toolName")]
pub enum ToolResultEvent {
    #[serde(rename = "bash")]
    Bash {
        #[serde(rename = "toolCallId")]
        tool_call_id: String,
        input: Map<String, Value>,
        content: Vec<Value>,
        #[serde(rename = "isError")]
        is_error: bool,
        details: Option<BashToolDetails>,
    },
    #[serde(rename = "edit")]
    Edit {
        #[serde(rename = "toolCallId")]
        tool_call_id: String,
        input: Map<String, Value>,
        content: Vec<Value>,
        #[serde(rename = "isError")]
        is_error: bool,
        details: Option<EditToolDetails>,
    },
    #[serde(rename = "ipython")]
    Ipython {
        #[serde(rename = "toolCallId")]
        tool_call_id: String,
        input: Map<String, Value>,
        content: Vec<Value>,
        #[serde(rename = "isError")]
        is_error: bool,
        details: Option<IpythonToolDetails>,
    },
    #[serde(untagged)]
    Custom {
        #[serde(rename = "toolName")]
        tool_name: String,
        #[serde(rename = "toolCallId")]
        tool_call_id: String,
        input: Map<String, Value>,
        content: Vec<Value>,
        #[serde(rename = "isError")]
        is_error: bool,
        details: Value,
    },
}

impl ToolResultEvent {
    pub fn tool_name(&self) -> &str {
        match self {
            ToolResultEvent::Bash { .. } => "bash",
            ToolResultEvent::Edit { .. } => "edit",
            ToolResultEvent::Ipython { .. } => "ipython",
            ToolResultEvent::Custom { tool_name, .. } => tool_name,
        }
    }

    pub fn tool_call_id(&self) -> &str {
        match self {
            ToolResultEvent::Bash { tool_call_id, .. } => tool_call_id,
            ToolResultEvent::Edit { tool_call_id, .. } => tool_call_id,
            ToolResultEvent::Ipython { tool_call_id, .. } => tool_call_id,
            ToolResultEvent::Custom { tool_call_id, .. } => tool_call_id,
        }
    }

    pub fn content(&self) -> &[Value] {
        match self {
            ToolResultEvent::Bash { content, .. } => content,
            ToolResultEvent::Edit { content, .. } => content,
            ToolResultEvent::Ipython { content, .. } => content,
            ToolResultEvent::Custom { content, .. } => content,
        }
    }

    pub fn is_error(&self) -> bool {
        match self {
            ToolResultEvent::Bash { is_error, .. } => *is_error,
            ToolResultEvent::Edit { is_error, .. } => *is_error,
            ToolResultEvent::Ipython { is_error, .. } => *is_error,
            ToolResultEvent::Custom { is_error, .. } => *is_error,
        }
    }

    pub fn details(&self) -> Value {
        match self {
            ToolResultEvent::Bash { details, .. } => serde_json::to_value(details).unwrap_or(Value::Null),
            ToolResultEvent::Edit { details, .. } => serde_json::to_value(details).unwrap_or(Value::Null),
            ToolResultEvent::Ipython { details, .. } => serde_json::to_value(details).unwrap_or(Value::Null),
            ToolResultEvent::Custom { details, .. } => details.clone(),
        }
    }
}

pub fn is_bash_tool_result(event: &ToolResultEvent) -> bool {
    event.tool_name() == "bash"
}

pub fn is_edit_tool_result(event: &ToolResultEvent) -> bool {
    event.tool_name() == "edit"
}

pub fn is_ipython_tool_result(event: &ToolResultEvent) -> bool {
    event.tool_name() == "ipython"
}

/// Type guard for narrowing `ToolCallEvent` by tool name.
pub fn is_tool_call_event_type(tool_name: &str, event: &ToolCallEvent) -> bool {
    event.tool_name() == tool_name
}

/// `InputEventResult`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "action")]
pub enum InputEventResult {
    #[serde(rename = "continue")]
    Continue,
    #[serde(rename = "transform")]
    Transform {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        images: Option<Vec<ImageContent>>,
    },
    #[serde(rename = "handled")]
    Handled,
}

// ---------------------------------------------------------------------------
// Registration surface
// ---------------------------------------------------------------------------

/// `RegisteredCommand`.
#[derive(Clone)]
pub struct RegisteredCommand {
    pub name: String,
    pub source_info: SourceInfo,
    pub description: Option<String>,
    pub get_argument_completions: Option<
        Arc<
            dyn Fn(String) -> Pin<Box<dyn std::future::Future<Output = Option<Vec<AutocompleteItem>>> + Send>>
                + Send
                + Sync,
        >,
    >,
    pub handler: Arc<
        dyn Fn(String, Arc<dyn ExtensionCommandContext>) -> Pin<Box<dyn std::future::Future<Output = Result<(), String>> + Send>>
            + Send
            + Sync,
    >,
}

impl std::fmt::Debug for RegisteredCommand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegisteredCommand")
            .field("name", &self.name)
            .field("description", &self.description)
            .finish_non_exhaustive()
    }
}

/// `ResolvedCommand`.
#[derive(Clone)]
pub struct ResolvedCommand {
    pub command: RegisteredCommand,
    pub invocation_name: String,
}

impl std::fmt::Debug for ResolvedCommand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResolvedCommand")
            .field("invocation_name", &self.invocation_name)
            .finish_non_exhaustive()
    }
}

/// Handler function type for events: `(event, ctx) => Promise<R | void> | R | void`.
pub type ExtensionHandler =
    Arc<dyn Fn(ExtensionEvent, Arc<dyn ExtensionContext>) -> Pin<Box<dyn std::future::Future<Output = Option<Value>> + Send>> + Send + Sync>;

/// `RegisteredTool`.
#[derive(Clone)]
pub struct RegisteredTool {
    pub definition: ToolDefinition,
    pub source_info: SourceInfo,
}

impl std::fmt::Debug for RegisteredTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegisteredTool")
            .field("definition", &self.definition)
            .finish_non_exhaustive()
    }
}

/// `ExtensionFlag`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExtensionFlag {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(rename = "type")]
    pub flag_type: String,
    #[serde(rename = "default", skip_serializing_if = "Option::is_none")]
    pub default: Option<Value>,
    pub extension_path: String,
}

/// `ExtensionShortcut`.
#[derive(Clone)]
pub struct ExtensionShortcut {
    pub shortcut: KeyId,
    pub description: Option<String>,
    pub handler: Arc<
        dyn Fn(Arc<dyn ExtensionContext>) -> Pin<Box<dyn std::future::Future<Output = Result<(), String>> + Send>>
            + Send
            + Sync,
    >,
    pub extension_path: String,
}

impl std::fmt::Debug for ExtensionShortcut {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExtensionShortcut")
            .field("shortcut", &self.shortcut)
            .field("extension_path", &self.extension_path)
            .finish_non_exhaustive()
    }
}

/// `ToolInfo`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolInfo {
    pub name: String,
    pub description: String,
    pub parameters: Value,
    #[serde(rename = "sourceInfo")]
    pub source_info: SourceInfo,
}

pub type HandlerFn = ExtensionHandler;

pub type SendMessageHandler =
    Arc<dyn Fn(CustomMessagePayload, Option<SendMessageOptions>) + Send + Sync>;
pub type SendUserMessageHandler = Arc<dyn Fn(Value, Option<SendUserMessageOptions>) + Send + Sync>;
pub type AppendEntryHandler = Arc<dyn Fn(String, Option<Value>) + Send + Sync>;
pub type SetSessionNameHandler =
    Arc<dyn Fn(String) -> Pin<Box<dyn std::future::Future<Output = ()> + Send>> + Send + Sync>;
pub type GetSessionNameHandler = Arc<dyn Fn() -> Option<String> + Send + Sync>;
pub type GetActiveToolsHandler = Arc<dyn Fn() -> Vec<String> + Send + Sync>;
pub type GetAllToolsHandler = Arc<dyn Fn() -> Vec<ToolInfo> + Send + Sync>;
pub type GetCommandsHandler = Arc<dyn Fn() -> Vec<SlashCommandInfo> + Send + Sync>;
pub type SetActiveToolsHandler = Arc<dyn Fn(Vec<String>) + Send + Sync>;
pub type RefreshToolsHandler = Arc<dyn Fn() + Send + Sync>;
pub type SetModelHandler =
    Arc<dyn Fn(Model) -> Pin<Box<dyn std::future::Future<Output = bool> + Send>> + Send + Sync>;
pub type GetThinkingLevelHandler = Arc<dyn Fn() -> ThinkingLevel + Send + Sync>;
pub type SetThinkingLevelHandler = Arc<dyn Fn(ThinkingLevel) + Send + Sync>;
pub type SetLabelHandler = Arc<dyn Fn(String, Option<String>) + Send + Sync>;

/// Configuration for registering a provider via `pi.registerProvider()`.
#[derive(Clone, Default)]
pub struct ProviderConfig {
    /// Display name for the provider in UI.
    pub name: Option<String>,
    /// Base URL for the API endpoint. Required when defining models.
    pub base_url: Option<String>,
    /// API key or environment variable name.
    pub api_key: Option<String>,
    /// API type. Required at provider or model level when defining models.
    pub api: Option<String>,
    /// Optional streamSimple handler for custom APIs.
    pub stream_simple: Option<
        Arc<
            dyn Fn(Model, Context, Option<SimpleStreamOptions>) -> AssistantMessageEventStream
                + Send
                + Sync,
        >,
    >,
    /// Custom headers to include in requests.
    pub headers: Option<Map<String, Value>>,
    /// If true, adds Authorization: Bearer header with the resolved API key.
    pub auth_header: Option<bool>,
    /// Models to register. If provided, replaces all existing models for this provider.
    pub models: Option<Vec<ProviderModelConfig>>,
    /// OAuth provider for /login support. The `id` is set automatically.
    pub oauth: Option<ProviderOAuthConfig>,
}

impl std::fmt::Debug for ProviderConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderConfig")
            .field("name", &self.name)
            .field("base_url", &self.base_url)
            .finish_non_exhaustive()
    }
}

/// The `oauth` block of `ProviderConfig`.
#[derive(Clone)]
pub struct ProviderOAuthConfig {
    /// Display name for the provider in login UI.
    pub name: String,
    /// Run the login flow, return credentials to persist.
    pub login: Arc<
        dyn Fn(OAuthLoginCallbacks) -> Pin<Box<dyn std::future::Future<Output = Result<OAuthCredentials, String>> + Send>>
            + Send
            + Sync,
    >,
    /// Refresh expired credentials, return updated credentials to persist.
    pub refresh_token: Arc<
        dyn Fn(OAuthCredentials) -> Pin<Box<dyn std::future::Future<Output = Result<OAuthCredentials, String>> + Send>>
            + Send
            + Sync,
    >,
    /// Convert credentials to API key string for the provider.
    pub get_api_key: Arc<dyn Fn(OAuthCredentials) -> String + Send + Sync>,
    /// Optional: modify models for this provider.
    pub modify_models: Option<Arc<dyn Fn(Vec<Model>, OAuthCredentials) -> Vec<Model> + Send + Sync>>,
}

impl std::fmt::Debug for ProviderOAuthConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderOAuthConfig").field("name", &self.name).finish()
    }
}

/// Configuration for a model within a provider.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderModelConfig {
    pub id: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    pub reasoning: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_level_map: Option<pi_ai::types::ThinkingLevelMap>,
    pub input: Vec<String>,
    pub cost: ProviderModelCost,
    pub context_window: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_input_tokens: Option<f64>,
    pub max_tokens: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub native_compaction: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<Map<String, Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compat: Option<Value>,
}

/// `cost: { input; output; cacheRead; cacheWrite }`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderModelCost {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
}

/// `ExtensionRuntimeState` - the shared state created by the loader, used
/// during registration and runtime.
///
/// The TypeScript object holds action methods that are replaced by
/// `bindCore()`. Rust keeps the same slots as `Option`s: calling one before the
/// bind throws the TypeScript "Extension runtime not initialized." error.
pub struct ExtensionRuntimeState {
    pub flag_values: indexmap::IndexMap<String, Value>,
    /// Extra env vars merged over process.env for pi.exec() subprocesses.
    pub get_exec_env: Option<Arc<dyn Fn() -> Option<Map<String, Value>> + Send + Sync>>,
    /// Provider registrations queued during extension loading.
    pub pending_provider_registrations: Vec<PendingProviderRegistration>,
    /// Stale-instance message set by `invalidate()`.
    pub stale_message: Option<String>,
    /// Action implementations installed by `runner.bindCore()`.
    pub actions: Option<ExtensionActions>,
    /// Provider action overrides installed by `runner.bindCore()`.
    pub provider_actions: Option<ProviderActions>,
}

impl Default for ExtensionRuntimeState {
    fn default() -> Self {
        Self {
            flag_values: indexmap::IndexMap::new(),
            get_exec_env: None,
            pending_provider_registrations: Vec::new(),
            stale_message: None,
            actions: None,
            provider_actions: None,
        }
    }
}

impl std::fmt::Debug for ExtensionRuntimeState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExtensionRuntimeState")
            .field("flag_values", &self.flag_values)
            .finish_non_exhaustive()
    }
}

/// `{ registerProvider?; unregisterProvider? }` passed to `bindCore()`.
#[derive(Clone, Default)]
pub struct ProviderActions {
    pub register_provider: Option<Arc<dyn Fn(String, ProviderConfig) + Send + Sync>>,
    pub unregister_provider: Option<Arc<dyn Fn(String) + Send + Sync>>,
}

impl std::fmt::Debug for ProviderActions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderActions").finish_non_exhaustive()
    }
}

/// The message thrown when an action is called before `bindCore()`.
pub const EXTENSION_RUNTIME_NOT_INITIALIZED: &str =
    "Extension runtime not initialized. Action methods cannot be called during extension loading.";

/// The default stale-instance message used by `invalidate()`.
pub const EXTENSION_RUNTIME_STALE_MESSAGE: &str =
    "This extension ctx is stale after session replacement or reload. Do not use a captured pi or command ctx after ctx.newSession(), ctx.fork(), ctx.switchSession(), or ctx.reload(). For newSession, fork, and switchSession, move post-replacement work into withSession and use the ctx passed to withSession. For reload, do not use the old ctx after await ctx.reload().";

/// Full runtime = state + actions. Created by the loader with throwing action
/// stubs, completed by `runner.initialize()`.
#[derive(Clone)]
pub struct ExtensionRuntime {
    pub state: Arc<std::sync::Mutex<ExtensionRuntimeState>>,
}

impl std::fmt::Debug for ExtensionRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExtensionRuntime").finish_non_exhaustive()
    }
}

impl ExtensionRuntime {
    pub fn new(state: ExtensionRuntimeState) -> Self {
        Self {
            state: Arc::new(std::sync::Mutex::new(state)),
        }
    }

    fn with_state<T>(&self, f: impl FnOnce(&mut ExtensionRuntimeState) -> T) -> T {
        let mut guard = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        f(&mut guard)
    }

    /// `runtime.assertActive()`.
    pub fn assert_active(&self) -> Result<(), String> {
        self.with_state(|state| match &state.stale_message {
            Some(message) => Err(message.clone()),
            None => Ok(()),
        })
    }

    /// `runtime.invalidate(message)`.
    pub fn invalidate(&self, message: Option<String>) {
        self.with_state(|state| {
            if state.stale_message.is_none() {
                state.stale_message = Some(message.unwrap_or_else(|| EXTENSION_RUNTIME_STALE_MESSAGE.to_string()));
            }
        });
    }

    pub fn flag_values_get(&self, name: &str) -> Option<Value> {
        self.with_state(|state| state.flag_values.get(name).cloned())
    }

    pub fn flag_values_set(&self, name: &str, value: Value) {
        self.with_state(|state| {
            state.flag_values.insert(name.to_string(), value);
        });
    }

    pub fn flag_values_has(&self, name: &str) -> bool {
        self.with_state(|state| state.flag_values.contains_key(name))
    }

    pub fn flag_values_snapshot(&self) -> indexmap::IndexMap<String, Value> {
        self.with_state(|state| state.flag_values.clone())
    }

    pub fn get_exec_env(&self) -> Option<Map<String, Value>> {
        let getter = self.with_state(|state| state.get_exec_env.clone())?;
        getter()
    }

    /// `runtime.registerProvider(name, config, extensionPath)`.
    pub fn register_provider(&self, name: &str, config: ProviderConfig, extension_path: Option<&str>) {
        let direct = self.with_state(|state| state.provider_actions.clone());
        if let Some(actions) = direct {
            if let Some(register) = actions.register_provider {
                register(name.to_string(), config);
                return;
            }
        }
        self.with_state(|state| {
            state.pending_provider_registrations.push(PendingProviderRegistration {
                name: name.to_string(),
                config,
                extension_path: extension_path.unwrap_or("<unknown>").to_string(),
            });
        });
    }

    /// `runtime.unregisterProvider(name, extensionPath)`.
    pub fn unregister_provider(&self, name: &str, extension_path: Option<&str>) {
        let direct = self.with_state(|state| state.provider_actions.clone());
        if let Some(actions) = direct {
            if let Some(unregister) = actions.unregister_provider {
                unregister(name.to_string());
                return;
            }
        }
        self.with_state(|state| {
            state
                .pending_provider_registrations
                .retain(|registration| registration.name != name);
            let _ = extension_path;
        });
    }

    /// Take the queued provider registrations (used by `bindCore()`).
    pub fn take_pending_provider_registrations(&self) -> Vec<PendingProviderRegistration> {
        self.with_state(|state| std::mem::take(&mut state.pending_provider_registrations))
    }

    /// Read the installed action implementations.
    pub fn actions(&self) -> Option<ExtensionActions> {
        self.with_state(|state| state.actions.clone())
    }

    fn require_actions(&self) -> Result<ExtensionActions, String> {
        self.actions().ok_or_else(|| EXTENSION_RUNTIME_NOT_INITIALIZED.to_string())
    }

    pub fn send_message(
        &self,
        message: CustomMessagePayload,
        options: Option<SendMessageOptions>,
    ) -> Result<(), String> {
        let actions = self.require_actions()?;
        (actions.send_message)(message, options);
        Ok(())
    }

    pub fn send_user_message(&self, content: Value, options: Option<SendUserMessageOptions>) -> Result<(), String> {
        let actions = self.require_actions()?;
        (actions.send_user_message)(content, options);
        Ok(())
    }

    pub fn append_entry(&self, custom_type: &str, data: Option<Value>) -> Result<(), String> {
        let actions = self.require_actions()?;
        (actions.append_entry)(custom_type.to_string(), data);
        Ok(())
    }

    pub fn set_session_name(
        &self,
        name: String,
    ) -> Result<Pin<Box<dyn std::future::Future<Output = ()> + Send>>, String> {
        let actions = self.require_actions()?;
        Ok((actions.set_session_name)(name))
    }

    pub fn get_session_name(&self) -> Result<Option<String>, String> {
        let actions = self.require_actions()?;
        Ok((actions.get_session_name)())
    }

    pub fn set_label(&self, entry_id: String, label: Option<String>) -> Result<(), String> {
        let actions = self.require_actions()?;
        (actions.set_label)(entry_id, label);
        Ok(())
    }

    pub fn get_active_tools(&self) -> Result<Vec<String>, String> {
        let actions = self.require_actions()?;
        Ok((actions.get_active_tools)())
    }

    pub fn get_all_tools(&self) -> Result<Vec<ToolInfo>, String> {
        let actions = self.require_actions()?;
        Ok((actions.get_all_tools)())
    }

    pub fn set_active_tools(&self, tool_names: Vec<String>) -> Result<(), String> {
        let actions = self.require_actions()?;
        (actions.set_active_tools)(tool_names);
        Ok(())
    }

    pub fn refresh_tools(&self) {
        if let Some(actions) = self.actions() {
            (actions.refresh_tools)();
        }
    }

    pub fn get_commands(&self) -> Result<Vec<SlashCommandInfo>, String> {
        let actions = self.require_actions()?;
        Ok((actions.get_commands)())
    }

    /// `runtime.setModel(model)` - rejects with "Extension runtime not initialized".
    pub fn set_model(&self, model: pi_ai::types::Model) -> Pin<Box<dyn std::future::Future<Output = Result<bool, String>> + Send>> {
        match self.actions() {
            Some(actions) => Box::pin(async move { Ok((actions.set_model)(model).await) }),
            None => Box::pin(async { Err("Extension runtime not initialized".to_string()) }),
        }
    }

    pub fn get_thinking_level(&self) -> Result<ThinkingLevel, String> {
        let actions = self.require_actions()?;
        Ok((actions.get_thinking_level)())
    }

    pub fn set_thinking_level(&self, level: ThinkingLevel) -> Result<(), String> {
        let actions = self.require_actions()?;
        (actions.set_thinking_level)(level);
        Ok(())
    }
}

/// Action implementations for `pi.*` API methods.
#[derive(Clone)]
pub struct ExtensionActions {
    pub send_message: SendMessageHandler,
    pub send_user_message: SendUserMessageHandler,
    pub append_entry: AppendEntryHandler,
    pub set_session_name: SetSessionNameHandler,
    pub get_session_name: GetSessionNameHandler,
    pub set_label: SetLabelHandler,
    pub get_active_tools: GetActiveToolsHandler,
    pub get_all_tools: GetAllToolsHandler,
    pub set_active_tools: SetActiveToolsHandler,
    pub refresh_tools: RefreshToolsHandler,
    pub get_commands: GetCommandsHandler,
    pub set_model: SetModelHandler,
    pub get_thinking_level: GetThinkingLevelHandler,
    pub set_thinking_level: SetThinkingLevelHandler,
}

impl std::fmt::Debug for ExtensionActions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExtensionActions").finish_non_exhaustive()
    }
}

/// Actions for `ExtensionContext` (`ctx.*` in event handlers).
#[derive(Clone)]
pub struct ExtensionContextActions {
    pub get_model: Arc<dyn Fn() -> Option<Model> + Send + Sync>,
    pub is_idle: Arc<dyn Fn() -> bool + Send + Sync>,
    pub get_signal: Arc<dyn Fn() -> Option<AbortSignal> + Send + Sync>,
    pub abort: Arc<dyn Fn() + Send + Sync>,
    pub has_pending_messages: Arc<dyn Fn() -> bool + Send + Sync>,
    pub shutdown: Arc<dyn Fn() + Send + Sync>,
    pub get_context_usage: Arc<dyn Fn() -> Option<ContextUsage> + Send + Sync>,
    pub compact: Arc<dyn Fn(Option<CompactOptions>) + Send + Sync>,
    pub get_system_prompt: Arc<dyn Fn() -> String + Send + Sync>,
}

impl std::fmt::Debug for ExtensionContextActions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExtensionContextActions").finish_non_exhaustive()
    }
}

/// Actions for `ExtensionCommandContext`.
#[derive(Clone)]
pub struct ExtensionCommandContextActions {
    pub wait_for_idle: Arc<dyn Fn() -> Pin<Box<dyn std::future::Future<Output = ()> + Send>> + Send + Sync>,
    pub new_session: Arc<dyn Fn(Option<NewSessionOptions>) -> Pin<Box<dyn std::future::Future<Output = CancelledResult> + Send>> + Send + Sync>,
    pub fork: Arc<dyn Fn(String, Option<ForkOptions>) -> Pin<Box<dyn std::future::Future<Output = CancelledResult> + Send>> + Send + Sync>,
    pub navigate_tree: Arc<dyn Fn(String, Option<NavigateTreeOptions>) -> Pin<Box<dyn std::future::Future<Output = CancelledResult> + Send>> + Send + Sync>,
    pub switch_session: Arc<dyn Fn(String, Option<SwitchSessionOptions>) -> Pin<Box<dyn std::future::Future<Output = CancelledResult> + Send>> + Send + Sync>,
    pub reload: Arc<dyn Fn() -> Pin<Box<dyn std::future::Future<Output = ()> + Send>> + Send + Sync>,
}

impl std::fmt::Debug for ExtensionCommandContextActions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExtensionCommandContextActions").finish_non_exhaustive()
    }
}

/// Loaded extension with all registered items.
///
/// The TypeScript extension object is a single mutable identity that the
/// `ExtensionAPI` keeps writing to after the factory returns, so the port keeps
/// it behind an `Arc<Mutex<_>>`.
pub struct Extension {
    pub path: String,
    pub resolved_path: String,
    pub source_info: SourceInfo,
    pub handlers: std::collections::HashMap<String, Vec<ExtensionHandler>>,
    pub tools: std::collections::HashMap<String, RegisteredTool>,
    pub message_renderers: std::collections::HashMap<String, MessageRenderer>,
    pub commands: indexmap::IndexMap<String, RegisteredCommand>,
    pub flags: std::collections::HashMap<String, ExtensionFlag>,
    pub shortcuts: std::collections::HashMap<KeyId, ExtensionShortcut>,
}

impl std::fmt::Debug for Extension {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Extension")
            .field("path", &self.path)
            .field("resolved_path", &self.resolved_path)
            .finish_non_exhaustive()
    }
}

/// Shared handle to one loaded extension.
pub type SharedExtension = Arc<std::sync::Mutex<Extension>>;

/// Result of loading extensions.
pub struct LoadExtensionsResult {
    pub extensions: Vec<SharedExtension>,
    pub errors: Vec<LoadExtensionError>,
    /// Shared runtime - actions are throwing stubs until `runner.initialize()`.
    pub runtime: ExtensionRuntime,
}

impl std::fmt::Debug for LoadExtensionsResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadExtensionsResult")
            .field("extensions", &self.extensions.len())
            .field("errors", &self.errors)
            .finish()
    }
}

/// `{ path; error }` load error entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoadExtensionError {
    pub path: String,
    pub error: String,
}

/// `ExtensionError`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExtensionError {
    pub extension_path: String,
    pub event: String,
    pub error: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stack: Option<String>,
}

/// `ExtensionFactory` - sync or async initialization.
pub type ExtensionFactory = Arc<
    dyn Fn(Arc<dyn ExtensionApi>) -> Pin<Box<dyn std::future::Future<Output = Result<(), String>> + Send>>
        + Send
        + Sync,
>;

/// The `ExtensionAPI` passed to extension factory functions.
///
/// The TypeScript interface has one overloaded `on()` per event type; Rust keeps
/// a single `on(event_type, handler)` because the handler receives the tagged
/// `ExtensionEvent` union.
pub trait ExtensionApi: Send + Sync {
    /// Subscribe to a lifecycle event.
    fn on(&self, event_type: &str, handler: ExtensionHandler);
    /// Register a tool that the LLM can call.
    fn register_tool(&self, tool: ToolDefinition);
    /// Register a custom command.
    fn register_command(&self, name: String, options: RegisterCommandOptions);
    /// Register a keyboard shortcut.
    fn register_shortcut(&self, shortcut: KeyId, options: RegisterShortcutOptions);
    /// Register a CLI flag.
    fn register_flag(&self, name: String, options: RegisterFlagOptions);
    /// Get the value of a registered CLI flag.
    fn get_flag(&self, name: &str) -> Option<Value>;
    /// Register a custom renderer for `CustomMessageEntry`.
    fn register_message_renderer(&self, custom_type: String, renderer: MessageRenderer);
    /// Send a custom message to the session.
    fn send_message(&self, message: CustomMessagePayload, options: Option<SendMessageOptions>);
    /// Send a user message to the agent. Always triggers a turn.
    fn send_user_message(&self, content: Value, options: Option<SendUserMessageOptions>);
    /// Append a custom entry to the session for state persistence.
    fn append_entry(&self, custom_type: String, data: Option<Value>);
    /// Set the session display name.
    fn set_session_name(&self, name: String) -> Pin<Box<dyn std::future::Future<Output = ()> + Send>>;
    /// Get the current session name, if set.
    fn get_session_name(&self) -> Option<String>;
    /// Set or clear a label on an entry.
    fn set_label(&self, entry_id: String, label: Option<String>);
    /// Execute a shell command.
    fn exec(&self, command: String, args: Vec<String>, options: Option<ExecOptions>) -> Pin<Box<dyn std::future::Future<Output = Result<ExecResult, String>> + Send>>;
    /// Get the list of currently active tool names.
    fn get_active_tools(&self) -> Vec<String>;
    /// Get all configured tools with parameter schema and source metadata.
    fn get_all_tools(&self) -> Vec<ToolInfo>;
    /// Set the active tools by name.
    fn set_active_tools(&self, tool_names: Vec<String>);
    /// Get available slash commands in the current session.
    fn get_commands(&self) -> Vec<SlashCommandInfo>;
    /// Set the current model. Returns false if no API key available.
    fn set_model(&self, model: Model) -> Pin<Box<dyn std::future::Future<Output = bool> + Send>>;
    /// Get current thinking level.
    fn get_thinking_level(&self) -> ThinkingLevel;
    /// Set thinking level (clamped to model capabilities).
    fn set_thinking_level(&self, level: ThinkingLevel);
    /// Register or override a model provider.
    fn register_provider(&self, name: String, config: ProviderConfig);
    /// Unregister a previously registered provider.
    fn unregister_provider(&self, name: String);
    /// Shared event bus for extension communication.
    fn events(&self) -> Arc<dyn EventBus>;
}

/// `Omit<RegisteredCommand, "name" | "sourceInfo">`.
#[derive(Clone, Default)]
pub struct RegisterCommandOptions {
    pub description: Option<String>,
    pub get_argument_completions: Option<
        Arc<
            dyn Fn(String) -> Pin<Box<dyn std::future::Future<Output = Option<Vec<AutocompleteItem>>> + Send>>
                + Send
                + Sync,
        >,
    >,
    pub handler: Option<
        Arc<
            dyn Fn(String, Arc<dyn ExtensionCommandContext>) -> Pin<Box<dyn std::future::Future<Output = Result<(), String>> + Send>>
                + Send
                + Sync,
        >,
    >,
}

impl std::fmt::Debug for RegisterCommandOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegisterCommandOptions")
            .field("description", &self.description)
            .finish_non_exhaustive()
    }
}

/// `{ description?; handler }` for `registerShortcut`.
#[derive(Clone)]
pub struct RegisterShortcutOptions {
    pub description: Option<String>,
    pub handler: Arc<
        dyn Fn(Arc<dyn ExtensionContext>) -> Pin<Box<dyn std::future::Future<Output = Result<(), String>> + Send>>
            + Send
            + Sync,
    >,
}

impl std::fmt::Debug for RegisterShortcutOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegisterShortcutOptions")
            .field("description", &self.description)
            .finish_non_exhaustive()
    }
}

/// `{ description?; type; default? }` for `registerFlag`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegisterFlagOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(rename = "type")]
    pub flag_type: String,
    #[serde(rename = "default", skip_serializing_if = "Option::is_none")]
    pub default: Option<Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extension_event_type_matches_the_typescript_discriminant() {
        let event = ExtensionEvent::SessionShutdown(SessionShutdownPayload {
            reason: "quit".to_string(),
            target_session_file: None,
        });
        assert_eq!(event.event_type(), "session_shutdown");
        let value = serde_json::to_value(&event).unwrap();
        assert_eq!(value["type"], serde_json::json!("session_shutdown"));
        assert_eq!(value["reason"], serde_json::json!("quit"));
        assert!(value.get("targetSessionFile").is_none());
    }

    #[test]
    fn resources_discover_result_omits_absent_arrays() {
        let result = ResourcesDiscoverResult::default();
        assert_eq!(serde_json::to_string(&result).unwrap(), "{}");
    }

    #[test]
    fn tool_call_event_reports_builtin_and_custom_names() {
        let bash = ToolCallEvent::Bash {
            tool_call_id: "c1".to_string(),
            input: BashToolInput::default(),
        };
        assert_eq!(bash.tool_name(), "bash");
        assert!(is_tool_call_event_type("bash", &bash));
        assert!(!is_tool_call_event_type("edit", &bash));

        let custom = ToolCallEvent::Custom {
            tool_name: "my_tool".to_string(),
            tool_call_id: "c2".to_string(),
            input: Map::new(),
        };
        assert_eq!(custom.tool_name(), "my_tool");
        assert_eq!(custom.tool_call_id(), "c2");
    }

    #[test]
    fn tool_result_type_guards_match_the_typescript() {
        let bash = ToolResultEvent::Bash {
            tool_call_id: "c1".to_string(),
            input: Map::new(),
            content: vec![serde_json::json!({"type": "text", "text": "ok"})],
            is_error: false,
            details: None,
        };
        assert!(is_bash_tool_result(&bash));
        assert!(!is_edit_tool_result(&bash));
        assert!(!is_ipython_tool_result(&bash));
    }

    #[test]
    fn input_event_result_serialises_the_action_tag() {
        assert_eq!(
            serde_json::to_string(&InputEventResult::Continue).unwrap(),
            r#"{"action":"continue"}"#
        );
        assert_eq!(
            serde_json::to_string(&InputEventResult::Handled).unwrap(),
            r#"{"action":"handled"}"#
        );
    }

    #[test]
    fn tool_call_event_result_omits_absent_fields() {
        assert_eq!(serde_json::to_string(&ToolCallEventResult::default()).unwrap(), "{}");
    }

    #[test]
    fn define_tool_is_an_identity_helper() {
        let tool = ToolDefinition {
            name: "t".to_string(),
            label: "T".to_string(),
            description: "d".to_string(),
            prompt_snippet: None,
            prompt_guidelines: None,
            parameters: serde_json::json!({"type": "object"}),
            render_shell: None,
            replay_built_in_tool_name: None,
            prepare_arguments: None,
            execution_mode: None,
            execute: Arc::new(|_, _, _, _, _| Box::pin(async { Err("unused".to_string()) })),
            render_call: None,
            render_result: None,
        };
        assert_eq!(define_tool(tool).name, "t");
    }

    #[test]
    fn widget_placement_uses_camel_case_literals() {
        assert_eq!(
            serde_json::to_string(&WidgetPlacement::AboveEditor).unwrap(),
            r#""aboveEditor""#
        );
        assert_eq!(
            serde_json::to_string(&WidgetPlacement::BelowEditor).unwrap(),
            r#""belowEditor""#
        );
    }
}
