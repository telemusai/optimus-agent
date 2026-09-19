//! Port of packages/coding-agent/src/core/extensions/runner.ts
//!
//! Extension runner - executes extensions and manages their lifecycle.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use serde_json::{Map, Value};
use futures::FutureExt;

use crate::core::diagnostics::ResourceDiagnostic;
use crate::core::slash_commands::SlashCommandInfo;

use super::types::{
    AbortSignal, BeforeAgentStartEventResult, CancelledResult, CompactOptions, ContextEventResult, ContextUsage,
    CustomMessagePayload, Extension, ExtensionActions, ExtensionCommandContext, ExtensionCommandContextActions,
    ExtensionContext, ExtensionContextActions, ExtensionError, ExtensionEvent, ExtensionFlag, ExtensionHandler,
    ExtensionRuntime, ExtensionShortcut, ExtensionUiContext, ExtensionUIDialogOptions, ForkOptions,
    InputEventResult, MessageEndEventResult, MessageRenderer, NavigateTreeOptions, NewSessionOptions,
    ProviderActions, ProviderConfig, RegisteredCommand, RegisteredTool, ReplacedSessionContext, ResolvedCommand,
    SendMessageOptions, SendUserMessageOptions, SessionBeforeCompactResult, SessionBeforeForkResult,
    SessionBeforeRefineResult, SessionBeforeSwitchResult, SessionBeforeTreeResult, SharedExtension, SwitchSessionOptions,
    Theme, ToolCallEvent, ToolCallEventResult, ToolInfo, ToolResultEvent, ToolResultEventResult, UserBashEventResult,
    WidgetPlacement, WorkingIndicatorOptions,
};
use super::types::{
    AutocompleteItem, AutocompleteProviderFactory, Component, EditorFactory, ExtensionWidgetOptions,
    ReadonlyFooterDataProvider, ReadonlySessionManager, SessionManager, TerminalInputHandler, ThemeInfo,
    SetThemeResult, ModelRegistry,
};

// Extension shortcuts compete with canonical keybinding ids from keybindings.json.
// Only editor-global shortcuts are reserved here. Picker-specific bindings are not.
pub const RESERVED_KEYBINDINGS_FOR_EXTENSION_CONFLICTS: [&str; 17] = [
    "app.interrupt",
    "app.clear",
    "app.exit",
    "app.suspend",
    "app.model.select",
    "app.tools.expand",
    "app.messages.expand",
    "app.edits.expand",
    "app.thinking.toggle",
    "app.subagents.focus",
    "app.editor.external",
    "app.message.followUp",
    "tui.input.submit",
    "tui.select.confirm",
    "tui.select.cancel",
    "tui.input.copy",
    "tui.editor.deleteToLineEnd",
];

/// `{ keybinding; restrictOverride }`.
#[derive(Debug, Clone, PartialEq)]
pub struct BuiltInKeyBinding {
    pub keybinding: String,
    pub restrict_override: bool,
}

/// `buildBuiltinKeybindings(resolvedKeybindings)`.
pub fn build_builtin_keybindings(resolved_keybindings: &Map<String, Value>) -> HashMap<String, BuiltInKeyBinding> {
    let mut builtin_keybindings: HashMap<String, BuiltInKeyBinding> = HashMap::new();
    for (keybinding, keys) in resolved_keybindings {
        if keys.is_null() {
            continue;
        }
        let key_list: Vec<String> = match keys {
            Value::Array(list) => list
                .iter()
                .filter_map(|value| value.as_str().map(str::to_string))
                .collect(),
            Value::String(value) => vec![value.clone()],
            _ => continue,
        };
        let restrict_override = RESERVED_KEYBINDINGS_FOR_EXTENSION_CONFLICTS.contains(&keybinding.as_str());
        for key in key_list {
            let normalized_key = key.to_lowercase();
            // If multiple actions bind the same key, the reserved action wins so
            // extensions remain blocked by reserved shortcuts regardless of
            // iteration order.
            let existing = builtin_keybindings.get(&normalized_key);
            if existing.map(|existing| existing.restrict_override).unwrap_or(false) && !restrict_override {
                continue;
            }
            builtin_keybindings.insert(
                normalized_key,
                BuiltInKeyBinding {
                    keybinding: keybinding.clone(),
                    restrict_override,
                },
            );
        }
    }
    builtin_keybindings
}

/// Combined result from all `before_agent_start` handlers.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BeforeAgentStartCombinedResult {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub messages: Option<Vec<CustomMessagePayload>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
}

/// `resources_discover` result entry `{ path; extensionPath }`.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResourcePathEntry {
    pub path: String,
    pub extension_path: String,
}

/// Result of `emitResourcesDiscover`.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResourcesDiscoverPaths {
    pub skill_paths: Vec<ResourcePathEntry>,
    pub prompt_paths: Vec<ResourcePathEntry>,
    pub theme_paths: Vec<ResourcePathEntry>,
}

/// `ExtensionErrorListener`.
pub type ExtensionErrorListener = Arc<dyn Fn(ExtensionError) + Send + Sync>;

/// `NewSessionHandler`.
pub type NewSessionHandler = Arc<
    dyn Fn(Option<NewSessionOptions>) -> std::pin::Pin<Box<dyn std::future::Future<Output = CancelledResult> + Send>>
        + Send
        + Sync,
>;
/// `ForkHandler`.
pub type ForkHandler = Arc<
    dyn Fn(String, Option<ForkOptions>) -> std::pin::Pin<Box<dyn std::future::Future<Output = CancelledResult> + Send>>
        + Send
        + Sync,
>;
/// `NavigateTreeHandler`.
pub type NavigateTreeHandler = Arc<
    dyn Fn(String, Option<NavigateTreeOptions>) -> std::pin::Pin<Box<dyn std::future::Future<Output = CancelledResult> + Send>>
        + Send
        + Sync,
>;
/// `SwitchSessionHandler`.
pub type SwitchSessionHandler = Arc<
    dyn Fn(String, Option<SwitchSessionOptions>) -> std::pin::Pin<Box<dyn std::future::Future<Output = CancelledResult> + Send>>
        + Send
        + Sync,
>;
/// `ReloadHandler`.
pub type ReloadHandler =
    Arc<dyn Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>> + Send + Sync>;
/// `ShutdownHandler`.
pub type ShutdownHandler = Arc<dyn Fn() + Send + Sync>;

/// Helper to emit `session_shutdown` to extensions.
/// Returns true if the event was emitted, false if there were no handlers.
pub async fn emit_session_shutdown_event(
    extension_runner: &Arc<ExtensionRunner>,
    event: ExtensionEvent,
) -> bool {
    if extension_runner.has_handlers("session_shutdown") {
        extension_runner.emit(event).await;
        return true;
    }
    false
}

/// `noOpUIContext` - every method is a no-op, exactly like the TypeScript object.
pub struct NoOpUiContext;

impl ExtensionUiContext for NoOpUiContext {
    fn select(
        &self,
        _title: String,
        _options: Vec<String>,
        _opts: Option<ExtensionUIDialogOptions>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Option<String>> + Send>> {
        Box::pin(async { None })
    }

    fn confirm(
        &self,
        _title: String,
        _message: String,
        _opts: Option<ExtensionUIDialogOptions>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = bool> + Send>> {
        Box::pin(async { false })
    }

    fn input(
        &self,
        _title: String,
        _placeholder: Option<String>,
        _opts: Option<ExtensionUIDialogOptions>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Option<String>> + Send>> {
        Box::pin(async { None })
    }

    fn notify(&self, _message: String, _kind: Option<String>) {}

    fn on_terminal_input(&self, _handler: TerminalInputHandler) -> Arc<dyn Fn() + Send + Sync> {
        Arc::new(|| {})
    }

    fn set_status(&self, _key: String, _text: Option<String>) {}
    fn set_working_message(&self, _message: Option<String>) {}
    fn set_working_visible(&self, _visible: bool) {}
    fn set_working_indicator(&self, _options: Option<WorkingIndicatorOptions>) {}
    fn set_hidden_thinking_label(&self, _label: Option<String>) {}

    fn set_widget_strings(
        &self,
        _key: String,
        _content: Option<Vec<String>>,
        _options: Option<ExtensionWidgetOptions>,
    ) {
    }

    fn set_widget_factory(
        &self,
        _key: String,
        _content: Option<super::types::WidgetFactory>,
        _options: Option<ExtensionWidgetOptions>,
    ) {
    }

    fn set_footer(&self, _factory: Option<super::types::FooterFactory>) {}
    fn set_header(&self, _factory: Option<super::types::HeaderFactory>) {}
    fn set_title(&self, _title: String) {}

    fn custom(&self, _factory: crate::core::extensions::types::CustomComponentFactory, _options: Option<Value>) -> super::types::CustomComponentResult {
        Box::pin(async { None })
    }

    fn paste_to_editor(&self, _text: String) {}
    fn set_editor_text(&self, _text: String) {}

    fn get_editor_text(&self) -> String {
        String::new()
    }

    fn editor(
        &self,
        _title: String,
        _prefill: Option<String>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Option<String>> + Send>> {
        Box::pin(async { None })
    }

    fn add_autocomplete_provider(&self, _factory: AutocompleteProviderFactory) {}
    fn set_editor_component(&self, _factory: Option<EditorFactory>) {}

    fn get_editor_component(&self) -> Option<EditorFactory> {
        None
    }

    fn theme(&self) -> Theme {
        Theme::default()
    }

    fn get_all_themes(&self) -> Vec<ThemeInfo> {
        Vec::new()
    }

    fn get_theme(&self, _name: String) -> Option<Theme> {
        None
    }

    fn set_theme(&self, _theme: Value) -> SetThemeResult {
        SetThemeResult {
            success: false,
            error: Some("UI not available".to_string()),
        }
    }

    fn get_tools_expanded(&self) -> bool {
        false
    }

    fn set_tools_expanded(&self, _expanded: bool) {}
}

/// A component that renders nothing.
pub struct NoOpComponent;

impl Component for NoOpComponent {
    fn render(&self, _width: usize) -> Vec<String> {
        Vec::new()
    }
}

/// Null session manager used when no session is bound yet.
pub struct NullSessionManager;

impl ReadonlySessionManager for NullSessionManager {
    fn get_session_id(&self) -> String {
        String::new()
    }
    fn get_session_file(&self) -> Option<String> {
        None
    }
    fn get_session_dir(&self) -> String {
        String::new()
    }
    fn get_branch(&self) -> Vec<super::types::SessionEntry> {
        Vec::new()
    }
}

impl SessionManager for NullSessionManager {}

/// Null model registry used when no registry is bound yet.
pub struct NullModelRegistry;

impl ModelRegistry for NullModelRegistry {
    fn register_provider(&self, _name: &str, _config: &ProviderConfig) {}
    fn unregister_provider(&self, _name: &str) {}

    fn get_api_key_and_headers(
        &self,
        _model: &pi_ai::types::Model,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Value, String>> + Send>> {
        Box::pin(async { Ok(Value::Null) })
    }
}


/// Bindable callbacks. `bindCore()` and `bindCommandContext()` replace these
/// after construction; the TypeScript reassigns plain fields, so the port keeps
/// one lock-protected record instead of unsafe field mutation.
pub struct RunnerCallbacks {
    pub get_model: Arc<dyn Fn() -> Option<pi_ai::types::Model> + Send + Sync>,
    pub is_idle: Arc<dyn Fn() -> bool + Send + Sync>,
    pub get_signal: Arc<dyn Fn() -> Option<AbortSignal> + Send + Sync>,
    pub wait_for_idle: Arc<dyn Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>> + Send + Sync>,
    pub abort: Arc<dyn Fn() + Send + Sync>,
    pub has_pending_messages: Arc<dyn Fn() -> bool + Send + Sync>,
    pub shutdown: ShutdownHandler,
    pub get_context_usage: Arc<dyn Fn() -> Option<ContextUsage> + Send + Sync>,
    pub compact: Arc<dyn Fn(Option<CompactOptions>) + Send + Sync>,
    pub get_system_prompt: Arc<dyn Fn() -> String + Send + Sync>,
    pub new_session: NewSessionHandler,
    pub fork: ForkHandler,
    pub navigate_tree: NavigateTreeHandler,
    pub switch_session: SwitchSessionHandler,
    pub reload: ReloadHandler,
}

impl Default for RunnerCallbacks {
    fn default() -> Self {
        Self {
            get_model: Arc::new(|| None),
            is_idle: Arc::new(|| true),
            get_signal: Arc::new(|| None),
            wait_for_idle: Arc::new(|| Box::pin(async {})),
            abort: Arc::new(|| {}),
            has_pending_messages: Arc::new(|| false),
            shutdown: Arc::new(|| {}),
            get_context_usage: Arc::new(|| None),
            compact: Arc::new(|_| {}),
            get_system_prompt: Arc::new(String::new),
            new_session: Arc::new(|_| Box::pin(async { CancelledResult { cancelled: false } })),
            fork: Arc::new(|_, _| Box::pin(async { CancelledResult { cancelled: false } })),
            navigate_tree: Arc::new(|_, _| Box::pin(async { CancelledResult { cancelled: false } })),
            switch_session: Arc::new(|_, _| Box::pin(async { CancelledResult { cancelled: false } })),
            reload: Arc::new(|| Box::pin(async {})),
        }
    }
}

impl std::fmt::Debug for RunnerCallbacks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunnerCallbacks").finish_non_exhaustive()
    }
}

/// `ExtensionRunner`.
pub struct ExtensionRunner {
    extensions: Vec<SharedExtension>,
    runtime: ExtensionRuntime,
    ui_context: Mutex<Option<Arc<dyn ExtensionUiContext>>>,
    cwd: String,
    session_manager: Arc<dyn SessionManager>,
    model_registry: Arc<dyn ModelRegistry>,
    error_listeners: Arc<Mutex<Vec<ExtensionErrorListener>>>,
    callbacks: Mutex<RunnerCallbacks>,
    shortcut_diagnostics: Mutex<Vec<ResourceDiagnostic>>,
    command_diagnostics: Mutex<Vec<ResourceDiagnostic>>,
    stale_message: Mutex<Option<String>>,
}

impl std::fmt::Debug for ExtensionRunner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExtensionRunner")
            .field("cwd", &self.cwd)
            .field("extensions", &self.extensions.len())
            .finish_non_exhaustive()
    }
}

impl ExtensionRunner {
    pub fn new(
        extensions: Vec<SharedExtension>,
        runtime: ExtensionRuntime,
        cwd: String,
        session_manager: Arc<dyn SessionManager>,
        model_registry: Arc<dyn ModelRegistry>,
    ) -> Self {
        Self {
            extensions,
            runtime,
            ui_context: Mutex::new(None),
            cwd,
            session_manager,
            model_registry,
            error_listeners: Arc::new(Mutex::new(Vec::new())),
            callbacks: Mutex::new(RunnerCallbacks::default()),
            shortcut_diagnostics: Mutex::new(Vec::new()),
            command_diagnostics: Mutex::new(Vec::new()),
            stale_message: Mutex::new(None),
        }
    }

    /// `bindCore(actions, contextActions, providerActions?)`.
    pub fn bind_core(
        &self,
        actions: ExtensionActions,
        context_actions: ExtensionContextActions,
        provider_actions: Option<ProviderActions>,
    ) {
        {
            let mut guard = self
                .runtime
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            guard.actions = Some(actions);
            if let Some(provider_actions) = provider_actions {
                guard.provider_actions = Some(provider_actions);
            }
        }
        let mut callbacks = self
            .callbacks
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        callbacks.get_model = context_actions.get_model.clone();
        callbacks.is_idle = context_actions.is_idle.clone();
        callbacks.get_signal = context_actions.get_signal.clone();
        callbacks.abort = context_actions.abort.clone();
        callbacks.has_pending_messages = context_actions.has_pending_messages.clone();
        callbacks.shutdown = context_actions.shutdown.clone();
        callbacks.get_context_usage = context_actions.get_context_usage.clone();
        callbacks.compact = context_actions.compact.clone();
        callbacks.get_system_prompt = context_actions.get_system_prompt.clone();
        drop(callbacks);

        for registration in self.runtime.take_pending_provider_registrations() {
            let result = {
                let guard = self
                    .runtime
                    .state
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                match guard
                    .provider_actions
                    .as_ref()
                    .and_then(|actions| actions.register_provider.clone())
                {
                    Some(register) => {
                        register(registration.name.clone(), registration.config.clone());
                        Ok(())
                    }
                    None => {
                        self.model_registry
                            .register_provider(&registration.name, &registration.config);
                        Ok(())
                    }
                }
            };
            if let Err(error) = result {
                self.emit_error(ExtensionError {
                    extension_path: registration.extension_path,
                    event: "register_provider".to_string(),
                    error,
                    stack: None,
                });
            }
        }
    }

    /// `bindCommandContext(actions?)`.
    pub fn bind_command_context(&self, actions: Option<ExtensionCommandContextActions>) {
        let mut callbacks = self
            .callbacks
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match actions {
            Some(actions) => {
                callbacks.wait_for_idle = actions.wait_for_idle;
                callbacks.new_session = actions.new_session;
                callbacks.fork = actions.fork;
                callbacks.navigate_tree = actions.navigate_tree;
                callbacks.switch_session = actions.switch_session;
                callbacks.reload = actions.reload;
            }
            None => {
                callbacks.wait_for_idle = Arc::new(|| Box::pin(async {}));
                callbacks.new_session = Arc::new(|_| Box::pin(async { CancelledResult { cancelled: false } }));
                callbacks.fork = Arc::new(|_, _| Box::pin(async { CancelledResult { cancelled: false } }));
                callbacks.navigate_tree = Arc::new(|_, _| Box::pin(async { CancelledResult { cancelled: false } }));
                callbacks.switch_session = Arc::new(|_, _| Box::pin(async { CancelledResult { cancelled: false } }));
                callbacks.reload = Arc::new(|| Box::pin(async {}));
            }
        }
    }

    pub fn set_ui_context(&self, ui_context: Option<Arc<dyn ExtensionUiContext>>) {
        *self.ui_context.lock().unwrap_or_else(|p| p.into_inner()) = ui_context;
    }

    pub fn get_ui_context(&self) -> Arc<dyn ExtensionUiContext> {
        self.ui_context.lock().unwrap_or_else(|p| p.into_inner()).clone().unwrap_or_else(|| Arc::new(NoOpUiContext))
    }

    /// `hasUI()` - true when a real (non-no-op) UI context is installed.
    pub fn has_ui(&self) -> bool {
        self.ui_context.lock().unwrap_or_else(|p| p.into_inner()).is_some()
    }

    pub fn get_extension_paths(&self) -> Vec<String> {
        self.extensions
            .iter()
            .map(|extension| extension.lock().unwrap_or_else(|p| p.into_inner()).path.clone())
            .collect()
    }

    /// Get all registered tools from all extensions (first registration per name wins).
    pub fn get_all_registered_tools(&self) -> Vec<RegisteredTool> {
        let mut tools_by_name: indexmap::IndexMap<String, RegisteredTool> = indexmap::IndexMap::new();
        for extension in &self.extensions {
            let guard = extension.lock().unwrap_or_else(|p| p.into_inner());
            for tool in guard.tools.values() {
                if !tools_by_name.contains_key(&tool.definition.name) {
                    tools_by_name.insert(tool.definition.name.clone(), tool.clone());
                }
            }
        }
        tools_by_name.into_values().collect()
    }

    /// Get a tool definition by name. Returns `None` if not found.
    pub fn get_tool_definition(&self, tool_name: &str) -> Option<super::types::ToolDefinition> {
        for extension in &self.extensions {
            let guard = extension.lock().unwrap_or_else(|p| p.into_inner());
            if let Some(tool) = guard.tools.get(tool_name) {
                return Some(tool.definition.clone());
            }
        }
        None
    }

    pub fn get_flags(&self) -> indexmap::IndexMap<String, ExtensionFlag> {
        let mut all_flags: indexmap::IndexMap<String, ExtensionFlag> = indexmap::IndexMap::new();
        for extension in &self.extensions {
            let guard = extension.lock().unwrap_or_else(|p| p.into_inner());
            for (name, flag) in &guard.flags {
                if !all_flags.contains_key(name) {
                    all_flags.insert(name.clone(), flag.clone());
                }
            }
        }
        all_flags
    }

    pub fn set_flag_value(&self, name: &str, value: Value) {
        self.runtime.flag_values_set(name, value);
    }

    pub fn get_flag_values(&self) -> indexmap::IndexMap<String, Value> {
        self.runtime.flag_values_snapshot()
    }

    /// `getShortcuts(resolvedKeybindings)`.
    pub fn get_shortcuts(&self, resolved_keybindings: &Map<String, Value>) -> indexmap::IndexMap<String, ExtensionShortcut> {
        let builtin_keybindings = build_builtin_keybindings(resolved_keybindings);
        let mut extension_shortcuts: indexmap::IndexMap<String, ExtensionShortcut> = indexmap::IndexMap::new();
        let mut diagnostics: Vec<ResourceDiagnostic> = Vec::new();
        let has_ui = self.has_ui();

        let mut add_diagnostic = |diagnostics: &mut Vec<ResourceDiagnostic>, message: String, extension_path: &str| {
            diagnostics.push(ResourceDiagnostic {
                diagnostic_type: "warning".to_string(),
                message: message.clone(),
                path: Some(extension_path.to_string()),
                collision: None,
            });
            if !has_ui {
                eprintln!("{message}");
            }
        };

        for extension in &self.extensions {
            let guard = extension.lock().unwrap_or_else(|p| p.into_inner());
            for (key, shortcut) in &guard.shortcuts {
                let normalized_key = key.to_lowercase();

                if let Some(built_in) = builtin_keybindings.get(&normalized_key) {
                    if built_in.restrict_override {
                        add_diagnostic(
                            &mut diagnostics,
                            format!(
                                "Extension shortcut '{}' from {} conflicts with built-in shortcut. Skipping.",
                                key, shortcut.extension_path
                            ),
                            &shortcut.extension_path,
                        );
                        continue;
                    }
                    add_diagnostic(
                        &mut diagnostics,
                        format!(
                            "Extension shortcut conflict: '{}' is built-in shortcut for {} and {}. Using {}.",
                            key, built_in.keybinding, shortcut.extension_path, shortcut.extension_path
                        ),
                        &shortcut.extension_path,
                    );
                }

                if let Some(existing) = extension_shortcuts.get(&normalized_key) {
                    add_diagnostic(
                        &mut diagnostics,
                        format!(
                            "Extension shortcut conflict: '{}' registered by both {} and {}. Using {}.",
                            key, existing.extension_path, shortcut.extension_path, shortcut.extension_path
                        ),
                        &shortcut.extension_path,
                    );
                }
                extension_shortcuts.insert(normalized_key, shortcut.clone());
            }
        }

        *self
            .shortcut_diagnostics
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = diagnostics;
        extension_shortcuts
    }

    pub fn get_shortcut_diagnostics(&self) -> Vec<ResourceDiagnostic> {
        self.shortcut_diagnostics
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// `invalidate(message?)`.
    pub fn invalidate(&self, message: Option<String>) {
        let mut stale = self
            .stale_message
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if stale.is_none() {
            let message = message.unwrap_or_else(|| super::types::EXTENSION_RUNTIME_STALE_MESSAGE.to_string());
            *stale = Some(message.clone());
            self.runtime.invalidate(Some(message));
        }
    }

    fn assert_active(&self) -> Result<(), String> {
        let stale = self
            .stale_message
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match stale.as_ref() {
            Some(message) => Err(message.clone()),
            None => Ok(()),
        }
    }

    pub fn on_error(&self, listener: ExtensionErrorListener) -> Arc<dyn Fn() + Send + Sync> {
        {
            let mut listeners = self
                .error_listeners
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            listeners.push(listener.clone());
        }
        let listeners = self.error_listeners.clone();
        Arc::new(move || {
            let mut guard = listeners.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            guard.retain(|existing| !Arc::ptr_eq(existing, &listener));
        })
    }

    pub fn emit_error(&self, error: ExtensionError) {
        let listeners = self
            .error_listeners
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        for listener in listeners {
            listener(error.clone());
        }
    }

    pub fn has_handlers(&self, event_type: &str) -> bool {
        for extension in &self.extensions {
            let guard = extension.lock().unwrap_or_else(|p| p.into_inner());
            if guard
                .handlers
                .get(event_type)
                .map(|handlers| !handlers.is_empty())
                .unwrap_or(false)
            {
                return true;
            }
        }
        false
    }

    pub fn get_message_renderer(&self, custom_type: &str) -> Option<MessageRenderer> {
        for extension in &self.extensions {
            let guard = extension.lock().unwrap_or_else(|p| p.into_inner());
            if let Some(renderer) = guard.message_renderers.get(custom_type) {
                return Some(renderer.clone());
            }
        }
        None
    }

    /// `resolveRegisteredCommands()`.
    fn resolve_registered_commands(&self) -> Vec<ResolvedCommand> {
        let mut commands: Vec<RegisteredCommand> = Vec::new();
        let mut counts: HashMap<String, usize> = HashMap::new();

        for extension in &self.extensions {
            let guard = extension.lock().unwrap_or_else(|p| p.into_inner());
            for command in guard.commands.values() {
                commands.push(command.clone());
                *counts.entry(command.name.clone()).or_insert(0) += 1;
            }
        }

        let mut seen: HashMap<String, usize> = HashMap::new();
        let mut taken_invocation_names: HashSet<String> = HashSet::new();
        let mut resolved: Vec<ResolvedCommand> = Vec::new();

        for command in commands {
            let occurrence = seen.get(&command.name).copied().unwrap_or(0) + 1;
            seen.insert(command.name.clone(), occurrence);

            let mut invocation_name = if counts.get(&command.name).copied().unwrap_or(0) > 1 {
                format!("{}:{}", command.name, occurrence)
            } else {
                command.name.clone()
            };

            if taken_invocation_names.contains(&invocation_name) {
                let mut suffix = occurrence;
                loop {
                    suffix += 1;
                    invocation_name = format!("{}:{}", command.name, suffix);
                    if !taken_invocation_names.contains(&invocation_name) {
                        break;
                    }
                }
            }

            taken_invocation_names.insert(invocation_name.clone());
            resolved.push(ResolvedCommand {
                command,
                invocation_name,
            });
        }

        resolved
    }

    pub fn get_registered_commands(&self) -> Vec<ResolvedCommand> {
        *self
            .command_diagnostics
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Vec::new();
        self.resolve_registered_commands()
    }

    pub fn get_command_diagnostics(&self) -> Vec<ResourceDiagnostic> {
        self.command_diagnostics
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    pub fn get_command(&self, name: &str) -> Option<ResolvedCommand> {
        self.resolve_registered_commands()
            .into_iter()
            .find(|command| command.invocation_name == name)
    }

    /// Request a graceful shutdown. Called by extension tools and event handlers.
    pub fn shutdown(&self) {
        let handler = self
            .callbacks
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .shutdown
            .clone();
        handler();
    }

    /// Create an `ExtensionContext` for use in event handlers and tool execution.
    /// Context values are resolved at call time, so changes via `bindCore`/`bindUI`
    /// are reflected.
    pub fn create_context(self: &Arc<Self>) -> Arc<dyn ExtensionContext> {
        Arc::new(RunnerContext {
            runner: self.clone(),
        })
    }

    /// Create an `ExtensionCommandContext` for command handlers.
    pub fn create_command_context(self: &Arc<Self>) -> Arc<dyn ExtensionCommandContext> {
        Arc::new(RunnerCommandContext {
            runner: self.clone(),
        })
    }

    fn is_session_before_event(event: &ExtensionEvent) -> bool {
        matches!(
            event.event_type(),
            "session_before_switch"
                | "session_before_fork"
                | "session_before_compact"
                | "session_before_refine"
                | "session_before_tree"
        )
    }

    /// `emit(event)` - generic emit for events without a dedicated method.
    pub async fn emit(self: &Arc<Self>, event: ExtensionEvent) -> Option<Value> {
        let ctx = match self.context_handle() {
            Some(ctx) => ctx,
            None => return None,
        };
        let mut result: Option<Value> = None;

        for extension in &self.extensions {
            let (extension_path, handlers) = {
                let guard = extension.lock().unwrap_or_else(|p| p.into_inner());
                (
                    guard.path.clone(),
                    guard.handlers.get(event.event_type()).cloned().unwrap_or_default(),
                )
            };
            if handlers.is_empty() {
                continue;
            }

            for handler in handlers {
                match invoke_handler(&handler, event.clone(), ctx.clone()).await {
                    Ok(handler_result) => {
                        if Self::is_session_before_event(&event) {
                            if let Some(handler_result) = handler_result {
                                let cancel = handler_result.get("cancel").and_then(Value::as_bool).unwrap_or(false);
                                let skip = handler_result.get("skip").and_then(Value::as_bool).unwrap_or(false);
                                result = Some(handler_result);
                                if cancel || skip {
                                    return result;
                                }
                            }
                        }
                    }
                    Err(error) => {
                        self.emit_error(ExtensionError {
                            extension_path: extension_path.clone(),
                            event: event.event_type().to_string(),
                            error,
                            stack: None,
                        });
                    }
                }
            }
        }

        result
    }

    /// `emitMessageEnd(event)`.
    pub async fn emit_message_end(self: &Arc<Self>, event: Value) -> Option<Value> {
        let ctx = self.context_handle()?;
        let mut current_message = event.get("message").cloned().unwrap_or(Value::Null);
        let mut modified = false;

        for extension in &self.extensions {
            let (extension_path, handlers) = {
                let guard = extension.lock().unwrap_or_else(|p| p.into_inner());
                (
                    guard.path.clone(),
                    guard.handlers.get("message_end").cloned().unwrap_or_default(),
                )
            };
            if handlers.is_empty() {
                continue;
            }

            for handler in handlers {
                let mut current_event = event.clone();
                if let Value::Object(map) = &mut current_event {
                    map.insert("message".to_string(), current_message.clone());
                }
                match invoke_handler(&handler, ExtensionEvent::MessageEnd(super::types::MessageEndPayload { message: current_message.clone() }), ctx.clone()).await {
                    Ok(Some(handler_result)) => {
                        let Some(message) = handler_result.get("message").cloned() else {
                            continue;
                        };
                        if message.is_null() {
                            continue;
                        }
                        let current_role = role_of(&current_message);
                        let new_role = role_of(&message);
                        if current_role != new_role {
                            self.emit_error(ExtensionError {
                                extension_path: extension_path.clone(),
                                event: "message_end".to_string(),
                                error: "message_end handlers must return a message with the same role".to_string(),
                                stack: None,
                            });
                            continue;
                        }
                        current_message = message;
                        modified = true;
                    }
                    Ok(None) => {}
                    Err(error) => {
                        self.emit_error(ExtensionError {
                            extension_path: extension_path.clone(),
                            event: "message_end".to_string(),
                            error,
                            stack: None,
                        });
                    }
                }
            }
        }

        if modified {
            Some(current_message)
        } else {
            None
        }
    }

    /// `emitToolResult(event)`.
    pub async fn emit_tool_result(self: &Arc<Self>, event: &ToolResultEvent) -> Option<ToolResultEventResult> {
        let ctx = self.context_handle()?;
        let mut current_event = event.clone();
        let mut modified = false;

        for extension in &self.extensions {
            let (extension_path, handlers) = {
                let guard = extension.lock().unwrap_or_else(|p| p.into_inner());
                (
                    guard.path.clone(),
                    guard.handlers.get("tool_result").cloned().unwrap_or_default(),
                )
            };
            if handlers.is_empty() {
                continue;
            }

            for handler in handlers {
                let event_value = serde_json::to_value(&current_event).unwrap_or(Value::Null);
                match invoke_handler(&handler, 
                    ExtensionEvent::ToolResult(current_event.clone()),
                    ctx.clone(),
                )
                .await
                {
                    Ok(Some(handler_result)) => {
                        let content = handler_result.get("content").cloned().filter(|v| !v.is_null());
                        let details = handler_result.get("details").cloned().filter(|v| !v.is_null());
                        let is_error = handler_result.get("isError").and_then(Value::as_bool);
                        if let Some(content) = content {
                            set_tool_result_content(&mut current_event, content);
                            modified = true;
                        }
                        if let Some(details) = details {
                            set_tool_result_details(&mut current_event, details);
                            modified = true;
                        }
                        if let Some(is_error) = is_error {
                            set_tool_result_is_error(&mut current_event, is_error);
                            modified = true;
                        }
                        let _ = event_value;
                    }
                    Ok(None) => {}
                    Err(error) => {
                        self.emit_error(ExtensionError {
                            extension_path: extension_path.clone(),
                            event: "tool_result".to_string(),
                            error,
                            stack: None,
                        });
                    }
                }
            }
        }

        if !modified {
            return None;
        }

        Some(ToolResultEventResult {
            content: Some(current_event.content().to_vec()),
            details: Some(current_event.details()),
            is_error: Some(current_event.is_error()),
        })
    }

    /// `emitToolCall(event)`.
    pub async fn emit_tool_call(self: &Arc<Self>, event: &ToolCallEvent) -> Option<ToolCallEventResult> {
        let ctx = self.context_handle()?;
        let mut result: Option<ToolCallEventResult> = None;

        for extension in &self.extensions {
            let (extension_path, handlers) = {
                let guard = extension.lock().unwrap_or_else(|p| p.into_inner());
                (
                    guard.path.clone(),
                    guard.handlers.get("tool_call").cloned().unwrap_or_default(),
                )
            };
            if handlers.is_empty() {
                continue;
            }

            for handler in handlers {
                match invoke_handler(&handler, ExtensionEvent::ToolCall(event.clone()), ctx.clone()).await {
                    Ok(Some(handler_result)) => {
                        let blocked = handler_result.get("block").and_then(Value::as_bool).unwrap_or(false);
                        let parsed = ToolCallEventResult {
                            block: handler_result.get("block").and_then(Value::as_bool),
                            reason: handler_result
                                .get("reason")
                                .and_then(Value::as_str)
                                .map(str::to_string),
                        };
                        result = Some(parsed);
                        if blocked {
                            return result;
                        }
                    }
                    // `emitToolCall` does not catch handler errors: a throwing
                    // handler rejects the whole emit, matching the TypeScript.
                    Ok(None) => {}
                    Err(error) => {
                        self.emit_error(ExtensionError {
                            extension_path: extension_path.clone(),
                            event: "tool_call".to_string(),
                            error,
                            stack: None,
                        });
                    }
                }
            }
        }

        result
    }

    /// `emitUserBash(event)`.
    pub async fn emit_user_bash(self: &Arc<Self>, event: Value) -> Option<UserBashEventResult> {
        let ctx = self.context_handle()?;

        for extension in &self.extensions {
            let (extension_path, handlers) = {
                let guard = extension.lock().unwrap_or_else(|p| p.into_inner());
                (
                    guard.path.clone(),
                    guard.handlers.get("user_bash").cloned().unwrap_or_default(),
                )
            };
            if handlers.is_empty() {
                continue;
            }

            for handler in handlers {
                let parsed = serde_json::from_value::<super::types::UserBashPayload>(event.clone())
                    .ok()
                    .map(ExtensionEvent::UserBash);
                let Some(extension_event) = parsed else {
                    continue;
                };
                match invoke_handler(&handler, extension_event, ctx.clone()).await {
                    Ok(Some(handler_result)) => {
                        return Some(UserBashEventResult {
                            operations: None,
                            result: handler_result
                                .get("result")
                                .and_then(|value| serde_json::from_value(value.clone()).ok()),
                        });
                    }
                    Ok(None) => {}
                    Err(error) => {
                        self.emit_error(ExtensionError {
                            extension_path: extension_path.clone(),
                            event: "user_bash".to_string(),
                            error,
                            stack: None,
                        });
                    }
                }
            }
        }

        None
    }

    /// `emitContext(messages)`.
    pub async fn emit_context(self: &Arc<Self>, messages: Vec<Value>) -> Vec<Value> {
        let Some(ctx) = self.context_handle() else {
            return messages;
        };
        // `structuredClone(messages)`.
        let mut current_messages = messages;

        for extension in &self.extensions {
            let (extension_path, handlers) = {
                let guard = extension.lock().unwrap_or_else(|p| p.into_inner());
                (
                    guard.path.clone(),
                    guard.handlers.get("context").cloned().unwrap_or_default(),
                )
            };
            if handlers.is_empty() {
                continue;
            }

            for handler in handlers {
                let event = ExtensionEvent::Context(super::types::ContextPayload {
                    messages: current_messages.clone(),
                });
                match invoke_handler(&handler, event, ctx.clone()).await {
                    Ok(Some(handler_result)) => {
                        if let Some(next) = handler_result.get("messages").cloned() {
                            if !next.is_null() {
                                if let Ok(parsed) = serde_json::from_value::<ContextEventResult>(handler_result) {
                                    if let Some(next) = parsed.messages {
                                        current_messages = next;
                                    }
                                }
                            }
                        }
                    }
                    Ok(None) => {}
                    Err(error) => {
                        self.emit_error(ExtensionError {
                            extension_path: extension_path.clone(),
                            event: "context".to_string(),
                            error,
                            stack: None,
                        });
                    }
                }
            }
        }

        current_messages
    }

    /// `emitBeforeProviderRequest(payload)`.
    pub async fn emit_before_provider_request(self: &Arc<Self>, payload: Value) -> Value {
        let Some(ctx) = self.context_handle() else {
            return payload;
        };
        let mut current_payload = payload;

        for extension in &self.extensions {
            let (extension_path, handlers) = {
                let guard = extension.lock().unwrap_or_else(|p| p.into_inner());
                (
                    guard.path.clone(),
                    guard
                        .handlers
                        .get("before_provider_request")
                        .cloned()
                        .unwrap_or_default(),
                )
            };
            if handlers.is_empty() {
                continue;
            }

            for handler in handlers {
                let event = ExtensionEvent::BeforeProviderRequest(super::types::BeforeProviderRequestPayload {
                    payload: current_payload.clone(),
                });
                match invoke_handler(&handler, event, ctx.clone()).await {
                    Ok(Some(handler_result)) => {
                        current_payload = handler_result;
                    }
                    Ok(None) => {}
                    Err(error) => {
                        self.emit_error(ExtensionError {
                            extension_path: extension_path.clone(),
                            event: "before_provider_request".to_string(),
                            error,
                            stack: None,
                        });
                    }
                }
            }
        }

        current_payload
    }

    /// `emitBeforeAgentStart(prompt, images, systemPrompt, systemPromptOptions)`.
    pub async fn emit_before_agent_start(
        self: &Arc<Self>,
        prompt: String,
        images: Option<Vec<pi_ai::types::ImageContent>>,
        system_prompt: String,
        system_prompt_options: crate::core::system_prompt::BuildSystemPromptOptions,
    ) -> Option<BeforeAgentStartCombinedResult> {
        let Some(ctx) = self.context_handle() else {
            return None;
        };
        let mut current_system_prompt = system_prompt.clone();
        let mut messages: Vec<CustomMessagePayload> = Vec::new();
        let mut system_prompt_modified = false;

        for extension in &self.extensions {
            let (extension_path, handlers) = {
                let guard = extension.lock().unwrap_or_else(|p| p.into_inner());
                (
                    guard.path.clone(),
                    guard
                        .handlers
                        .get("before_agent_start")
                        .cloned()
                        .unwrap_or_default(),
                )
            };
            if handlers.is_empty() {
                continue;
            }

            for handler in handlers {
                let event = ExtensionEvent::BeforeAgentStart(super::types::BeforeAgentStartPayload {
                    prompt: prompt.clone(),
                    images: images.clone(),
                    system_prompt: current_system_prompt.clone(),
                    system_prompt_options: system_prompt_options.clone(),
                });
                match invoke_handler(&handler, event, ctx.clone()).await {
                    Ok(Some(handler_result)) => {
                        if let Ok(result) = serde_json::from_value::<BeforeAgentStartEventResult>(handler_result) {
                            if let Some(message) = result.message {
                                messages.push(message);
                            }
                            if let Some(system_prompt) = result.system_prompt {
                                current_system_prompt = system_prompt;
                                system_prompt_modified = true;
                            }
                        }
                    }
                    Ok(None) => {}
                    Err(error) => {
                        self.emit_error(ExtensionError {
                            extension_path: extension_path.clone(),
                            event: "before_agent_start".to_string(),
                            error,
                            stack: None,
                        });
                    }
                }
            }
        }

        if !messages.is_empty() || system_prompt_modified {
            return Some(BeforeAgentStartCombinedResult {
                messages: if messages.is_empty() { None } else { Some(messages) },
                system_prompt: if system_prompt_modified {
                    Some(current_system_prompt)
                } else {
                    None
                },
            });
        }

        None
    }

    /// `emitResourcesDiscover(cwd, reason)`.
    pub async fn emit_resources_discover(self: &Arc<Self>, cwd: String, reason: String) -> ResourcesDiscoverPaths {
        let Some(ctx) = self.context_handle() else {
            return ResourcesDiscoverPaths::default();
        };
        let mut skill_paths: Vec<ResourcePathEntry> = Vec::new();
        let mut prompt_paths: Vec<ResourcePathEntry> = Vec::new();
        let mut theme_paths: Vec<ResourcePathEntry> = Vec::new();

        for extension in &self.extensions {
            let (extension_path, handlers) = {
                let guard = extension.lock().unwrap_or_else(|p| p.into_inner());
                (
                    guard.path.clone(),
                    guard
                        .handlers
                        .get("resources_discover")
                        .cloned()
                        .unwrap_or_default(),
                )
            };
            if handlers.is_empty() {
                continue;
            }

            for handler in handlers {
                let event = ExtensionEvent::ResourcesDiscover(super::types::ResourcesDiscoverPayload {
                    cwd: cwd.clone(),
                    reason: reason.clone(),
                });
                match invoke_handler(&handler, event, ctx.clone()).await {
                    Ok(Some(handler_result)) => {
                        if let Ok(result) =
                            serde_json::from_value::<super::types::ResourcesDiscoverResult>(handler_result)
                        {
                            if let Some(paths) = result.skill_paths {
                                skill_paths.extend(paths.into_iter().map(|path| ResourcePathEntry {
                                    path,
                                    extension_path: extension_path.clone(),
                                }));
                            }
                            if let Some(paths) = result.prompt_paths {
                                prompt_paths.extend(paths.into_iter().map(|path| ResourcePathEntry {
                                    path,
                                    extension_path: extension_path.clone(),
                                }));
                            }
                            if let Some(paths) = result.theme_paths {
                                theme_paths.extend(paths.into_iter().map(|path| ResourcePathEntry {
                                    path,
                                    extension_path: extension_path.clone(),
                                }));
                            }
                        }
                    }
                    Ok(None) => {}
                    Err(error) => {
                        self.emit_error(ExtensionError {
                            extension_path: extension_path.clone(),
                            event: "resources_discover".to_string(),
                            error,
                            stack: None,
                        });
                    }
                }
            }
        }

        ResourcesDiscoverPaths {
            skill_paths,
            prompt_paths,
            theme_paths,
        }
    }

    /// Emit `input` event. Transforms chain, "handled" short-circuits.
    pub async fn emit_input(
        self: &Arc<Self>,
        text: String,
        images: Option<Vec<pi_ai::types::ImageContent>>,
        source: String,
    ) -> InputEventResult {
        let Some(ctx) = self.context_handle() else {
            return InputEventResult::Continue;
        };
        let original_text = text.clone();
        let original_images = images.clone();
        let mut current_text = text;
        let mut current_images = images;

        for extension in &self.extensions {
            let (extension_path, handlers) = {
                let guard = extension.lock().unwrap_or_else(|p| p.into_inner());
                (
                    guard.path.clone(),
                    guard.handlers.get("input").cloned().unwrap_or_default(),
                )
            };
            for handler in handlers {
                let event = ExtensionEvent::Input(super::types::InputPayload {
                    text: current_text.clone(),
                    images: current_images.clone(),
                    source: source.clone(),
                });
                match invoke_handler(&handler, event, ctx.clone()).await {
                    Ok(Some(handler_result)) => {
                        let action = handler_result.get("action").and_then(Value::as_str);
                        match action {
                            Some("handled") => return InputEventResult::Handled,
                            Some("transform") => {
                                if let Some(text) = handler_result.get("text").and_then(Value::as_str) {
                                    current_text = text.to_string();
                                }
                                if let Some(images) = handler_result.get("images") {
                                    if let Ok(parsed) =
                                        serde_json::from_value::<Vec<pi_ai::types::ImageContent>>(images.clone())
                                    {
                                        current_images = Some(parsed);
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                    Ok(None) => {}
                    Err(error) => {
                        self.emit_error(ExtensionError {
                            extension_path: extension_path.clone(),
                            event: "input".to_string(),
                            error,
                            stack: None,
                        });
                    }
                }
            }
        }

        if current_text != original_text || current_images != original_images {
            InputEventResult::Transform {
                text: current_text,
                images: current_images,
            }
        } else {
            InputEventResult::Continue
        }
    }

    /// Borrow this runner as an `Arc` for context construction.
    fn context_handle(self: &Arc<Self>) -> Option<Arc<dyn ExtensionContext>> {
        Some(self.create_context())
    }
}

// TypeScript catches synchronous throws and asynchronous handler rejections.
async fn invoke_handler(handler: &ExtensionHandler, event: ExtensionEvent, context: Arc<dyn ExtensionContext>) -> Result<Option<Value>, String> {
    std::panic::AssertUnwindSafe(async { handler(event, context).await }).catch_unwind().await
        .map_err(|error| {
            error.downcast_ref::<String>().cloned()
                .or_else(|| error.downcast_ref::<&str>().map(|message| (*message).to_string()))
                .unwrap_or_else(|| "Extension handler panicked".to_string())
        })
}

/// The `role` field of a message value.
fn role_of(message: &Value) -> Option<String> {
    message
        .get("role")
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn set_tool_result_content(event: &mut ToolResultEvent, content: Value) {
    let parsed = serde_json::from_value::<Vec<Value>>(content).unwrap_or_default();
    match event {
        ToolResultEvent::Bash { content: slot, .. } => *slot = parsed,
        ToolResultEvent::Edit { content: slot, .. } => *slot = parsed,
        ToolResultEvent::Ipython { content: slot, .. } => *slot = parsed,
        ToolResultEvent::Custom { content: slot, .. } => *slot = parsed,
    }
}

fn set_tool_result_details(event: &mut ToolResultEvent, details: Value) {
    match event {
        ToolResultEvent::Bash { details: slot, .. } => {
            *slot = serde_json::from_value(details).ok();
        }
        ToolResultEvent::Edit { details: slot, .. } => {
            *slot = serde_json::from_value(details).ok();
        }
        ToolResultEvent::Ipython { details: slot, .. } => {
            *slot = serde_json::from_value(details).ok();
        }
        ToolResultEvent::Custom { details: slot, .. } => *slot = details,
    }
}

fn set_tool_result_is_error(event: &mut ToolResultEvent, is_error: bool) {
    match event {
        ToolResultEvent::Bash { is_error: slot, .. } => *slot = is_error,
        ToolResultEvent::Edit { is_error: slot, .. } => *slot = is_error,
        ToolResultEvent::Ipython { is_error: slot, .. } => *slot = is_error,
        ToolResultEvent::Custom { is_error: slot, .. } => *slot = is_error,
    }
}

/// `createContext()` result - `ctx.*` in event handlers and tool execution.
///
/// Context values are resolved at call time through the runner, so changes made
/// by `bindCore`/`setUIContext` are reflected, and the stale-instance guard runs
/// on every access exactly like the TypeScript getters.
pub struct RunnerContext {
    pub runner: Arc<ExtensionRunner>,
}

impl RunnerContext {
    fn guard(&self) -> Result<(), String> {
        self.runner.assert_active()
    }
}

impl ExtensionContext for RunnerContext {
    fn ui(&self) -> Arc<dyn ExtensionUiContext> {
        self.runner.get_ui_context()
    }

    fn has_ui(&self) -> bool {
        self.runner.has_ui()
    }

    fn cwd(&self) -> String {
        self.runner.cwd.clone()
    }

    fn session_manager(&self) -> Arc<dyn ReadonlySessionManager> {
        self.runner.session_manager.clone()
    }

    fn model_registry(&self) -> Arc<dyn ModelRegistry> {
        self.runner.model_registry.clone()
    }

    fn model(&self) -> Option<pi_ai::types::Model> {
        let getter = self
            .runner
            .callbacks
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get_model
            .clone();
        getter()
    }

    fn is_idle(&self) -> bool {
        let getter = self
            .runner
            .callbacks
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .is_idle
            .clone();
        getter()
    }

    fn signal(&self) -> Option<AbortSignal> {
        let getter = self
            .runner
            .callbacks
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get_signal
            .clone();
        getter()
    }

    fn abort(&self) {
        let abort = self
            .runner
            .callbacks
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .abort
            .clone();
        abort();
    }

    fn has_pending_messages(&self) -> bool {
        let getter = self
            .runner
            .callbacks
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .has_pending_messages
            .clone();
        getter()
    }

    fn shutdown(&self) {
        self.runner.shutdown();
    }

    fn get_context_usage(&self) -> Option<ContextUsage> {
        let getter = self
            .runner
            .callbacks
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get_context_usage
            .clone();
        getter()
    }

    fn compact(&self, options: Option<CompactOptions>) {
        let compact = self
            .runner
            .callbacks
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .compact
            .clone();
        compact(options);
    }

    fn get_system_prompt(&self) -> String {
        let getter = self
            .runner
            .callbacks
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get_system_prompt
            .clone();
        getter()
    }
}

/// `createCommandContext()` result.
pub struct RunnerCommandContext {
    pub runner: Arc<ExtensionRunner>,
}

impl ExtensionContext for RunnerCommandContext {
    fn ui(&self) -> Arc<dyn ExtensionUiContext> {
        self.runner.get_ui_context()
    }
    fn has_ui(&self) -> bool {
        self.runner.has_ui()
    }
    fn cwd(&self) -> String {
        self.runner.cwd.clone()
    }
    fn session_manager(&self) -> Arc<dyn ReadonlySessionManager> {
        self.runner.session_manager.clone()
    }
    fn model_registry(&self) -> Arc<dyn ModelRegistry> {
        self.runner.model_registry.clone()
    }
    fn model(&self) -> Option<pi_ai::types::Model> {
        let getter = self
            .runner
            .callbacks
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get_model
            .clone();
        getter()
    }
    fn is_idle(&self) -> bool {
        let getter = self
            .runner
            .callbacks
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .is_idle
            .clone();
        getter()
    }
    fn signal(&self) -> Option<AbortSignal> {
        let getter = self
            .runner
            .callbacks
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get_signal
            .clone();
        getter()
    }
    fn abort(&self) {
        let abort = self
            .runner
            .callbacks
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .abort
            .clone();
        abort();
    }
    fn has_pending_messages(&self) -> bool {
        let getter = self
            .runner
            .callbacks
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .has_pending_messages
            .clone();
        getter()
    }
    fn shutdown(&self) {
        self.runner.shutdown();
    }
    fn get_context_usage(&self) -> Option<ContextUsage> {
        let getter = self
            .runner
            .callbacks
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get_context_usage
            .clone();
        getter()
    }
    fn compact(&self, options: Option<CompactOptions>) {
        let compact = self
            .runner
            .callbacks
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .compact
            .clone();
        compact(options);
    }
    fn get_system_prompt(&self) -> String {
        let getter = self
            .runner
            .callbacks
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get_system_prompt
            .clone();
        getter()
    }
}

impl ExtensionCommandContext for RunnerCommandContext {
    fn wait_for_idle(&self) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>> {
        if self.runner.assert_active().is_err() {
            return Box::pin(async {});
        }
        let handler = self
            .runner
            .callbacks
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .wait_for_idle
            .clone();
        handler()
    }

    fn new_session(
        &self,
        options: Option<NewSessionOptions>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = CancelledResult> + Send>> {
        if self.runner.assert_active().is_err() {
            return Box::pin(async { CancelledResult { cancelled: false } });
        }
        let handler = self
            .runner
            .callbacks
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .new_session
            .clone();
        handler(options)
    }

    fn fork(
        &self,
        entry_id: String,
        options: Option<ForkOptions>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = CancelledResult> + Send>> {
        if self.runner.assert_active().is_err() {
            return Box::pin(async { CancelledResult { cancelled: false } });
        }
        let handler = self
            .runner
            .callbacks
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .fork
            .clone();
        handler(entry_id, options)
    }

    fn navigate_tree(
        &self,
        target_id: String,
        options: Option<NavigateTreeOptions>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = CancelledResult> + Send>> {
        if self.runner.assert_active().is_err() {
            return Box::pin(async { CancelledResult { cancelled: false } });
        }
        let handler = self
            .runner
            .callbacks
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .navigate_tree
            .clone();
        handler(target_id, options)
    }

    fn switch_session(
        &self,
        session_path: String,
        options: Option<SwitchSessionOptions>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = CancelledResult> + Send>> {
        if self.runner.assert_active().is_err() {
            return Box::pin(async { CancelledResult { cancelled: false } });
        }
        let handler = self
            .runner
            .callbacks
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .switch_session
            .clone();
        handler(session_path, options)
    }

    fn reload(&self) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>> {
        if self.runner.assert_active().is_err() {
            return Box::pin(async {});
        }
        let handler = self
            .runner
            .callbacks
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .reload
            .clone();
        handler()
    }
}

impl ReplacedSessionContext for RunnerCommandContext {
    fn send_message(
        &self,
        message: CustomMessagePayload,
        options: Option<SendMessageOptions>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>> {
        let runtime = self.runner.runtime.clone();
        Box::pin(async move {
            let _ = runtime.send_message(message, options);
        })
    }

    fn send_user_message(
        &self,
        content: Value,
        options: Option<SendUserMessageOptions>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>> {
        let runtime = self.runner.runtime.clone();
        Box::pin(async move {
            let _ = runtime.send_user_message(content, options);
        })
    }
}

/// `ExtensionRunner` construction helper that keeps the runner in an `Arc`.
pub fn create_extension_runner(
    extensions: Vec<SharedExtension>,
    runtime: ExtensionRuntime,
    cwd: String,
    session_manager: Arc<dyn SessionManager>,
    model_registry: Arc<dyn ModelRegistry>,
) -> Arc<ExtensionRunner> {
    // Jev comparison observer (guarded): with default Off settings this is
    // one cheap settings read and no extension is added. See
    // `core::jev_bridge`. Repair-overlap note: the ONLY lane-B edit in this
    // file; integration may relocate the call to the session assembly site.
    let mut extensions = extensions;
    crate::core::jev_bridge::maybe_register_jev_observer(&mut extensions);
    Arc::new(ExtensionRunner::new(
        extensions,
        runtime,
        cwd,
        session_manager,
        model_registry,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::extensions::loader::{create_extension_runtime, register_extension_factory};
    use crate::core::extensions::types::{
        create_event_bus, CancelledResult, ExtensionActions, ExtensionContextActions, ExtensionEvent,
        ExtensionHandler, ExtensionShortcut, SessionShutdownPayload,
    };
    use serde_json::json;

    fn null_session_manager() -> Arc<dyn SessionManager> {
        Arc::new(NullSessionManager)
    }

    fn null_model_registry() -> Arc<dyn ModelRegistry> {
        Arc::new(NullModelRegistry)
    }

    fn context_actions() -> ExtensionContextActions {
        ExtensionContextActions {
            get_model: Arc::new(|| None),
            is_idle: Arc::new(|| true),
            get_signal: Arc::new(|| None),
            abort: Arc::new(|| {}),
            has_pending_messages: Arc::new(|| false),
            shutdown: Arc::new(|| {}),
            get_context_usage: Arc::new(|| None),
            compact: Arc::new(|_| {}),
            get_system_prompt: Arc::new(|| "prompt".to_string()),
        }
    }

    fn actions() -> ExtensionActions {
        ExtensionActions {
            send_message: Arc::new(|_: CustomMessagePayload, _| {}),
            send_user_message: Arc::new(|_: Value, _| {}),
            append_entry: Arc::new(|_: String, _: Option<Value>| {}),
            set_session_name: Arc::new(|_: String| Box::pin(async {})),
            get_session_name: Arc::new(|| None),
            set_label: Arc::new(|_: String, _: Option<String>| {}),
            get_active_tools: Arc::new(Vec::new),
            get_all_tools: Arc::new(Vec::new),
            set_active_tools: Arc::new(|_: Vec<String>| {}),
            refresh_tools: Arc::new(|| {}),
            get_commands: Arc::new(Vec::new),
            set_model: Arc::new(|_: pi_ai::types::Model| Box::pin(async { true })),
            get_thinking_level: Arc::new(|| pi_agent_core::types::ThinkingLevel::Off),
            set_thinking_level: Arc::new(|_: pi_agent_core::types::ThinkingLevel| {}),
        }
    }

    fn make_runner() -> Arc<ExtensionRunner> {
        let runtime = create_extension_runtime();
        let runner = create_extension_runner(
            Vec::new(),
            runtime.clone(),
            "/cwd".to_string(),
            null_session_manager(),
            null_model_registry(),
        );
        runner.bind_core(actions(), context_actions(), None);
        runner
    }

    /// Builds an in-memory extension the way `create_extension` does for an
    /// inline (`<inline>`) source, so tests can register handlers directly.
    fn extension_with_handler(event_type: &str, handler: ExtensionHandler) -> SharedExtension {
        let extension = Arc::new(Mutex::new(Extension {
            path: "<inline>".to_string(),
            resolved_path: "<inline>".to_string(),
            source_info: crate::core::source_info::create_synthetic_source_info(
                "<inline>",
                &crate::core::source_info::SyntheticSourceInfoOptions {
                    source: "inline".to_string(),
                    ..Default::default()
                },
            ),
            handlers: HashMap::new(),
            tools: HashMap::new(),
            message_renderers: HashMap::new(),
            commands: indexmap::IndexMap::new(),
            flags: HashMap::new(),
            shortcuts: HashMap::new(),
        }));
        extension
            .lock()
            .unwrap()
            .handlers
            .entry(event_type.to_string())
            .or_default()
            .push(handler);
        extension
    }

    #[test]
    fn builtin_keybindings_prefer_reserved_actions() {
        let mut resolved = Map::new();
        resolved.insert("app.interrupt".to_string(), json!("ctrl+c"));
        resolved.insert("app.custom".to_string(), json!(["ctrl+c", "ctrl+d"]));
        resolved.insert("app.unset".to_string(), Value::Null);
        let builtin = build_builtin_keybindings(&resolved);

        assert!(builtin.get("ctrl+c").unwrap().restrict_override);
        assert_eq!(builtin.get("ctrl+c").unwrap().keybinding, "app.interrupt");
        assert_eq!(builtin.get("ctrl+d").unwrap().keybinding, "app.custom");
        assert!(!builtin.get("ctrl+d").unwrap().restrict_override);
        assert!(!builtin.contains_key("app.unset"));
    }

    #[test]
    fn shortcuts_report_conflicts_and_reserved_skips() {
        let runtime = create_extension_runtime();
        let extension = extension_with_handler("session_start", Arc::new(|_, _| Box::pin(async { None })));
        {
            let mut guard = extension.lock().unwrap();
            guard.shortcuts.insert(
                "ctrl+x".to_string(),
                ExtensionShortcut {
                    shortcut: "ctrl+x".to_string(),
                    description: None,
                    handler: Arc::new(|_| Box::pin(async { Ok(()) })),
                    extension_path: "a.ts".to_string(),
                },
            );
            guard.shortcuts.insert(
                "ctrl+c".to_string(),
                ExtensionShortcut {
                    shortcut: "ctrl+c".to_string(),
                    description: None,
                    handler: Arc::new(|_| Box::pin(async { Ok(()) })),
                    extension_path: "a.ts".to_string(),
                },
            );
        }
        let runner = create_extension_runner(
            vec![extension],
            runtime,
            "/cwd".to_string(),
            null_session_manager(),
            null_model_registry(),
        );
        let mut resolved = Map::new();
        resolved.insert("app.interrupt".to_string(), json!("ctrl+c"));
        let shortcuts = runner.get_shortcuts(&resolved);
        assert!(shortcuts.contains_key("ctrl+x"));
        assert!(!shortcuts.contains_key("ctrl+c"));

        let diagnostics = runner.get_shortcut_diagnostics();
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(
            diagnostics[0].message,
            "Extension shortcut 'ctrl+c' from a.ts conflicts with built-in shortcut. Skipping."
        );
        assert_eq!(diagnostics[0].path.as_deref(), Some("a.ts"));
    }

    #[test]
    fn reserved_shortcut_conflict_is_case_insensitive() {
        let runtime = create_extension_runtime();
        let extension = extension_with_handler("session_start", Arc::new(|_, _| Box::pin(async { None })));
        {
            let mut guard = extension.lock().unwrap();
            guard.shortcuts.insert(
                "Ctrl+C".to_string(),
                ExtensionShortcut {
                    shortcut: "Ctrl+C".to_string(),
                    description: None,
                    handler: Arc::new(|_| Box::pin(async { Ok(()) })),
                    extension_path: "b.ts".to_string(),
                },
            );
        }
        let runner = create_extension_runner(
            vec![extension],
            runtime,
            "/cwd".to_string(),
            null_session_manager(),
            null_model_registry(),
        );
        let mut resolved = Map::new();
        resolved.insert("app.clear".to_string(), json!("ctrl+c"));
        assert!(runner.get_shortcuts(&resolved).is_empty());
    }

    #[test]
    fn extension_shortcut_conflicts_keep_the_last_registration() {
        let runtime = create_extension_runtime();
        let first = extension_with_handler("session_start", Arc::new(|_, _| Box::pin(async { None })));
        let second = extension_with_handler("session_start", Arc::new(|_, _| Box::pin(async { None })));
        for (extension, path) in [(&first, "a.ts"), (&second, "b.ts")] {
            extension.lock().unwrap().shortcuts.insert(
                "ctrl+x".to_string(),
                ExtensionShortcut {
                    shortcut: "ctrl+x".to_string(),
                    description: None,
                    handler: Arc::new(|_| Box::pin(async { Ok(()) })),
                    extension_path: path.to_string(),
                },
            );
        }
        let runner = create_extension_runner(
            vec![first, second],
            runtime,
            "/cwd".to_string(),
            null_session_manager(),
            null_model_registry(),
        );
        let shortcuts = runner.get_shortcuts(&Map::new());
        assert_eq!(shortcuts.get("ctrl+x").unwrap().extension_path, "b.ts");
        let diagnostics = runner.get_shortcut_diagnostics();
        assert_eq!(
            diagnostics[0].message,
            "Extension shortcut conflict: 'ctrl+x' registered by both a.ts and b.ts. Using b.ts."
        );
    }

    #[test]
    fn registered_commands_get_suffixed_invocation_names() {
        let runtime = create_extension_runtime();
        let extension = extension_with_handler("session_start", Arc::new(|_, _| Box::pin(async { None })));
        {
            let mut guard = extension.lock().unwrap();
            for name in ["dup", "dup", "solo"] {
                let key = format!("{name}-{}", guard.commands.len());
                let source_info = guard.source_info.clone();
                guard.commands.insert(
                    key,
                    crate::core::extensions::types::RegisteredCommand {
                        name: name.to_string(),
                        source_info,
                        description: None,
                        get_argument_completions: None,
                        handler: Arc::new(|_, _| Box::pin(async { Ok(()) })),
                    },
                );
            }
        }
        let runner = create_extension_runner(
            vec![extension],
            runtime,
            "/cwd".to_string(),
            null_session_manager(),
            null_model_registry(),
        );
        let commands = runner.get_registered_commands();
        let names: Vec<&str> = commands
            .iter()
            .map(|command| command.invocation_name.as_str())
            .collect();
        assert_eq!(names, ["dup:1", "dup:2", "solo"]);
        assert!(runner.get_command("dup:2").is_some());
        assert!(runner.get_command("dup:3").is_none());
    }

    #[tokio::test]
    async fn session_before_events_short_circuit_on_cancel_and_skip() {
        let runtime = create_extension_runtime();
        let handler: ExtensionHandler = Arc::new(|_, _| Box::pin(async { Some(json!({"cancel": true})) }));
        let extension = extension_with_handler("session_before_fork", handler);
        let runner = create_extension_runner(
            vec![extension],
            runtime,
            "/cwd".to_string(),
            null_session_manager(),
            null_model_registry(),
        );
        let result = runner
            .emit(ExtensionEvent::SessionShutdown(SessionShutdownPayload {
                reason: "quit".to_string(),
                target_session_file: None,
            }))
            .await;
        assert_eq!(result, None);

        let skip: ExtensionHandler = Arc::new(|_, _| Box::pin(async { Some(json!({"skip": true})) }));
        let runtime = create_extension_runtime();
        let extension = extension_with_handler("session_before_refine", skip);
        let runner = create_extension_runner(
            vec![extension],
            runtime,
            "/cwd".to_string(),
            null_session_manager(),
            null_model_registry(),
        );
        let event = ExtensionEvent::SessionBeforeRefine(crate::core::extensions::types::SessionBeforeRefinePayload {
            preparation: crate::core::extensions::types::RefinePreparation {
                trigger: "auto".to_string(),
                instructions: None,
                scope: "local".to_string(),
                planning_state: crate::core::refinement::refinement::HarnessState {
                    schema: 1.0,
                    entries: indexmap::IndexMap::new(),
                    refinements: Vec::new(),
                },
                history: Vec::new(),
                conversation_text: String::new(),
            },
        });
        let result = runner.emit(event).await;
        assert_eq!(result, Some(json!({"skip": true})));
    }

    #[tokio::test]
    async fn handler_errors_are_reported_and_do_not_stop_later_handlers() {
        let runtime = create_extension_runtime();
        let failing: ExtensionHandler = Arc::new(|_, _| Box::pin(async { panic!("boom") }));
        let ok: ExtensionHandler = Arc::new(|_, _| Box::pin(async { Some(json!({"cancel": true})) }));
        let first = extension_with_handler("session_before_switch", failing);
        let second = extension_with_handler("session_before_switch", ok);
        let runner = create_extension_runner(
            vec![first, second],
            runtime,
            "/cwd".to_string(),
            null_session_manager(),
            null_model_registry(),
        );
        let errors: Arc<Mutex<Vec<ExtensionError>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = errors.clone();
        runner.on_error(Arc::new(move |error| sink.lock().unwrap().push(error)));

        let result = runner
            .emit(ExtensionEvent::SessionBeforeSwitch(
                crate::core::extensions::types::SessionBeforeSwitchPayload {
                    reason: "new".to_string(),
                    target_session_file: None,
                },
            ))
            .await;
        assert_eq!(result, Some(json!({"cancel": true})));
        let errors = errors.lock().unwrap();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].event, "session_before_switch");
        assert_eq!(errors[0].error, "boom");
    }

    #[tokio::test]
    async fn context_handlers_chain_messages() {
        let runtime = create_extension_runtime();
        let handler: ExtensionHandler = Arc::new(|event, _| {
            Box::pin(async move {
                let ExtensionEvent::Context(payload) = event else {
                    return None;
                };
                let mut messages = payload.messages;
                messages.push(json!({"role": "user", "content": "added"}));
                Some(json!({"messages": messages}))
            })
        });
        let extension = extension_with_handler("context", handler);
        let runner = create_extension_runner(
            vec![extension],
            runtime,
            "/cwd".to_string(),
            null_session_manager(),
            null_model_registry(),
        );
        let result = runner.emit_context(vec![json!({"role": "user", "content": "first"})]).await;
        assert_eq!(result.len(), 2);
        assert_eq!(result[1]["content"], json!("added"));
    }

    #[tokio::test]
    async fn before_agent_start_chains_system_prompts_and_collects_messages() {
        let runtime = create_extension_runtime();
        let handler: ExtensionHandler = Arc::new(|event, _| {
            Box::pin(async move {
                let ExtensionEvent::BeforeAgentStart(payload) = event else {
                    return None;
                };
                Some(json!({
                    "message": {
                        "customType": "note",
                        "content": payload.system_prompt,
                        "display": false
                    },
                    "systemPrompt": format!("{} +next", payload.system_prompt)
                }))
            })
        });
        let extension = extension_with_handler("before_agent_start", handler);
        let runner = create_extension_runner(
            vec![extension],
            runtime,
            "/cwd".to_string(),
            null_session_manager(),
            null_model_registry(),
        );
        let result = runner
            .emit_before_agent_start(
                "hello".to_string(),
                None,
                "base".to_string(),
                crate::core::system_prompt::BuildSystemPromptOptions {
                    cwd: "/cwd".to_string(),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(result.system_prompt.as_deref(), Some("base +next"));
        assert_eq!(result.messages.unwrap()[0].content, json!("base"));
    }

    #[tokio::test]
    async fn input_events_transform_then_continue() {
        let runtime = create_extension_runtime();
        let handler: ExtensionHandler = Arc::new(|event, _| {
            Box::pin(async move {
                let ExtensionEvent::Input(payload) = event else {
                    return None;
                };
                Some(json!({"action": "transform", "text": format!("{}!", payload.text)}))
            })
        });
        let extension = extension_with_handler("input", handler);
        let runner = create_extension_runner(
            vec![extension],
            runtime,
            "/cwd".to_string(),
            null_session_manager(),
            null_model_registry(),
        );
        let result = runner
            .emit_input("hi".to_string(), None, "interactive".to_string())
            .await;
        assert_eq!(
            result,
            crate::core::extensions::types::InputEventResult::Transform {
                text: "hi!".to_string(),
                images: None
            }
        );
    }

    #[tokio::test]
    async fn message_end_rejects_a_different_role_and_reports_the_error() {
        let runtime = create_extension_runtime();
        let handler: ExtensionHandler = Arc::new(|_, _| {
            Box::pin(async { Some(json!({"message": {"role": "assistant", "content": []}})) })
        });
        let extension = extension_with_handler("message_end", handler);
        let runner = create_extension_runner(
            vec![extension],
            runtime,
            "/cwd".to_string(),
            null_session_manager(),
            null_model_registry(),
        );
        let errors: Arc<Mutex<Vec<ExtensionError>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = errors.clone();
        runner.on_error(Arc::new(move |error| sink.lock().unwrap().push(error)));

        let result = runner
            .emit_message_end(json!({"message": {"role": "user", "content": []}}))
            .await;
        assert_eq!(result, None);
        assert_eq!(
            errors.lock().unwrap()[0].error,
            "message_end handlers must return a message with the same role"
        );
    }

    #[test]
    fn invalidation_freezes_the_runner_and_the_runtime() {
        let runner = make_runner();
        assert!(runner.assert_active().is_ok());
        runner.invalidate(Some("stale now".to_string()));
        assert_eq!(runner.assert_active().unwrap_err(), "stale now");
        assert_eq!(runner.runtime.assert_active().unwrap_err(), "stale now");
        // A second invalidate keeps the first message.
        runner.invalidate(Some("other".to_string()));
        assert_eq!(runner.assert_active().unwrap_err(), "stale now");
    }

    #[test]
    fn flag_values_are_shared_with_the_runtime() {
        let runner = make_runner();
        runner.set_flag_value("flag", json!("cli"));
        assert_eq!(runner.get_flag_values().get("flag"), Some(&json!("cli")));
    }

    #[tokio::test]
    async fn session_shutdown_helper_reports_whether_handlers_existed() {
        let runtime = create_extension_runtime();
        let runner = create_extension_runner(
            Vec::new(),
            runtime,
            "/cwd".to_string(),
            null_session_manager(),
            null_model_registry(),
        );
        let event = ExtensionEvent::SessionShutdown(SessionShutdownPayload {
            reason: "quit".to_string(),
            target_session_file: None,
        });
        assert!(!emit_session_shutdown_event(&runner, event.clone()).await);

        let runtime = create_extension_runtime();
        let extension = extension_with_handler("session_shutdown", Arc::new(|_, _| Box::pin(async { None })));
        let runner = create_extension_runner(
            vec![extension],
            runtime,
            "/cwd".to_string(),
            null_session_manager(),
            null_model_registry(),
        );
        assert!(emit_session_shutdown_event(&runner, event).await);
    }

    #[tokio::test]
    async fn registered_extension_factories_load_through_the_loader() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ext.ts").to_string_lossy().to_string();
        let factory: crate::core::extensions::types::ExtensionFactory =
            Arc::new(|api| {
                Box::pin(async move {
                    api.on("session_start", Arc::new(|_, _| Box::pin(async { None })));
                    Ok(())
                })
            });
        register_extension_factory(&path, factory);
        let result =
            crate::core::extensions::loader::load_extensions(&[path.clone()], "/cwd", None).await;
        assert!(result.errors.is_empty());
        assert_eq!(result.extensions.len(), 1);
        let runner = create_extension_runner(
            result.extensions,
            result.runtime,
            "/cwd".to_string(),
            null_session_manager(),
            null_model_registry(),
        );
        assert!(runner.has_handlers("session_start"));
    }

    #[test]
    fn command_context_reports_cancelled_false_by_default() {
        let runner = make_runner();
        let ctx = runner.create_command_context();
        let result = futures::executor::block_on(ctx.new_session(None));
        assert_eq!(result, CancelledResult { cancelled: false });
    }

    #[test]
    fn runner_context_reads_live_callback_values() {
        let runner = make_runner();
        let ctx = runner.create_context();
        assert_eq!(ctx.get_system_prompt(), "prompt");
        assert!(ctx.is_idle());
        assert_eq!(ctx.cwd(), "/cwd");
        assert!(!ctx.has_ui());
        let _ = create_event_bus();
    }
}
