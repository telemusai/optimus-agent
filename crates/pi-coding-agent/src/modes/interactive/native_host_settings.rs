//! Native mounting and callbacks for TypeScript `showSettingsSelector`.
use super::*;
use crate::core::session_action_store::IdleEvictionMinutes;
use crate::core::settings_manager::WarningSettings;
use crate::modes::interactive::components::settings_selector::{
    SettingsCallbacks, SettingsConfig, SettingsSelectorComponent,
};
use crate::modes::interactive::theme::theme::{get_available_themes, set_theme};

#[derive(Clone)]
pub(super) enum Change {
    AutoCompact(bool),
    IdleEvictionMinutes(IdleEvictionMinutes),
    ShowImages(bool),
    AutoResizeImages(bool),
    BlockImages(bool),
    EnableSkillCommands(bool),
    EnableBuiltinSkills(bool),
    SteeringMode(String),
    FollowUpMode(String),
    Transport(String),
    ThinkingLevel(String),
    Theme(String),
    HideThinkingBlock(bool),
    MermaidRenderingMode(String),
    TreeFilterMode(String),
    ShowHardwareCursor(bool),
    EditorPaddingX(f64),
    AutocompleteMaxVisible(f64),
    QuietStartup(bool),
    ClearOnShrink(bool),
    ShowTerminalProgress(bool),
    Fullscreen(bool),
    Warnings(WarningSettings),
    ThemePreview(String),
    Close,
}

impl Change {
    /// Whether the change also has to reach the daemon. These are the five
    /// callbacks TypeScript sends through `this.agentConnection`
    /// (interactive-mode.ts:7804, :7832, :7836, :7842, :7847, :7852).
    pub(super) fn touches_the_daemon(&self) -> bool {
        matches!(
            self,
            Change::AutoCompact(_)
                | Change::SteeringMode(_)
                | Change::FollowUpMode(_)
                | Change::Transport(_)
                | Change::ThinkingLevel(_)
                | Change::EnableBuiltinSkills(_)
        )
    }
}

pub(super) fn create(
    mode: &InteractiveMode,
    state: &wire::AgentConnectionState,
    send: &mpsc::Sender<HostEvent>,
) -> Result<SettingsSelectorComponent, String> {
    let manager = mode.settings_manager();
    let settings = manager.lock().map_err(|e| e.to_string())?;
    let config = SettingsConfig {
        idle_eviction_minutes: match settings.get_idle_eviction_minutes() {
            crate::core::settings_manager::IdleEvictionMinutes::Minutes(n) => {
                IdleEvictionMinutes::Minutes(n)
            }
            crate::core::settings_manager::IdleEvictionMinutes::Off => IdleEvictionMinutes::Off,
        },
        show_images: settings.get_show_images(),
        auto_resize_images: settings.get_image_auto_resize(),
        block_images: settings.get_block_images(),
        enable_skill_commands: settings.get_enable_skill_commands(),
        enable_builtin_skills: settings.get_enable_builtin_skills(),
        transport: settings.get_transport(),
        mermaid_rendering_mode: settings.get_mermaid_rendering_mode(),
        tree_filter_mode: settings.get_tree_filter_mode(),
        show_hardware_cursor: settings.get_show_hardware_cursor(),
        editor_padding_x: settings.get_editor_padding_x(),
        autocomplete_max_visible: settings.get_autocomplete_max_visible(),
        quiet_startup: settings.get_quiet_startup(),
        clear_on_shrink: settings.get_clear_on_shrink(),
        show_terminal_progress: settings.get_show_terminal_progress(),
        warnings: settings.get_warnings(),
        auto_compact: state.auto_compaction_enabled,
        steering_mode: state.steering_mode.clone(),
        follow_up_mode: state.follow_up_mode.clone(),
        thinking_level: state.thinking_level.as_str().into(),
        available_thinking_levels: available_thinking_levels(state)
            .iter()
            .map(|level| level.as_str().into())
            .collect(),
        current_theme: settings.get_theme().unwrap_or_else(|| "prime".into()),
        available_themes: get_available_themes(),
        hide_thinking_block: mode.hide_thinking_block,
        fullscreen: mode.fullscreen_enabled,
    };
    macro_rules! callback {
        ($variant:ident) => {{
            let send = send.clone();
            Box::new(move |value| {
                let _ = send.send(HostEvent::Setting(Change::$variant(value)));
            })
        }};
    }
    let close = send.clone();
    Ok(SettingsSelectorComponent::new(
        config,
        SettingsCallbacks {
            on_auto_compact_change: callback!(AutoCompact),
            on_idle_eviction_minutes_change: callback!(IdleEvictionMinutes),
            on_show_images_change: callback!(ShowImages),
            on_auto_resize_images_change: callback!(AutoResizeImages),
            on_block_images_change: callback!(BlockImages),
            on_enable_skill_commands_change: callback!(EnableSkillCommands),
            on_enable_builtin_skills_change: callback!(EnableBuiltinSkills),
            on_steering_mode_change: callback!(SteeringMode),
            on_follow_up_mode_change: callback!(FollowUpMode),
            on_transport_change: callback!(Transport),
            on_thinking_level_change: callback!(ThinkingLevel),
            on_theme_change: callback!(Theme),
            on_hide_thinking_block_change: callback!(HideThinkingBlock),
            on_mermaid_rendering_mode_change: callback!(MermaidRenderingMode),
            on_tree_filter_mode_change: callback!(TreeFilterMode),
            on_show_hardware_cursor_change: callback!(ShowHardwareCursor),
            on_editor_padding_x_change: callback!(EditorPaddingX),
            on_autocomplete_max_visible_change: callback!(AutocompleteMaxVisible),
            on_quiet_startup_change: callback!(QuietStartup),
            on_clear_on_shrink_change: callback!(ClearOnShrink),
            on_show_terminal_progress_change: callback!(ShowTerminalProgress),
            on_fullscreen_change: callback!(Fullscreen),
            on_warnings_change: callback!(Warnings),
            on_theme_preview: Some(callback!(ThemePreview)),
            on_cancel: Box::new(move || {
                let _ = close.send(HostEvent::Setting(Change::Close));
            }),
        },
    ))
}

pub(super) fn fullscreen(
    enabled: bool,
    mode: &Rc<RefCell<InteractiveMode>>,
    editor: &Rc<RefCell<CustomEditor>>,
    ui: &Rc<RefCell<TUI>>,
    transcript: &Rc<RefCell<Transcript>>,
) {
    mode.borrow_mut().fullscreen_enabled = enabled;
    if enabled {
        let dock = Rc::new(RefCell::new(pi_tui::tui::Container::new()));
        let surfaces = transcript.borrow().extension_surfaces.clone();
        if let Some(surfaces) = &surfaces {
            dock.borrow_mut().add_child(Rc::new(RefCell::new(
                native_extensions::Widgets(surfaces.clone(), false),
            )));
        }
        dock.borrow_mut().add_child(editor.clone());
        if let Some(surfaces) = &surfaces {
            dock.borrow_mut().add_child(Rc::new(RefCell::new(
                native_extensions::Widgets(surfaces.clone(), true),
            )));
        }
        if let Some(bar) = &transcript.borrow().subagents { dock.borrow_mut().add_child(bar.clone()); }
        let tray = Tray(mode.clone(), editor.clone());
        if let Some(surfaces) = surfaces {
            dock.borrow_mut().add_child(Rc::new(RefCell::new(
                native_extensions::Statuses(surfaces, tray),
            )));
        } else {
            dock.borrow_mut().add_child(Rc::new(RefCell::new(tray)));
        }
        let mouse = mode
            .borrow()
            .settings_manager()
            .lock()
            .map(|s| s.get_fullscreen_mouse())
            .unwrap_or(true);
        ui.borrow_mut()
            .enter_fullscreen(pi_tui::tui::FullscreenOptions {
                scroll: vec![transcript.clone()],
                dock,
                mouse,
                viewport_controls: true,
            });
    } else {
        ui.borrow_mut()
            .exit_fullscreen(pi_tui::tui::ExitFullscreenOptions {
                flush: true,
                leave_alt_screen: true,
            });
    }
    ui.borrow_mut().request_render();
}

/// The daemon calls a settings change needs.
///
/// Narrowing the surface to these six methods keeps the local/remote split and
/// its failure path testable without standing in for the whole connection
/// contract. `AgentConnection` is the only implementor in production.
pub(super) trait SettingsDaemon {
    fn set_auto_compaction_enabled(
        &self,
        enabled: bool,
    ) -> pi_ai::types::BoxFuture<Result<(), String>>;
    fn set_steering_mode(&self, mode: &str) -> pi_ai::types::BoxFuture<Result<(), String>>;
    fn set_follow_up_mode(&self, mode: &str) -> pi_ai::types::BoxFuture<Result<(), String>>;
    fn set_transport(
        &self,
        transport: pi_ai::types::Transport,
    ) -> pi_ai::types::BoxFuture<Result<(), String>>;
    fn set_thinking_level(
        &self,
        level: pi_agent_core::types::ThinkingLevel,
    ) -> pi_ai::types::BoxFuture<Result<(), String>>;
    fn reload(&self) -> pi_ai::types::BoxFuture<Result<(), String>>;
}

impl<T: wire::AgentConnection + ?Sized> SettingsDaemon for T {
    fn set_auto_compaction_enabled(
        &self,
        enabled: bool,
    ) -> pi_ai::types::BoxFuture<Result<(), String>> {
        wire::AgentConnection::set_auto_compaction_enabled(self, enabled)
    }
    fn set_steering_mode(&self, mode: &str) -> pi_ai::types::BoxFuture<Result<(), String>> {
        wire::AgentConnection::set_steering_mode(self, mode)
    }
    fn set_follow_up_mode(&self, mode: &str) -> pi_ai::types::BoxFuture<Result<(), String>> {
        wire::AgentConnection::set_follow_up_mode(self, mode)
    }
    fn set_transport(
        &self,
        transport: pi_ai::types::Transport,
    ) -> pi_ai::types::BoxFuture<Result<(), String>> {
        wire::AgentConnection::set_transport(self, transport)
    }
    fn set_thinking_level(
        &self,
        level: pi_agent_core::types::ThinkingLevel,
    ) -> pi_ai::types::BoxFuture<Result<(), String>> {
        wire::AgentConnection::set_thinking_level(self, level)
    }
    fn reload(&self) -> pi_ai::types::BoxFuture<Result<(), String>> {
        wire::AgentConnection::reload(self)
    }
}

/// The daemon-owned half of a settings change.
///
/// TypeScript never awaits these calls: every settings callback patches the
/// local state first and then fires the remote call as a floating promise whose
/// rejection is only reported through `showError`
/// (interactive-mode.ts:7804, :7836, :7842, :7847, :7852). This function is
/// therefore run off the owner loop.
pub(super) async fn apply_remote<T: SettingsDaemon + ?Sized>(
    change: &Change,
    connection: &T,
) -> Result<(), String> {
    match change {
        Change::AutoCompact(value) => connection.set_auto_compaction_enabled(*value).await?,
        Change::SteeringMode(value) => connection.set_steering_mode(value).await?,
        Change::FollowUpMode(value) => connection.set_follow_up_mode(value).await?,
        Change::Transport(value) => connection.set_transport(value.clone()).await?,
        Change::ThinkingLevel(value) => {
            let level = serde_json::from_value(serde_json::Value::String(value.clone()))
                .map_err(|e| e.to_string())?;
            connection.set_thinking_level(level).await?;
        }
        // `void this.handleReloadCommand()` (interactive-mode.ts:7832).
        Change::EnableBuiltinSkills(_) => connection.reload().await?,
        _ => {}
    }
    Ok(())
}

/// Runs the daemon half of a change off the owner loop, like the floating
/// promise TypeScript creates. A rejection is reported through `showError`
/// (interactive-mode.ts:7804-7806, :7836-7838, :7842-7844, :7847-7849) without
/// touching the local state that was already applied.
pub(super) fn spawn_remote<T: SettingsDaemon + Send + Sync + 'static + ?Sized>(
    change: Change,
    connection: Arc<T>,
    send: mpsc::Sender<HostEvent>,
) {
    if !change.touches_the_daemon() {
        return;
    }
    tokio::spawn(async move {
        // `as_ref()` reaches the `SettingsDaemon` blanket impl on the connection
        // itself; `Arc<T>` deliberately does not implement it.
        match apply_remote(&change, connection.as_ref()).await {
            Ok(()) => {
                if matches!(change, Change::ThinkingLevel(_)) {
                    // `onThinkingLevelChange` patches the state in the `.then`
                    // of the remote call (interactive-mode.ts:7852-7861), so a
                    // rejected change must not publish a level the daemon never
                    // accepted. The continuation travels back as an event; the
                    // loop must not await another RPC to finish it.
                    let _ = send.send(HostEvent::SettingAccepted(change));
                }
            }
            // `showError` from the `.catch` (interactive-mode.ts:7805, :7837,
            // :7843, :7848, :7860): reported, never rolled back.
            Err(error) => {
                let _ = send.send(HostEvent::Error(error));
            }
        }
    });
}

/// Applies one settings change exactly as the owner loop must:
/// fire the daemon work first, then apply the local state synchronously.
///
/// TypeScript order is local-first in the callback body and fire-and-forget for
/// the remote call (`patchConnectionState` then
/// `void agentConnection.set*.catch(showError)`). Doing the remote work first
/// without awaiting it keeps that same observable order - the local state is
/// already applied when the loop continues, and a rejection can only be
/// reported afterwards (interactive-mode.ts:7803-7806).
pub(super) fn apply_change<T: SettingsDaemon + Send + Sync + 'static + ?Sized>(
    change: Change,
    mode: &Rc<RefCell<InteractiveMode>>,
    editor: &Rc<RefCell<CustomEditor>>,
    ui: &Rc<RefCell<TUI>>,
    transcript: &Rc<RefCell<Transcript>>,
    connection: &Arc<T>,
    send: &mpsc::Sender<HostEvent>,
) {
    spawn_remote(change.clone(), connection.clone(), send.clone());
    if let Err(error) = apply_local(change, mode, editor, ui, transcript) {
        mode.borrow_mut().show_error(&error);
    }
}

/// The `.then` continuation of a daemon-owned change.
///
/// `onThinkingLevelChange` is the only settings callback that patches the
/// connection state AFTER the remote call resolves
/// (interactive-mode.ts:7852-7857), so the new level must not be published when
/// the daemon rejects the change.
pub(super) fn apply_remote_applied(
    change: &Change,
    mode: &Rc<RefCell<InteractiveMode>>,
    ui: &Rc<RefCell<TUI>>,
) {
    if let Change::ThinkingLevel(value) = change {
        mode.borrow_mut().patch_connection_state(|state| {
            state.thinking_level =
                serde_json::from_value(serde_json::Value::String(value.clone())).unwrap_or_default()
        });
    }
    ui.borrow_mut().request_render();
}

/// The local half of a settings change: the optimistic state patch, the
/// persistence call, and the live effect.
///
/// Order matches TypeScript: the callbacks that own connection state patch it
/// BEFORE the remote call (`patchConnectionState` at interactive-mode.ts:7803,
/// :7835, :7840), every callback persists through its `settingsManager.set*`
/// call, and a rejected remote call never rolls the local state back.
pub(super) fn apply_local(
    change: Change,
    mode: &Rc<RefCell<InteractiveMode>>,
    editor: &Rc<RefCell<CustomEditor>>,
    ui: &Rc<RefCell<TUI>>,
    transcript: &Rc<RefCell<Transcript>>,
) -> Result<(), String> {
    match &change {
        Change::AutoCompact(value) => mode
            .borrow_mut()
            .patch_connection_state(|state| state.auto_compaction_enabled = *value),
        Change::SteeringMode(value) => mode
            .borrow_mut()
            .patch_connection_state(|state| state.steering_mode = value.clone()),
        Change::FollowUpMode(value) => mode
            .borrow_mut()
            .patch_connection_state(|state| state.follow_up_mode = value.clone()),
        _ => {}
    }
    {
        let manager = mode.borrow().settings_manager().clone();
        let mut settings = manager.lock().map_err(|e| e.to_string())?;
        match &change {
            Change::AutoCompact(value) => settings.set_compaction_enabled(*value),
            Change::IdleEvictionMinutes(value) => settings.set_idle_eviction_minutes(match value {
                IdleEvictionMinutes::Minutes(n) => {
                    crate::core::settings_manager::IdleEvictionMinutes::Minutes(*n)
                }
                IdleEvictionMinutes::Off => crate::core::settings_manager::IdleEvictionMinutes::Off,
            }),
            Change::ShowImages(value) => settings.set_show_images(*value),
            Change::AutoResizeImages(value) => settings.set_image_auto_resize(*value),
            Change::BlockImages(value) => settings.set_block_images(*value),
            Change::EnableSkillCommands(value) => settings.set_enable_skill_commands(*value),
            Change::EnableBuiltinSkills(value) => settings.set_enable_builtin_skills(*value),
            Change::SteeringMode(value) => settings.set_steering_mode(value),
            Change::FollowUpMode(value) => settings.set_follow_up_mode(value),
            Change::Transport(value) => settings.set_transport(value.clone()),
            Change::Theme(value) => settings.set_theme(value),
            Change::HideThinkingBlock(value) => settings.set_hide_thinking_block(*value),
            Change::MermaidRenderingMode(value) => settings.set_mermaid_rendering_mode(value),
            Change::TreeFilterMode(value) => settings.set_tree_filter_mode(value),
            Change::ShowHardwareCursor(value) => settings.set_show_hardware_cursor(*value),
            Change::EditorPaddingX(value) => settings.set_editor_padding_x(*value),
            Change::AutocompleteMaxVisible(value) => settings.set_autocomplete_max_visible(*value),
            Change::QuietStartup(value) => settings.set_quiet_startup(*value),
            Change::ClearOnShrink(value) => settings.set_clear_on_shrink(*value),
            Change::ShowTerminalProgress(value) => settings.set_show_terminal_progress(*value),
            Change::Fullscreen(value) => settings.set_fullscreen(*value),
            Change::Warnings(value) => settings.set_warnings(value.clone()),
            _ => {}
        }
    }
    match change {
        Change::ShowImages(enabled) => {
            for tool in transcript.borrow().all_tools() {
                tool.borrow_mut().set_show_images(enabled);
            }
        }
        Change::HideThinkingBlock(hidden) => {
            mode.borrow_mut().hide_thinking_block = hidden;
            for assistant in transcript.borrow().all_assistants() {
                assistant.borrow_mut().set_hide_thinking_block(hidden);
            }
        }
        Change::EditorPaddingX(padding) => editor.borrow_mut().editor_mut().set_padding_x(padding),
        Change::AutocompleteMaxVisible(maximum) => editor
            .borrow_mut()
            .editor_mut()
            .set_autocomplete_max_visible(maximum),
        Change::ShowHardwareCursor(enabled) => ui.borrow_mut().set_show_hardware_cursor(enabled),
        Change::ClearOnShrink(enabled) => ui.borrow_mut().set_clear_on_shrink(enabled),
        Change::Fullscreen(enabled) => fullscreen(enabled, mode, editor, ui, transcript),
        Change::Theme(name) | Change::ThemePreview(name) => {
            let result = set_theme(&name, true);
            if !result.success {
                mode.borrow_mut().show_error(&format!(
                    "Failed to load theme \"{name}\": {}\nFell back to dark theme.",
                    result.error.unwrap_or_default()
                ));
            }
            ui.borrow_mut().invalidate();
        }
        Change::EnableSkillCommands(_) => {
            native_autocomplete::configure(
                &mut editor.borrow_mut(),
                mode.clone(),
                &mode.borrow().get_current_cwd(),
            );
        }
        _ => {}
    }
    ui.borrow_mut().request_render();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A daemon that can reject, or park until released. This is the "remote
    /// call fails / never returns" case the TypeScript handles with a floating
    /// promise (interactive-mode.ts:7804-7806).
    #[derive(Default)]
    struct ParkedDaemon {
        calls: std::sync::Mutex<Vec<String>>,
        /// `Some(true)` rejects, `Some(false)` accepts, `None` never resolves.
        outcome: std::sync::Mutex<Option<bool>>,
        /// The release signal must be ASYNC: a blocking `recv()` inside the
        /// spawned task would park the single `#[tokio::test]` runtime thread,
        /// so nothing else on that runtime could ever run.
        release: std::sync::Mutex<Option<tokio::sync::oneshot::Receiver<()>>>,
    }

    impl ParkedDaemon {
        fn rejecting() -> Self {
            let daemon = Self::default();
            *daemon.outcome.lock().unwrap() = Some(true);
            daemon
        }

        fn parked(receiver: tokio::sync::oneshot::Receiver<()>) -> Self {
            let daemon = Self::default();
            *daemon.release.lock().unwrap() = Some(receiver);
            daemon
        }

        fn calls(&self) -> Vec<String> {
            self.calls.lock().unwrap().clone()
        }

        fn outcome(&self) -> pi_ai::types::BoxFuture<Result<(), String>> {
            let outcome = *self.outcome.lock().unwrap();
            let receiver = self.release.lock().unwrap().take();
            Box::pin(async move {
                if let Some(receiver) = receiver {
                    // A dropped sender also releases the call, so the test can
                    // never leave the task parked forever.
                    let _ = receiver.await;
                }
                match outcome {
                    Some(true) => Err("daemon refused the change".to_string()),
                    _ => Ok(()),
                }
            })
        }
    }

    impl SettingsDaemon for ParkedDaemon {
        fn set_auto_compaction_enabled(
            &self,
            _enabled: bool,
        ) -> pi_ai::types::BoxFuture<Result<(), String>> {
            self.calls
                .lock()
                .unwrap()
                .push("set_auto_compaction_enabled".into());
            self.outcome()
        }
        fn set_steering_mode(&self, _mode: &str) -> pi_ai::types::BoxFuture<Result<(), String>> {
            self.calls.lock().unwrap().push("set_steering_mode".into());
            self.outcome()
        }
        fn set_follow_up_mode(&self, _mode: &str) -> pi_ai::types::BoxFuture<Result<(), String>> {
            self.calls.lock().unwrap().push("set_follow_up_mode".into());
            self.outcome()
        }
        fn set_transport(
            &self,
            _transport: pi_ai::types::Transport,
        ) -> pi_ai::types::BoxFuture<Result<(), String>> {
            self.calls.lock().unwrap().push("set_transport".into());
            self.outcome()
        }
        fn set_thinking_level(
            &self,
            _level: pi_agent_core::types::ThinkingLevel,
        ) -> pi_ai::types::BoxFuture<Result<(), String>> {
            self.calls.lock().unwrap().push("set_thinking_level".into());
            self.outcome()
        }
        fn reload(&self) -> pi_ai::types::BoxFuture<Result<(), String>> {
            self.calls.lock().unwrap().push("reload".into());
            self.outcome()
        }
    }

    /// Drives the spawned remote task to completion.
    ///
    /// `#[tokio::test]` defaults to a current-thread runtime, so a spawned task
    /// only advances when the test yields. A real await here also proves the
    /// task really runs: an `apply_change` that never spawned it would time out.
    async fn settle() {
        for _ in 0..50 {
            tokio::task::yield_now().await;
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        }
    }

    /// Yields until the sender produces a message, or the deadline passes.
    async fn next_event(
        receive: &mut mpsc::Receiver<HostEvent>,
        timeout: std::time::Duration,
    ) -> Option<HostEvent> {
        let deadline = std::time::Instant::now() + timeout;
        loop {
            match receive.try_recv() {
                Ok(event) => return Some(event),
                Err(std::sync::mpsc::TryRecvError::Disconnected) => return None,
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
            }
            if std::time::Instant::now() >= deadline {
                return None;
            }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
    }

    fn settings_mode(session_id: &str) -> Rc<RefCell<InteractiveMode>> {
        let mode = Rc::new(RefCell::new(super::super::tests::stash_mode(session_id)));
        // `patchConnectionState` writes the live connection state, so the mode
        // needs one exactly as the attached host does.
        mode.borrow_mut()
            .apply_connection_state_snapshot(local::AgentConnectionState {
                session_id: session_id.into(),
                ..Default::default()
            });
        mode
    }

    fn host(
        mode: &Rc<RefCell<InteractiveMode>>,
    ) -> (
        Rc<RefCell<CustomEditor>>,
        Rc<RefCell<TUI>>,
        Rc<RefCell<Transcript>>,
    ) {
        let ui = Rc::new(RefCell::new(TUI::new(
            Box::new(pi_tui::terminal::ProcessTerminal::new()),
            Some(false),
        )));
        let editor = Rc::new(RefCell::new(CustomEditor::new(
            ui.clone(),
            editor_theme(),
            CustomEditorOptions::default(),
        )));
        let transcript = Rc::new(RefCell::new(Transcript::new(mode.clone())));
        (editor, ui, transcript)
    }

    /// DEFECT A. A rejected daemon call must not undo the local setting, and the
    /// apply path must not wait on the daemon.
    ///
    /// TypeScript patches the connection state first and only reports the
    /// rejection (`patchConnectionState` then
    /// `void setSteeringMode(...).catch(showError)`,
    /// interactive-mode.ts:7835-7838). The audited port awaited the RPC before it
    /// touched local state (`native_host.rs:1670-1682` at the audit's HEAD), so a
    /// failure skipped the persist entirely.
    #[tokio::test]
    async fn a_rejected_daemon_change_still_applies_the_local_setting() {
        let mode = settings_mode("settings-failure");
        let (editor, ui, transcript) = host(&mode);
        let daemon = Arc::new(ParkedDaemon::rejecting());
        let (send, mut receive) = mpsc::channel();

        let before = mode
            .borrow()
            .with_settings(|settings| settings.get_steering_mode());
        assert_ne!(
            before, "all-at-once",
            "test needs a different starting value"
        );

        apply_change(
            Change::SteeringMode("all-at-once".into()),
            &mode,
            &editor,
            &ui,
            &transcript,
            &daemon,
            &send,
        );

        // Local state is already applied when the loop continues - the spawned
        // remote task has not even been polled yet.
        assert_eq!(
            mode.borrow()
                .with_settings(|settings| settings.get_steering_mode()),
            "all-at-once",
            "the local setting must be persisted before the remote call settles"
        );
        assert_eq!(
            mode.borrow()
                .connection_state
                .as_ref()
                .map(|state| state.steering_mode.clone()),
            Some("all-at-once".into()),
            "patchConnectionState must run optimistically"
        );

        // The rejection is reported, not rolled back.
        let reported = next_event(&mut receive, std::time::Duration::from_secs(5))
            .await
            .expect("the failed remote call must report through the host");
        match reported {
            HostEvent::Error(error) => assert!(
                error.contains("daemon refused"),
                "the reported error must carry the daemon's message: {error}"
            ),
            _ => panic!("expected the daemon error event, got a different HostEvent variant"),
        }
        assert_eq!(daemon.calls(), vec!["set_steering_mode"]);
        assert_eq!(
            mode.borrow()
                .with_settings(|settings| settings.get_steering_mode()),
            "all-at-once",
            "a rejected remote call must not undo the persisted setting"
        );
    }

    /// DEFECT A, blocking half. A daemon call that has not resolved must not
    /// stop the apply path. The daemon call is released inside the test, so the
    /// case can never park the shared suite.
    #[tokio::test]
    async fn applying_a_settings_change_never_waits_for_the_daemon() {
        let mode = settings_mode("settings-parked");
        let (editor, ui, transcript) = host(&mode);
        let (release, parked) = tokio::sync::oneshot::channel::<()>();
        let daemon = Arc::new(ParkedDaemon::parked(parked));
        let (send, mut receive) = mpsc::channel();

        mode.borrow_mut()
            .apply_connection_state_snapshot(local::AgentConnectionState {
                session_id: "settings-parked".into(),
                thinking_level: pi_agent_core::types::ThinkingLevel::Low,
                ..Default::default()
            });

        // The call is parked at this instant: nothing has been released yet.
        apply_change(
            Change::ThinkingLevel("high".into()),
            &mode,
            &editor,
            &ui,
            &transcript,
            &daemon,
            &send,
        );

        // `apply_change` already returned, so the owner loop did not wait for the
        // daemon. The `.then` half must not have run either.
        assert_eq!(
            mode.borrow()
                .connection_state
                .as_ref()
                .map(|state| state.thinking_level),
            Some(pi_agent_core::types::ThinkingLevel::Low),
            "the level must stay unpublished while the daemon call is parked"
        );

        // Let the spawned task start and reach the parked call.
        settle().await;
        assert_eq!(
            daemon.calls(),
            vec!["set_thinking_level"],
            "the daemon call must have been issued off the owner loop"
        );
        assert!(
            next_event(&mut receive, std::time::Duration::from_millis(50))
                .await
                .is_none(),
            "no result can arrive while the daemon call is parked"
        );

        // Always release, so the task completes and this test cannot hang.
        let _ = release.send(());
        let event = next_event(&mut receive, std::time::Duration::from_secs(5))
            .await
            .expect("the released daemon call must report its continuation");
        match event {
            HostEvent::SettingAccepted(Change::ThinkingLevel(level)) => assert_eq!(level, "high"),
            _ => panic!("the released accepted change must report its continuation event"),
        }
    }

    /// The `.then` continuation is separate from the local apply: a rejected
    /// thinking-level change must not publish the level the daemon refused
    /// (interactive-mode.ts:7852-7856 patches inside `.then`).
    #[tokio::test]
    async fn a_rejected_thinking_level_is_reported_without_publishing_the_level() {
        let mode = settings_mode("settings-thinking");
        let (editor, ui, transcript) = host(&mode);
        let daemon = Arc::new(ParkedDaemon::rejecting());
        let (send, mut receive) = mpsc::channel();

        mode.borrow_mut()
            .apply_connection_state_snapshot(local::AgentConnectionState {
                session_id: "settings-thinking".into(),
                thinking_level: pi_agent_core::types::ThinkingLevel::Low,
                ..Default::default()
            });

        apply_change(
            Change::ThinkingLevel("high".into()),
            &mode,
            &editor,
            &ui,
            &transcript,
            &daemon,
            &send,
        );

        let reported = next_event(&mut receive, std::time::Duration::from_secs(5))
            .await
            .expect("the failed remote call must report through the host");
        assert!(
            matches!(reported, HostEvent::Error(_)),
            "a rejected thinking-level change must report a HostEvent::Error, not another variant"
        );
        assert_eq!(
            mode.borrow()
                .connection_state
                .as_ref()
                .map(|state| state.thinking_level),
            Some(pi_agent_core::types::ThinkingLevel::Low),
            "the refused level must not be published"
        );

        // The accepting path publishes it through the `.then` continuation.
        apply_remote_applied(&Change::ThinkingLevel("high".into()), &mode, &ui);
        assert_eq!(
            mode.borrow()
                .connection_state
                .as_ref()
                .map(|state| state.thinking_level),
            Some(pi_agent_core::types::ThinkingLevel::High)
        );
    }

    /// The `.then` continuation the owner loop runs for an accepted change: the
    /// host must publish the new level only through this event, so the loop
    /// never awaits the RPC to finish it.
    #[tokio::test]
    async fn an_accepted_thinking_level_reports_the_continuation_event() {
        let mode = settings_mode("settings-accepted");
        let (editor, ui, transcript) = host(&mode);
        let daemon = Arc::new(ParkedDaemon::default());
        let (send, mut receive) = mpsc::channel();

        mode.borrow_mut()
            .apply_connection_state_snapshot(local::AgentConnectionState {
                session_id: "settings-accepted".into(),
                thinking_level: pi_agent_core::types::ThinkingLevel::Low,
                ..Default::default()
            });

        apply_change(
            Change::ThinkingLevel("high".into()),
            &mode,
            &editor,
            &ui,
            &transcript,
            &daemon,
            &send,
        );

        let event = next_event(&mut receive, std::time::Duration::from_secs(5))
            .await
            .expect("an accepted change must send its continuation event");
        match event {
            HostEvent::SettingAccepted(Change::ThinkingLevel(level)) => {
                assert_eq!(level, "high")
            }
            _ => panic!("expected the thinking-level continuation event"),
        }

        // Still unpublished until the loop runs the continuation.
        assert_eq!(
            mode.borrow()
                .connection_state
                .as_ref()
                .map(|state| state.thinking_level),
            Some(pi_agent_core::types::ThinkingLevel::Low)
        );
        apply_remote_applied(&Change::ThinkingLevel("high".into()), &mode, &ui);
        assert_eq!(
            mode.borrow()
                .connection_state
                .as_ref()
                .map(|state| state.thinking_level),
            Some(pi_agent_core::types::ThinkingLevel::High)
        );
    }
}
