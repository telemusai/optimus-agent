//! Native owner-thread dialogs for the remaining interactive slash commands.
use super::*;

// Jev surfaces. Registered here (inside the lane-C-owned file) with `#[path]`
// sibling declarations, so the new files need no edit to `native_host.rs`.
#[path = "jev_menu.rs"]
pub(crate) mod jev_menu;
#[path = "jev_menu_component.rs"]
pub(crate) mod jev_menu_component;
#[path = "jev_key_input.rs"]
pub(crate) mod jev_key_input;
#[path = "jev_footer.rs"]
pub(crate) mod jev_footer;
#[path = "jev_host.rs"]
pub(crate) mod jev_host;
use crate::core::{auth_storage::AuthStorage, settings_manager::SettingsManager};
use crate::modes::interactive::components::{
    scoped_models_selector::{ModelsCallbacks, ModelsConfig, ScopedModelsSelectorComponent},
    tree_selector::TreeSelectorComponent,
    user_message_selector::{UserMessageItem, UserMessageSelectorComponent},
};
use tokio::sync::{mpsc as async_mpsc, oneshot};

pub(super) enum Dialog {
    Logout(
        Vec<crate::modes::interactive::components::oauth_selector::AuthSelectorProvider>,
        oneshot::Sender<Option<String>>,
    ),
    Select(String, Vec<String>, oneshot::Sender<Option<String>>),
    Input(String, bool, oneshot::Sender<Option<String>>),
    Fork(
        Vec<wire::AgentConnectionUserMessage>,
        oneshot::Sender<Option<String>>,
    ),
    Tree(
        wire::AgentConnectionWatchSessionTree,
        Option<String>,
        oneshot::Sender<Option<String>>,
    ),
    Models(
        Vec<wire::AgentConnectionModel>,
        Option<Vec<String>>,
        async_mpsc::UnboundedSender<ScopeChange>,
    ),
    // SHARED FILE EDIT (modes/interactive/native_host_commands.rs, jev-ui lane):
    // the two `/jev` overlays. Additive `Dialog` variants; every other producer
    // of `Dialog` is untouched.
    Jev(
        pi_jev::types::JevMode,
        async_mpsc::UnboundedSender<jev_host::JevMenuEvent>,
    ),
    JevKey(async_mpsc::UnboundedSender<jev_host::JevKeyEvent>),
}

pub(super) enum ScopeChange {
    Change(Option<Vec<String>>),
    Persist(Option<Vec<String>>),
    Close,
}

struct LogoutPicker {
    picker: crate::modes::interactive::components::oauth_selector::OAuthSelectorComponent,
    reply: Option<oneshot::Sender<Option<String>>>,
    send: mpsc::Sender<HostEvent>,
}
impl TuiComponent for LogoutPicker {
    fn render(&mut self, width: f64) -> Vec<String> {
        self.picker.render(width)
    }
    fn invalidate(&mut self) {
        self.picker.invalidate();
    }
    fn handle_input(&mut self, data: &str) {
        self.picker.handle_input(data);
        let selected = self.picker.selected_provider.take();
        if selected.is_some() || self.picker.cancelled {
            let _ = self.send.send(HostEvent::CloseCommandDialog);
            if let Some(reply) = self.reply.take() {
                let _ = reply.send(selected.map(|p| p.id));
            }
        }
    }
}

pub(super) fn mount(
    dialog: Dialog,
    ui: Rc<RefCell<TUI>>,
    mode: &InteractiveMode,
    connection: Arc<dyn wire::AgentConnection>,
    send: mpsc::Sender<HostEvent>,
) -> Rc<RefCell<dyn TuiComponent>> {
    // SHARED FILE EDIT (native_host_commands.rs, jev-ui lane): the two `/jev`
    // overlay mounts. Additive branches ahead of the existing ones.
    if let Dialog::Jev(active_mode, events) = dialog {
        return Rc::new(RefCell::new(jev_host::JevMenuOverlay::new(
            active_mode, events, send,
        )));
    }
    if let Dialog::JevKey(events) = dialog {
        return Rc::new(RefCell::new(jev_host::JevKeyOverlay::new(events, send)));
    }
    if let Dialog::Logout(providers, reply) = dialog {
        use crate::modes::interactive::components::oauth_selector::*;
        let rows = ui.clone();
        return Rc::new(RefCell::new(LogoutPicker {
            picker: OAuthSelectorComponent::new(
                "logout",
                Box::new(AuthStorage::create(None, None)),
                providers,
                None,
                OAuthSelectorOptions {
                    get_rows: Some(Rc::new(move || rows.borrow().terminal_rows() as f64)),
                    ..Default::default()
                },
            ),
            reply: Some(reply),
            send,
        }));
    }
    if let Dialog::Models(models, enabled, changes) = dialog {
        let persist = changes.clone();
        let close = changes.clone();
        return Rc::new(RefCell::new(ScopedModelsSelectorComponent::new(
            ModelsConfig {
                all_models: models,
                enabled_model_ids: enabled,
            },
            ModelsCallbacks {
                on_change: Box::new(move |ids| {
                    let _ = changes.send(ScopeChange::Change(ids));
                }),
                on_persist: Box::new(move |ids| {
                    let _ = persist.send(ScopeChange::Persist(ids));
                }),
                on_cancel: Box::new(move || {
                    let _ = send.send(HostEvent::CloseCommandDialog);
                    let _ = close.send(ScopeChange::Close);
                }),
            },
        )));
    }
    let (reply, content) = match dialog {
        // Handled above; listed so a future `Dialog` variant forces an update here.
        Dialog::Jev(..) | Dialog::JevKey(..) => unreachable!(),
        Dialog::Select(title, values, reply) => (reply, (Some(title), values, None, None, false)),
        Dialog::Input(title, multiline, reply) => {
            (reply, (Some(title), Vec::new(), None, None, multiline))
        }
        Dialog::Fork(messages, reply) => (reply, (None, Vec::new(), Some(messages), None, false)),
        Dialog::Tree(tree, selected, reply) => {
            (reply, (selected, Vec::new(), None, Some(tree), false))
        }
        Dialog::Models(..) | Dialog::Logout(..) => unreachable!(),
    };
    let reply = Rc::new(RefCell::new(Some(reply)));
    let selected = reply.clone();
    let selected_send = send.clone();
    let accept: Box<dyn FnMut(&str)> = Box::new(move |value| {
        let _ = selected_send.send(HostEvent::CloseCommandDialog);
        if let Some(reply) = selected.borrow_mut().take() {
            let _ = reply.send(Some(value.into()));
        }
    });
    let label_send = send.clone();
    let cancel: Box<dyn FnMut()> = Box::new(move || {
        let _ = send.send(HostEvent::CloseCommandDialog);
        if let Some(reply) = reply.borrow_mut().take() {
            let _ = reply.send(None);
        }
    });
    let (title, values, messages, tree, multiline) = content;
    if let Some(messages) = messages {
        let initial = messages.last().map(|m| m.entry_id.clone());
        return Rc::new(RefCell::new(UserMessageSelectorComponent::new(
            messages
                .into_iter()
                .map(|m| UserMessageItem {
                    id: m.entry_id,
                    text: m.text,
                    timestamp: None,
                })
                .collect(),
            accept,
            cancel,
            initial.as_deref(),
        )));
    }
    if let Some(tree) = tree {
        let labels = connection.clone();
        let filter = mode
            .settings_manager()
            .lock()
            .ok()
            .map(|s| s.get_tree_filter_mode());
        let filter = filter.as_ref().map(|f| f.as_str()).and_then(|f| {
            crate::modes::interactive::components::tree_selector::FILTER_MODES
                .iter()
                .copied()
                .find(|value| *value == f)
        });
        return Rc::new(RefCell::new(TreeSelectorComponent::new(
            &tree.tree,
            tree.leaf_id,
            ui.borrow().terminal_rows() as f64,
            accept,
            cancel,
            Some(Box::new(move |id, label| {
                let connection = labels.clone();
                let send = label_send.clone();
                tokio::spawn(async move {
                    if let Err(error) = connection
                        .set_session_entry_label(&id, label.as_deref())
                        .await
                    {
                        let _ = send.send(HostEvent::Error(format!(
                            "Could not set session label: {error}"
                        )));
                    }
                });
            })),
            title,
            filter,
        )));
    }
    if !values.is_empty() {
        return Rc::new(RefCell::new(ExtensionSelectorComponent::new(
            title.as_deref().unwrap_or("Select"),
            values,
            accept,
            cancel,
            ExtensionSelectorOptions::default(),
        )));
    }
    if multiline {
        let mut accept = accept;
        Rc::new(RefCell::new(ExtensionEditorComponent::new(
            ui,
            Arc::new(AppKeybindingsManager),
            title.as_deref().unwrap_or("Input"),
            None,
            Box::new(move |value| accept(&value)),
            cancel,
            Default::default(),
        )))
    } else {
        Rc::new(RefCell::new(ExtensionInputComponent::new(
            title.as_deref().unwrap_or("Input"),
            None,
            accept,
            cancel,
            ExtensionInputOptions::default(),
        )))
    }
}

async fn select(send: &mpsc::Sender<HostEvent>, title: &str, choices: &[&str]) -> Option<String> {
    let (reply, wait) = oneshot::channel();
    send.send(HostEvent::CommandDialog(Dialog::Select(
        title.into(),
        choices.iter().map(|s| s.to_string()).collect(),
        reply,
    )))
    .ok()?;
    wait.await.ok().flatten()
}

async fn input(send: &mpsc::Sender<HostEvent>, title: &str, multiline: bool) -> Option<String> {
    let (reply, wait) = oneshot::channel();
    send.send(HostEvent::CommandDialog(Dialog::Input(
        title.into(),
        multiline,
        reply,
    )))
    .ok()?;
    wait.await.ok().flatten()
}

pub(super) async fn run(
    connection: &Arc<dyn wire::AgentConnection>,
    send: &mpsc::Sender<HostEvent>,
    name: &str,
    args: &str,
) -> Result<CommandOutput, String> {
    match name {
        "fork" => {
            let messages = connection.get_user_messages_for_forking().await?;
            if messages.is_empty() {
                return Ok(CommandOutput::Status("No messages to fork from".into()));
            }
            let (reply, wait) = oneshot::channel();
            send.send(HostEvent::CommandDialog(Dialog::Fork(messages, reply)))
                .map_err(|e| e.to_string())?;
            if let Some(id) = wait.await.ok().flatten() {
                let result = connection.fork(&id, None).await?;
                if !result
                    .get("cancelled")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
                {
                    let _ = send.send(HostEvent::RefreshSnapshot(
                        connection.get_initial_snapshot().await?,
                    ));
                    let _ = send.send(HostEvent::EditorText(
                        result
                            .get("selectedText")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .into(),
                    ));
                    return Ok(CommandOutput::Status("Forked to new session".into()));
                }
            }
        }
        "tree" => {
            let mut selected = None;
            loop {
                let started = Instant::now();
                let tree = connection.get_session_tree().await?;
                if tree.tree.is_empty() {
                    return Ok(CommandOutput::Status("No entries in session".into()));
                }
                let leaf = tree.leaf_id.clone();
                let (reply, wait) = oneshot::channel();
                send.send(HostEvent::CommandDialog(Dialog::Tree(
                    tree, selected, reply,
                )))
                .map_err(|e| e.to_string())?;
                let _ = send.send(HostEvent::MenuTiming(started));
                let Some(id) = wait.await.ok().flatten() else {
                    break;
                };
                if leaf.as_deref() == Some(&id) {
                    return Ok(CommandOutput::Status("Already at this point".into()));
                }
                selected = Some(id.clone());
                let state = connection.get_state().await?;
                let settings = SettingsManager::create(&state.cwd, None);
                let mut options = wire::AgentConnectionNavigateTreeOptions::default();
                let mut cancelled = false;
                if !settings.get_branch_summary_skip_prompt() {
                    loop {
                        let Some(choice) = select(
                            send,
                            "Summarize branch?",
                            &["No summary", "Summarize", "Summarize with custom prompt"],
                        )
                        .await
                        else {
                            cancelled = true;
                            break;
                        };
                        options.summarize = Some(choice != "No summary");
                        if choice == "Summarize with custom prompt" {
                            let Some(text) =
                                input(send, "Custom summarization instructions", true).await
                            else {
                                continue;
                            };
                            options.custom_instructions = Some(text);
                        }
                        break;
                    }
                }
                if cancelled {
                    continue;
                }
                let summarize = options.summarize == Some(true);
                let cancel = tokio_util::sync::CancellationToken::new();
                if summarize {
                    let _ = send.send(HostEvent::CommandBusy(
                        "Summarizing branch...".into(),
                        cancel.clone(),
                    ));
                }
                let navigation = connection.navigate_tree(&id, Some(options));
                tokio::pin!(navigation);
                let result = tokio::select! {
                    result = &mut navigation => result,
                    _ = cancel.cancelled(), if summarize => {
                        let _ = connection.abort_branch_summary().await;
                        navigation.await
                    }
                };
                if summarize {
                    let _ = send.send(HostEvent::CloseCommandDialog);
                }
                let result = result?;
                if result.aborted == Some(true) {
                    let _ = send.send(HostEvent::Status("Branch summarization cancelled".into()));
                    continue;
                }
                if result.cancelled {
                    return Ok(CommandOutput::Status("Navigation cancelled".into()));
                }
                let _ = send.send(HostEvent::RefreshSnapshot(
                    connection.get_initial_snapshot().await?,
                ));
                let _ = send.send(HostEvent::EditorText(
                    result.editor_text.unwrap_or_default(),
                ));
                break;
            }
        }
        "scoped-models" => {
            let models = connection.get_model_catalog().await?.models;
            if models.is_empty() {
                return Ok(CommandOutput::Status("No models available".into()));
            }
            let state = connection.get_state().await?;
            let mut settings = SettingsManager::create(&state.cwd, None);
            let enabled = if !state.scoped_models.is_empty() {
                Some(
                    state
                        .scoped_models
                        .iter()
                        .map(|s| format!("{}/{}", s.model.provider, s.model.id))
                        .collect(),
                )
            } else {
                settings.get_enabled_models().map(|patterns| {
                    crate::core::model_resolver::resolve_model_scope_from_models(&patterns, &models)
                        .into_iter()
                        .map(|s| format!("{}/{}", s.model.provider, s.model.id))
                        .collect()
                })
            };
            let (changes, mut receive) = async_mpsc::unbounded_channel();
            send.send(HostEvent::CommandDialog(Dialog::Models(
                models.clone(),
                enabled,
                changes,
            )))
            .map_err(|e| e.to_string())?;
            while let Some(change) = receive.recv().await {
                match change {
                    ScopeChange::Change(ids) => {
                        let scoped: Vec<wire::AgentConnectionScopedModel> = ids
                            .filter(|ids| !ids.is_empty() && ids.len() < models.len())
                            .unwrap_or_default()
                            .into_iter()
                            .filter_map(|id| {
                                models
                                    .iter()
                                    .find(|m| format!("{}/{}", m.provider, m.id) == id)
                                    .cloned()
                            })
                            .map(|model| wire::AgentConnectionScopedModel {
                                model,
                                thinking_level: None,
                            })
                            .collect();
                        if let Err(error) = connection.set_scoped_models(scoped.clone()).await {
                            let _ = send.send(HostEvent::Error(error));
                        } else {
                            let _ = send
                                .send(HostEvent::ScopeChanged(state.session_id.clone(), scoped));
                        }
                    }
                    ScopeChange::Persist(ids) => {
                        settings.set_enabled_models(ids.filter(|ids| ids.len() != models.len()));
                        settings.flush().await;
                        let errors = settings.drain_errors(None);
                        if let Some(error) = errors.first() {
                            let _ = send.send(HostEvent::Error(error.error.to_string()));
                        } else {
                            let _ = send.send(HostEvent::Status(
                                "Model selection saved to settings".into(),
                            ));
                        }
                        let _ = send.send(HostEvent::ReloadSettings);
                    }
                    ScopeChange::Close => break,
                }
            }
        }
        "logout" => {
            let mut auth = AuthStorage::create(None, None);
            use crate::modes::interactive::components::oauth_selector::{
                AuthSelectorCategory, AuthSelectorProvider,
            };
            let oauth = auth.get_oauth_providers();
            let mut providers: Vec<_> = auth
                .list()
                .into_iter()
                .map(|id| {
                    let is_oauth = matches!(
                        auth.get(&id),
                        Some(crate::core::auth_storage::AuthCredential::OAuth { .. })
                    );
                    AuthSelectorProvider {
                        name: oauth
                            .iter()
                            .find(|p| p.id == id)
                            .map(|p| p.name.clone())
                            .unwrap_or_else(|| id.clone()),
                        category: Some(if id.starts_with("mcp:") || id == "serper" {
                            AuthSelectorCategory::Service
                        } else {
                            AuthSelectorCategory::Provider
                        }),
                        auth_type: if is_oauth { "oauth" } else { "api_key" }.into(),
                        id,
                    }
                })
                .collect();
            if !providers.iter().any(|p| p.id == "prime-inference")
                && auth.get_auth_status("prime-inference").source.as_deref() == Some("prime_cli")
            {
                providers.push(AuthSelectorProvider {
                    id: "prime-inference".into(),
                    name: "Prime Inference".into(),
                    auth_type: "api_key".into(),
                    category: None,
                });
            }
            providers.sort_by(|a, b| a.name.cmp(&b.name));
            if providers.is_empty() {
                return Ok(CommandOutput::Status("No stored credentials to remove. /logout only removes credentials saved by /login; environment variables and models.json config are unchanged.".into()));
            }
            let (reply, wait) = oneshot::channel();
            send.send(HostEvent::CommandDialog(Dialog::Logout(providers, reply)))
                .map_err(|e| e.to_string())?;
            if let Some(provider) = wait.await.ok().flatten() {
                let message =
                    crate::modes::interactive::auth_flows::logout_provider(&mut auth, &provider)?;
                let _ = send.send(HostEvent::AuthChanged);
                if provider.starts_with("mcp:") {
                    return reload_mcp(connection, message).await;
                }
                return Ok(CommandOutput::Status(message));
            }
        }
        // SHARED FILE EDIT (native_host_commands.rs, jev-ui lane): the `/jev`
        // dispatch arm.
        "jev" => return jev_host::run(connection, send, args).await,
        "mcp" => return mcp(connection, send, args).await,
        "share" => return share(connection, send).await,
        "traces" => return traces(connection, send, args).await,
        "monitor" => return monitor(send, args).await,
        "update" => {
            let _ = send.send(HostEvent::RunUpdate(
                crate::core::prompt_templates::parse_command_args(args),
            ));
        }
        "btw" => {
            let _ = send.send(HostEvent::SideQuestion(args.into()));
        }
        "debug" => {
            let _ = send.send(HostEvent::Debug);
        }
        _ => return Err(format!("Unknown local command: /{name}")),
    }
    Ok(CommandOutput::Nothing)
}

async fn monitor(send: &mpsc::Sender<HostEvent>, args: &str) -> Result<CommandOutput, String> {
    use crate::core::performance_monitor::{parse_monitor_command, MonitorCommand, PerformanceMonitor};
    let monitor = PerformanceMonitor::from_environment(crate::config::get_agent_dir());
    let command = parse_monitor_command(args)?;
    let requested = match command {
        MonitorCommand::Status => return Ok(CommandOutput::Panel(monitor.status_text()?)),
        MonitorCommand::Set(enabled) => Some(enabled),
        MonitorCommand::Select => {
            let title = format!("Performance monitoring: {}", if monitor.enabled()? { "ON" } else { "OFF" });
            select(send, &title, &["ON", "OFF"]).await.map(|choice| choice == "ON")
        }
    };
    let Some(enabled) = requested else { return Ok(CommandOutput::Nothing); };
    monitor.set_enabled(enabled)?;
    Ok(CommandOutput::Status(monitor.status_text()?))
}

async fn reload_mcp(
    connection: &Arc<dyn wire::AgentConnection>,
    message: String,
) -> Result<CommandOutput, String> {
    let state = connection.get_state().await?;
    if state.is_streaming || state.is_compacting {
        return Ok(CommandOutput::Status(format!(
            "{message} The change was saved. Run /reload after the current turn to activate it."
        )));
    }
    match connection.reload().await {
        Ok(()) => Ok(CommandOutput::Status(message)),
        Err(error) => Ok(CommandOutput::Warning(format!(
            "{message} The change remains saved, but it is not active in this session. {error}"
        ))),
    }
}

struct Credentials(AuthStorage);
impl crate::core::mcp::mcp_command::McpCredentialStore for Credentials {
    fn remove_verified(&mut self, provider: &str) -> Result<(), String> {
        self.0.remove_verified(provider)
    }
}
async fn mcp(
    connection: &Arc<dyn wire::AgentConnection>,
    send: &mpsc::Sender<HostEvent>,
    args: &str,
) -> Result<CommandOutput, String> {
    let args = crate::core::prompt_templates::parse_command_args(args);
    let mut auth = AuthStorage::create(None, None);
    match args.first().map(String::as_str) {
        Some("login") => {
            if args.len() != 2 {
                return Err("Usage: /mcp login <name> (e.g. /mcp login linear)".into());
            }
            let id = format!("mcp:{}", args[1]);
            if crate::core::auth_storage::get_oauth_provider(&id).is_none() {
                return Err(format!("Unknown MCP integration: {}", args[1]));
            }
            let _ = send.send(HostEvent::BeginLogin(id, true));
            return Ok(CommandOutput::Nothing);
        }
        Some("logout") => {
            if args.len() != 2 {
                return Err("Usage: /mcp logout <name>".into());
            }
            let id = format!("mcp:{}", args[1]);
            if auth.get(&id).is_none() {
                return Ok(CommandOutput::Status(format!(
                    "{} is not connected.",
                    args[1]
                )));
            }
            auth.logout(&id)?;
            auth.remove_verified(&id)?;
            let _ = send.send(HostEvent::AuthChanged);
            return reload_mcp(connection, format!("Disconnected {}.", args[1])).await;
        }
        _ => {}
    }
    let state = connection.get_state().await?;
    let mut settings = SettingsManager::create(&state.cwd, None);
    let result = crate::core::mcp::mcp_command::run_mcp_management_command(
        &args,
        &mut settings,
        Some(&mut Credentials(auth)),
    )
    .await?;
    if result.changed {
        let _ = send.send(HostEvent::AuthChanged);
        return reload_mcp(connection, result.message).await;
    }
    if result.action == "list" {
        let auth = AuthStorage::create(None, None);
        let builtins = pi_ai::mcp::catalog::builtin_mcp_catalog()
            .iter()
            .map(|e| {
                format!(
                    "{} ({}): {}",
                    e.label,
                    e.server,
                    if auth.get(&format!("mcp:{}", e.server)).is_some() {
                        "connected"
                    } else {
                        "not connected"
                    }
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        return Ok(CommandOutput::Status(format!(
            "Built-in MCP integrations:\n{builtins}\n\nUser-configured MCP servers:\n{}",
            result.message
        )));
    }
    Ok(CommandOutput::Status(result.message))
}

async fn share(
    connection: &Arc<dyn wire::AgentConnection>,
    send: &mpsc::Sender<HostEvent>,
) -> Result<CommandOutput, String> {
    let auth = tokio::process::Command::new("gh")
        .args(["auth", "status"])
        .output()
        .await
        .map_err(|_| {
            "GitHub CLI (gh) is not installed. Install it from https://cli.github.com/".to_string()
        })?;
    if !auth.status.success() {
        return Err("GitHub CLI is not logged in. Run 'gh auth login' first.".into());
    }
    let directory = tempfile::tempdir().map_err(|e| e.to_string())?;
    let file = directory.path().join("session.html");
    connection
        .export_to_html(file.to_str())
        .await
        .map_err(|e| format!("Failed to export session: {e}"))?;
    let cancel = tokio_util::sync::CancellationToken::new();
    let _ = send.send(HostEvent::CommandBusy(
        "Creating gist...".into(),
        cancel.clone(),
    ));
    let mut command = tokio::process::Command::new("gh");
    command
        .args(["gist", "create", "--public=false"])
        .arg(&file)
        .kill_on_drop(true);
    let result = tokio::select! {
        result = command.output() => result.map_err(|e| e.to_string()),
        _ = cancel.cancelled() => { let _ = send.send(HostEvent::CloseCommandDialog); return Ok(CommandOutput::Status("Share cancelled".into())); }
    };
    let _ = send.send(HostEvent::CloseCommandDialog);
    let output = result?;
    if !output.status.success() {
        return Err(format!(
            "Failed to create gist: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let url = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let id = url
        .rsplit('/')
        .next()
        .filter(|s| !s.is_empty())
        .ok_or("Failed to parse gist ID from gh output")?;
    Ok(CommandOutput::Status(format!(
        "Share URL: {}\nGist: {url}",
        crate::config::get_share_viewer_url(id)
    )))
}

async fn traces(
    connection: &Arc<dyn wire::AgentConnection>,
    send: &mpsc::Sender<HostEvent>,
    args: &str,
) -> Result<CommandOutput, String> {
    use crate::core::agent_traces::*;
    let state = connection.get_state().await?;
    let mut settings = SettingsManager::create(&state.cwd, None);
    let mut auth = AuthStorage::create(None, None);
    let command = args.trim().to_lowercase();
    let command = if command.is_empty() {
        "status"
    } else {
        &command
    };
    if matches!(command, "off" | "disable") {
        settings.set_agent_traces_enabled(false);
        settings.flush().await;
        if let Some(error) = settings.drain_errors(None).first() {
            return Err(error.error.to_string());
        }
        let _ = send.send(HostEvent::ReloadSettings);
        return Ok(CommandOutput::Status("Trace sharing disabled.".into()));
    }
    if command == "login" {
        trace_login(send, &mut auth).await?;
        return Ok(CommandOutput::Nothing);
    }
    let mut credential = get_prime_agent_trace_credential(&mut auth, true, None).await;
    if matches!(command, "on" | "enable") && credential.is_none() {
        if !trace_login(send, &mut auth).await? {
            return Ok(CommandOutput::Nothing);
        }
        credential = get_prime_agent_trace_credential(&mut auth, true, None).await;
    }
    if command == "status" {
        return Ok(CommandOutput::Panel(format!("Trace Sharing\n\nAutomatic uploads: {}\nCredential: {}\nEndpoint: {}\nSession file: {}\n\nCommands: /traces on, /traces off, /traces preview, /traces upload-current, /traces upload-all, /traces login", if settings.get_agent_traces_enabled() { "Enabled" } else { "Disabled" }, credential.map(|c| c.label).unwrap_or_else(|| "Not configured".into()), crate::core::prime_inference_auth::resolve_prime_agent_traces_base_url(None), state.session_file.as_deref().unwrap_or("In-memory"))));
    }
    if command == "preview" {
        let preview = preview_agent_trace_file(&AgentTracePreviewOptions {
            session_file: state.session_file,
            ..Default::default()
        })
        .await;
        let text = match preview {
            AgentTracePreviewResult::Ready { session_file, session_id, trace_id, size, max_bytes, uploadable, endpoint, content_preview, truncated, parent_session_id, git_repo, git_commit, .. } => {
                let mut text = format!("Trace Preview\nNothing has been uploaded by this command.\n\nFile: {session_file}\nSize: {size} bytes\nUploadable: {}\nEndpoint: {endpoint}\nSession ID: {session_id}\nTrace ID: {trace_id}", if uploadable { "Yes".into() } else { format!("No (limit {max_bytes} bytes)") });
                for (label, value) in [("Parent session", parent_session_id), ("Git repository", git_repo), ("Git commit", git_commit)] { if let Some(value) = value { text.push_str(&format!("\n{label}: {value}")); } }
                text.push_str(&format!("\n\nRaw JSONL payload preview\n{content_preview}"));
                if truncated { text.push_str("\n\nPreview truncated; upload sends the complete file."); }
                text
            }
            AgentTracePreviewResult::NoSessionFile => "Trace preview is unavailable until the current session has a persisted assistant response.".into(),
            AgentTracePreviewResult::EmptySession => "The current trace is empty.".into(),
            AgentTracePreviewResult::InvalidSession { message } | AgentTracePreviewResult::Failed { message } => format!("Trace preview failed: {message}."),
        };
        return Ok(CommandOutput::Panel(text));
    }
    if !matches!(
        command,
        "on" | "enable" | "upload" | "upload-current" | "upload-all"
    ) {
        return Err(
            "Usage: /traces [status|on|off|preview|upload|upload-current|upload-all|login]".into(),
        );
    }
    if credential.is_none() {
        return Err("Trace sharing needs a Prime API key. Run /traces login.".into());
    }
    let enable = matches!(command, "on" | "enable");
    if enable {
        settings.set_agent_traces_enabled(true);
        settings.flush().await;
        if let Some(error) = settings.drain_errors(None).first() {
            return Err(error.error.to_string());
        }
        let _ = send.send(HostEvent::ReloadSettings);
    }
    let cancel = tokio_util::sync::CancellationToken::new();
    let _ = send.send(HostEvent::CommandBusy(
        "Uploading traces...".into(),
        cancel.clone(),
    ));
    let upload = AgentTraceUploadOptions {
        session_file: state.session_file,
        auth_storage: Arc::new(tokio::sync::Mutex::new(auth)),
        require_enabled: false,
        base_url: None,
        config_path: None,
        fetch_fn: None,
        reload_config: false,
        request_timeout_ms: None,
        signal: Some(cancel),
        agent_traces_enabled: Arc::new(move || enable),
        reload_settings: Arc::new(|| Box::pin(async { Ok(()) })),
    };
    let message = if command == "upload-all" {
        let progress_send = send.clone();
        let result = upload_all_agent_traces(&AgentTraceUploadAllOptions {
            upload,
            session_dir: state.session_dir,
            concurrency: None,
            on_progress: Some(Arc::new(move |p| {
                if p.total > 0
                    && (p.completed == 0 || p.completed == p.total || p.completed % 10 == 0)
                {
                    let _ = progress_send.send(HostEvent::Status(format!(
                        "Uploading traces: {}/{}",
                        p.completed, p.total
                    )));
                }
            })),
        })
        .await;
        format!(
            "Trace upload complete: {} uploaded, {} skipped, {} failed; {} bytes.",
            result.uploaded, result.skipped, result.failed, result.bytes_stored
        )
    } else {
        let result = upload_agent_trace_file(upload).await;
        let message = format_upload(&result);
        if let AgentTraceUploadResult::Failed { .. } = result {
            let _ = send.send(HostEvent::CloseCommandDialog);
            return Err(message);
        }
        if enable {
            format!("Trace sharing enabled. {message}")
        } else {
            message
        }
    };
    let _ = send.send(HostEvent::CloseCommandDialog);
    Ok(CommandOutput::Status(message))
}

async fn trace_login(
    send: &mpsc::Sender<HostEvent>,
    auth: &mut AuthStorage,
) -> Result<bool, String> {
    let Some(key) = input(send, "Prime Agent Traces: enter Prime API key", false).await else {
        return Ok(false);
    };
    let key = key.trim();
    if key.is_empty() {
        return Err("API key cannot be empty".into());
    }
    let base = crate::core::prime_inference_auth::resolve_prime_agent_traces_base_url(None);
    let access = crate::core::prime_inference_auth::check_prime_agent_traces_access(
        key, &base, None, None, None,
    )
    .await?;
    if let crate::core::prime_inference_auth::PrimeInferenceAccessResult::Failed {
        message, ..
    } = access
    {
        return Err(message);
    }
    auth.set(
        "prime-agent-traces",
        crate::core::auth_storage::AuthCredential::ApiKey {
            key: key.into(),
            prime_team: None,
        },
    );
    auth.reload();
    if !matches!(auth.get("prime-agent-traces"), Some(crate::core::auth_storage::AuthCredential::ApiKey { key: saved, .. }) if saved == key)
    {
        return Err("Could not save trace credentials".into());
    }
    let _ = send.send(HostEvent::AuthChanged);
    let _ = send.send(HostEvent::Status("Logged in to Prime Agent Traces".into()));
    Ok(true)
}

fn format_upload(result: &crate::core::agent_traces::AgentTraceUploadResult) -> String {
    use crate::core::agent_traces::AgentTraceUploadResult::*;
    match result {
        Uploaded { bytes_stored, .. } => format!("Trace uploaded ({bytes_stored} bytes)."),
        Disabled => "Trace sharing is disabled.".into(),
        Unchanged => "Trace is already uploaded; no new content since the last upload.".into(),
        MissingCredentials => "Trace sharing needs a Prime API key. Run /traces login.".into(),
        NoSessionFile => "Current session has no persisted trace yet.".into(),
        EmptySession => "Current session trace is empty.".into(),
        InvalidSession { message } => format!("Trace upload skipped: {message}."),
        TooLarge { size, max_bytes } => format!("Trace upload skipped: session file is {size} bytes; limit is {max_bytes} bytes."),
        Failed { status_code: Some(404), .. } => "Trace upload endpoint was not found. The platform API may not be deployed yet, or PRIME_AGENT_TRACES_BASE_URL points at the wrong API.".into(),
        Failed { status_code, message, .. } => format!("Trace upload failed: {}{message}.", status_code.map(|s| format!("HTTP {s}: ")).unwrap_or_default()),
    }
}

pub(super) async fn debug(
    ui: &Rc<RefCell<TUI>>,
    connection: &Arc<dyn wire::AgentConnection>,
    send: &mpsc::Sender<HostEvent>,
) -> Result<(), String> {
    let (width, height) = {
        let ui = ui.borrow();
        (ui.terminal.columns(), ui.terminal.rows())
    };
    let lines = ui.borrow_mut().render(width as f64);
    let messages = connection.get_messages().await?;
    let mut text = format!("Debug output at {}\nTerminal: {width}x{height}\nTotal lines: {}\n\n=== All rendered lines with visible widths ===\n", chrono::Utc::now().to_rfc3339(), lines.len());
    for (index, line) in lines.iter().enumerate() {
        text.push_str(&format!(
            "[{index}] (w={}) {}\n",
            pi_tui::utils::visible_width(line),
            serde_json::to_string(line).map_err(|e| e.to_string())?
        ));
    }
    text.push_str("\n=== Agent messages (JSONL) ===\n");
    for message in messages {
        text.push_str(&serde_json::to_string(&message).map_err(|e| e.to_string())?);
        text.push('\n');
    }
    let path = std::path::PathBuf::from(crate::config::get_debug_log_path());
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    std::fs::write(&path, text).map_err(|e| e.to_string())?;
    let _ = send.send(HostEvent::Panel(format!(
        "Debug log written\n{}",
        path.display()
    )));
    Ok(())
}

/// Returns relaunch arguments only when the interactive process must retire.
/// All subprocesses inherit the terminal after raw mode has been released.
pub(super) async fn update(
    args: &[String],
    options: &InteractiveModeSeamOptions,
    mode: &Rc<RefCell<InteractiveMode>>,
    ui: &Rc<RefCell<TUI>>,
    connection: &Arc<dyn wire::AgentConnection>,
) -> Result<Option<Vec<String>>, String> {
    use crate::cli::daemon_update_restart::*;
    let executable = std::env::current_exe().map_err(|e| e.to_string())?;
    let includes_self = update_args_include_self(args);
    // `commandName === "update"` busy gate (interactive-mode.ts:5001-5012): a
    // non-self update must not tear down the terminal while a turn, compaction,
    // or shell command is running.
    let busy = {
        let mode = mode.borrow();
        mode.is_agent_compacting() || mode.is_agent_streaming() || mode.is_bash_running()
    };
    if !includes_self && busy {
        mode.borrow_mut().show_warning("Wait for the current work to finish before updating.");
        return Ok(None);
    }
    let cwd = mode.borrow().get_current_cwd();
    let state = connection.get_state().await?;
    let socket = resolve_interactive_update_daemon_socket_path(
        args,
        &resolve_daemon_update_restart_socket_path(options.daemon_socket_path.as_deref()),
    );
    let child_args = if includes_self {
        build_update_child_args(args, &socket)
    } else {
        args.to_vec()
    };
    ui.borrow_mut().terminal.drain_input(1000, 50);
    ui.borrow_mut().stop(TuiStopOptions::default());
    let mut child = std::process::Command::new(&executable);
    child.arg("update").args(child_args).current_dir(&cwd);
    if includes_self {
        child.env(crate::config::SELF_UPDATE_INTERACTIVE_CHILD_ENV, "1");
    }
    let result = child.status();
    let not_attempted = result
        .as_ref()
        .is_ok_and(|s| s.code() == Some(crate::config::SELF_UPDATE_NOT_ATTEMPTED_EXIT_CODE));
    if includes_self && !not_attempted {
        if result.as_ref().is_ok_and(|s| s.success()) {
            // Detach before coordination so this client does not hold the old
            // worker resident while the replacement supervisor is starting.
            let _ = connection.dispose().await;
            match launch_daemon_update_restart_coordinator(
                LaunchDaemonUpdateRestartCoordinatorOptions {
                    socket_path: socket,
                    agent_dir: crate::config::get_agent_dir(),
                    cwd: Some(cwd),
                    origin_active_session_id: state.active_session_id,
                    timeout_ms: None,
                },
            )
            .await
            {
                Ok(status) => {
                    let report = build_daemon_update_restart_report(&status);
                    for message in report.info {
                        println!("{message}");
                    }
                    for warning in report.warnings {
                        eprintln!("Warning: {warning}");
                    }
                }
                Err(error) => eprintln!("Updated, but daemon restart coordination failed: {error}"),
            }
        } else {
            match result {
                Err(error) => eprintln!("Update failed: {error}"),
                Ok(status) => eprintln!("Update exited with {status}"),
            }
            eprintln!("Relaunching optimus-rust...");
        }
        return Ok(Some(build_update_relaunch_args(
            &std::env::args().skip(1).collect::<Vec<_>>(),
            state.session_file.as_deref(),
        )));
    }
    ui.borrow_mut().start();
    ui.borrow_mut().request_render_forced();
    let status = result.map_err(|e| format!("Update failed: {e}"))?;
    if status.success() || not_attempted {
        Ok(None)
    } else {
        Err(format!("Update exited with {status}"))
    }
}
