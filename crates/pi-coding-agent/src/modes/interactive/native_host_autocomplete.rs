//! Adapter between the shared completion provider and the native terminal editor.

use std::{cell::RefCell, rc::{Rc, Weak}, sync::{Arc, mpsc}, time::Duration};

use pi_tui::autocomplete::{self, CombinedAutocompleteProvider};
use pi_tui::components::{editor, select_list::SelectItem};

use super::{local, wire, CustomEditor, InteractiveMode};

struct NativeAutocomplete {
    provider: CombinedAutocompleteProvider,
    catalogue: Option<PendingCatalogue>,
}

struct PendingCatalogue {
    receive: mpsc::Receiver<Result<Vec<wire::AgentConnectionSlashCommand>, String>>,
    task: tokio::task::JoinHandle<()>,
    mode: Weak<RefCell<InteractiveMode>>,
    cwd: String,
    fd: Option<String>,
}

impl Drop for PendingCatalogue {
    fn drop(&mut self) { self.task.abort(); }
}

impl NativeAutocomplete {
    fn new(provider: CombinedAutocompleteProvider) -> Self {
        Self { provider, catalogue: None }
    }

    fn refresh_catalogue(&mut self) {
        let Some(pending) = &self.catalogue else { return; };
        let result = match pending.receive.try_recv() {
            Ok(result) => result,
            Err(mpsc::TryRecvError::Empty) => return,
            Err(mpsc::TryRecvError::Disconnected) => Err("command catalogue request closed".into()),
        };
        let pending = self.catalogue.take().expect("pending catalogue");
        let Some(mode) = pending.mode.upgrade() else { return; };
        match result {
            Ok(commands) => {
                mode.borrow_mut().connection_commands = commands.into_iter().map(|command| {
                    local::AgentConnectionSlashCommand {
                        name: command.name, registered_name: command.registered_name,
                        description: command.description, argument_hint: command.argument_hint,
                        source: command.source,
                        source_info: local::AgentConnectionSourceInfo {
                            path: command.source_info.path, source: command.source_info.source,
                            scope: command.source_info.scope, origin: command.source_info.origin,
                            base_dir: command.source_info.base_dir,
                        },
                    }
                }).collect();
                self.provider = CombinedAutocompleteProvider::new(
                    command_entries(mode), &pending.cwd, pending.fd.as_deref());
            }
            Err(error) => mode.borrow_mut().show_warning(&format!("Could not load extension commands: {error}")),
        }
    }
}

// The editor stores UTF-8 byte offsets; the shared provider uses scalar offsets.
fn scalar_column(lines: &[String], line: usize, column: usize) -> usize {
    lines.get(line).map_or(0, |text| {
        text.char_indices()
            .take_while(|(offset, _)| *offset < column)
            .count()
    })
}

impl editor::EditorAutocompleteProvider for NativeAutocomplete {
    fn get_suggestions(
        &mut self,
        lines: &[String],
        line: usize,
        column: usize,
        force: bool,
    ) -> Option<editor::AutocompleteSuggestions> {
        // The shared provider currently performs bounded synchronous filesystem
        // scans; its future does not depend on a Tokio reactor.
        self.refresh_catalogue();
        let result = futures::executor::block_on(self.provider.get_suggestions(
            lines,
            line,
            scalar_column(lines, line, column),
            &autocomplete::AbortSignal::new(),
            force,
        ))?;
        Some(editor::AutocompleteSuggestions {
            items: result
                .items
                .into_iter()
                .map(|item| SelectItem {
                    value: item.value,
                    label: item.label,
                    description: item.description,
                    argument_hint: item.argument_hint,
                    source_tag: item.source_tag,
                    takes_argument: item.takes_argument,
                })
                .collect(),
            prefix: result.prefix,
            kind: result.kind,
        })
    }

    fn apply_completion(
        &mut self,
        lines: &[String],
        line: usize,
        column: usize,
        item: &SelectItem,
        prefix: &str,
    ) -> editor::ApplyCompletionResult {
        let result = self.provider.apply_completion(
            lines,
            line,
            scalar_column(lines, line, column),
            &autocomplete::AutocompleteItem {
                value: item.value.clone(),
                label: item.label.clone(),
                description: item.description.clone(),
                argument_hint: item.argument_hint.clone(),
                source_tag: item.source_tag.clone(),
                takes_argument: item.takes_argument,
            },
            prefix,
        );
        let cursor_col = result.lines.get(result.cursor_line).map_or(0, |text| {
            text.char_indices()
                .nth(result.cursor_col)
                .map_or(text.len(), |(offset, _)| offset)
        });
        editor::ApplyCompletionResult {
            lines: result.lines,
            cursor_line: result.cursor_line,
            cursor_col,
        }
    }

    fn should_trigger_file_completion(&self, lines: &[String], line: usize, column: usize) -> bool {
        self.provider
            .should_trigger_file_completion(lines, line, scalar_column(lines, line, column))
    }
}

/// Port of `getModelArgumentCompletions` (model-autocomplete.ts): the value is
/// `provider/id`, the label the model id, the description the provider.
fn get_model_argument_completions(
    prefix: &str,
    models: &[local::AgentConnectionModel],
) -> Option<Vec<super::AutocompleteItem>> {
    if models.is_empty() {
        return None;
    }
    let term = prefix.trim().to_lowercase();
    let filtered: Vec<&local::AgentConnectionModel> = models
        .iter()
        .filter(|model| {
            term.is_empty()
                || format!("{} {}", model.id, model.provider).to_lowercase().contains(&term)
        })
        .collect();
    if filtered.is_empty() {
        return None;
    }
    Some(
        filtered
            .into_iter()
            .map(|model| super::AutocompleteItem {
                value: format!("{}/{}", model.provider, model.id),
                label: model.id.clone(),
                description: Some(model.provider.clone()),
            })
            .collect(),
    )
}

fn command_entries(mode: Rc<RefCell<InteractiveMode>>) -> Vec<autocomplete::CommandEntry> {
    // Port of interactive-mode.ts:1438-1441: the effort hint lists the live
    // levels; without thinking support the registry default stays.
    // Without thinking support the registry default hint stays
    // (interactive-mode.ts:1434-1442 keeps the catalog hint otherwise).
    let effort_hint = {
        let levels = mode.borrow().get_available_thinking_levels();
        if levels.is_empty() {
            Some("[level]".to_string())
        } else {
            let joined =
                levels.iter().map(|level| level.as_str()).collect::<Vec<_>>().join("/");
            Some(format!("[{joined}]"))
        }
    };
    let mut commands: Vec<_> = crate::core::slash_commands::builtin_slash_commands()
        .iter()
        .map(|command| {
            let mode = mode.clone();
            let name = command.name.clone();
            let completions =
                matches!(name.as_str(), "effort" | "heartbeat" | "model").then(|| {
                Box::new(move |prefix: &str| {
                    let mode = mode.borrow();
                    let items = if name == "effort" {
                        mode.get_thinking_level_completions(prefix)
                    } else if name == "heartbeat" {
                        mode.get_heartbeat_argument_completions(prefix)
                    } else {
                        // Port of `getModelArgumentCompletions`
                        // (interactive-mode.ts:1428-1432, model-autocomplete.ts).
                        get_model_argument_completions(prefix, &mode.get_cached_model_candidates())
                    };
                    items
                        .unwrap_or_default()
                        .into_iter()
                        .map(|item| autocomplete::AutocompleteItem {
                            value: item.value,
                            label: item.label,
                            description: item.description,
                            ..Default::default()
                        })
                        .collect()
                }) as Box<dyn Fn(&str) -> Vec<autocomplete::AutocompleteItem>>
            });
            autocomplete::CommandEntry::Slash(autocomplete::SlashCommand {
                name: command.name.clone(),
                aliases: command.aliases.clone().unwrap_or_default(),
                description: Some(command.description.clone()),
                argument_hint: if command.name == "effort" {
                    effort_hint.clone()
                } else {
                    command.argument_hint.clone()
                },
                takes_argument: command.takes_argument,
                source_tag: None,
                get_argument_completions: completions,
            })
        })
        .collect();
    append_connection_commands(&mut commands, &mode.borrow());
    commands
}

fn append_connection_commands(commands: &mut Vec<autocomplete::CommandEntry>, mode: &InteractiveMode) {
    let skills_enabled = mode.settings_manager().lock()
        .map(|settings| settings.get_enable_skill_commands()).unwrap_or(false);
    for command in &mode.connection_commands {
        if (command.source == "skill" && !skills_enabled)
            || crate::core::slash_commands::is_builtin_slash_command_name(&command.name)
            || commands.iter().any(|entry| entry.name() == command.name) { continue; }
        commands.push(autocomplete::CommandEntry::Slash(autocomplete::SlashCommand {
            name: command.name.clone(), aliases: vec![], description: command.description.clone(),
            argument_hint: command.argument_hint.clone(), takes_argument: None,
            source_tag: mode.get_autocomplete_source_label(Some(&command.source_info)),
            get_argument_completions: None,
        }));
    }
}

pub(super) fn configure(editor: &mut CustomEditor, mode: Rc<RefCell<InteractiveMode>>, cwd: &str) {
    let fd = crate::utils::tools_manager::get_tool_path(crate::utils::tools_manager::FD);
    let mut provider = NativeAutocomplete::new(CombinedAutocompleteProvider::new(
        command_entries(mode.clone()), cwd, fd.as_deref()));
    // The controller stores the same connection used by the native owner. Fetch
    // off-thread: a slow worker must never block keyboard input or menu dismissal.
    let connection = mode.borrow().agent_connection
        .downcast_ref::<Arc<dyn wire::AgentConnection>>().cloned();
    if let (Some(connection), Ok(runtime)) = (connection, tokio::runtime::Handle::try_current()) {
        let (send, receive) = mpsc::sync_channel(64 * 1024);
        let task = runtime.spawn(async move {
            let result = tokio::time::timeout(Duration::from_secs(5), connection.get_commands())
                .await.unwrap_or_else(|_| Err("command catalogue timed out".into()));
            let _ = send.send(result);
        });
        provider.catalogue = Some(PendingCatalogue {
            receive, task, mode: Rc::downgrade(&mode), cwd: cwd.into(), fd,
        });
    }
    editor
        .editor_mut()
        .set_autocomplete_provider(Rc::new(RefCell::new(provider)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use pi_tui::components::editor::EditorAutocompleteProvider;

    fn test_mode() -> Rc<RefCell<InteractiveMode>> {
        use super::super::{InteractiveModeOptions, InteractiveModeUiServices};
        use std::sync::Mutex;
        Rc::new(RefCell::new(InteractiveMode::new(InteractiveModeOptions {
            migrated_providers: None, model_fallback_message: None, startup_notice: None,
            initial_message: None, initial_images: None, initial_messages: None, initial_prompts: None,
            verbose: false, agent_connection: Arc::new(()), daemon_socket_path: None,
            local_session_host: None, bind_local_session_extensions: false,
            ui_services: Some(InteractiveModeUiServices {
                settings_manager: Arc::new(Mutex::new(local::SettingsManager::in_memory(serde_json::Map::new()))),
                model_registry: Arc::new(Mutex::new(local::ModelRegistry::in_memory())),
                get_initial_cwd: Box::new(|| ".".into()), get_initial_session_name: Box::new(|| None),
                get_themes: Box::new(Vec::new), refresh_mcp_providers: None,
            }),
            on_shutdown: None, return_to_agents_view: false, force_fullscreen: false,
            agents_view_owns_startup_notices: false, session_depth: None, session_has_children: false,
            prompt_stash_store: None, prompt_stash_session_id: None,
        }).unwrap()))
    }

    #[tokio::test]
    async fn live_catalogue_adds_telegram_without_hiding_builtins_or_blocking_input() {
        let mode = test_mode();
        let (send, receive) = mpsc::sync_channel(64 * 1024);
        let mut provider = NativeAutocomplete::new(CombinedAutocompleteProvider::new(
            command_entries(mode.clone()), ".", None));
        provider.catalogue = Some(PendingCatalogue {
            receive, task: tokio::spawn(std::future::pending()),
            mode: Rc::downgrade(&mode), cwd: ".".into(), fd: None,
        });
        // An unfinished catalogue request cannot block the terminal owner.
        let start = std::time::Instant::now();
        let before = provider.get_suggestions(&["/eff".into()], 0, 4, false).unwrap();
        assert!(before.items.iter().any(|item| item.value == "effort"));
        assert!(start.elapsed() < Duration::from_secs(1));
        send.send(Ok(vec![
            wire::AgentConnectionSlashCommand { name: "telegram".into(), source: "extension".into(), ..Default::default() },
            wire::AgentConnectionSlashCommand { name: "model".into(), source: "extension".into(), ..Default::default() },
        ])).unwrap();
        let after = provider.get_suggestions(&["/tel".into()], 0, 4, false).unwrap();
        assert!(after.items.iter().any(|item| item.value == "telegram"));
        assert!(mode.borrow().is_recognized_slash_command("telegram"));
        let all = command_entries(mode.clone());
        assert_eq!(all.iter().filter(|item| item.name() == "model").count(), 1);
        // Settings reconfiguration keeps the learned catalogue.
        assert!(command_entries(mode).iter().any(|item| item.name() == "telegram"));
    }

    #[test]
    fn completions_translate_unicode_cursor_offsets_in_both_directions() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("résumé.txt"), "fixture").unwrap();
        let mut provider = NativeAutocomplete::new(CombinedAutocompleteProvider::new(
            vec![],
            root.path().to_str().unwrap(),
            None,
        ));
        let lines = vec!["explain é ./rés".to_string()];
        let suggestions = provider
            .get_suggestions(&lines, 0, lines[0].len(), true)
            .unwrap();
        let item = suggestions
            .items
            .iter()
            .find(|item| item.value.contains("résumé"))
            .unwrap();
        let completed =
            provider.apply_completion(&lines, 0, lines[0].len(), item, &suggestions.prefix);
        assert_eq!(completed.lines, ["explain é ./résumé.txt"]);
        assert_eq!(completed.cursor_col, completed.lines[0].len());
    }

    #[test]
    fn native_editor_shows_and_accepts_shared_slash_completions() {
        use pi_tui::tui::Component;
        let ui = Rc::new(RefCell::new(pi_tui::tui::TUI::new(
            Box::new(pi_tui::terminal::ProcessTerminal::new()),
            None,
        )));
        let mut editor = CustomEditor::new(ui, super::super::editor_theme(), Default::default());
        let command = autocomplete::CommandEntry::Slash(autocomplete::SlashCommand {
            name: "model".into(),
            aliases: vec![],
            description: Some("Select model".into()),
            argument_hint: None,
            source_tag: None,
            takes_argument: Some(true),
            get_argument_completions: None,
        });
        editor
            .editor_mut()
            .set_autocomplete_provider(Rc::new(RefCell::new(NativeAutocomplete::new(
                CombinedAutocompleteProvider::new(vec![command], ".", None),
            ))));
        for key in ["/", "m", "o"] {
            editor.handle_input(key);
        }
        editor.editor_mut().poll_autocomplete();
        assert!(editor.editor().is_showing_autocomplete());
        editor.handle_input("\t");
        assert_eq!(editor.editor().get_text(), "/model ");
    }
    // ==================== T14 terminal-parity suite ====================

    fn t14_a04_test_model(id: &str, provider: &str) -> wire::AgentConnectionModel {
        pi_ai::types::Model::new(id, id, "openai-completions", provider, "https://example.test")
    }

    /// A-04: `/model <Tab>` completes the cached model candidates like the
    /// TypeScript `getModelArgumentCompletions` wiring
    /// (interactive-mode.ts:1428-1432, model-autocomplete.ts).
    #[test]
    fn t14_a04_model_argument_completions_list_cached_candidates() {
        let mode = test_mode();
        mode.borrow_mut().connection_model_catalog = vec![
            t14_a04_test_model("mock-a", "prov-a"),
            t14_a04_test_model("mock-b", "prov-a"),
            t14_a04_test_model("mock-c", "prov-b"),
        ];
        let mut provider = NativeAutocomplete::new(CombinedAutocompleteProvider::new(
            command_entries(mode.clone()),
            ".",
            None,
        ));
        let suggestions = provider
            .get_suggestions(&["/model ".to_string()], 0, 7, false)
            .expect("suggestions");
        let values: Vec<&str> = suggestions.items.iter().map(|item| item.value.as_str()).collect();
        assert!(
            values.contains(&"prov-a/mock-a") && values.contains(&"prov-b/mock-c"),
            "expected cached model candidates for /model, got {values:?}"
        );
    }

    /// A-05: the `/effort` hint lists the live thinking levels like the
    /// TypeScript (`interactive-mode.ts:1434-1442`), and keeps the registry
    /// default when the model has none.
    #[test]
    fn t14_a05_effort_hint_lists_the_live_levels() {
        use pi_agent_core::types::ThinkingLevel;
        let mode = test_mode();
        mode.borrow_mut().connection_state = Some(local::AgentConnectionState {
            thinking_level: ThinkingLevel::Low,
            available_thinking_levels: vec![
                ThinkingLevel::Off,
                ThinkingLevel::Minimal,
                ThinkingLevel::Low,
                ThinkingLevel::Medium,
                ThinkingLevel::High,
            ],
            ..Default::default()
        });
        let hint = match command_entries(mode.clone())
            .into_iter()
            .find(|entry| entry.name() == "effort")
            .expect("effort entry")
        {
            autocomplete::CommandEntry::Slash(command) => command.argument_hint,
            _ => panic!("effort must be a slash entry"),
        };
        assert_eq!(
            hint.as_deref(),
            Some("[off/minimal/low/medium/high]"),
            "the effort hint must list the live levels"
        );

        // No thinking support keeps the registry default.
        let mode = test_mode();
        let fallback = match command_entries(mode)
            .into_iter()
            .find(|entry| entry.name() == "effort")
            .expect("effort entry")
        {
            autocomplete::CommandEntry::Slash(command) => command.argument_hint,
            _ => panic!("effort must be a slash entry"),
        };
        assert_eq!(fallback.as_deref(), Some("[level]"));
    }

}
