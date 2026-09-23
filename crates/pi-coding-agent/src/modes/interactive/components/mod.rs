pub mod agent_message;
pub mod armin;
pub mod assistant_message;
pub mod bash_execution;
pub mod bordered_loader;
pub mod branch_summary_message;
pub mod centered_overlay;
pub mod collapsible_error;
pub mod compaction_outcome_message;
pub mod compaction_summary_message;
pub mod config_selector;
pub mod configuration_menu;
pub mod context_tree_format;
pub mod conversation_components;
pub mod countdown_timer;
pub mod custom_editor;
pub mod custom_message;
pub mod daxnuts;
pub mod diff;
pub mod dynamic_border;
pub mod earendil_announcement;
pub mod edit_summary;
pub mod expandable_custom_message;
pub mod extension_editor;
pub mod extension_input;
pub mod extension_selector;
pub mod feature_hint;
pub mod footer;
pub mod heartbeat_manager;
pub mod injected_prompt_message;
pub mod ipython_cell;
pub mod keybinding_hints;
pub mod login_dialog;
pub mod menu_panel;
pub mod mermaid;
pub mod modal_back;
pub mod model_selector;
pub mod oauth_selector;
pub mod prime_onboarding_splash;
pub mod prime_team_selector;
pub mod prompt_highlight;
pub mod refinement_outcome_message;
pub mod scoped_models_selector;
pub mod settings_selector;
pub mod show_images_selector;
pub mod shell_completion;
pub mod side_question;
pub mod skill_invocation_message;
pub mod slash_command_message;
pub mod slash_command_result_message;
pub mod subagent_summary_line;
pub mod theme_selector;
pub mod thinking_selector;
pub mod tool_execution;
pub mod tool_panel;
pub mod tree_selector;
pub mod user_message;
pub mod user_message_selector;
pub mod visual_truncate;

// ---------------------------------------------------------------------------
// Port of index.ts - UI Components for extensions
//
// The TypeScript module is a pure re-export surface. The port re-exports the
// same names from the mapped modules so every `components/<file>.js` import
// target resolves to the same path segment.
// ---------------------------------------------------------------------------

pub use agent_message::AgentMessageComponent;
pub use armin::ArminComponent;
pub use assistant_message::AssistantMessageComponent;
pub use bash_execution::BashExecutionComponent;
pub use bordered_loader::BorderedLoader;
pub use branch_summary_message::BranchSummaryMessageComponent;
pub use compaction_outcome_message::{
    CompactionOutcomeMessageComponent, MalformedCompactionOutcomeMessageComponent,
};
pub use compaction_summary_message::CompactionSummaryMessageComponent;
pub use configuration_menu::{ConfigurationMenuComponent, ConfigurationMenuTab};
pub use custom_editor::CustomEditor;
pub use custom_message::CustomMessageComponent;
pub use daxnuts::DaxnutsComponent;
pub use diff::{render_diff, RenderDiffOptions};
pub use dynamic_border::DynamicBorder;
pub use extension_editor::ExtensionEditorComponent;
pub use extension_input::ExtensionInputComponent;
pub use extension_selector::ExtensionSelectorComponent;
pub use footer::FooterComponent;
pub use injected_prompt_message::{is_injected_prompt_message, InjectedPromptMessageComponent};
pub use ipython_cell::{
    get_ipython_code_from_args, IPythonCellComponent, IPythonCellContentBlock, IPythonCellState,
};
pub use keybinding_hints::{key_hint, key_text, raw_key_hint};
pub use login_dialog::LoginDialogComponent;
pub use model_selector::ModelSelectorComponent;
pub use oauth_selector::OAuthSelectorComponent;
pub use prime_onboarding_splash::PrimeOnboardingSplashComponent;
pub use scoped_models_selector::{ModelsCallbacks, ModelsConfig, ScopedModelsSelectorComponent};
pub use settings_selector::{SettingsCallbacks, SettingsConfig, SettingsSelectorComponent};
pub use show_images_selector::ShowImagesSelectorComponent;
pub use skill_invocation_message::SkillInvocationMessageComponent;
pub use subagent_summary_line::SubagentSummaryLine;
pub use theme_selector::ThemeSelectorComponent;
pub use thinking_selector::ThinkingSelectorComponent;
pub use tool_execution::{ToolExecutionComponent, ToolExecutionOptions};
pub use tool_panel::{tool_panel_content_width, tool_panel_line, ToolPanel, TOOL_PANEL_PADDING_X};
pub use tree_selector::TreeSelectorComponent;
pub use user_message::UserMessageComponent;
pub use user_message_selector::UserMessageSelectorComponent;
pub use visual_truncate::{truncate_to_visual_lines, VisualTruncateResult};

#[cfg(test)]
mod tests {
    use super::*;
    use super::tree_selector::FILTER_MODES;

    #[test]
    fn index_re_exports_resolve() {
        // index.ts re-exports these names; the port keeps the same names at the
        // same module path so every `components/<file>.js` import target resolves.
        let _ = std::any::type_name::<ArminComponent>();
        let _ = std::any::type_name::<DaxnutsComponent>();
        let _ = std::any::type_name::<ExtensionEditorComponent>();
        let _ = std::any::type_name::<ThinkingSelectorComponent>();
        let _ = std::any::type_name::<TreeSelectorComponent>();
        let _ = std::any::type_name::<heartbeat_manager::HeartbeatManagerComponent>();
        let _ = std::any::type_name::<heartbeat_manager::HeartbeatManagerOptions>();
        let _ = std::any::type_name::<prime_team_selector::PrimeTeamSelectorComponent>();
        let _ =
            std::any::type_name::<refinement_outcome_message::RefinementOutcomeMessageComponent>();
        let _ = std::any::type_name::<
            refinement_outcome_message::MalformedRefinementOutcomeMessageComponent,
        >();
        let _ = std::any::type_name::<slash_command_message::SlashCommandMessageComponent>();
        let _ = std::any::type_name::<dyn expandable_custom_message::ExpandableCustomMessageBox>();
        let _ = std::any::type_name::<dyn modal_back::BackGuardInput>();
        assert_eq!(
            FILTER_MODES,
            ["default", "no-tools", "user-only", "labeled-only", "all"]
        );
    }
}
