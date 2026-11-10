//! Port of packages/coding-agent/src/modes/interactive/components/settings-selector.ts

use std::cell::RefCell;
use std::rc::Rc;

use pi_tui::components::select_list::{SelectItem, SelectList, SelectListLayoutOptions};
use pi_tui::components::settings_list::{
    SettingItem, SettingsList, SettingsListOptions, SettingsListTheme as TuiSettingsListTheme,
    SubmenuDone,
};
use pi_tui::components::spacer::Spacer;
use pi_tui::components::text::Text;
use pi_tui::tui::Component;

use crate::core::session_action_store::IdleEvictionMinutes;
use crate::core::settings_manager::WarningSettings;
use crate::modes::interactive::theme::theme::{
    get_select_list_theme, get_settings_list_theme, theme,
};

use super::show_images_selector::{to_tui_select_list_theme, DynamicBorder};

/// `SETTINGS_SUBMENU_SELECT_LIST_LAYOUT`.
pub fn settings_submenu_select_list_layout() -> SelectListLayoutOptions {
    SelectListLayoutOptions {
        min_primary_column_width: Some(12),
        max_primary_column_width: Some(32),
        ..Default::default()
    }
}

/// `THINKING_DESCRIPTIONS`.
pub const THINKING_DESCRIPTIONS: [(&str, &str); 7] = [
    ("off", "No reasoning"),
    ("minimal", "Very brief reasoning"),
    ("low", "Light reasoning"),
    ("medium", "Moderate reasoning"),
    ("high", "Deep reasoning"),
    ("xhigh", "Very deep reasoning"),
    ("max", "Maximum reasoning"),
];

fn thinking_description(level: &str) -> &'static str {
    THINKING_DESCRIPTIONS
        .iter()
        .find(|(key, _)| *key == level)
        .map(|(_, value)| *value)
        .unwrap_or("")
}

/// Port of `SettingsConfig`.
#[derive(Debug, Clone, PartialEq)]
pub struct SettingsConfig {
    pub auto_compact: bool,
    pub idle_eviction_minutes: IdleEvictionMinutes,
    pub show_images: bool,
    pub auto_resize_images: bool,
    pub block_images: bool,
    pub enable_skill_commands: bool,
    pub enable_builtin_skills: bool,
    pub steering_mode: String,
    pub follow_up_mode: String,
    pub transport: String,
    pub thinking_level: String,
    pub available_thinking_levels: Vec<String>,
    pub current_theme: String,
    pub available_themes: Vec<String>,
    pub hide_thinking_block: bool,
    pub mermaid_rendering_mode: String,
    pub tree_filter_mode: String,
    pub show_hardware_cursor: bool,
    pub editor_padding_x: f64,
    pub autocomplete_max_visible: f64,
    pub quiet_startup: bool,
    pub clear_on_shrink: bool,
    pub show_terminal_progress: bool,
    pub fullscreen: bool,
    pub warnings: WarningSettings,
}

/// Port of `SettingsCallbacks`.
pub struct SettingsCallbacks {
    pub on_auto_compact_change: Box<dyn FnMut(bool)>,
    pub on_idle_eviction_minutes_change: Box<dyn FnMut(IdleEvictionMinutes)>,
    pub on_show_images_change: Box<dyn FnMut(bool)>,
    pub on_auto_resize_images_change: Box<dyn FnMut(bool)>,
    pub on_block_images_change: Box<dyn FnMut(bool)>,
    pub on_enable_skill_commands_change: Box<dyn FnMut(bool)>,
    pub on_enable_builtin_skills_change: Box<dyn FnMut(bool)>,
    pub on_steering_mode_change: Box<dyn FnMut(String)>,
    pub on_follow_up_mode_change: Box<dyn FnMut(String)>,
    pub on_transport_change: Box<dyn FnMut(String)>,
    pub on_thinking_level_change: Box<dyn FnMut(String)>,
    pub on_theme_change: Box<dyn FnMut(String)>,
    pub on_theme_preview: Option<Box<dyn FnMut(String)>>,
    pub on_hide_thinking_block_change: Box<dyn FnMut(bool)>,
    pub on_mermaid_rendering_mode_change: Box<dyn FnMut(String)>,
    pub on_tree_filter_mode_change: Box<dyn FnMut(String)>,
    pub on_show_hardware_cursor_change: Box<dyn FnMut(bool)>,
    pub on_editor_padding_x_change: Box<dyn FnMut(f64)>,
    pub on_autocomplete_max_visible_change: Box<dyn FnMut(f64)>,
    pub on_quiet_startup_change: Box<dyn FnMut(bool)>,
    pub on_clear_on_shrink_change: Box<dyn FnMut(bool)>,
    pub on_show_terminal_progress_change: Box<dyn FnMut(bool)>,
    pub on_fullscreen_change: Box<dyn FnMut(bool)>,
    pub on_warnings_change: Box<dyn FnMut(WarningSettings)>,
    pub on_cancel: Box<dyn FnMut()>,
}

/// Wrapper used by the submenu factories. The TypeScript submenus close over the
/// single `callbacks` object; Rust closure fields cannot be aliased, so the
/// callbacks live in a shared cell and each submenu borrows it at call time.
pub struct SharedCallbacks(Rc<RefCell<SettingsCallbacks>>);

impl SharedCallbacks {
    fn new(callbacks: SettingsCallbacks) -> Self {
        Self(Rc::new(RefCell::new(callbacks)))
    }

    fn rc(&self) -> Rc<RefCell<SettingsCallbacks>> {
        Rc::clone(&self.0)
    }
}

/// `Warnings` and `onWarningsChange` share the live value, exactly like the
/// `currentWarnings` binding in the TypeScript constructor.
#[derive(Clone)]
pub struct SharedWarnings(Rc<RefCell<WarningSettings>>);

impl SharedWarnings {
    fn new(warnings: WarningSettings) -> Self {
        Self(Rc::new(RefCell::new(warnings)))
    }

    fn get(&self) -> WarningSettings {
        self.0.borrow().clone()
    }

    fn set(&self, warnings: WarningSettings) {
        *self.0.borrow_mut() = warnings;
    }
}

fn true_false_item(id: &str, label: &str, description: &str, value: bool) -> SettingItem {
    SettingItem {
        id: id.to_string(),
        label: label.to_string(),
        description: Some(description.to_string()),
        current_value: if value {
            "true".to_string()
        } else {
            "false".to_string()
        },
        values: Some(vec!["true".to_string(), "false".to_string()]),
        submenu: None,
    }
}

/// `String(config.idleEvictionMinutes)`.
fn idle_eviction_minutes_text(value: IdleEvictionMinutes) -> String {
    match value {
        IdleEvictionMinutes::Minutes(minutes) => js_number_string(minutes),
        IdleEvictionMinutes::Off => "off".to_string(),
    }
}

/// `String(n)` for a JS number.
fn js_number_string(value: f64) -> String {
    if value.is_finite() && value.fract() == 0.0 {
        format!("{}", value as i64)
    } else {
        format!("{value}")
    }
}

/// Port of `WarningSettingsSubmenu`.
pub struct WarningSettingsSubmenu {
    settings_list: SettingsList,
    state: WarningSettings,
}

impl WarningSettingsSubmenu {
    pub fn new(
        warnings: WarningSettings,
        on_change: Rc<RefCell<Box<dyn FnMut(WarningSettings)>>>,
        on_cancel: Box<dyn FnMut()>,
    ) -> Self {
        let state = warnings;

        let items: Vec<SettingItem> = vec![SettingItem {
            id: "anthropic-extra-usage".to_string(),
            label: "Anthropic extra usage".to_string(),
            description: Some(
                "Warn when Anthropic subscription auth may use paid extra usage".to_string(),
            ),
            current_value: if state.anthropic_extra_usage.unwrap_or(true) {
                "true".to_string()
            } else {
                "false".to_string()
            },
            values: Some(vec!["true".to_string(), "false".to_string()]),
            submenu: None,
        }];

        let item_count = items.len();
        let state_for_change = state.clone();
        let settings_list = SettingsList::new(
            items,
            item_count.min(10),
            to_tui_settings_list_theme(get_settings_list_theme()),
            Box::new(move |id: &str, new_value: &str| {
                if id == "anthropic-extra-usage" {
                    let updated = WarningSettings {
                        anthropic_extra_usage: Some(new_value == "true"),
                        ..state_for_change.clone()
                    };
                    (*on_change.borrow_mut())(updated);
                }
            }),
            on_cancel,
            SettingsListOptions::default(),
        );

        Self {
            settings_list,
            state,
        }
    }

    /// The `state` this submenu tracks (`this.state`).
    pub fn state(&self) -> &WarningSettings {
        &self.state
    }
}

impl Component for WarningSettingsSubmenu {
    fn render(&mut self, width: f64) -> Vec<String> {
        self.settings_list.render(width)
    }

    fn handle_input(&mut self, data: &str) {
        self.settings_list.handle_input(data);
    }

    fn invalidate(&mut self) {
        self.settings_list.invalidate();
    }
}

/// Port of `SelectSubmenu`.
pub struct SelectSubmenu {
    leading: Vec<Box<dyn Component>>,
    select_list: SelectList,
    trailing_spacer: Spacer,
    trailing_hint: Text,
}

impl SelectSubmenu {
    pub fn new(
        title: &str,
        description: &str,
        options: Vec<SelectItem>,
        current_value: &str,
        on_select: Box<dyn FnMut(&str)>,
        on_cancel: Box<dyn FnMut()>,
        on_selection_change: Option<Box<dyn FnMut(&str)>>,
    ) -> Self {
        let mut leading: Vec<Box<dyn Component>> = Vec::new();

        leading.push(Box::new(Text::new(
            theme().bold(&theme().fg("accent", title)),
            0,
            0,
            None,
        )));

        if !description.is_empty() {
            leading.push(Box::new(Spacer::new(1)));
            leading.push(Box::new(Text::new(
                theme().fg("muted", description),
                0,
                0,
                None,
            )));
        }

        leading.push(Box::new(Spacer::new(1)));

        let option_count = options.len();
        let current_value = current_value.to_string();
        let mut select_list = SelectList::new(
            options,
            option_count.min(10),
            to_tui_select_list_theme(get_select_list_theme()),
            settings_submenu_select_list_layout(),
        );

        let current_index = select_list
            .filtered_items()
            .iter()
            .position(|option| option.value == current_value);
        if let Some(current_index) = current_index {
            select_list.set_selected_index(current_index);
        }

        let mut on_select = on_select;
        select_list.on_select = Some(Box::new(move |item: &SelectItem| {
            on_select(&item.value);
        }));

        select_list.on_cancel = Some(on_cancel);

        if let Some(on_selection_change) = on_selection_change {
            let mut on_selection_change = on_selection_change;
            select_list.on_selection_change = Some(Box::new(move |item: &SelectItem| {
                on_selection_change(&item.value);
            }));
        }

        Self {
            leading,
            select_list,
            trailing_spacer: Spacer::new(1),
            trailing_hint: Text::new(
                theme().fg("dim", "  Enter to select \u{00b7} esc to go back"),
                0,
                0,
                None,
            ),
        }
    }
}

impl Component for SelectSubmenu {
    fn render(&mut self, width: f64) -> Vec<String> {
        let mut lines: Vec<String> = Vec::new();
        for child in self.leading.iter_mut() {
            lines.extend(child.render(width));
        }
        lines.extend(self.select_list.render(width));
        lines.extend(self.trailing_spacer.render(width));
        lines.extend(self.trailing_hint.render(width));
        lines
    }

    fn handle_input(&mut self, data: &str) {
        self.select_list.handle_input(data);
    }

    fn invalidate(&mut self) {
        for child in self.leading.iter_mut() {
            child.invalidate();
        }
        self.select_list.invalidate();
        self.trailing_spacer.invalidate();
        self.trailing_hint.invalidate();
    }
}

/// `getSettingsListTheme()` closures are `Send + Sync`; the pi-tui component is
/// not.
fn to_tui_settings_list_theme(
    theme: crate::modes::interactive::theme::theme::SettingsListTheme,
) -> TuiSettingsListTheme {
    TuiSettingsListTheme {
        label: Box::new(move |text: &str, selected: bool| (theme.label)(text, selected)),
        value: Box::new(move |text: &str, selected: bool| (theme.value)(text, selected)),
        description: Box::new(move |text: &str| (theme.description)(text)),
        cursor: theme.cursor,
        hint: Box::new(move |text: &str| (theme.hint)(text)),
    }
}

/// Port of `SettingsSelectorComponent`.
pub struct SettingsSelectorComponent {
    leading_border: DynamicBorder,
    settings_list: SettingsList,
    trailing_border: DynamicBorder,
}

impl SettingsSelectorComponent {
    pub fn new(config: SettingsConfig, callbacks: SettingsCallbacks) -> Self {
        let callbacks = SharedCallbacks::new(callbacks);
        let current_warnings = SharedWarnings::new(config.warnings.clone());

        let mut idle_eviction_values: Vec<f64> = vec![30.0, 60.0, 90.0, 180.0, 360.0];
        if let IdleEvictionMinutes::Minutes(minutes) = config.idle_eviction_minutes {
            if !idle_eviction_values.contains(&minutes) {
                idle_eviction_values.push(minutes);
                idle_eviction_values
                    .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            }
        }

        let mut items: Vec<SettingItem> = vec![
            true_false_item(
                "autocompact",
                "Auto-compact",
                "Automatically compact context when it gets too large",
                config.auto_compact,
            ),
            SettingItem {
                id: "idle-eviction-minutes".to_string(),
                label: "Idle worker eviction".to_string(),
                description: Some(
                    "Stop fully idle agent trees after this many minutes (global daemon policy)"
                        .to_string(),
                ),
                current_value: idle_eviction_minutes_text(config.idle_eviction_minutes),
                values: Some(
                    std::iter::once("off".to_string())
                        .chain(idle_eviction_values.iter().map(|value| js_number_string(*value)))
                        .collect(),
                ),
                submenu: None,
            },
            SettingItem {
                id: "steering-mode".to_string(),
                label: "Steering mode".to_string(),
                description: Some(
                    "Enter while streaming queues steering messages. 'one-at-a-time': deliver one, wait for response. 'all': deliver all at once.".to_string(),
                ),
                current_value: config.steering_mode.clone(),
                values: Some(vec!["one-at-a-time".to_string(), "all".to_string()]),
                submenu: None,
            },
            SettingItem {
                id: "follow-up-mode".to_string(),
                label: "Follow-up mode".to_string(),
                description: Some(
                    "Alt+Enter queues follow-up messages until agent stops. 'one-at-a-time': deliver one, wait for response. 'all': deliver all at once.".to_string(),
                ),
                current_value: config.follow_up_mode.clone(),
                values: Some(vec!["one-at-a-time".to_string(), "all".to_string()]),
                submenu: None,
            },
            SettingItem {
                id: "transport".to_string(),
                label: "Transport".to_string(),
                description: Some(
                    "Preferred transport for providers that support multiple transports".to_string(),
                ),
                current_value: config.transport.clone(),
                values: Some(vec![
                    "sse".to_string(),
                    "websocket".to_string(),
                    "websocket-cached".to_string(),
                    "auto".to_string(),
                ]),
                submenu: None,
            },
            true_false_item(
                "hide-thinking",
                "Hide thinking",
                "Hide thinking blocks in assistant responses",
                config.hide_thinking_block,
            ),
            SettingItem {
                id: "mermaid-rendering".to_string(),
                label: "Mermaid diagrams".to_string(),
                description: Some("Render Mermaid code blocks as Unicode diagrams".to_string()),
                current_value: config.mermaid_rendering_mode.clone(),
                values: Some(vec!["off".to_string(), "final".to_string(), "streaming".to_string()]),
                submenu: None,
            },
            true_false_item(
                "quiet-startup",
                "Quiet startup",
                "Disable verbose printing at startup",
                config.quiet_startup,
            ),
            SettingItem {
                id: "tree-filter-mode".to_string(),
                label: "Tree filter mode".to_string(),
                description: Some("Default filter when opening /tree".to_string()),
                current_value: config.tree_filter_mode.clone(),
                values: Some(vec![
                    "default".to_string(),
                    "no-tools".to_string(),
                    "user-only".to_string(),
                    "labeled-only".to_string(),
                    "all".to_string(),
                ]),
                submenu: None,
            },
            {
                // `warnings` submenu
                let warnings = SharedWarnings::clone(&current_warnings);
                let shared = callbacks.rc();
                SettingItem {
                    id: "warnings".to_string(),
                    label: "Warnings".to_string(),
                    description: Some("Enable or disable individual warnings".to_string()),
                    current_value: "configure".to_string(),
                    values: None,
                    submenu: Some(Box::new(
                        move |_current_value: &str, done: SubmenuDone| -> Box<dyn Component> {
                            let warnings = warnings.clone();
                            let shared = Rc::clone(&shared);
                            let done_for_cancel = Rc::clone(&done);
                            let on_change: Rc<RefCell<Box<dyn FnMut(WarningSettings)>>> =
                                Rc::new(RefCell::new(Box::new(move |warnings: WarningSettings| {
                                    (*shared.borrow_mut().on_warnings_change)(warnings);
                                })));
                            Box::new(WarningSettingsSubmenu::new(
                                warnings.get(),
                                on_change,
                                Box::new(move || done_for_cancel(None)),
                            ))
                        },
                    )),
                }
            },
            {
                // `thinking` submenu
                let available_levels = config.available_thinking_levels.clone();
                let current_level = config.thinking_level.clone();
                let shared = callbacks.rc();
                SettingItem {
                    id: "thinking".to_string(),
                    label: "Thinking level".to_string(),
                    description: Some("Reasoning depth for thinking-capable models".to_string()),
                    current_value: current_level,
                    values: None,
                    submenu: Some(Box::new(
                        move |current_value: &str, done: SubmenuDone| -> Box<dyn Component> {
                            let options: Vec<SelectItem> = available_levels
                                .iter()
                                .map(|level| SelectItem {
                                    value: level.clone(),
                                    label: level.clone(),
                                    description: Some(thinking_description(level).to_string()),
                                    ..Default::default()
                                })
                                .collect();
                            let shared_select = Rc::clone(&shared);
                            let done_select = Rc::clone(&done);
                            let done_cancel = Rc::clone(&done);
                            Box::new(SelectSubmenu::new(
                                "Thinking Level",
                                "Select reasoning depth for thinking-capable models",
                                options,
                                current_value,
                                Box::new(move |value: &str| {
                                    (*shared_select.borrow_mut().on_thinking_level_change)(value.to_string());
                                    done_select(Some(value.to_string()));
                                }),
                                Box::new(move || done_cancel(None)),
                                None,
                            ))
                        },
                    )),
                }
            },
            {
                // `theme` submenu
                let shared = callbacks.rc();
                let available_themes = config.available_themes.clone();
                SettingItem {
                    id: "theme".to_string(),
                    label: "Theme".to_string(),
                    description: Some("Color theme for the interface".to_string()),
                    current_value: config.current_theme.clone(),
                    values: None,
                    submenu: Some(Box::new(
                        move |current_value: &str, done: SubmenuDone| -> Box<dyn Component> {
                            let options: Vec<SelectItem> = available_themes
                                .iter()
                                .map(|name| SelectItem {
                                    value: name.clone(),
                                    label: name.clone(),
                                    ..Default::default()
                                })
                                .collect();
                            let shared_select = Rc::clone(&shared);
                            let shared_preview = Rc::clone(&shared);
                            let shared_cancel = Rc::clone(&shared);
                            let done_select = Rc::clone(&done);
                            let done_cancel = Rc::clone(&done);
                            let preview_cancel_value = current_value.to_string();
                            Box::new(SelectSubmenu::new(
                                "Theme",
                                "Select color theme",
                                options,
                                current_value,
                                Box::new(move |value: &str| {
                                    (*shared_select.borrow_mut().on_theme_change)(value.to_string());
                                    done_select(Some(value.to_string()));
                                }),
                                Box::new(move || {
                                    // Restore original theme on cancel
                                    if let Some(preview) = shared_cancel.borrow_mut().on_theme_preview.as_mut() {
                                        preview(preview_cancel_value.clone());
                                    }
                                    done_cancel(None);
                                }),
                                Some(Box::new(move |value: &str| {
                                    // Preview theme on selection change
                                    if let Some(preview) = shared_preview.borrow_mut().on_theme_preview.as_mut() {
                                        preview(value.to_string());
                                    }
                                })),
                            ))
                        },
                    )),
                }
            },
        ];

        items.insert(
            1,
            true_false_item(
                "show-images",
                "Show image metadata",
                "Show image type and dimensions in terminal",
                config.show_images,
            ),
        );

        // Image auto-resize toggle (always available, affects both attached and
        // read images)
        items.insert(
            2,
            true_false_item(
                "auto-resize-images",
                "Auto-resize images",
                "Resize large images to 2000x2000 max for better model compatibility",
                config.auto_resize_images,
            ),
        );

        // Block images toggle (always available, insert after auto-resize-images)
        let auto_resize_index = items
            .iter()
            .position(|item| item.id == "auto-resize-images")
            .unwrap_or(0);
        items.insert(
            auto_resize_index + 1,
            true_false_item(
                "block-images",
                "Block images",
                "Prevent images from being sent to LLM providers",
                config.block_images,
            ),
        );

        // Skill commands toggle (insert after block-images)
        let block_images_index = items
            .iter()
            .position(|item| item.id == "block-images")
            .unwrap_or(0);
        items.insert(
            block_images_index + 1,
            true_false_item(
                "skill-commands",
                "Skill commands",
                "Register skills as /skill:name commands",
                config.enable_skill_commands,
            ),
        );

        // Built-in skills toggle (insert after skill-commands)
        let skill_commands_index = items
            .iter()
            .position(|item| item.id == "skill-commands")
            .unwrap_or(0);
        items.insert(
            skill_commands_index + 1,
            true_false_item(
                "builtin-skills",
                "Built-in skills",
                "Load built-in skills shipped with Optimus (takes effect after reload)",
                config.enable_builtin_skills,
            ),
        );

        // Hardware cursor toggle (insert after builtin-skills)
        let builtin_skills_index = items
            .iter()
            .position(|item| item.id == "builtin-skills")
            .unwrap_or(0);
        items.insert(
            builtin_skills_index + 1,
            true_false_item(
                "show-hardware-cursor",
                "Show hardware cursor",
                "Show the terminal cursor while still positioning it for IME support",
                config.show_hardware_cursor,
            ),
        );

        // Editor padding toggle (insert after show-hardware-cursor)
        let hardware_cursor_index = items
            .iter()
            .position(|item| item.id == "show-hardware-cursor")
            .unwrap_or(0);
        items.insert(
            hardware_cursor_index + 1,
            SettingItem {
                id: "editor-padding".to_string(),
                label: "Editor padding".to_string(),
                description: Some("Horizontal padding for input editor (0-3)".to_string()),
                current_value: js_number_string(config.editor_padding_x),
                values: Some(vec![
                    "0".to_string(),
                    "1".to_string(),
                    "2".to_string(),
                    "3".to_string(),
                ]),
                submenu: None,
            },
        );

        // Autocomplete max visible toggle (insert after editor-padding)
        let editor_padding_index = items
            .iter()
            .position(|item| item.id == "editor-padding")
            .unwrap_or(0);
        items.insert(
            editor_padding_index + 1,
            SettingItem {
                id: "autocomplete-max-visible".to_string(),
                label: "Autocomplete max items".to_string(),
                description: Some("Max visible items in autocomplete dropdown (3-20)".to_string()),
                current_value: js_number_string(config.autocomplete_max_visible),
                values: Some(vec![
                    "3".to_string(),
                    "5".to_string(),
                    "7".to_string(),
                    "10".to_string(),
                    "15".to_string(),
                    "20".to_string(),
                ]),
                submenu: None,
            },
        );

        // Clear on shrink toggle (insert after autocomplete-max-visible)
        let autocomplete_index = items
            .iter()
            .position(|item| item.id == "autocomplete-max-visible")
            .unwrap_or(0);
        items.insert(
            autocomplete_index + 1,
            true_false_item(
                "clear-on-shrink",
                "Clear on shrink",
                "Clear empty rows when content shrinks (may cause flicker)",
                config.clear_on_shrink,
            ),
        );

        // Terminal progress toggle (insert after clear-on-shrink)
        let clear_on_shrink_index = items
            .iter()
            .position(|item| item.id == "clear-on-shrink")
            .unwrap_or(0);
        items.insert(
            clear_on_shrink_index + 1,
            true_false_item(
                "terminal-progress",
                "Terminal progress",
                "Show OSC 9;4 progress indicators in the terminal tab bar",
                config.show_terminal_progress,
            ),
        );

        // Fullscreen toggle (insert after terminal-progress)
        let terminal_progress_index = items
            .iter()
            .position(|item| item.id == "terminal-progress")
            .unwrap_or(0);
        items.insert(
            terminal_progress_index + 1,
            true_false_item(
                "fullscreen",
                "Fullscreen rendering",
                "Alternate-screen UI with scrollable transcript and pinned prompt",
                config.fullscreen,
            ),
        );

        let settings_list = SettingsList::new(
            items,
            10,
            to_tui_settings_list_theme(get_settings_list_theme()),
            Box::new({
                let shared = callbacks.rc();
                move |id: &str, new_value: &str| {
                    let mut callbacks = shared.borrow_mut();
                    match id {
                        "autocompact" => (callbacks.on_auto_compact_change)(new_value == "true"),
                        "idle-eviction-minutes" => {
                            let value = if new_value == "off" {
                                IdleEvictionMinutes::Off
                            } else {
                                IdleEvictionMinutes::Minutes(js_string_to_number(new_value))
                            };
                            (callbacks.on_idle_eviction_minutes_change)(value);
                        }
                        "show-images" => (callbacks.on_show_images_change)(new_value == "true"),
                        "auto-resize-images" => {
                            (callbacks.on_auto_resize_images_change)(new_value == "true")
                        }
                        "block-images" => (callbacks.on_block_images_change)(new_value == "true"),
                        "skill-commands" => {
                            (callbacks.on_enable_skill_commands_change)(new_value == "true")
                        }
                        "builtin-skills" => {
                            (callbacks.on_enable_builtin_skills_change)(new_value == "true")
                        }
                        "steering-mode" => {
                            (callbacks.on_steering_mode_change)(new_value.to_string())
                        }
                        "follow-up-mode" => {
                            (callbacks.on_follow_up_mode_change)(new_value.to_string())
                        }
                        "transport" => (callbacks.on_transport_change)(new_value.to_string()),
                        "hide-thinking" => {
                            (callbacks.on_hide_thinking_block_change)(new_value == "true")
                        }
                        "mermaid-rendering" => {
                            (callbacks.on_mermaid_rendering_mode_change)(new_value.to_string())
                        }
                        "quiet-startup" => (callbacks.on_quiet_startup_change)(new_value == "true"),
                        "tree-filter-mode" => {
                            (callbacks.on_tree_filter_mode_change)(new_value.to_string())
                        }
                        "show-hardware-cursor" => {
                            (callbacks.on_show_hardware_cursor_change)(new_value == "true")
                        }
                        "editor-padding" => {
                            (callbacks.on_editor_padding_x_change)(js_string_to_number(new_value))
                        }
                        "autocomplete-max-visible" => (callbacks
                            .on_autocomplete_max_visible_change)(
                            js_string_to_number(new_value)
                        ),
                        "clear-on-shrink" => {
                            (callbacks.on_clear_on_shrink_change)(new_value == "true")
                        }
                        "terminal-progress" => {
                            (callbacks.on_show_terminal_progress_change)(new_value == "true")
                        }
                        "fullscreen" => (callbacks.on_fullscreen_change)(new_value == "true"),
                        _ => {}
                    }
                }
            }),
            Box::new({
                let shared = callbacks.rc();
                move || {
                    (*shared.borrow_mut().on_cancel)();
                }
            }),
            SettingsListOptions {
                enable_search: Some(true),
            },
        );

        Self {
            leading_border: DynamicBorder::new(None),
            settings_list,
            trailing_border: DynamicBorder::new(None),
        }
    }

    /// Port of `getSettingsList`.
    pub fn get_settings_list(&mut self) -> &mut SettingsList {
        &mut self.settings_list
    }
}

/// `parseInt(value, 10)`.
fn js_string_to_number(value: &str) -> f64 {
    let trimmed = value.trim_start();
    let mut digits = String::new();
    for (index, ch) in trimmed.chars().enumerate() {
        if index == 0 && (ch == '-' || ch == '+') {
            digits.push(ch);
            continue;
        }
        if ch.is_ascii_digit() {
            digits.push(ch);
            continue;
        }
        break;
    }
    digits.parse::<f64>().unwrap_or(f64::NAN)
}

impl Component for SettingsSelectorComponent {
    fn render(&mut self, width: f64) -> Vec<String> {
        let mut lines = self.leading_border.render(width);
        lines.extend(self.settings_list.render(width));
        lines.extend(self.trailing_border.render(width));
        lines
    }

    fn handle_input(&mut self, data: &str) {
        self.settings_list.handle_input(data);
    }

    fn invalidate(&mut self) {
        self.leading_border.invalidate();
        self.settings_list.invalidate();
        self.trailing_border.invalidate();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modes::interactive::theme::theme::init_theme;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;

    fn config() -> SettingsConfig {
        init_theme(Some("prime"), false);
        SettingsConfig {
            auto_compact: true,
            idle_eviction_minutes: IdleEvictionMinutes::Minutes(90.0),
            show_images: true,
            auto_resize_images: true,
            block_images: false,
            enable_skill_commands: true,
            enable_builtin_skills: true,
            steering_mode: "one-at-a-time".to_string(),
            follow_up_mode: "one-at-a-time".to_string(),
            transport: "sse".to_string(),
            thinking_level: "medium".to_string(),
            available_thinking_levels: vec![
                "off".to_string(),
                "medium".to_string(),
                "high".to_string(),
            ],
            current_theme: "dark".to_string(),
            available_themes: vec!["dark".to_string(), "light".to_string()],
            hide_thinking_block: false,
            mermaid_rendering_mode: "final".to_string(),
            tree_filter_mode: "default".to_string(),
            show_hardware_cursor: false,
            editor_padding_x: 1.0,
            autocomplete_max_visible: 10.0,
            quiet_startup: false,
            clear_on_shrink: false,
            show_terminal_progress: true,
            fullscreen: false,
            warnings: WarningSettings {
                anthropic_extra_usage: Some(true),
            },
        }
    }

    fn callbacks() -> SettingsCallbacks {
        SettingsCallbacks {
            on_auto_compact_change: Box::new(|_| {}),
            on_idle_eviction_minutes_change: Box::new(|_| {}),
            on_show_images_change: Box::new(|_| {}),
            on_auto_resize_images_change: Box::new(|_| {}),
            on_block_images_change: Box::new(|_| {}),
            on_enable_skill_commands_change: Box::new(|_| {}),
            on_enable_builtin_skills_change: Box::new(|_| {}),
            on_steering_mode_change: Box::new(|_| {}),
            on_follow_up_mode_change: Box::new(|_| {}),
            on_transport_change: Box::new(|_| {}),
            on_thinking_level_change: Box::new(|_| {}),
            on_theme_change: Box::new(|_| {}),
            on_theme_preview: Some(Box::new(|_| {})),
            on_hide_thinking_block_change: Box::new(|_| {}),
            on_mermaid_rendering_mode_change: Box::new(|_| {}),
            on_tree_filter_mode_change: Box::new(|_| {}),
            on_show_hardware_cursor_change: Box::new(|_| {}),
            on_editor_padding_x_change: Box::new(|_| {}),
            on_autocomplete_max_visible_change: Box::new(|_| {}),
            on_quiet_startup_change: Box::new(|_| {}),
            on_clear_on_shrink_change: Box::new(|_| {}),
            on_show_terminal_progress_change: Box::new(|_| {}),
            on_fullscreen_change: Box::new(|_| {}),
            on_warnings_change: Box::new(|_| {}),
            on_cancel: Box::new(|| {}),
        }
    }

    fn item_ids(selector: &mut SettingsSelectorComponent) -> Vec<String> {
        selector
            .get_settings_list()
            .items()
            .iter()
            .map(|item| item.id.clone())
            .collect()
    }

    #[test]
    fn thinking_descriptions_match_the_typescript_table() {
        assert_eq!(THINKING_DESCRIPTIONS.len(), 7);
        assert_eq!(thinking_description("off"), "No reasoning");
        assert_eq!(thinking_description("xhigh"), "Very deep reasoning");
        assert_eq!(thinking_description("max"), "Maximum reasoning");
        assert_eq!(thinking_description("nope"), "");
    }

    #[test]
    fn items_are_inserted_in_the_typescript_order() {
        let mut selector = SettingsSelectorComponent::new(config(), callbacks());
        let ids = item_ids(&mut selector);
        let expected_prefix = vec![
            "autocompact",
            "show-images",
            "auto-resize-images",
            "block-images",
            "skill-commands",
            "builtin-skills",
            "show-hardware-cursor",
            "editor-padding",
            "autocomplete-max-visible",
            "clear-on-shrink",
            "terminal-progress",
            "fullscreen",
        ];
        assert_eq!(&ids[..expected_prefix.len()], expected_prefix.as_slice());
        assert!(ids.contains(&"idle-eviction-minutes".to_string()));
        assert_eq!(ids[ids.len() - 3..], ["warnings", "thinking", "theme"]);
    }

    #[test]
    fn idle_eviction_values_include_an_unlisted_current_value_in_sorted_order() {
        let mut config = config();
        config.idle_eviction_minutes = IdleEvictionMinutes::Minutes(45.0);
        let mut selector = SettingsSelectorComponent::new(config, callbacks());
        let item = selector
            .get_settings_list()
            .items()
            .iter()
            .find(|item| item.id == "idle-eviction-minutes")
            .unwrap()
            .clone();
        assert_eq!(item.current_value, "45");
        assert_eq!(
            item.values.clone().unwrap(),
            vec![
                "off".to_string(),
                "30".to_string(),
                "45".to_string(),
                "60".to_string(),
                "90".to_string(),
                "180".to_string(),
                "360".to_string()
            ]
        );
    }

    #[test]
    fn an_off_value_keeps_the_off_current_value() {
        let mut config = config();
        config.idle_eviction_minutes = IdleEvictionMinutes::Off;
        let mut selector = SettingsSelectorComponent::new(config, callbacks());
        let item = selector
            .get_settings_list()
            .items()
            .iter()
            .find(|item| item.id == "idle-eviction-minutes")
            .unwrap()
            .clone();
        assert_eq!(item.current_value, "off");
        assert_eq!(item.values.clone().unwrap()[0], "off");
    }

    #[test]
    fn parse_int_matches_javascript() {
        assert_eq!(js_string_to_number("10"), 10.0);
        assert_eq!(js_string_to_number("  -3abc"), -3.0);
        assert!(js_string_to_number("abc").is_nan());
        assert!(js_string_to_number("").is_nan());
        assert!(js_string_to_number("+").is_nan());
        assert_eq!(js_string_to_number("12.5"), 12.0);
    }

    #[test]
    fn the_list_cycles_values_and_reports_the_change() {
        let calls: Arc<AtomicUsize> = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&calls);
        let mut callbacks = callbacks();
        callbacks.on_auto_compact_change = Box::new(move |_| {
            counter.fetch_add(1, Ordering::SeqCst);
        });
        let mut selector = SettingsSelectorComponent::new(config(), callbacks);
        // Enter on the selected row (Auto-compact) cycles true -> false.
        selector.handle_input("\r");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn cancel_is_forwarded() {
        let cancelled = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&cancelled);
        let mut callbacks = callbacks();
        callbacks.on_cancel = Box::new(move || flag.store(true, Ordering::SeqCst));
        let mut selector = SettingsSelectorComponent::new(config(), callbacks);
        selector.handle_input("\u{1b}");
        assert!(cancelled.load(Ordering::SeqCst));
    }

    #[test]
    fn borders_and_list_are_rendered_in_order() {
        let mut selector = SettingsSelectorComponent::new(config(), callbacks());
        let lines = selector.render(60.0);
        assert_eq!(pi_tui::utils::strip_ansi(&lines[0]), "\u{2500}".repeat(60));
        assert_eq!(
            pi_tui::utils::strip_ansi(&lines[lines.len() - 1]),
            "\u{2500}".repeat(60)
        );
        assert!(lines.len() > 3);
    }

    #[test]
    fn warnings_submenu_updates_the_shared_state() {
        let received: Rc<RefCell<Vec<WarningSettings>>> = Rc::new(RefCell::new(Vec::new()));
        let sink = Rc::clone(&received);
        let mut callbacks = callbacks();
        callbacks.on_warnings_change = Box::new(move |warnings| sink.borrow_mut().push(warnings));
        let mut selector = SettingsSelectorComponent::new(config(), callbacks);

        // Select the `warnings` row and open the submenu.
        let warnings_index = selector
            .get_settings_list()
            .items()
            .iter()
            .position(|item| item.id == "warnings")
            .unwrap();
        for _ in 0..warnings_index {
            selector.handle_input("\u{1b}[B");
        }
        assert_eq!(
            selector.get_settings_list().selected_index(),
            warnings_index
        );
        selector.handle_input("\r");

        // Change the single submenu item.
        let output = selector.render(60.0);
        assert!(output
            .iter()
            .any(|line| line.contains("Anthropic extra usage")));
        selector.handle_input("\r");
        let recorded = received.borrow();
        assert_eq!(recorded.len(), 1);
        assert_eq!(recorded[0].anthropic_extra_usage, Some(false));
    }

    #[test]
    fn the_select_submenu_marks_the_current_value_and_selects_it() {
        init_theme(Some("prime"), false);
        let selected = Rc::new(RefCell::new(Vec::<String>::new()));
        let sink = Rc::clone(&selected);
        let mut submenu = SelectSubmenu::new(
            "Theme",
            "Select color theme",
            vec![
                SelectItem {
                    value: "dark".to_string(),
                    label: "dark".to_string(),
                    ..Default::default()
                },
                SelectItem {
                    value: "light".to_string(),
                    label: "light".to_string(),
                    ..Default::default()
                },
            ],
            "light",
            Box::new(move |value: &str| sink.borrow_mut().push(value.to_string())),
            Box::new(|| {}),
            None,
        );
        submenu.handle_input("\r");
        assert_eq!(*selected.borrow(), vec!["light".to_string()]);
        let lines = submenu.render(40.0);
        assert!(lines.iter().any(|line| line.contains("Select color theme")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Enter to select \u{00b7} esc to go back")));
    }

    #[test]
    fn the_select_submenu_omits_an_empty_description_block() {
        init_theme(Some("prime"), false);
        let mut submenu = SelectSubmenu::new(
            "Theme",
            "",
            vec![SelectItem {
                value: "dark".to_string(),
                label: "dark".to_string(),
                ..Default::default()
            }],
            "dark",
            Box::new(|_| {}),
            Box::new(|| {}),
            None,
        );
        let lines = submenu.render(40.0);
        assert!(lines[0].contains("Theme"));
    }
}
