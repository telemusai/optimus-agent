//! Port of packages/coding-agent/src/modes/interactive/components/config-selector.ts
//!
//! TUI component for managing package resources (enable/disable).

use std::path::{Component as PathComponent, Path};

use pi_tui::components::input::Input;
use pi_tui::components::spacer::Spacer;
use pi_tui::keybindings::get_keybindings;
use pi_tui::keys::matches_key;
use pi_tui::tui::{Component, Focusable};
use pi_tui::utils::{truncate_to_width, visible_width};

use crate::config::CONFIG_DIR_NAME;
use crate::core::package_manager::{PathMetadata, ResolvedPaths, ResolvedResource};
use crate::core::settings_manager::{PackageSource, SettingsManager};
use crate::modes::interactive::theme::theme::theme;

use super::keybinding_hints::raw_key_hint;
use super::show_images_selector::DynamicBorder;

/// `ResourceType`
pub type ResourceType = &'static str;

/// `RESOURCE_TYPE_LABELS`
pub const RESOURCE_TYPE_LABELS: [(&str, &str); 4] = [
    ("extensions", "Extensions"),
    ("skills", "Skills"),
    ("prompts", "Prompts"),
    ("themes", "Themes"),
];

fn resource_type_label(resource_type: ResourceType) -> &'static str {
    RESOURCE_TYPE_LABELS
        .iter()
        .find(|(key, _)| *key == resource_type)
        .map(|(_, label)| *label)
        .unwrap_or("")
}

/// Port of `ResourceItem`.
#[derive(Debug, Clone, PartialEq)]
pub struct ResourceItem {
    pub path: String,
    pub enabled: bool,
    pub metadata: PathMetadata,
    pub resource_type: ResourceType,
    pub display_name: String,
    pub group_key: String,
    pub subgroup_key: String,
}

/// Port of `ResourceSubgroup`.
#[derive(Debug, Clone, PartialEq)]
pub struct ResourceSubgroup {
    pub r#type: ResourceType,
    pub label: String,
    pub items: Vec<ResourceItem>,
}

/// Port of `ResourceGroup`.
#[derive(Debug, Clone, PartialEq)]
pub struct ResourceGroup {
    pub key: String,
    pub label: String,
    pub scope: String,
    pub origin: String,
    pub source: String,
    pub subgroups: Vec<ResourceSubgroup>,
}

/// Port of `getGroupLabel`.
pub fn get_group_label(metadata: &PathMetadata) -> String {
    if metadata.origin == "package" {
        return format!("{} ({})", metadata.source, metadata.scope);
    }
    // Top-level resources
    if metadata.source == "builtin" {
        return "Built-in".to_string();
    }
    if metadata.source == "auto" {
        return if metadata.scope == "user" {
            format!("User (~/{CONFIG_DIR_NAME}/)")
        } else {
            format!("Project ({CONFIG_DIR_NAME}/)")
        };
    }
    if metadata.scope == "user" {
        "User settings".to_string()
    } else {
        "Project settings".to_string()
    }
}

/// Port of `buildGroups`.
pub fn build_groups(resolved: &ResolvedPaths) -> Vec<ResourceGroup> {
    // `new Map<string, ResourceGroup>()` preserves insertion order.
    let mut group_keys: Vec<String> = Vec::new();
    let mut groups: Vec<ResourceGroup> = Vec::new();

    let mut add_to_group = |resources: &[ResolvedResource], resource_type: ResourceType| {
        for res in resources {
            let group_key = format!(
                "{}:{}:{}",
                res.metadata.origin, res.metadata.scope, res.metadata.source
            );

            let group_index = match group_keys.iter().position(|key| *key == group_key) {
                Some(index) => index,
                None => {
                    group_keys.push(group_key.clone());
                    groups.push(ResourceGroup {
                        key: group_key.clone(),
                        label: get_group_label(&res.metadata),
                        scope: res.metadata.scope.clone(),
                        origin: res.metadata.origin.clone(),
                        source: res.metadata.source.clone(),
                        subgroups: Vec::new(),
                    });
                    groups.len() - 1
                }
            };

            let subgroup_key = format!("{group_key}:{resource_type}");
            let group = &mut groups[group_index];

            let subgroup_index = match group
                .subgroups
                .iter()
                .position(|sg| sg.r#type == resource_type)
            {
                Some(index) => index,
                None => {
                    group.subgroups.push(ResourceSubgroup {
                        r#type: resource_type,
                        label: resource_type_label(resource_type).to_string(),
                        items: Vec::new(),
                    });
                    group.subgroups.len() - 1
                }
            };

            let file_name = basename(&res.path);
            let parent_folder = basename(&dirname(&res.path));
            let display_name = if resource_type == "extensions" && parent_folder != "extensions" {
                format!("{parent_folder}/{file_name}")
            } else if resource_type == "skills" && file_name == "SKILL.md" {
                parent_folder
            } else {
                file_name
            };
            groups[group_index].subgroups[subgroup_index]
                .items
                .push(ResourceItem {
                    path: res.path.clone(),
                    enabled: res.enabled,
                    metadata: res.metadata.clone(),
                    resource_type,
                    display_name,
                    group_key: group_key.clone(),
                    subgroup_key: subgroup_key.clone(),
                });
        }
    };

    add_to_group(&resolved.extensions, "extensions");
    add_to_group(&resolved.skills, "skills");
    add_to_group(&resolved.prompts, "prompts");
    add_to_group(&resolved.themes, "themes");

    // Sort groups: packages first, then top-level; user before project
    groups.sort_by(|a, b| {
        if a.origin != b.origin {
            return if a.origin == "package" {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            };
        }
        if a.scope != b.scope {
            return if a.scope == "user" {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            };
        }
        a.source.cmp(&b.source)
    });

    // Sort subgroups within each group by type order, and items by name
    let type_order = |resource_type: ResourceType| -> usize {
        match resource_type {
            "extensions" => 0,
            "skills" => 1,
            "prompts" => 2,
            _ => 3,
        }
    };
    for group in groups.iter_mut() {
        group
            .subgroups
            .sort_by_key(|subgroup| type_order(subgroup.r#type));
        for subgroup in group.subgroups.iter_mut() {
            // `localeCompare` on the port: plain lexicographic order.
            subgroup
                .items
                .sort_by(|a, b| a.display_name.cmp(&b.display_name));
        }
    }

    groups
}

/// Port of `FlatEntry`.
#[derive(Debug, Clone, PartialEq)]
pub enum FlatEntry {
    Group {
        group: ResourceGroup,
    },
    Subgroup {
        subgroup: ResourceSubgroup,
        group: ResourceGroup,
    },
    Item {
        item: ResourceItem,
    },
}

fn entry_is_item(entry: &FlatEntry) -> bool {
    matches!(entry, FlatEntry::Item { .. })
}

/// Port of `ConfigSelectorHeader`.
pub struct ConfigSelectorHeader;

impl Component for ConfigSelectorHeader {
    fn render(&mut self, width: f64) -> Vec<String> {
        let width = width.max(0.0).floor() as usize;
        let title = theme().bold("Resource Configuration");
        let sep = theme().fg("muted", " \u{00b7} ");
        let hint = raw_key_hint("space", "toggle") + &sep + &raw_key_hint("esc", "close");
        let hint_width = visible_width(&hint);
        let title_width = visible_width(&title);
        let spacing = width.saturating_sub(title_width + hint_width).max(1);

        vec![
            truncate_to_width(
                &format!("{title}{}{hint}", " ".repeat(spacing)),
                width as f64,
                "",
                false,
            ),
            theme().fg("muted", "Type to filter resources"),
        ]
    }

    fn invalidate(&mut self) {}
}

/// `path.basename` (POSIX/win32 agnostic: split on `/` and `\\`).
fn basename(path: &str) -> String {
    let trimmed = path.trim_end_matches(['/', '\\']);
    match trimmed.rsplit(['/', '\\']).next() {
        Some(name) if !name.is_empty() => name.to_string(),
        _ => String::new(),
    }
}

/// `path.dirname`.
fn dirname(path: &str) -> String {
    match path.rfind(['/', '\\']) {
        Some(0) => path[..1].to_string(),
        Some(index) => path[..index].to_string(),
        None => ".".to_string(),
    }
}

/// `path.join`.
fn join_path(base: &str, name: &str) -> String {
    if base.is_empty() {
        return name.to_string();
    }
    let separator = if base.contains('\\') && !base.contains('/') {
        "\\"
    } else {
        "/"
    };
    if base.ends_with('/') || base.ends_with('\\') {
        format!("{base}{name}")
    } else {
        format!("{base}{separator}{name}")
    }
}

/// `path.relative(from, to)`.
fn relative(from: &str, to: &str) -> String {
    let from_parts = normalized_parts(from);
    let to_parts = normalized_parts(to);
    let shared = from_parts
        .iter()
        .zip(to_parts.iter())
        .take_while(|(a, b)| a == b)
        .count();
    let mut parts: Vec<String> = Vec::new();
    for _ in shared..from_parts.len() {
        parts.push("..".to_string());
    }
    parts.extend(to_parts[shared..].iter().cloned());
    parts.join("/")
}

fn normalized_parts(path: &str) -> Vec<String> {
    let mut parts: Vec<String> = Vec::new();
    for piece in path.split(['/', '\\']) {
        match piece {
            "" | "." => continue,
            ".." => {
                parts.pop();
            }
            other => parts.push(other.to_string()),
        }
    }
    let _ = Path::new(path)
        .components()
        .next()
        .map(|c| c != PathComponent::RootDir);
    parts
}

/// Port of `ResourceList`.
pub struct ResourceList {
    groups: Vec<ResourceGroup>,
    flat_items: Vec<FlatEntry>,
    filtered_items: Vec<FlatEntry>,
    selected_index: usize,
    search_input: Input,
    max_visible: usize,
    settings_manager: SettingsManager,
    cwd: String,
    agent_dir: String,
    focused: bool,
    pub on_cancel: Option<Box<dyn FnMut()>>,
    pub on_exit: Option<Box<dyn FnMut()>>,
    pub on_toggle: Option<Box<dyn FnMut(&ResourceItem, bool)>>,
}

impl ResourceList {
    pub fn new(
        groups: Vec<ResourceGroup>,
        settings_manager: SettingsManager,
        cwd: String,
        agent_dir: String,
    ) -> Self {
        let mut list = Self {
            groups,
            flat_items: Vec::new(),
            filtered_items: Vec::new(),
            selected_index: 0,
            search_input: Input::new(),
            max_visible: 15,
            settings_manager,
            cwd,
            agent_dir,
            focused: false,
            on_cancel: None,
            on_exit: None,
            on_toggle: None,
        };
        list.build_flat_list();
        list.filtered_items = list.flat_items.clone();
        list
    }

    /// Port of `buildFlatList`.
    fn build_flat_list(&mut self) {
        self.flat_items = Vec::new();
        for group in self.groups.iter() {
            self.flat_items.push(FlatEntry::Group {
                group: group.clone(),
            });
            for subgroup in group.subgroups.iter() {
                self.flat_items.push(FlatEntry::Subgroup {
                    subgroup: subgroup.clone(),
                    group: group.clone(),
                });
                for item in subgroup.items.iter() {
                    self.flat_items.push(FlatEntry::Item { item: item.clone() });
                }
            }
        }
        // Start selection on first item (not header)
        let first_item = self
            .flat_items
            .iter()
            .position(entry_is_item)
            .map(|index| index as i64)
            .unwrap_or(-1);
        self.selected_index = if first_item < 0 {
            0
        } else {
            first_item as usize
        };
    }

    /// Port of `findNextItem`.
    fn find_next_item(&self, from_index: usize, direction: i64) -> usize {
        let mut index = from_index as i64 + direction;
        while index >= 0 && (index as usize) < self.filtered_items.len() {
            if entry_is_item(&self.filtered_items[index as usize]) {
                return index as usize;
            }
            index += direction;
        }
        from_index // Stay at current if no item found
    }

    /// Port of `filterItems`.
    fn filter_items(&mut self, query: &str) {
        if query.trim().is_empty() {
            self.filtered_items = self.flat_items.clone();
            self.select_first_item();
            return;
        }

        let lower_query = query.to_lowercase();
        let mut matching_items: Vec<(String, ResourceType)> = Vec::new();

        for entry in self.flat_items.iter() {
            if let FlatEntry::Item { item } = entry {
                if item.display_name.to_lowercase().contains(&lower_query)
                    || item.resource_type.to_lowercase().contains(&lower_query)
                    || item.path.to_lowercase().contains(&lower_query)
                {
                    matching_items.push((item.path.clone(), item.resource_type));
                }
            }
        }

        // Find which subgroups and groups contain matching items
        let mut matching_subgroups: Vec<String> = Vec::new();
        let mut matching_groups: Vec<String> = Vec::new();
        for group in self.groups.iter() {
            for subgroup in group.subgroups.iter() {
                for item in subgroup.items.iter() {
                    if matching_items.contains(&(item.path.clone(), item.resource_type)) {
                        let subgroup_key = format!("{}:{}", group.key, subgroup.r#type);
                        if !matching_subgroups.contains(&subgroup_key) {
                            matching_subgroups.push(subgroup_key);
                        }
                        if !matching_groups.contains(&group.key) {
                            matching_groups.push(group.key.clone());
                        }
                    }
                }
            }
        }

        self.filtered_items = Vec::new();
        for entry in self.flat_items.iter() {
            match entry {
                FlatEntry::Group { group } if matching_groups.contains(&group.key) => {
                    self.filtered_items.push(entry.clone())
                }
                FlatEntry::Subgroup { subgroup, group }
                    if matching_subgroups
                        .contains(&format!("{}:{}", group.key, subgroup.r#type)) =>
                {
                    self.filtered_items.push(entry.clone())
                }
                FlatEntry::Item { item }
                    if matching_items.contains(&(item.path.clone(), item.resource_type)) =>
                {
                    self.filtered_items.push(entry.clone())
                }
                _ => {}
            }
        }

        self.select_first_item();
    }

    /// Port of `selectFirstItem`.
    fn select_first_item(&mut self) {
        let first_item_index = self
            .filtered_items
            .iter()
            .position(entry_is_item)
            .map(|index| index as i64)
            .unwrap_or(-1);
        self.selected_index = if first_item_index >= 0 {
            first_item_index as usize
        } else {
            0
        };
    }

    /// Port of `updateItem`.
    pub fn update_item(&mut self, item: &mut ResourceItem, enabled: bool) {
        item.enabled = enabled;
        // Update in groups too
        for group in self.groups.iter_mut() {
            for subgroup in group.subgroups.iter_mut() {
                let found = subgroup
                    .items
                    .iter_mut()
                    .find(|i| i.path == item.path && i.resource_type == item.resource_type);
                if let Some(found) = found {
                    found.enabled = enabled;
                }
            }
        }
        // TypeScript keeps shared item references; Rust owns copies in each view.
        for entry in self
            .flat_items
            .iter_mut()
            .chain(self.filtered_items.iter_mut())
        {
            if let FlatEntry::Item { item: visible } = entry {
                if visible.path == item.path && visible.resource_type == item.resource_type {
                    visible.enabled = enabled;
                }
            }
        }
    }

    /// Port of `toggleResource`.
    fn toggle_resource(&mut self, item: &ResourceItem, enabled: bool) {
        if item.metadata.origin == "top-level" {
            self.toggle_top_level_resource(item, enabled);
        } else {
            self.toggle_package_resource(item, enabled);
        }
    }

    /// Port of `getTopLevelBaseDir`.
    fn get_top_level_base_dir(&self, scope: &str) -> String {
        if scope == "project" {
            join_path(&self.cwd, CONFIG_DIR_NAME)
        } else {
            self.agent_dir.clone()
        }
    }

    /// Port of `getResourcePattern`.
    fn get_resource_pattern(&self, item: &ResourceItem) -> String {
        // Built-in resources live under the package install dir; their override
        // patterns are matched relative to metadata.baseDir, not the config dir.
        if item.metadata.source == "builtin" {
            if let Some(base_dir) = &item.metadata.base_dir {
                return relative(base_dir, &item.path);
            }
        }
        let scope = item.metadata.scope.as_str();
        let base_dir = self.get_top_level_base_dir(scope);
        relative(&base_dir, &item.path)
    }

    /// Port of `getPackageResourcePattern`.
    fn get_package_resource_pattern(&self, item: &ResourceItem) -> String {
        let base_dir = match &item.metadata.base_dir {
            Some(base_dir) => base_dir.clone(),
            None => dirname(&item.path),
        };
        relative(&base_dir, &item.path)
    }

    /// Port of `toggleTopLevelResource`.
    fn toggle_top_level_resource(&mut self, item: &ResourceItem, enabled: bool) {
        let scope = item.metadata.scope.clone();
        let settings = if scope == "project" {
            self.settings_manager.get_project_settings()
        } else {
            self.settings_manager.get_global_settings()
        };

        let array_key = item.resource_type;
        let current = string_array_from_settings(&settings, array_key);

        // Generate pattern for this resource
        let pattern = self.get_resource_pattern(item);
        let disable_pattern = format!("-{pattern}");
        let enable_pattern = format!("+{pattern}");

        // Filter out existing patterns for this resource
        let mut updated: Vec<String> = current
            .into_iter()
            .filter(|p| {
                let stripped = match p.chars().next() {
                    Some('!') | Some('+') | Some('-') => &p[1..],
                    _ => p.as_str(),
                };
                stripped != pattern
            })
            .collect();

        if enabled {
            updated.push(enable_pattern);
        } else {
            updated.push(disable_pattern);
        }

        self.apply_top_level_paths(&scope, array_key, updated);
    }

    fn apply_top_level_paths(
        &mut self,
        scope: &str,
        array_key: ResourceType,
        updated: Vec<String>,
    ) {
        if scope == "project" {
            match array_key {
                "extensions" => self.settings_manager.set_project_extension_paths(updated),
                "skills" => self.settings_manager.set_project_skill_paths(updated),
                "prompts" => self
                    .settings_manager
                    .set_project_prompt_template_paths(updated),
                "themes" => self.settings_manager.set_project_theme_paths(updated),
                _ => {}
            }
        } else {
            match array_key {
                "extensions" => self.settings_manager.set_extension_paths(updated),
                "skills" => self.settings_manager.set_skill_paths(updated),
                "prompts" => self.settings_manager.set_prompt_template_paths(updated),
                "themes" => self.settings_manager.set_theme_paths(updated),
                _ => {}
            }
        }
    }

    /// Port of `togglePackageResource`.
    fn toggle_package_resource(&mut self, item: &ResourceItem, enabled: bool) {
        let scope = item.metadata.scope.clone();
        let settings = if scope == "project" {
            self.settings_manager.get_project_settings()
        } else {
            self.settings_manager.get_global_settings()
        };

        let mut packages: Vec<PackageSource> = packages_from_settings(&settings);
        let pkg_index = packages.iter().position(|pkg| match pkg {
            PackageSource::Source(source) => *source == item.metadata.source,
            PackageSource::Filtered(filtered) => filtered.source == item.metadata.source,
        });

        let Some(pkg_index) = pkg_index else {
            return;
        };

        // Convert string to object form if needed
        if let PackageSource::Source(source) = packages[pkg_index].clone() {
            packages[pkg_index] =
                PackageSource::Filtered(crate::core::settings_manager::FilteredPackageSource {
                    source,
                    extensions: None,
                    skills: None,
                    prompts: None,
                    themes: None,
                });
        }

        // Get the resource array for this type
        let array_key = item.resource_type;
        let current = match &packages[pkg_index] {
            PackageSource::Filtered(filtered) => filtered_array(filtered, array_key),
            PackageSource::Source(_) => Vec::new(),
        };

        // Generate pattern relative to package root
        let pattern = self.get_package_resource_pattern(item);
        let disable_pattern = format!("-{pattern}");
        let enable_pattern = format!("+{pattern}");

        // Filter out existing patterns for this resource
        let mut updated: Vec<String> = current
            .into_iter()
            .filter(|p| {
                let stripped = match p.chars().next() {
                    Some('!') | Some('+') | Some('-') => &p[1..],
                    _ => p.as_str(),
                };
                stripped != pattern
            })
            .collect();

        if enabled {
            updated.push(enable_pattern);
        } else {
            updated.push(disable_pattern);
        }

        if let PackageSource::Filtered(filtered) = &mut packages[pkg_index] {
            let value = if updated.is_empty() {
                None
            } else {
                Some(updated.clone())
            };
            match array_key {
                "extensions" => filtered.extensions = value,
                "skills" => filtered.skills = value,
                "prompts" => filtered.prompts = value,
                "themes" => filtered.themes = value,
                _ => {}
            }
        }

        // Clean up empty filter object
        let has_filters = match &packages[pkg_index] {
            PackageSource::Filtered(filtered) => {
                filtered.extensions.is_some()
                    || filtered.skills.is_some()
                    || filtered.prompts.is_some()
                    || filtered.themes.is_some()
            }
            PackageSource::Source(_) => false,
        };
        if !has_filters {
            if let PackageSource::Filtered(filtered) = &packages[pkg_index] {
                packages[pkg_index] = PackageSource::Source(filtered.source.clone());
            }
        }

        let values: Vec<serde_json::Value> = packages
            .iter()
            .map(|pkg| serde_json::to_value(pkg).unwrap_or(serde_json::Value::Null))
            .collect();
        if scope == "project" {
            self.settings_manager.set_project_packages(values);
        } else {
            self.settings_manager.set_packages(values);
        }
    }
}

/// The typed settings arrays the TypeScript reads by key (`settings[arrayKey] ?? []`).
fn string_array_from_settings(
    settings: &crate::core::settings_manager::Settings,
    key: &str,
) -> Vec<String> {
    match settings.get(key) {
        Some(serde_json::Value::Array(items)) => items
            .iter()
            .filter_map(|item| item.as_str().map(|text| text.to_string()))
            .collect(),
        _ => Vec::new(),
    }
}

/// `settings.packages ?? []` as the `PackageSource[]` union.
fn packages_from_settings(
    settings: &crate::core::settings_manager::Settings,
) -> Vec<PackageSource> {
    match settings.get("packages") {
        Some(serde_json::Value::Array(items)) => items
            .iter()
            .filter_map(|item| serde_json::from_value::<PackageSource>(item.clone()).ok())
            .collect(),
        _ => Vec::new(),
    }
}

fn filtered_array(
    filtered: &crate::core::settings_manager::FilteredPackageSource,
    key: ResourceType,
) -> Vec<String> {
    match key {
        "extensions" => filtered.extensions.clone(),
        "skills" => filtered.skills.clone(),
        "prompts" => filtered.prompts.clone(),
        "themes" => filtered.themes.clone(),
        _ => None,
    }
    .unwrap_or_default()
}

impl Component for ResourceList {
    fn render(&mut self, width: f64) -> Vec<String> {
        let width = width.max(0.0).floor() as usize;
        let mut lines: Vec<String> = Vec::new();

        // Search input
        lines.extend(self.search_input.render(width as f64));
        lines.push(String::new());

        if self.filtered_items.is_empty() {
            lines.push(theme().fg("muted", "  No resources found"));
            return lines;
        }

        // Calculate visible range
        let start_index = ((self.selected_index as i64 - (self.max_visible / 2) as i64)
            .min(self.filtered_items.len().saturating_sub(self.max_visible) as i64))
        .max(0) as usize;
        let end_index = (start_index + self.max_visible).min(self.filtered_items.len());

        for i in start_index..end_index {
            let entry = &self.filtered_items[i];
            let is_selected = i == self.selected_index;

            match entry {
                FlatEntry::Group { group } => {
                    // Main group header (no cursor)
                    let group_line = theme().fg("accent", &theme().bold(&group.label));
                    lines.push(truncate_to_width(
                        &format!("  {group_line}"),
                        width as f64,
                        "",
                        false,
                    ));
                }
                FlatEntry::Subgroup { subgroup, .. } => {
                    // Subgroup header (indented, no cursor)
                    let subgroup_line = theme().fg("muted", &subgroup.label);
                    lines.push(truncate_to_width(
                        &format!("    {subgroup_line}"),
                        width as f64,
                        "",
                        false,
                    ));
                }
                FlatEntry::Item { item } => {
                    // Resource item (cursor only on items)
                    let cursor = if is_selected { "> " } else { "  " };
                    let checkbox = if item.enabled {
                        theme().fg("success", "[x]")
                    } else {
                        theme().fg("dim", "[ ]")
                    };
                    let name = if is_selected {
                        theme().bold(&item.display_name)
                    } else {
                        item.display_name.clone()
                    };
                    lines.push(truncate_to_width(
                        &format!("{cursor}    {checkbox} {name}"),
                        width as f64,
                        "...",
                        false,
                    ));
                }
            }
        }

        // Scroll indicator
        if start_index > 0 || end_index < self.filtered_items.len() {
            let item_count = self
                .filtered_items
                .iter()
                .filter(|entry| entry_is_item(entry))
                .count();
            let current_item_index = self.filtered_items[..self.selected_index]
                .iter()
                .filter(|entry| entry_is_item(entry))
                .count()
                + 1;
            lines.push(theme().fg("dim", &format!("  ({current_item_index}/{item_count})")));
        }

        lines
    }

    fn handle_input(&mut self, data: &str) {
        let kb = get_keybindings();

        if self.filtered_items.is_empty()
            && (kb.matches(data, "tui.select.up")
                || kb.matches(data, "tui.select.down")
                || kb.matches(data, "tui.select.pageUp")
                || kb.matches(data, "tui.select.pageDown"))
        {
            return;
        }

        if kb.matches(data, "tui.select.up") {
            self.selected_index = self.find_next_item(self.selected_index, -1);
            return;
        }
        if kb.matches(data, "tui.select.down") {
            self.selected_index = self.find_next_item(self.selected_index, 1);
            return;
        }
        if kb.matches(data, "tui.select.pageUp") {
            // Jump up by maxVisible, then find nearest item
            let mut target = self.selected_index.saturating_sub(self.max_visible);
            while target < self.filtered_items.len() && !entry_is_item(&self.filtered_items[target])
            {
                target += 1;
            }
            if target < self.filtered_items.len() {
                self.selected_index = target;
            }
            return;
        }
        if kb.matches(data, "tui.select.pageDown") {
            // Jump down by maxVisible, then find nearest item
            let mut target = (self.selected_index + self.max_visible)
                .min(self.filtered_items.len().saturating_sub(1))
                as i64;
            while target >= 0
                && !self
                    .filtered_items
                    .get(target as usize)
                    .map(entry_is_item)
                    .unwrap_or(false)
            {
                target -= 1;
            }
            if target >= 0 {
                self.selected_index = target as usize;
            }
            return;
        }
        if kb.matches(data, "tui.select.cancel") {
            if let Some(callback) = self.on_cancel.as_mut() {
                callback();
            }
            return;
        }
        if matches_key(data, "ctrl+c") {
            if let Some(callback) = self.on_exit.as_mut() {
                callback();
            }
            return;
        }
        let space_toggles = data == " " && self.search_input.get_value().is_empty();
        if space_toggles || kb.matches(data, "tui.select.confirm") {
            let entry = self.filtered_items.get(self.selected_index).cloned();
            if let Some(FlatEntry::Item { mut item }) = entry {
                let new_enabled = !item.enabled;
                self.toggle_resource(&item, new_enabled);
                self.update_item(&mut item, new_enabled);
                if let Some(callback) = self.on_toggle.as_mut() {
                    callback(&item, new_enabled);
                }
            }
            return;
        }

        // Pass to search input
        self.search_input.handle_input(data);
        let value = self.search_input.get_value().to_string();
        self.filter_items(&value);
    }

    fn invalidate(&mut self) {}

    fn as_focusable(&mut self) -> Option<&mut dyn Focusable> {
        Some(self)
    }
}

impl Focusable for ResourceList {
    fn focused(&self) -> bool {
        self.focused
    }

    fn set_focused(&mut self, focused: bool) {
        self.focused = focused;
        self.search_input.set_focused(focused);
    }
}

/// Port of `ConfigSelectorComponent`.
pub struct ConfigSelectorComponent {
    /// `Spacer(1)`, `DynamicBorder`, `Spacer(1)`, header, `Spacer(1)`.
    leading: Vec<Box<dyn Component>>,
    resource_list: ResourceList,
    /// `Spacer(1)` and the bottom `DynamicBorder`.
    trailing: Vec<Box<dyn Component>>,
    focused: bool,
}

impl ConfigSelectorComponent {
    pub fn new(
        resolved_paths: &ResolvedPaths,
        settings_manager: SettingsManager,
        cwd: String,
        agent_dir: String,
        on_close: Box<dyn FnMut()>,
        on_exit: Box<dyn FnMut()>,
        request_render: Box<dyn FnMut()>,
    ) -> Self {
        let groups = build_groups(resolved_paths);

        // Add header
        let leading: Vec<Box<dyn Component>> = vec![
            Box::new(Spacer::new(1)),
            Box::new(DynamicBorder::new(None)),
            Box::new(Spacer::new(1)),
            Box::new(ConfigSelectorHeader),
            Box::new(Spacer::new(1)),
        ];

        // Resource list
        let mut resource_list = ResourceList::new(groups, settings_manager, cwd, agent_dir);
        resource_list.on_cancel = Some(on_close);
        resource_list.on_exit = Some(on_exit);
        let mut request_render = request_render;
        resource_list.on_toggle = Some(Box::new(move |_item, _enabled| request_render()));

        // Bottom border
        let trailing: Vec<Box<dyn Component>> =
            vec![Box::new(Spacer::new(1)), Box::new(DynamicBorder::new(None))];

        Self {
            leading,
            resource_list,
            trailing,
            focused: false,
        }
    }

    /// Port of `getResourceList`.
    pub fn get_resource_list(&mut self) -> &mut ResourceList {
        &mut self.resource_list
    }

    /// The groups the list was built from, exposed for tests.
    pub fn groups(&self) -> &[ResourceGroup] {
        &self.resource_list.groups
    }

    pub fn set_request_render(&mut self, request_render: Box<dyn FnMut()>) {
        let mut request_render = request_render;
        self.resource_list.on_toggle = Some(Box::new(move |_item, _enabled| request_render()));
    }
}

impl Component for ConfigSelectorComponent {
    fn render(&mut self, width: f64) -> Vec<String> {
        let mut lines: Vec<String> = Vec::new();
        // `Spacer(1)`, `DynamicBorder`, `Spacer(1)`, header, `Spacer(1)`,
        // `resourceList`, `Spacer(1)`, `DynamicBorder` - the container order.
        for child in self.leading.iter_mut() {
            lines.extend(child.render(width));
        }
        lines.extend(self.resource_list.render(width));
        for child in self.trailing.iter_mut() {
            lines.extend(child.render(width));
        }
        lines
    }

    fn handle_input(&mut self, data: &str) {
        self.resource_list.handle_input(data);
    }

    fn invalidate(&mut self) {
        for child in self.leading.iter_mut() {
            child.invalidate();
        }
        self.resource_list.invalidate();
        for child in self.trailing.iter_mut() {
            child.invalidate();
        }
    }

    fn as_focusable(&mut self) -> Option<&mut dyn Focusable> {
        Some(self)
    }
}

impl Focusable for ConfigSelectorComponent {
    fn focused(&self) -> bool {
        self.focused
    }

    fn set_focused(&mut self, focused: bool) {
        self.focused = focused;
        self.resource_list.set_focused(focused);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::diagnostics::ResourceDiagnostic;
    use std::cell::RefCell;
    use std::rc::Rc;

    fn metadata(origin: &str, scope: &str, source: &str, base_dir: Option<&str>) -> PathMetadata {
        PathMetadata {
            source: source.to_string(),
            scope: scope.to_string(),
            origin: origin.to_string(),
            base_dir: base_dir.map(|value| value.to_string()),
        }
    }

    fn resource(path: &str, metadata: PathMetadata) -> ResolvedResource {
        ResolvedResource {
            path: path.to_string(),
            enabled: true,
            metadata,
        }
    }

    fn paths() -> ResolvedPaths {
        ResolvedPaths {
            extensions: vec![resource(
                "/agent/extensions/foo/index.ts",
                metadata("top-level", "user", "auto", None),
            )],
            skills: vec![resource(
                "/agent/skills/pdf/SKILL.md",
                metadata("top-level", "user", "auto", None),
            )],
            prompts: vec![resource(
                "/pkg/prompts/hello.md",
                metadata("package", "user", "npm:thing", Some("/pkg")),
            )],
            themes: vec![resource(
                "/agent/themes/dark.json",
                metadata("top-level", "project", "settings", None),
            )],
            diagnostics: Vec::<ResourceDiagnostic>::new(),
        }
    }

    #[test]
    fn group_labels_match_the_typescript_cases() {
        assert_eq!(
            get_group_label(&metadata("package", "user", "npm:thing", None)),
            "npm:thing (user)"
        );
        assert_eq!(
            get_group_label(&metadata("top-level", "user", "builtin", None)),
            "Built-in"
        );
        assert_eq!(
            get_group_label(&metadata("top-level", "user", "auto", None)),
            format!("User (~/{CONFIG_DIR_NAME}/)")
        );
        assert_eq!(
            get_group_label(&metadata("top-level", "project", "auto", None)),
            format!("Project ({CONFIG_DIR_NAME}/)")
        );
        assert_eq!(
            get_group_label(&metadata("top-level", "user", "settings", None)),
            "User settings"
        );
        assert_eq!(
            get_group_label(&metadata("top-level", "project", "settings", None)),
            "Project settings"
        );
    }

    #[test]
    fn packages_sort_before_top_level_and_user_before_project() {
        let groups = build_groups(&paths());
        assert!(groups.len() >= 3);
        assert_eq!(groups[0].origin, "package");
        assert!(groups
            .iter()
            .filter(|group| group.origin == "top-level")
            .all(|group| group.scope == "user" || group.scope == "project"));
    }

    #[test]
    fn display_names_use_the_folder_for_nested_extensions_and_skill_files() {
        let groups = build_groups(&paths());
        let items: Vec<&ResourceItem> = groups
            .iter()
            .flat_map(|group| group.subgroups.iter())
            .flat_map(|subgroup| subgroup.items.iter())
            .collect();
        assert!(items.iter().any(|item| item.display_name == "foo/index.ts"));
        assert!(items.iter().any(|item| item.display_name == "pdf"));
        assert!(items.iter().any(|item| item.display_name == "hello.md"));
    }

    #[test]
    fn subgroups_sort_by_type_order_and_items_by_name() {
        let mut input = paths();
        input.extensions = vec![
            resource("/a/zzz.ts", metadata("top-level", "user", "auto", None)),
            resource("/a/aaa.ts", metadata("top-level", "user", "auto", None)),
        ];
        let groups = build_groups(&input);
        let group = groups
            .iter()
            .find(|group| group.subgroups.iter().any(|s| s.r#type == "extensions"))
            .unwrap();
        let extensions = group
            .subgroups
            .iter()
            .find(|subgroup| subgroup.r#type == "extensions")
            .unwrap();
        assert_eq!(
            extensions
                .items
                .iter()
                .map(|i| i.display_name.clone())
                .collect::<Vec<_>>(),
            vec!["a/aaa.ts".to_string(), "a/zzz.ts".to_string()]
        );
        let order: Vec<&str> = group.subgroups.iter().map(|s| s.r#type).collect();
        let mut sorted = order.clone();
        sorted.sort_by_key(|t| match *t {
            "extensions" => 0,
            "skills" => 1,
            "prompts" => 2,
            _ => 3,
        });
        assert_eq!(order, sorted);
    }

    #[test]
    fn relative_and_path_helpers_match_node_semantics() {
        assert_eq!(basename("/a/b/c.ts"), "c.ts");
        assert_eq!(basename("/a/b/"), "b");
        assert_eq!(dirname("/a/b/c.ts"), "/a/b");
        assert_eq!(dirname("c.ts"), ".");
        assert_eq!(join_path("/a", "b"), "/a/b");
        assert_eq!(relative("/a/b", "/a/b/c/d.ts"), "c/d.ts");
        assert_eq!(relative("/a/b/c", "/a/b/d.ts"), "../d.ts");
        assert_eq!(relative("/a/b", "/a/b"), "");
    }

    #[test]
    fn filtering_keeps_matching_groups_subgroups_and_items() {
        let groups = build_groups(&paths());
        let mut list = ResourceList::new(
            groups,
            SettingsManager::in_memory(Default::default()),
            "/work".to_string(),
            "/agent".to_string(),
        );
        list.filter_items("pdf");
        assert!(list.filtered_items.iter().all(|entry| match entry {
            FlatEntry::Group { group } => group.subgroups.iter().any(|s| s.r#type == "skills"),
            FlatEntry::Subgroup { subgroup, .. } => subgroup.r#type == "skills",
            FlatEntry::Item { item } => item.display_name == "pdf",
        }));
        assert_eq!(list.selected_index, 2);
        assert!(entry_is_item(&list.filtered_items[list.selected_index]));

        list.filter_items("nothing-matches-this");
        assert!(list.filtered_items.is_empty());
    }

    #[test]
    fn an_empty_filter_restores_every_entry_and_selects_the_first_item() {
        let groups = build_groups(&paths());
        let mut list = ResourceList::new(
            groups,
            SettingsManager::in_memory(Default::default()),
            "/work".to_string(),
            "/agent".to_string(),
        );
        let total = list.flat_items.len();
        list.filter_items("  ");
        assert_eq!(list.filtered_items.len(), total);
        assert!(entry_is_item(&list.filtered_items[list.selected_index]));
    }

    #[test]
    fn navigation_skips_header_entries() {
        let groups = build_groups(&paths());
        let mut list = ResourceList::new(
            groups,
            SettingsManager::in_memory(Default::default()),
            "/work".to_string(),
            "/agent".to_string(),
        );
        for _ in 0..list.filtered_items.len() {
            list.handle_input("\u{1b}[B");
            assert!(entry_is_item(&list.filtered_items[list.selected_index]));
        }
        for _ in 0..list.filtered_items.len() {
            list.handle_input("\u{1b}[A");
            assert!(entry_is_item(&list.filtered_items[list.selected_index]));
        }
    }

    fn empty_resource_list() -> ResourceList {
        ResourceList::new(
            Vec::new(),
            SettingsManager::in_memory(Default::default()),
            "/work".to_string(),
            "/agent".to_string(),
        )
    }

    #[test]
    fn empty_resource_navigation_is_a_noop_and_still_renders() {
        crate::modes::interactive::theme::theme::init_theme(Some("prime"), false);
        let mut list = empty_resource_list();
        for key in ["\u{1b}[A", "\u{1b}[B", "\u{1b}[5~", "\u{1b}[6~"] {
            list.handle_input(key);
            assert_eq!(list.selected_index, 0);
            assert!(list.render(60.0).iter().any(|line| line.contains("No resources found")));
        }
    }

    #[test]
    fn filtered_empty_resource_navigation_is_a_noop_and_still_renders() {
        crate::modes::interactive::theme::theme::init_theme(Some("prime"), false);
        let groups = build_groups(&paths());
        let mut list = ResourceList::new(
            groups,
            SettingsManager::in_memory(Default::default()),
            "/work".to_string(),
            "/agent".to_string(),
        );
        list.filter_items("nothing-matches-this");
        for key in ["\u{1b}[A", "\u{1b}[B", "\u{1b}[5~", "\u{1b}[6~"] {
            list.handle_input(key);
            assert_eq!(list.selected_index, 0);
            assert!(list.render(60.0).iter().any(|line| line.contains("No resources found")));
        }
    }

    #[test]
    fn cancellation_precedes_ctrl_c_exit_unless_rebound() {
        let previous = pi_tui::keybindings::get_keybindings();
        pi_tui::keybindings::set_keybindings(pi_tui::keybindings::KeybindingsManager::new(
            pi_tui::keybindings::tui_keybindings(),
            Default::default(),
        ));
        let groups = build_groups(&paths());
        let mut list = ResourceList::new(
            groups,
            SettingsManager::in_memory(Default::default()),
            "/work".to_string(),
            "/agent".to_string(),
        );
        let cancelled = Rc::new(RefCell::new(false));
        let exited = Rc::new(RefCell::new(false));
        let cancel_flag = Rc::clone(&cancelled);
        let exit_flag = Rc::clone(&exited);
        list.on_cancel = Some(Box::new(move || *cancel_flag.borrow_mut() = true));
        list.on_exit = Some(Box::new(move || *exit_flag.borrow_mut() = true));
        list.handle_input("\u{1b}");
        assert!(*cancelled.borrow());
        *cancelled.borrow_mut() = false;
        list.handle_input("\u{3}");
        assert!(*cancelled.borrow());
        assert!(!*exited.borrow());
        let mut configured = pi_tui::keybindings::get_keybindings();
        let mut bindings = configured.get_user_bindings();
        bindings.insert("tui.select.cancel".to_string(), vec!["escape".to_string()]);
        configured.set_user_bindings(bindings);
        pi_tui::keybindings::set_keybindings(configured);
        list.handle_input("\u{3}");
        assert!(*exited.borrow());
        pi_tui::keybindings::set_keybindings(previous);
    }

    #[test]
    fn space_toggles_an_item_and_reports_the_new_state() {
        let groups = build_groups(&paths());
        let mut list = ResourceList::new(
            groups,
            SettingsManager::in_memory(Default::default()),
            "/work".to_string(),
            "/agent".to_string(),
        );
        let toggled: Rc<RefCell<Vec<(String, bool)>>> = Rc::new(RefCell::new(Vec::new()));
        let sink = Rc::clone(&toggled);
        list.on_toggle = Some(Box::new(move |item, enabled| {
            assert_eq!(item.enabled, enabled);
            sink.borrow_mut().push((item.path.clone(), enabled))
        }));
        list.handle_input(" ");
        let recorded = toggled.borrow();
        assert_eq!(recorded.len(), 1);
        assert!(!recorded[0].1, "an enabled item toggles to disabled");
        drop(recorded);
        list.handle_input(" ");
        let recorded = toggled.borrow();
        assert_eq!(recorded.len(), 2);
        assert_eq!(recorded[0].0, recorded[1].0);
        assert!(recorded[1].1, "the same item toggles back to enabled");
    }

    #[test]
    fn active_search_inserts_spaces_but_enter_still_toggles_package_and_top_level_items() {
        for package in [true, false] {
            let mut input = paths();
            if package {
                input.extensions.clear();
                input.skills.clear();
                input.themes.clear();
                input.prompts[0].path = "/pkg/prompts/my note.md".into();
            } else {
                input.extensions[0].path = "/agent/extensions/my item/index.ts".into();
                input.skills.clear();
                input.prompts.clear();
                input.themes.clear();
            }
            let mut settings = serde_json::Map::new();
            if package {
                settings.insert("packages".into(), serde_json::json!(["npm:thing"]));
            }
            let mut list = ResourceList::new(
                build_groups(&input),
                SettingsManager::in_memory(settings),
                "/work".to_string(),
                "/agent".to_string(),
            );
            let toggled = Rc::new(RefCell::new(0usize));
            let sink = Rc::clone(&toggled);
            list.on_toggle = Some(Box::new(move |_item, _enabled| *sink.borrow_mut() += 1));
            let settings_before = list.settings_manager.get_global_settings();

            for ch in "my".chars() {
                list.handle_input(&ch.to_string());
            }
            list.handle_input(" ");

            assert_eq!(list.search_input.get_value(), "my ");
            assert_eq!(*toggled.borrow(), 0);
            assert_eq!(list.settings_manager.get_global_settings(), settings_before);

            list.handle_input("\u{7f}");
            assert_eq!(list.search_input.get_value(), "my");
            list.handle_input("\r");
            assert_eq!(*toggled.borrow(), 1, "Enter remains intentional activation");
            assert_ne!(list.settings_manager.get_global_settings(), settings_before);
        }
    }

    #[test]
    fn typing_filters_through_the_search_input() {
        let groups = build_groups(&paths());
        let mut list = ResourceList::new(
            groups,
            SettingsManager::in_memory(Default::default()),
            "/work".to_string(),
            "/agent".to_string(),
        );
        for ch in "pdf".chars() {
            list.handle_input(&ch.to_string());
        }
        assert_eq!(list.search_input.get_value(), "pdf");
        assert!(list.filtered_items.len() < list.flat_items.len());
    }

    #[test]
    fn the_selector_forwards_focus_and_input_to_its_resource_list() {
        crate::modes::interactive::theme::theme::init_theme(Some("prime"), false);
        let mut selector = ConfigSelectorComponent::new(
            &paths(),
            SettingsManager::in_memory(Default::default()),
            "/work".to_string(),
            "/agent".to_string(),
            Box::new(|| {}),
            Box::new(|| {}),
            Box::new(|| {}),
        );
        assert!(!selector.focused());
        selector.set_focused(true);
        assert!(selector.focused());
        assert!(selector.get_resource_list().focused());
        let lines = selector.render(40.0);
        assert!(lines
            .iter()
            .any(|line| line.contains("Resource Configuration")));
        assert!(lines.iter().any(|line| line.contains("\u{2500}")));
    }

    #[test]
    fn the_header_truncates_to_the_viewport() {
        crate::modes::interactive::theme::theme::init_theme(Some("prime"), false);
        let mut header = ConfigSelectorHeader;
        let lines = header.render(10.0);
        assert_eq!(visible_width(&lines[0]), 10);
        assert_eq!(lines[1], theme().fg("muted", "Type to filter resources"));
    }
}
